"""Unit tests for the pure-logic pieces (no network, no model)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import fields
from pathlib import Path

import pytest

from nbclaw.agent_runner import MAX_CHAT_SESSIONS, AgentRunner
from nbclaw.commands import CronAddError, parse_cron_add
from nbclaw.config import _TOML_SCHEMA, Config, build_config
from nbclaw.nl_schedule import ParseError, _build, _extract_json
from nbclaw.scheduler import (
    MAX_CRONS_PER_CONVERSATION,
    Scheduler,
    next_fire,
    parse_duration,
    slugify,
    validate_schedule,
)
from nbclaw.signal_client import Conversation, SignalClient, _parse_sse, _to_message
from swival import Result
from swival.mcp_client import McpManager


# --- duration / schedule parsing -------------------------------------
def test_parse_duration():
    assert parse_duration("30s") == 30
    assert parse_duration("5m") == 300
    assert parse_duration("2h") == 7200
    assert parse_duration("1d") == 86400
    with pytest.raises(ValueError):
        parse_duration("nonsense")
    # A zero duration would fire every tick — reject it.
    with pytest.raises(ValueError):
        parse_duration("0s")
    with pytest.raises(ValueError):
        parse_duration("0m")


def test_every_validation():
    # '@every' with no/blank argument is a clean ValueError, not IndexError.
    for bad in ("@every", "@every   ", "@every nope", "@every 0s"):
        with pytest.raises(ValueError):
            validate_schedule(bad)
    # And a well-formed one still works.
    now = 1_000_000.0
    assert next_fire("@every 90s", now) == now + 90


def test_next_fire_forms():
    now = 1_000_000.0
    assert next_fire("@every 1m", now) == now + 60
    assert next_fire("@hourly", now) > now
    # 5-field cron must produce a strictly future time.
    assert next_fire("*/5 * * * *", now) > now


def test_validate_schedule_rejects_garbage():
    with pytest.raises(ValueError):
        validate_schedule("not a schedule at all")


# --- cron add parsing -------------------------------------------------
def test_parse_cron_add_ok():
    name, sched, prompt = parse_cron_add("standup 0 9 * * 1-5 | summarize git log")
    assert name == "standup"
    assert sched == "0 9 * * 1-5"
    assert prompt == "summarize git log"


def test_parse_cron_add_every():
    name, sched, prompt = parse_cron_add("ping @every 30m | check the server")
    assert name == "ping"
    assert sched == "@every 30m"
    assert prompt == "check the server"


def test_parse_cron_add_errors():
    with pytest.raises(CronAddError):
        parse_cron_add("noseparator here")
    with pytest.raises(CronAddError):
        parse_cron_add("name 0 9 * * * |   ")


# --- scheduler persistence + ticking ---------------------------------
def test_scheduler_roundtrip(tmp_path):
    path = tmp_path / "crons.json"
    sched = Scheduler(path)
    conv = Conversation(recipient="+100")
    sched.add("a", "@every 1h", "do a", conv, now=1000.0)
    assert path.exists()

    reloaded = Scheduler(path)
    assert "a" in reloaded.jobs
    job = reloaded.jobs["a"]
    assert job.prompt == "do a"
    assert job.conversation.recipient == "+100"


def test_scheduler_due(tmp_path):
    sched = Scheduler(tmp_path / "c.json")
    conv = Conversation(group_id="grp==")
    now = time.time()
    sched.add("soon", "@every 1m", "tick", conv, now=now - 120)  # overdue
    fired = sched.due(now)
    assert [j.name for j in fired] == ["soon"]
    # Claimed but not finalized: a re-tick doesn't fire it again, yet it's
    # still due (next_run unchanged) so a crash before completion re-fires it.
    assert sched.due(now) == []
    assert sched.jobs["soon"].next_run <= now
    # Completing it advances the schedule into the future.
    sched.complete("soon", now=now)
    assert sched.jobs["soon"].next_run > now
    assert sched.jobs["soon"].last_run == now
    assert sched.due(now) == []


def test_scheduler_list_for_is_per_conversation(tmp_path):
    sched = Scheduler(tmp_path / "c.json")
    alice = Conversation(recipient="+1")
    bob = Conversation(group_id="grp==")
    sched.add("a1", "@daily", "p", alice)
    sched.add("a2", "@daily", "p", alice)
    sched.add("b1", "@daily", "p", bob)
    assert {j.name for j in sched.list_for(alice.key)} == {"a1", "a2"}
    assert {j.name for j in sched.list_for(bob.key)} == {"b1"}


def test_scheduler_caps_crons_per_conversation(tmp_path):
    sched = Scheduler(tmp_path / "c.json")
    conv = Conversation(recipient="+1")
    for i in range(MAX_CRONS_PER_CONVERSATION):
        sched.add(f"job{i}", "@daily", "p", conv)
    with pytest.raises(ValueError, match="too many scheduled tasks"):
        sched.add("overflow", "@daily", "p", conv)
    assert "overflow" not in sched.jobs
    # The cap is per conversation: a different one can still schedule.
    other = sched.add("fresh", "@daily", "p", Conversation(recipient="+2"))
    assert other.name == "fresh"


def test_scheduler_rejects_conversation_with_no_destination(tmp_path):
    sched = Scheduler(tmp_path / "c.json")
    with pytest.raises(ValueError, match="nowhere to deliver"):
        sched.add("nodest", "@daily", "p", Conversation())
    assert sched.jobs == {}


def test_scheduler_skips_persisted_crons_with_no_destination(tmp_path):
    # Nothing could ever deliver it, so it must not be loaded at all.
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "orphan",
                        "schedule": "@daily",
                        "prompt": "p",
                        "recipient": None,
                        "group_id": None,
                        "next_run": 0.0,
                        "created": 0.0,
                    },
                    {
                        "name": "fine",
                        "schedule": "@daily",
                        "prompt": "p",
                        "recipient": "+1",
                        "group_id": None,
                        "next_run": 0.0,
                        "created": 0.0,
                    },
                    ["not", "an", "object"],
                ]
            }
        )
    )
    sched = Scheduler(path)
    assert set(sched.jobs) == {"fine"}


def test_scheduler_validates_recurring_schedule_with_explicit_next_run(tmp_path):
    # A recurring job is rejected now, not when complete() can't advance it.
    sched = Scheduler(tmp_path / "c.json")
    conv = Conversation(recipient="+1")
    with pytest.raises(ValueError):
        sched.add("bad", "not-a-schedule", "p", conv, next_run=4_000_000_000.0)
    assert sched.jobs == {}
    # A one-shot still gets to carry its placeholder schedule.
    job = sched.add("ok", "once", "p", conv, once=True, next_run=4_000_000_000.0)
    assert job.schedule == "once"
    assert job.next_run == 4_000_000_000.0


def test_scheduler_remove(tmp_path):
    sched = Scheduler(tmp_path / "c.json")
    sched.add("x", "@daily", "p", Conversation(recipient="+1"))
    assert sched.remove("x") is True
    assert sched.remove("x") is False


def test_scheduler_unique_names(tmp_path):
    sched = Scheduler(tmp_path / "c.json")
    conv = Conversation(recipient="+1")
    a = sched.add("dup", "@daily", "p", conv)
    b = sched.add("dup", "@daily", "p", conv)
    assert a.name == "dup"
    assert b.name == "dup-2"


def test_scheduler_once_fires_then_removed(tmp_path):
    sched = Scheduler(tmp_path / "c.json")
    now = time.time()
    sched.add(
        "ping", "once", "go", Conversation(recipient="+1"), once=True, next_run=now - 1
    )
    fired = sched.due(now)
    assert [j.name for j in fired] == ["ping"]
    assert "ping" in sched.jobs  # still present until the firing completes
    sched.complete("ping", now=now)
    assert "ping" not in sched.jobs  # one-shot removed after completion
    # And it persists as removed.
    assert "ping" not in Scheduler(tmp_path / "c.json").jobs


def test_complete_ignores_reused_name(tmp_path):
    # Fire r, then delete and re-add a new cron also named r. Completing the
    # old firing must not clobber the new job.
    sched = Scheduler(tmp_path / "c.json")
    now = time.time()
    old = sched.add(
        "r", "@every 1m", "old", Conversation(recipient="+1"), now=now - 120
    )
    sched.due(now)  # claim the old firing
    sched.remove("r")
    new = sched.add("r", "@daily", "new", Conversation(recipient="+1"), now=now)
    new_next = new.next_run
    assert new.created != old.created

    sched.complete("r", created=old.created)  # stale completion
    assert sched.jobs["r"].prompt == "new"
    assert sched.jobs["r"].next_run == new_next  # untouched
    assert sched.jobs["r"].last_run is None


def test_release_makes_firing_due_again(tmp_path):
    sched = Scheduler(tmp_path / "c.json")
    now = time.time()
    job = sched.add(
        "r", "@every 1m", "tick", Conversation(recipient="+1"), now=now - 120
    )
    assert [j.name for j in sched.due(now)] == ["r"]
    assert sched.due(now) == []  # claimed
    sched.release("r", created=job.created)  # delivery failed: retry it
    assert [j.name for j in sched.due(now)] == ["r"]
    assert sched.jobs["r"].next_run <= now  # schedule not advanced


def test_scheduler_uncompleted_firing_refires_after_reload(tmp_path):
    # A crash between due() and complete() must not lose the firing: the job
    # was never advanced/saved, so a fresh scheduler still sees it as due.
    path = tmp_path / "c.json"
    sched = Scheduler(path)
    now = time.time()
    sched.add("r", "@every 1m", "tick", Conversation(recipient="+1"), now=now - 120)
    assert [j.name for j in sched.due(now)] == ["r"]  # claimed, not completed
    reloaded = Scheduler(path)  # simulates a restart
    assert [j.name for j in reloaded.due(now)] == ["r"]


# --- conversation routing --------------------------------------------
def test_conversation_send_params():
    assert Conversation(recipient="+33").send_params("hi") == {
        "message": "hi",
        "recipient": ["+33"],
    }
    assert Conversation(group_id="g==").send_params("yo") == {
        "message": "yo",
        "groupId": "g==",
    }


def test_conversation_requires_destination():
    # Neither recipient nor group_id set -> a clear error, not {"recipient": [None]}.
    with pytest.raises(ValueError):
        Conversation().routing_params()


# --- SSE / envelope parsing ------------------------------------------
def test_to_message_direct():
    env = {
        "params": {
            "envelope": {
                "sourceNumber": "+33600000000",
                "sourceUuid": "uuid-1",
                "timestamp": 123,
                "dataMessage": {"message": "hello there"},
            }
        }
    }
    msg = _to_message(env)
    assert msg is not None
    assert msg.text == "hello there"
    assert msg.source == "+33600000000"
    assert msg.conversation.recipient == "+33600000000"
    assert msg.conversation.group_id is None


def test_send_reaction_direct(monkeypatch):
    env = {
        "params": {
            "envelope": {
                "sourceNumber": "+33600000000",
                "sourceUuid": "uuid-1",
                "timestamp": 123,
                "dataMessage": {"message": "hello there"},
            }
        }
    }
    msg = _to_message(env)
    client = SignalClient("http://signal.local")
    calls = []

    async def fake_rpc(method, params=None):
        calls.append((method, params))
        return {}

    monkeypatch.setattr(client, "_rpc", fake_rpc)
    try:
        asyncio.run(client.send_reaction(msg, "👀"))
    finally:
        asyncio.run(client.aclose())

    assert calls == [
        (
            "sendReaction",
            {
                "emoji": "👀",
                "targetAuthor": "+33600000000",
                "targetTimestamp": 123,
                "recipient": ["+33600000000"],
            },
        )
    ]


def test_send_reaction_remove(monkeypatch):
    env = {
        "params": {
            "envelope": {
                "sourceNumber": "+33600000000",
                "sourceUuid": "uuid-1",
                "timestamp": 123,
                "dataMessage": {"message": "hello there"},
            }
        }
    }
    msg = _to_message(env)
    client = SignalClient("http://signal.local")
    calls = []

    async def fake_rpc(method, params=None):
        calls.append((method, params))
        return {}

    monkeypatch.setattr(client, "_rpc", fake_rpc)
    try:
        asyncio.run(client.send_reaction(msg, "👀", remove=True))
    finally:
        asyncio.run(client.aclose())

    assert calls == [
        (
            "sendReaction",
            {
                "emoji": "👀",
                "targetAuthor": "+33600000000",
                "targetTimestamp": 123,
                "recipient": ["+33600000000"],
                "remove": True,
            },
        )
    ]


def test_to_message_group():
    env = {
        "params": {
            "envelope": {
                "sourceNumber": "+1",
                "dataMessage": {
                    "message": "in group",
                    "groupInfo": {"groupId": "GRP=="},
                },
            }
        }
    }
    msg = _to_message(env)
    assert msg.conversation.group_id == "GRP=="


def test_send_reaction_group(monkeypatch):
    env = {
        "params": {
            "envelope": {
                "sourceNumber": "+1",
                "timestamp": 456,
                "dataMessage": {
                    "message": "in group",
                    "groupInfo": {"groupId": "GRP=="},
                },
            }
        }
    }
    msg = _to_message(env)
    client = SignalClient("http://signal.local")
    calls = []

    async def fake_rpc(method, params=None):
        calls.append((method, params))
        return {}

    monkeypatch.setattr(client, "_rpc", fake_rpc)
    try:
        asyncio.run(client.send_reaction(msg))
    finally:
        asyncio.run(client.aclose())

    assert calls == [
        (
            "sendReaction",
            {
                "emoji": "👀",
                "targetAuthor": "+1",
                "targetTimestamp": 456,
                "groupId": "GRP==",
            },
        )
    ]


def test_to_message_ignores_receipts():
    assert _to_message({"params": {"envelope": {"receiptMessage": {}}}}) is None
    assert (
        _to_message({"params": {"envelope": {"dataMessage": {"message": ""}}}}) is None
    )


def test_to_message_ignores_blank_messages():
    # Not a command, and an empty prompt for the agent: ignored on both channels.
    env = {"sourceNumber": "+1", "timestamp": 1, "dataMessage": {"message": "   \t "}}
    assert _to_message({"params": {"envelope": env}}) is None
    sync = {
        "sourceNumber": "+1",
        "syncMessage": {"sentMessage": {"message": " ", "destinationNumber": "+1"}},
    }
    assert _to_message({"params": {"envelope": sync}}) is None


def test_to_message_ignores_messages_with_nowhere_to_reply():
    # No sender and no group: nothing could answer it, and trying would raise.
    env = {"timestamp": 1, "dataMessage": {"message": "hi"}}
    assert _to_message({"params": {"envelope": env}}) is None


def test_to_message_survives_malformed_shapes():
    # signal-cli is another process: a wrong-shaped payload must not kill the stream.
    malformed = [
        [1, 2, 3],
        "just a string",
        {"params": "not an object"},
        {"params": {"envelope": 7}},
        {"params": {"envelope": {"sourceNumber": "+1", "dataMessage": "a string"}}},
        {
            "params": {
                "envelope": {"sourceNumber": "+1", "dataMessage": {"message": 42}}
            }
        },
        {"params": {"envelope": {"syncMessage": "a string"}}},
        {"params": {"envelope": {"syncMessage": {"sentMessage": ["x"]}}}},
    ]
    for payload in malformed:
        assert _to_message(payload) is None, payload

    # A bad sub-object is ignored without losing the message itself.
    msg = _to_message(
        {
            "params": {
                "envelope": {
                    "sourceNumber": "+1",
                    "dataMessage": {"message": "hi", "groupInfo": "bogus"},
                }
            }
        }
    )
    assert msg is not None
    assert msg.text == "hi"
    assert msg.conversation.recipient == "+1"


def test_to_message_note_to_self():
    # Bot runs on your own number: messaging yourself arrives as a sync sentMessage.
    env = {
        "envelope": {
            "source": "+33695226193",
            "sourceNumber": "+33695226193",
            "sourceUuid": "self-uuid",
            "syncMessage": {
                "sentMessage": {
                    "destinationNumber": "+33695226193",
                    "destinationUuid": "self-uuid",
                    "timestamp": 999,
                    "message": "  hi me  ",
                }
            },
        }
    }
    msg = _to_message(env)
    assert msg is not None
    assert msg.text == "hi me"
    assert msg.source == "+33695226193"
    assert msg.conversation.recipient == "+33695226193"  # replies to Note to Self


def test_to_message_ignores_sync_to_others():
    # A message you sent to a friend must NOT be treated as a bot command.
    env = {
        "envelope": {
            "sourceNumber": "+33695226193",
            "sourceUuid": "self-uuid",
            "syncMessage": {
                "sentMessage": {
                    "destinationNumber": "+15551112222",
                    "destinationUuid": "friend-uuid",
                    "message": "see you soon",
                }
            },
        }
    }
    assert _to_message(env) is None


def test_to_message_ignores_sync_group_send():
    env = {
        "envelope": {
            "sourceNumber": "+33695226193",
            "syncMessage": {
                "sentMessage": {
                    "message": "hello group",
                    "groupInfo": {"groupId": "G=="},
                }
            },
        }
    }
    assert _to_message(env) is None


def test_parse_sse_stream():
    class FakeResp:
        async def aiter_lines(self):
            for line in [
                ": keepalive",
                'data: {"method":"receive","params":{"envelope":{"dataMessage":{"message":"hi"}}}}',
                "",
            ]:
                yield line

    async def collect():
        out = []
        async for ev in _parse_sse(FakeResp()):
            out.append(ev)
        return out

    events = asyncio.run(collect())
    assert len(events) == 1
    assert events[0]["params"]["envelope"]["dataMessage"]["message"] == "hi"


def test_parse_sse_rejects_unterminated_event():
    # A backend that streams data: lines and never sends the blank separator
    # must not accumulate without bound — the parser bails so events() reconnects.
    chunk = "A" * 100_000

    class FloodResp:
        async def aiter_lines(self):
            while True:
                yield "data: " + chunk

    async def drain():
        async for _ in _parse_sse(FloodResp()):
            pass

    with pytest.raises(ValueError, match="too large"):
        asyncio.run(drain())


# --- config -----------------------------------------------------------
def test_session_kwargs_safe_vs_autonomous():
    safe = Config(safe=True, model="m").session_kwargs()
    assert safe["commands"] == "none"
    assert safe["files"] == "none"

    auto = Config(safe=False, model="m").session_kwargs()
    assert auto["commands"] == "all"
    assert auto["files"] == "all"
    # yolo is deliberately NOT forced — it makes small models tool-loop on chat.
    assert "yolo" not in auto


def test_write_instructions_default(tmp_path):
    cfg = Config(model="m", state_dir=tmp_path)
    res = cfg.write_instructions()
    assert res.action == "written"
    assert res.path is not None and res.path.name == "AGENTS.md"
    text = res.path.read_text()
    assert "Signal" in text
    assert "Do NOT" in text  # the no-tools-for-chat instruction


def test_write_instructions_disabled(tmp_path):
    cfg = Config(model="m", state_dir=tmp_path, instructions="")
    res = cfg.write_instructions()
    assert res.action == "disabled"
    assert res.path is None


def test_write_instructions_custom(tmp_path):
    cfg = Config(model="m", state_dir=tmp_path, instructions="be terse")
    res = cfg.write_instructions()
    assert res.action == "written"
    assert "be terse" in res.path.read_text()


def test_write_instructions_preserves_foreign_agents_md(tmp_path):
    # A custom workspace may be a real project: don't clobber its AGENTS.md.
    workspace = tmp_path / "project"
    workspace.mkdir()
    user_file = workspace / "AGENTS.md"
    user_file.write_text("# my project rules\n")
    cfg = Config(model="m", state_dir=tmp_path, workspace_dir=workspace)
    res = cfg.write_instructions()
    assert res.action == "skipped"
    assert user_file.read_text() == "# my project rules\n"


def test_write_instructions_rewrites_own_file_in_custom_workspace(tmp_path):
    # A file we wrote ourselves (carries the marker) is ours to update.
    workspace = tmp_path / "project"
    cfg = Config(
        model="m", state_dir=tmp_path, workspace_dir=workspace, instructions="v1"
    )
    first = cfg.write_instructions()
    assert first.action == "written"
    cfg.instructions = "v2"
    second = cfg.write_instructions()
    assert second.action == "written"
    assert "v2" in second.path.read_text()


def test_slugify():
    assert slugify("Summarize my git log!") == "summarize-my-git-log"
    assert slugify("a b c d e f") == "a-b-c-d"  # capped at 4 words
    assert slugify("!!!") == "task"


# --- natural-language schedule parsing (model output -> ParsedSchedule) ----
def test_extract_json_from_fenced_answer():
    ans = 'sure!\n```json\n{"name":"x","type":"recurring","cron":"0 9 * * *","prompt":"hi"}\n```'
    obj = _extract_json(ans)
    assert obj["cron"] == "0 9 * * *"


def test_extract_json_takes_the_first_object():
    # A second "{" in the answer used to defeat the greedy match.
    assert _extract_json('one {"a": 1} and another {"b": 2}') == {"a": 1}
    assert _extract_json('{"ok": true} then prose with {curly} braces') == {"ok": True}
    # Braces inside strings and nested objects still belong to the first object.
    assert _extract_json('note {x} then {"p":{"deep":"}"},"n":1}') == {
        "p": {"deep": "}"},
        "n": 1,
    }
    assert _extract_json("no json here at all") is None
    assert _extract_json("{not json}") is None
    # A bare {} in prose parses but can't be the schedule; keep looking.
    assert _extract_json('maybe { } — no: {"cron": "0 9 * * *"}') == {
        "cron": "0 9 * * *"
    }
    assert _extract_json("empty { } only") is None


def test_build_recurring():
    now = 1_000_000.0
    p = _build(
        {
            "name": "Git Standup",
            "type": "recurring",
            "cron": "0 9 * * 1-5",
            "prompt": "do it",
        },
        now,
    )
    assert p.once is False
    assert p.schedule == "0 9 * * 1-5"
    assert p.name == "git-standup"
    assert p.prompt == "do it"


def test_build_once_future():
    now = 1_000_000.0
    future = "2099-01-01 09:00"
    p = _build({"name": "wake", "type": "once", "at": future, "prompt": "morning"}, now)
    assert p.once is True
    assert p.at_epoch > now
    assert "once at" in p.describe()


def test_build_once_in_past_rejected():
    now = 4_000_000_000.0  # far future "now"
    with pytest.raises(ParseError):
        _build({"type": "once", "at": "2000-01-01 09:00", "prompt": "x"}, now)


def test_build_rejects_missing_prompt():
    with pytest.raises(ParseError):
        _build({"type": "recurring", "cron": "0 9 * * *"}, 1_000_000.0)


def test_build_rejects_bad_cron():
    with pytest.raises(ParseError):
        _build({"type": "recurring", "cron": "not a cron", "prompt": "x"}, 1_000_000.0)


def test_session_kwargs_swival_override():
    cfg = Config(model="m", swival={"max_turns": 7, "temperature": 0.1})
    kw = cfg.session_kwargs()
    assert kw["max_turns"] == 7
    assert kw["temperature"] == 0.1


def _write_toml(tmp_path, body: str) -> Path:
    path = tmp_path / "nbclaw.toml"
    path.write_text(body)
    return path


def test_toml_allow_must_be_a_list_of_strings(tmp_path):
    # A string here would make the sender check a substring match.
    for body in ('allow = "+1234567890"\n', 'allow = ["+1", 2]\n'):
        path = _write_toml(tmp_path, body)
        with pytest.raises(SystemExit, match="must be a list of strings"):
            build_config(["--config", str(path)])


def test_toml_rejects_mistyped_values(tmp_path):
    cases = [
        ("max_turns = true\n", "an integer"),  # bool is a subclass of int
        ('safe = "yes"\n', "true or false"),
        ("swival = 3\n", "a table"),
        ("mcp_servers = { a = 1 }\n", "a table of tables"),
        ("signal_url = 8080\n", "a string"),
        ("bogus = 1\n", "Unknown config key"),
    ]
    for body, expected in cases:
        path = _write_toml(tmp_path, body)
        with pytest.raises(SystemExit, match=expected):
            build_config(["--config", str(path)])


def test_toml_accepts_well_typed_values(tmp_path):
    path = _write_toml(
        tmp_path,
        'allow = ["+1"]\n'
        "allow_all = false\n"
        "max_turns = 5\n"
        "shutdown_timeout = 2\n"
        'state_dir = "~/nb"\n'
        "[swival]\n"
        "temperature = 0.1\n",
    )
    cfg = build_config(["--config", str(path)])
    assert cfg.allow == ["+1"]
    assert cfg.allow_all is False
    assert cfg.max_turns == 5
    assert cfg.shutdown_timeout == 2
    assert cfg.state_dir == Path.home() / "nb"
    assert cfg.swival == {"temperature": 0.1}


def test_toml_schema_covers_every_config_field():
    # A field missing from the schema can't be set from TOML at all.
    assert set(_TOML_SCHEMA) == {f.name for f in fields(Config)}


# --- agent session bounding ------------------------------------------
_DONE = Result(answer="done", exhausted=False, messages=[], report=None)


class _FakeSession:
    def __init__(self, **kwargs):
        self.closed = False

    def ask(self, prompt):
        return _DONE

    def close(self):
        self.closed = True


def test_agent_runner_bounds_session_count(monkeypatch):
    monkeypatch.setattr("nbclaw.agent_runner.Session", _FakeSession)
    runner = AgentRunner({})
    overflow = 8
    created = [
        runner._session_for(f"user:+{i}") for i in range(MAX_CHAT_SESSIONS + overflow)
    ]
    assert len(runner._sessions) == MAX_CHAT_SESSIONS
    # The earliest keys were evicted, and exactly those sessions were closed.
    assert sum(1 for s in created if s.closed) == overflow
    assert "user:+0" not in runner._sessions
    assert f"user:+{MAX_CHAT_SESSIONS + overflow - 1}" in runner._sessions


def test_agent_runner_evicts_least_recently_used(monkeypatch):
    monkeypatch.setattr("nbclaw.agent_runner.Session", _FakeSession)
    runner = AgentRunner({})
    for i in range(MAX_CHAT_SESSIONS):
        runner._session_for(f"k{i}")
    runner._session_for("k0")  # touch the oldest so it's most-recently-used
    runner._session_for("fresh")  # forces one eviction
    assert "k0" in runner._sessions  # spared: just used
    assert "k1" not in runner._sessions  # evicted as the new oldest
    assert "fresh" in runner._sessions


def test_agent_runner_closes_and_discards_failed_mcp_setup(monkeypatch):
    created = []

    class FailingSession(_FakeSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

        def _setup(self):
            raise RuntimeError("setup failed")

    monkeypatch.setattr("nbclaw.agent_runner.Session", FailingSession)
    runner = AgentRunner({"mcp_servers": {"test": {}}})

    with pytest.raises(RuntimeError, match="setup failed"):
        runner._session_for("user:+1")

    assert created[0].closed is True
    assert runner._sessions == {}


def test_cron_sessions_do_not_leak_swival_event_loop_fds(monkeypatch):
    class McpSession:
        def __init__(self, **kwargs):
            self.manager = McpManager({})
            self.manager.start()

        def run(self, prompt):
            return Result(answer="done", exhausted=False, messages=[], report=None)

        def close(self):
            self.manager.close()

    monkeypatch.setattr("nbclaw.agent_runner.Session", McpSession)
    runner = AgentRunner({})

    # Warm up one-time imports/threading state before measuring repeated cron
    # one-shots, each of which owns a fresh Swival MCP event loop.
    for _ in range(3):
        runner._once_blocking("tick")
    before = len(os.listdir("/dev/fd"))
    for _ in range(10):
        runner._once_blocking("tick")
    after = len(os.listdir("/dev/fd"))

    assert after - before <= 2, f"leaked {after - before} fds over 10 cron sessions"


def test_reset_closes_an_idle_session(monkeypatch):
    monkeypatch.setattr("nbclaw.agent_runner.Session", _FakeSession)
    runner = AgentRunner({})
    session = runner._session_for("user:+1")

    assert runner.reset("user:+1") is True
    assert session.closed is True
    assert runner.reset("user:+1") is False  # nothing left to clear


def test_reset_defers_closing_a_session_being_used(monkeypatch):
    # Closing a session mid-ask() would pull swival state from under the worker.
    started = threading.Event()
    finish = threading.Event()

    class BlockingSession(_FakeSession):
        def ask(self, prompt):
            started.set()
            assert finish.wait(5)
            return _DONE

    monkeypatch.setattr("nbclaw.agent_runner.Session", BlockingSession)
    runner = AgentRunner({})
    worker = threading.Thread(target=runner._ask_blocking, args=("user:+1", "hi"))
    worker.start()
    try:
        assert started.wait(5)
        session = runner._in_use["user:+1"]

        assert runner.reset("user:+1") is True
        assert session.closed is False  # still in use — not closed yet
        assert "user:+1" not in runner._sessions  # but the next message is fresh
    finally:
        finish.set()
        worker.join(5)

    assert session.closed is True  # closed once the worker handed it back
    assert runner._retired == set()
    assert runner._in_use == {}


def test_reset_discards_a_session_that_is_still_being_created(monkeypatch):
    # /clear can land while a session is being built, before it's in the cache.
    creating = threading.Event()
    cleared = threading.Event()
    made = []

    class SlowSession(_FakeSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            made.append(self)
            creating.set()
            assert cleared.wait(5)

    monkeypatch.setattr("nbclaw.agent_runner.Session", SlowSession)
    runner = AgentRunner({})
    worker = threading.Thread(target=runner._ask_blocking, args=("user:+1", "hi"))
    worker.start()
    try:
        assert creating.wait(5)
        assert runner.reset("user:+1") is True
    finally:
        cleared.set()
        worker.join(5)

    # The call still ran, but nothing was kept for the next message.
    assert "user:+1" not in runner._sessions
    assert runner._creating == set()
    assert runner._retired == set()
    assert runner._in_use == {}
    assert made[0].closed is True
    assert runner.reset("user:+1") is False


def test_close_all_closes_retired_sessions(monkeypatch):
    monkeypatch.setattr("nbclaw.agent_runner.Session", _FakeSession)
    runner = AgentRunner({})
    session = runner._session_for("user:+1", in_use=True)
    runner.reset("user:+1")  # retired, not closed: still checked out

    runner.close_all()
    assert session.closed is True
    assert runner._retired == set()
