"""Integration tests: daemon dispatch end-to-end.

Command flows are tested with a fake Signal transport (no model needed).
A single live test drives the real local model through the agent worker.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from nbclaw.config import Config
from nbclaw.daemon import MAX_REPLY_CHARS, Daemon, Job, split_reply
from nbclaw.signal_client import Conversation, IncomingMessage


class FakeSignal:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []  # (conv.key, message)
        self.reactions: list[tuple[str, int, str]] = []  # (author, timestamp, emoji)

    async def version(self):
        return "fake"

    async def send(self, conv, message):
        self.sent.append((conv.key, message))

    async def send_reaction(self, msg, emoji="👀"):
        self.reactions.append((msg.source or msg.source_uuid, msg.timestamp, emoji))

    async def send_typing(self, conv, *, stop=False):
        pass

    async def aclose(self):
        pass

    @property
    def last(self):
        return self.sent[-1][1] if self.sent else None


def make_daemon(tmp_path, **overrides) -> Daemon:
    cfg = Config(
        model="ornith-1.0-9b",
        state_dir=tmp_path,
        allow=["+15550000001"],
        **overrides,
    )
    d = Daemon(cfg)
    d.signal = FakeSignal()
    return d


def incoming(text, source="+15550000001"):
    return IncomingMessage(
        conversation=Conversation(recipient=source),
        text=text,
        source=source,
        source_uuid="uuid",
        timestamp=1,
    )


def test_help_command(tmp_path):
    d = make_daemon(tmp_path)
    asyncio.run(d._dispatch(incoming("/help")))
    assert "commands" in d.signal.last.lower()


def test_status_command(tmp_path):
    d = make_daemon(tmp_path)
    asyncio.run(d._dispatch(incoming("/status")))
    assert "model: ornith-1.0-9b" in d.signal.last


def test_clear_reset_and_new_aliases(tmp_path):
    class ResetAgent:
        def __init__(self):
            self.reset_keys = []

        def reset(self, key):
            self.reset_keys.append(key)
            return True

    d = make_daemon(tmp_path)
    d.agent = ResetAgent()

    for command in ("/clear", "/reset", "/new"):
        asyncio.run(d._dispatch(incoming(command)))

    assert d.agent.reset_keys == ["user:+15550000001"] * 3
    assert d.signal.last == "context cleared."


def test_cron_lifecycle(tmp_path):
    d = make_daemon(tmp_path)

    asyncio.run(d._dispatch(incoming("/cron add nightly @daily | say hi")))
    assert "scheduled 'nightly'" in d.signal.last
    assert "nightly" in d.scheduler.jobs

    asyncio.run(d._dispatch(incoming("/cron list")))
    assert "nightly" in d.signal.last

    asyncio.run(d._dispatch(incoming("/cron del nightly")))
    assert "cancelled" in d.signal.last
    assert "nightly" not in d.scheduler.jobs


def test_cron_management_is_scoped_to_conversation(tmp_path):
    d = make_daemon(tmp_path)
    owner = "+15550000001"
    other = "+15550000002"

    asyncio.run(d._dispatch(incoming("/cron add mine @daily | say hi", source=owner)))
    assert "mine" in d.scheduler.jobs

    # A different conversation can't see, delete, or run someone else's cron.
    asyncio.run(d._dispatch(incoming("/cron list", source=other)))
    assert "mine" not in d.signal.last
    assert "no scheduled tasks" in d.signal.last

    asyncio.run(d._dispatch(incoming("/cron del mine", source=other)))
    assert "no cron named 'mine'" in d.signal.last
    assert "mine" in d.scheduler.jobs  # untouched

    asyncio.run(d._dispatch(incoming("/cron run mine", source=other)))
    assert "no cron named 'mine'" in d.signal.last
    assert d.queue.qsize() == 0  # nothing was enqueued to fire

    # The owning conversation still sees and controls its own cron.
    asyncio.run(d._dispatch(incoming("/cron list", source=owner)))
    assert "mine" in d.signal.last
    asyncio.run(d._dispatch(incoming("/cron del mine", source=owner)))
    assert "cancelled" in d.signal.last
    assert "mine" not in d.scheduler.jobs


def test_cron_add_rejected_when_conversation_full(tmp_path):
    from nbclaw.scheduler import MAX_CRONS_PER_CONVERSATION

    d = make_daemon(tmp_path)
    conv = Conversation(recipient="+15550000001")
    for i in range(MAX_CRONS_PER_CONVERSATION):
        d.scheduler.add(f"job{i}", "@daily", "p", conv)

    asyncio.run(d._dispatch(incoming("/cron add toomany @daily | nope")))
    assert "too many scheduled tasks" in d.signal.last
    assert "toomany" not in d.scheduler.jobs


def test_cron_add_bad_schedule(tmp_path):
    d = make_daemon(tmp_path)
    asyncio.run(d._dispatch(incoming("/cron add x bogus-schedule | do it")))
    assert "cron add" in d.signal.last
    assert "x" not in d.scheduler.jobs


def test_unauthorized_sender_ignored(tmp_path):
    d = make_daemon(tmp_path)
    assert d._authorized(incoming("hi", source="+19998887777")) is False
    assert d._authorized(incoming("hi", source="+15550000001")) is True


def test_chat_message_is_queued(tmp_path):
    d = make_daemon(tmp_path)
    asyncio.run(d._dispatch(incoming("what is 2+2?")))
    assert d.queue.qsize() == 1
    job = d.queue.get_nowait()
    assert job.id == 1
    assert job.mode == "chat"
    assert job.prompt == "what is 2+2?"


async def _queue_two_chats(d: Daemon) -> None:
    await d._dispatch(incoming("first"))
    await d._dispatch(incoming("second"))


def test_queue_command_lists_this_conversation_only(tmp_path):
    d = make_daemon(tmp_path)
    other = Conversation(recipient="+15550000002")
    asyncio.run(d._enqueue(Job(other, "secret other chat", "chat")))
    asyncio.run(_queue_two_chats(d))

    asyncio.run(d._dispatch(incoming("/queue")))

    assert "#2 chat" in d.signal.last
    assert "#3 chat" in d.signal.last
    assert "secret other chat" not in d.signal.last


def test_cancel_without_id_cancels_this_conversation_only(tmp_path):
    d = make_daemon(tmp_path)
    other = Conversation(recipient="+15550000002")
    asyncio.run(d._enqueue(Job(other, "keep", "chat")))
    asyncio.run(_queue_two_chats(d))

    asyncio.run(d._dispatch(incoming("/cancel")))

    assert d.signal.last == "cancelled queued job #2."
    remaining = list(d.queue._queue)
    assert [(job.id, job.prompt) for job in remaining] == [(1, "keep"), (3, "second")]


def test_cancel_all_cancels_only_this_conversation(tmp_path):
    d = make_daemon(tmp_path)
    other = Conversation(recipient="+15550000002")
    asyncio.run(d._enqueue(Job(other, "keep", "chat")))
    asyncio.run(_queue_two_chats(d))

    asyncio.run(d._dispatch(incoming("/cancel all")))

    assert d.signal.last == "cancelled 2 queued jobs."
    remaining = list(d.queue._queue)
    assert [(job.id, job.prompt) for job in remaining] == [(1, "keep")]


def test_cancel_running_job_reports_not_cancellable(tmp_path):
    d = make_daemon(tmp_path)
    d.current_job = Job(Conversation(recipient="+15550000001"), "busy", "chat", id=7)

    asyncio.run(d._dispatch(incoming("/cancel 7")))

    assert "already running" in d.signal.last


def test_received_message_gets_reaction_before_dispatch(tmp_path):
    class OneMessageSignal(FakeSignal):
        async def events(self):
            yield incoming("what is 2+2?")

    d = make_daemon(tmp_path)
    d.signal = OneMessageSignal()

    asyncio.run(d._consume_events())

    assert d.signal.reactions == [("+15550000001", 1, "👀")]
    assert d.queue.qsize() == 1


class FakeAgent:
    def __init__(self, answer="done"):
        self.answer = answer

    async def once(self, prompt):
        return self.answer

    async def chat(self, key, prompt):
        return self.answer

    def close_all(self):
        pass


async def _safe_send_long_answer(d: Daemon, text: str) -> None:
    await d._safe_send(Conversation(recipient="+15550000001"), text)


def test_split_reply_numbers_chunks_under_limit():
    text = "alpha " * MAX_REPLY_CHARS

    chunks = split_reply(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= MAX_REPLY_CHARS for chunk in chunks)
    assert chunks[0].startswith("(1/")
    assert chunks[-1].startswith(f"({len(chunks)}/{len(chunks)})")


def test_safe_send_splits_long_replies(tmp_path):
    d = make_daemon(tmp_path)
    text = "x" * (MAX_REPLY_CHARS + 100)

    asyncio.run(_safe_send_long_answer(d, text))

    messages = [message for _, message in d.signal.sent]
    assert len(messages) == 2
    assert messages[0].startswith("(1/2) ")
    assert messages[1].startswith("(2/2) ")
    assert all(len(message) <= MAX_REPLY_CHARS for message in messages)


def test_scheduled_cron_finalized_after_delivery(tmp_path):
    d = make_daemon(tmp_path)
    d.agent = FakeAgent()
    conv = Conversation(recipient="+15550000001")
    now = time.time()
    d.scheduler.add("r", "@every 1m", "tick", conv, now=now - 120)
    due = d.scheduler.due(now)  # how the scheduler loop claims it
    assert [j.name for j in due] == ["r"]

    job = Job(conv, "tick", "cron", label="r", finalize=True)
    asyncio.run(d._run_one_job(job))

    assert any("[r]" in m for _, m in d.signal.sent)  # reply delivered
    assert d.scheduler.jobs["r"].next_run > now  # schedule advanced after delivery


class FailingSignal(FakeSignal):
    async def send(self, conv, message):
        raise RuntimeError("signal down")


def test_cron_not_finalized_when_delivery_fails(tmp_path):
    d = make_daemon(tmp_path)
    d.agent = FakeAgent()
    d.signal = FailingSignal()
    conv = Conversation(recipient="+15550000001")
    now = time.time()
    job_def = d.scheduler.add("r", "@every 1m", "tick", conv, now=now - 120)
    d.scheduler.due(now)  # claim it as the scheduler loop would

    job = Job(
        conv, "tick", "cron", label="r", finalize=True, cron_created=job_def.created
    )
    asyncio.run(d._run_one_job(job))

    # Delivery failed: schedule not advanced and the firing is due again.
    assert d.scheduler.jobs["r"].next_run <= now
    assert [j.name for j in d.scheduler.due(now)] == ["r"]


def test_manual_cron_run_does_not_finalize(tmp_path):
    d = make_daemon(tmp_path)
    d.agent = FakeAgent()
    conv = Conversation(recipient="+15550000001")
    now = time.time()
    d.scheduler.add("r", "@every 1m", "tick", conv, now=now - 120)

    job = Job(conv, "tick", "cron", label="r")  # finalize defaults to False
    asyncio.run(d._run_one_job(job))

    assert (
        d.scheduler.jobs["r"].next_run <= now
    )  # a one-off run leaves the schedule alone


def _model_up() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:1234/v1/models", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _model_up(), reason="local LM Studio not reachable")
def test_live_agent_roundtrip(tmp_path):
    d = make_daemon(tmp_path, safe=True)  # read-only is enough and quick
    conv = Conversation(recipient="+15550000001")
    job = Job(conv, "Reply with exactly the single word: PONG", "chat")
    asyncio.run(d._run_one_job(job))
    assert d.signal.sent, "no reply was sent"
    assert "PONG" in d.signal.last.upper()
    d.agent.close_all()
