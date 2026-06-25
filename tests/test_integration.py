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
from nbclaw.daemon import Daemon, Job
from nbclaw.signal_client import Conversation, IncomingMessage


class FakeSignal:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []  # (conv.key, message)

    async def version(self):
        return "fake"

    async def send(self, conv, message):
        self.sent.append((conv.key, message))

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
    assert job.mode == "chat"
    assert job.prompt == "what is 2+2?"


class FakeAgent:
    async def once(self, prompt):
        return "done"

    async def chat(self, key, prompt):
        return "done"

    def close_all(self):
        pass


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
