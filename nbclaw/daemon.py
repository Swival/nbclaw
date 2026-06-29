"""The nbclaw daemon: wires Signal, the agent, and the scheduler together.

Architecture (all asyncio, single process):

* ``_consume_events``  reads the signal-cli SSE stream. Fast slash commands are
  answered inline; anything that needs the model is pushed onto a job queue.
* ``_run_jobs``        a single worker drains the queue and runs the agent. One
  worker keeps the local model from being asked to do two things at once.
* ``_run_scheduler``   wakes periodically, asks the scheduler for due crons, and
  enqueues them as jobs.

Funnelling every model call through one queue makes the whole thing naturally
serialized and easy to reason about for a 24/7 process.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections import deque
from dataclasses import dataclass, field

from . import commands, nl_schedule
from .agent_runner import AgentRunner
from .config import Config
from .scheduler import CronJob, Scheduler, fmt_time, validate_schedule
from .signal_client import Conversation, IncomingMessage, SignalClient

log = logging.getLogger("nbclaw")

MAX_REPLY_CHARS = 3500
_REPLY_CHUNK_PREFIX_RESERVE = 64


def _split_command(text: str) -> tuple[str, str]:
    """Split ``"verb the rest"`` into (lowercased verb, remainder)."""
    parts = text.split(None, 1)
    verb = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    return verb, rest


def split_reply(text: str, max_chars: int = MAX_REPLY_CHARS) -> list[str]:
    """Split a long Signal reply into numbered chunks."""
    if len(text) <= max_chars:
        return [text]
    body_limit = max_chars - _REPLY_CHUNK_PREFIX_RESERVE
    if body_limit < 1:
        raise ValueError("max_chars is too small")

    chunks: list[str] = []
    remaining = text
    while len(remaining) > body_limit:
        split_at = remaining.rfind("\n", 0, body_limit + 1)
        if split_at < body_limit // 2:
            split_at = remaining.rfind(" ", 0, body_limit + 1)
        if split_at < body_limit // 2:
            split_at = body_limit
        chunk = remaining[:split_at].rstrip()
        if not chunk:
            chunk = remaining[:body_limit]
            split_at = body_limit
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)

    total = len(chunks)
    return [f"({i}/{total}) {chunk}" for i, chunk in enumerate(chunks, 1)]


def _truncate_prompt(prompt: str) -> str:
    """Shorten a prompt for a one-line summary."""
    return prompt if len(prompt) <= 60 else prompt[:57] + "…"


@dataclass
class Job:
    conversation: Conversation
    prompt: str
    mode: str  # "chat" or "cron"
    label: str = ""  # for logging (e.g. cron name)
    # True only for scheduler-fired crons: finalize the schedule (advance/remove)
    # after this job's reply is delivered. Manual "/cron run" leaves it False so
    # a one-off run never touches the schedule.
    finalize: bool = False
    # The fired cron's ``created`` stamp, so finalization targets that exact job
    # even if the name was deleted and re-added while this sat in the queue.
    cron_created: float | None = None
    id: int = 0
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None


class Daemon:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.signal = SignalClient(config.signal_url)
        self.agent = AgentRunner(config.session_kwargs())
        self.scheduler = Scheduler(config.crons_path())
        # ``None`` is the shutdown sentinel that lets the worker drain and exit.
        self.queue: asyncio.Queue[Job | None] = asyncio.Queue()
        self.current_job: Job | None = None
        self._next_job_id = 1
        self.start_time = time.time()
        self._stop = asyncio.Event()

    async def _enqueue(self, job: Job) -> Job:
        job.id = self._next_job_id
        self._next_job_id += 1
        job.queued_at = time.time()
        await self.queue.put(job)
        return job

    # --- lifecycle -----------------------------------------------------
    async def run(self) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.config.resolved_workspace().mkdir(parents=True, exist_ok=True)
        written = self.config.write_instructions()
        if written.action == "written":
            log.info("agent instructions written to %s", written.path)
        elif written.action == "skipped":
            log.warning(
                "leaving existing %s untouched (not written by nbclaw); "
                "the agent will use it as-is",
                written.path,
            )

        try:
            version = await self.signal.version()
            log.info("signal-cli %s at %s", version, self.config.signal_url)
        except Exception as exc:
            log.error("cannot reach signal-cli at %s: %s", self.config.signal_url, exc)
            raise SystemExit(1)

        if self.config.mcp_servers:
            # MCP servers are spawned lazily, per conversation, on the first
            # message (swival starts them in Session._setup). Log them here so
            # there's confirmation at boot that the config was picked up — the
            # actual "MCP <name> N tool(s)" line lands on the first message.
            log.info(
                "MCP servers configured (spawned on first message): %s",
                ", ".join(self.config.mcp_servers),
            )
        else:
            log.info("no MCP servers configured")

        if not self.config.allow and not self.config.allow_all:
            log.warning(
                "no --allow senders configured and --allow-all not set: "
                "ALL incoming messages will be ignored. Set an allowlist."
            )

        self._install_signal_handlers()
        await self._announce_online()

        events_task = asyncio.create_task(self._consume_events(), name="events")
        jobs_task = asyncio.create_task(self._run_jobs(), name="jobs")
        scheduler_task = asyncio.create_task(self._run_scheduler(), name="scheduler")
        log.info(
            "nbclaw online — provider=%s model=%s crons=%d workspace=%s",
            self.config.provider,
            self.config.model or "(auto)",
            len(self.scheduler.jobs),
            self.config.resolved_workspace(),
        )

        await self._stop.wait()
        log.info("shutting down…")
        # Stop taking in new work first, then let the worker drain what is
        # already queued (including the in-flight job) before closing resources.
        events_task.cancel()
        scheduler_task.cancel()
        await self.queue.put(None)  # wake the worker so it can finish and exit
        try:
            await asyncio.wait_for(jobs_task, timeout=self.config.shutdown_timeout)
        except asyncio.TimeoutError:
            log.warning(
                "job did not finish within %ss; abandoning it",
                self.config.shutdown_timeout,
            )
            jobs_task.cancel()
        await asyncio.gather(
            events_task, scheduler_task, jobs_task, return_exceptions=True
        )
        self.agent.close_all()
        await self.signal.aclose()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:  # pragma: no cover - non-unix
                pass

    async def _announce_online(self) -> None:
        if not self.config.notify:
            return
        conv = Conversation(recipient=self.config.notify)
        try:
            await self.signal.send(conv, "nbclaw is online.")
        except Exception as exc:
            log.warning("could not send startup notice: %s", exc)

    # --- authorization -------------------------------------------------
    def _authorized(self, msg: IncomingMessage) -> bool:
        if self.config.allow_all:
            return True
        allow = self.config.allow
        return (msg.source in allow) or (msg.source_uuid in allow)

    # --- event consumption ---------------------------------------------
    async def _consume_events(self) -> None:
        async for msg in self.signal.events():
            if not self._authorized(msg):
                log.info("ignoring message from unauthorized sender %s", msg.source)
                continue
            log.info("[%s] %s", msg.conversation.key, msg.text)
            await self.signal.send_reaction(msg)
            try:
                await self._dispatch(msg)
            except Exception as exc:
                log.exception("dispatch failed")
                await self._safe_send(msg.conversation, f"error: {exc}")

    async def _dispatch(self, msg: IncomingMessage) -> None:
        text = msg.text
        if not text.startswith("/"):
            await self._enqueue(Job(msg.conversation, text, "chat"))
            return
        await self._handle_command(msg.conversation, text)

    # --- command handling ----------------------------------------------
    async def _handle_command(self, conv: Conversation, text: str) -> None:
        cmd, rest = _split_command(text[1:])

        if cmd in ("help", "h", "?"):
            await self.signal.send(conv, commands.HELP_TEXT)
        elif cmd == "status":
            await self.signal.send(conv, self._status_text())
        elif cmd in ("clear", "reset", "new"):
            had = self.agent.reset(conv.key)
            await self.signal.send(
                conv, "context cleared." if had else "no context to clear."
            )
        elif cmd == "queue":
            await self.signal.send(conv, self._queue_text(conv))
        elif cmd == "cancel":
            await self.signal.send(conv, self._cancel_text(conv, rest))
        elif cmd == "cron":
            await self._handle_cron(conv, rest)
        else:
            await self.signal.send(
                conv, f"unknown command: /{cmd}\n\n{commands.HELP_TEXT}"
            )

    async def _handle_cron(self, conv: Conversation, rest: str) -> None:
        sub, sub_rest = _split_command(rest)

        if sub in ("list", "ls", ""):
            await self.signal.send(conv, self._cron_list_text(conv))
        elif sub in ("del", "rm", "delete", "cancel"):
            name = sub_rest.strip()
            if await self._lookup_cron(conv, name) is None:
                return
            self.scheduler.remove(name)
            await self.signal.send(conv, f"cancelled cron '{name}'.")
        elif sub == "run":
            name = sub_rest.strip()
            job = await self._lookup_cron(conv, name)
            if job is None:
                return
            queued = await self._enqueue(
                Job(job.conversation, job.prompt, "cron", label=name)
            )
            await self.signal.send(conv, f"queued cron '{name}' as #{queued.id}.")
        elif sub in ("help", "h", "?"):
            await self.signal.send(conv, commands.HELP_TEXT)
        elif sub == "add" and "|" in sub_rest:
            await self._cron_add_structured(conv, sub_rest)  # power-user form
        elif sub == "add":
            await self._cron_add_nl(conv, sub_rest)
        else:
            # Not a known subcommand: treat the whole thing as natural language,
            # so "/cron remind me to stretch every 2 hours" just works.
            await self._cron_add_nl(conv, rest)

    async def _cron_add_structured(self, conv: Conversation, body: str) -> None:
        try:
            name, schedule, prompt = commands.parse_cron_add(body)
            validate_schedule(schedule)
            job = self.scheduler.add(name, schedule, prompt, conv)
        except ValueError as exc:
            await self.signal.send(conv, f"cron add: {exc}")
            return
        await self.signal.send(
            conv,
            f"scheduled '{job.name}' ({schedule}). next run: {fmt_time(job.next_run)}",
        )

    async def _cron_add_nl(self, conv: Conversation, text: str) -> None:
        text = text.strip()
        if not text:
            await self.signal.send(
                conv,
                "tell me what to schedule, e.g.\n/cron remind me to stretch every 2 hours",
            )
            return
        await self.signal.send_typing(conv)
        try:
            parsed = await asyncio.to_thread(
                nl_schedule.parse_request,
                text,
                self.config.parser_session_kwargs(),
                time.time(),
            )
        except nl_schedule.ParseError as exc:
            await self.signal.send_typing(conv, stop=True)
            await self.signal.send(
                conv,
                f"couldn't schedule that: {exc}\n\n"
                "tip: try rephrasing, or use /cron add <name> <cron> | <prompt>",
            )
            return
        except Exception as exc:
            await self.signal.send_typing(conv, stop=True)
            log.exception("nl schedule failed")
            await self.signal.send(conv, f"scheduling error: {exc}")
            return
        await self.signal.send_typing(conv, stop=True)

        try:
            if parsed.once:
                job = self.scheduler.add(
                    parsed.name,
                    "once",
                    parsed.prompt,
                    conv,
                    once=True,
                    next_run=parsed.at_epoch,
                )
            else:
                job = self.scheduler.add(
                    parsed.name, parsed.schedule, parsed.prompt, conv
                )
        except ValueError as exc:
            await self.signal.send(conv, f"couldn't schedule that: {exc}")
            return
        await self.signal.send(
            conv,
            f"scheduled '{job.name}' — {parsed.describe()}.\n"
            f"next run: {fmt_time(job.next_run)}\n"
            f"task: {parsed.prompt}",
        )

    async def _lookup_cron(self, conv: Conversation, name: str) -> CronJob | None:
        """Find a cron owned by ``conv``, else reply that it doesn't exist.

        Managing a cron is scoped to the conversation it belongs to, so a sender
        can't delete or trigger someone else's. A cron owned by another
        conversation reads as absent, which keeps its existence from leaking.
        """
        job = self.scheduler.get_for(name, conv.key)
        if job is None:
            await self.signal.send(conv, f"no cron named '{name}'.")
        return job

    def _cron_list_text(self, conv: Conversation) -> str:
        jobs = self.scheduler.list_for(conv.key)
        if not jobs:
            return "no scheduled tasks."
        lines = ["scheduled tasks:"]
        for job in jobs:
            prompt = _truncate_prompt(job.prompt)
            lines.append(
                f"• {job.name} [{job.schedule}] next {fmt_time(job.next_run)}\n    {prompt}"
            )
        return "\n".join(lines)

    def _status_text(self) -> str:
        uptime = int(time.time() - self.start_time)
        h, rem = divmod(uptime, 3600)
        m, s = divmod(rem, 60)
        return (
            "nbclaw status\n"
            f"provider: {self.config.provider}\n"
            f"model: {self.config.model or '(auto)'}\n"
            f"mode: {'safe (read-only)' if self.config.safe else 'autonomous'}\n"
            f"uptime: {h}h{m:02d}m{s:02d}s\n"
            f"active crons: {len(self.scheduler.jobs)}"
        )

    def _queue_text(self, conv: Conversation) -> str:
        lines = ["queue:"]
        visible = False
        if (
            self.current_job is not None
            and self.current_job.conversation.key == conv.key
        ):
            visible = True
            lines.append(f"running: {self._job_summary(self.current_job)}")
        for job in self._pending_jobs_for(conv):
            visible = True
            lines.append(f"queued: {self._job_summary(job)}")
        if not visible:
            return "queue is empty."
        return "\n".join(lines)

    def _cancel_text(self, conv: Conversation, rest: str) -> str:
        arg = rest.strip()
        if not arg:
            return self._cancel_matching(conv, None)
        if arg.lower() == "all":
            return self._cancel_all(conv)
        try:
            job_id = int(arg.removeprefix("#"))
        except ValueError:
            return "usage: /cancel [job-id|all]"
        return self._cancel_matching(conv, job_id)

    def _cancel_all(self, conv: Conversation) -> str:
        removed = self._remove_pending_jobs(conv, None, all_matches=True)
        if removed:
            noun = "job" if removed == 1 else "jobs"
            return f"cancelled {removed} queued {noun}."
        return self._nothing_queued(conv)

    def _cancel_matching(self, conv: Conversation, job_id: int | None) -> str:
        removed = self._remove_pending_jobs(conv, job_id, all_matches=False)
        if removed:
            return f"cancelled queued job #{removed}."
        if job_id is None:
            return self._nothing_queued(conv)
        if self.current_job is not None and self.current_job.id == job_id:
            if self.current_job.conversation.key == conv.key:
                return f"job #{job_id} is already running and can't be cancelled."
            return f"no queued job #{job_id}."
        return f"no queued job #{job_id}."

    def _nothing_queued(self, conv: Conversation) -> str:
        if (
            self.current_job is not None
            and self.current_job.conversation.key == conv.key
        ):
            return "nothing queued to cancel. Current job is already running."
        return "nothing queued to cancel."

    def _pending_jobs_for(self, conv: Conversation) -> list[Job]:
        return [
            job
            for job in list(self.queue._queue)
            if job is not None and job.conversation.key == conv.key
        ]

    def _remove_pending_jobs(
        self, conv: Conversation, job_id: int | None, *, all_matches: bool
    ) -> int:
        kept: deque[Job | None] = deque()
        removed = 0
        removed_id = 0
        for item in self.queue._queue:
            matches = (
                item is not None
                and item.conversation.key == conv.key
                and (job_id is None or item.id == job_id)
                and (all_matches or removed == 0)
            )
            if matches:
                removed += 1
                removed_id = item.id
                self.queue.task_done()
            else:
                kept.append(item)
        self.queue._queue = kept
        return removed if all_matches else removed_id

    def _job_summary(self, job: Job) -> str:
        kind = f"cron '{job.label}'" if job.mode == "cron" else "chat"
        age_from = job.started_at if job.started_at is not None else job.queued_at
        age = self._format_age(time.time() - age_from)
        prompt = _truncate_prompt(job.prompt)
        return f"#{job.id} {kind}, {age} ago — {prompt}"

    @staticmethod
    def _format_age(seconds: float) -> str:
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m{seconds:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h{minutes:02d}m"

    # --- job worker ----------------------------------------------------
    async def _run_jobs(self) -> None:
        while True:
            job = await self.queue.get()
            if job is None:  # shutdown sentinel
                self.queue.task_done()
                return
            job.started_at = time.time()
            self.current_job = job
            try:
                await self._run_one_job(job)
            except Exception:
                log.exception("job failed")
            finally:
                self.current_job = None
                self.queue.task_done()

    async def _run_one_job(self, job: Job) -> None:
        conv = job.conversation
        await self.signal.send_typing(conv)
        try:
            if job.mode == "cron":
                log.info("running cron '%s'", job.label)
                answer = await self.agent.once(job.prompt)
                answer = f"[{job.label}]\n{answer}"
            else:
                answer = await self.agent.chat(conv.key, job.prompt)
        except Exception as exc:
            log.exception("agent error")
            answer = f"agent error: {exc}"
        finally:
            await self.signal.send_typing(conv, stop=True)
        delivered = await self._safe_send(conv, answer)
        # Advance/remove the schedule only once the firing was actually
        # delivered, so a crash or a failed send re-fires this cron later.
        if job.finalize:
            if delivered:
                self.scheduler.complete(job.label, created=job.cron_created)
            else:
                self.scheduler.release(job.label, created=job.cron_created)

    # --- scheduler loop ------------------------------------------------
    async def _run_scheduler(self) -> None:
        while True:
            # The scheduler isn't thread-safe; due() runs on the event loop
            # alongside every add/remove/list, so they never race.
            for job in self.scheduler.due(time.time()):
                log.info("cron '%s' is due", job.name)
                await self._enqueue(
                    Job(
                        job.conversation,
                        job.prompt,
                        "cron",
                        label=job.name,
                        finalize=True,
                        cron_created=job.created,
                    )
                )
            await asyncio.sleep(15)

    # --- helpers -------------------------------------------------------
    async def _safe_send(self, conv: Conversation, text: str) -> bool:
        """Send a reply, returning whether it was delivered."""
        try:
            for chunk in split_reply(text):
                await self.signal.send(conv, chunk)
            return True
        except Exception as exc:
            log.error("failed to send reply to %s: %s", conv.key, exc)
            return False
