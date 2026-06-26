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
from dataclasses import dataclass

from . import commands, nl_schedule
from .agent_runner import AgentRunner
from .config import Config
from .scheduler import Scheduler, fmt_time, validate_schedule
from .signal_client import Conversation, IncomingMessage, SignalClient

log = logging.getLogger("nbclaw")


def _split_command(text: str) -> tuple[str, str]:
    """Split ``"verb the rest"`` into (lowercased verb, remainder)."""
    parts = text.split(None, 1)
    verb = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    return verb, rest


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


class Daemon:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.signal = SignalClient(config.signal_url)
        self.agent = AgentRunner(config.session_kwargs())
        self.scheduler = Scheduler(config.crons_path())
        # ``None`` is the shutdown sentinel that lets the worker drain and exit.
        self.queue: asyncio.Queue[Job | None] = asyncio.Queue()
        self.start_time = time.time()
        self._stop = asyncio.Event()

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
            await self.queue.put(Job(msg.conversation, text, "chat"))
            return
        await self._handle_command(msg.conversation, text)

    # --- command handling ----------------------------------------------
    async def _handle_command(self, conv: Conversation, text: str) -> None:
        cmd, rest = _split_command(text[1:])

        if cmd in ("help", "h", "?"):
            await self.signal.send(conv, commands.HELP_TEXT)
        elif cmd == "status":
            await self.signal.send(conv, self._status_text())
        elif cmd in ("clear", "reset"):
            had = self.agent.reset(conv.key)
            await self.signal.send(
                conv, "context cleared." if had else "no context to clear."
            )
        elif cmd == "cron":
            await self._handle_cron(conv, rest)
        else:
            await self.signal.send(
                conv, f"unknown command: /{cmd}\n\n{commands.HELP_TEXT}"
            )

    async def _handle_cron(self, conv: Conversation, rest: str) -> None:
        sub, sub_rest = _split_command(rest)

        if sub in ("list", "ls", ""):
            await self.signal.send(conv, self._cron_list_text())
        elif sub in ("del", "rm", "delete", "cancel"):
            name = sub_rest.strip()
            ok = self.scheduler.remove(name)
            await self.signal.send(
                conv, f"cancelled cron '{name}'." if ok else f"no cron named '{name}'."
            )
        elif sub == "run":
            name = sub_rest.strip()
            job = self.scheduler.get(name)
            if job is None:
                await self.signal.send(conv, f"no cron named '{name}'.")
                return
            await self.signal.send(conv, f"running cron '{name}' now…")
            await self.queue.put(Job(job.conversation, job.prompt, "cron", label=name))
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
        except ValueError as exc:
            await self.signal.send(conv, f"cron add: {exc}")
            return
        job = self.scheduler.add(name, schedule, prompt, conv)
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
            job = self.scheduler.add(parsed.name, parsed.schedule, parsed.prompt, conv)
        await self.signal.send(
            conv,
            f"scheduled '{job.name}' — {parsed.describe()}.\n"
            f"next run: {fmt_time(job.next_run)}\n"
            f"task: {parsed.prompt}",
        )

    def _cron_list_text(self) -> str:
        jobs = self.scheduler.list()
        if not jobs:
            return "no scheduled tasks."
        lines = ["scheduled tasks:"]
        for job in jobs:
            prompt = job.prompt if len(job.prompt) <= 60 else job.prompt[:57] + "…"
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

    # --- job worker ----------------------------------------------------
    async def _run_jobs(self) -> None:
        while True:
            job = await self.queue.get()
            if job is None:  # shutdown sentinel
                self.queue.task_done()
                return
            try:
                await self._run_one_job(job)
            except Exception:
                log.exception("job failed")
            finally:
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
                await self.queue.put(
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
            await self.signal.send(conv, text)
            return True
        except Exception as exc:
            log.error("failed to send reply to %s: %s", conv.key, exc)
            return False
