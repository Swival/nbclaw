"""Persisted recurring-task scheduler ("crons").

A cron is a named, scheduled prompt bound to a Signal conversation. When it
fires, the prompt is run through the agent and the answer is delivered back to
that conversation.

Schedules accept three forms:

* A standard 5-field cron expression: ``"0 9 * * 1-5"``
* ``@every <duration>``: ``@every 30m``, ``@every 90s``, ``@every 2h``, ``@every 1d``
* Named shortcuts: ``@hourly``, ``@daily``, ``@weekly``, ``@monthly``
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from croniter import croniter

from .signal_client import Conversation

log = logging.getLogger("nbclaw.cron")

# Most crons one conversation may keep at once. They come from untrusted Signal
# senders and persist on disk, so this caps what a sender can pile up — both the
# storage and the agent runs each cron triggers when it fires.
MAX_CRONS_PER_CONVERSATION = 50

_NAMED = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
}

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text: str) -> int:
    """Parse ``30m`` / ``2h`` / ``90s`` / ``1d`` into a positive number of seconds."""
    match = re.fullmatch(r"(\d+)\s*([smhd])", text.strip().lower())
    if not match:
        raise ValueError(f"bad duration {text!r} (use e.g. 30s, 5m, 2h, 1d)")
    seconds = int(match.group(1)) * _DURATION_UNITS[match.group(2)]
    if seconds <= 0:
        raise ValueError(f"duration must be greater than zero, got {text!r}")
    return seconds


def next_fire(schedule: str, after: float) -> float:
    """Compute the next epoch timestamp at/after ``after`` for a schedule."""
    schedule = schedule.strip()
    lowered = schedule.lower()
    if lowered in _NAMED:
        return croniter(_NAMED[lowered], after).get_next(float)
    if lowered.startswith("@every"):
        parts = schedule.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            raise ValueError("'@every' needs a duration, e.g. @every 30m")
        return after + parse_duration(parts[1])
    # Standard cron expression — validates by construction.
    return croniter(schedule, after).get_next(float)


def validate_schedule(schedule: str) -> None:
    """Raise ValueError if the schedule string isn't understood."""
    next_fire(schedule, time.time())


def slugify(text: str) -> str:
    """A short kebab-case identifier derived from arbitrary text."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    slug = "-".join(words[:4])
    return slug or "task"


TIME_FMT = "%Y-%m-%d %H:%M"


def fmt_time(ts: float) -> str:
    """Format an epoch timestamp as local 'YYYY-MM-DD HH:MM'."""
    return time.strftime(TIME_FMT, time.localtime(ts))


@dataclass
class CronJob:
    name: str
    schedule: str
    prompt: str
    # Conversation to deliver results to.
    recipient: str | None
    group_id: str | None
    next_run: float
    last_run: float | None = None
    created: float = 0.0
    # One-shot jobs are removed after they fire instead of being rescheduled.
    once: bool = False

    @property
    def conversation(self) -> Conversation:
        return Conversation(recipient=self.recipient, group_id=self.group_id)


class Scheduler:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.jobs: dict[str, CronJob] = {}
        # Jobs handed out by due() but not yet finalized by complete(). Held in
        # memory only: after a crash it's empty, so an interrupted firing is
        # picked up again on the next tick instead of being lost.
        self._claimed: set[str] = set()
        self._load()

    # --- persistence ---------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.error("could not read crons from %s: %s", self.path, exc)
            return
        for item in raw.get("jobs", []):
            try:
                job = CronJob(**item)
                self.jobs[job.name] = job
            except TypeError as exc:
                log.error("skipping malformed cron entry %s: %s", item, exc)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"jobs": [asdict(j) for j in self.jobs.values()]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    # --- mutation ------------------------------------------------------
    def add(
        self,
        name: str,
        schedule: str,
        prompt: str,
        conv: Conversation,
        *,
        now: float | None = None,
        once: bool = False,
        next_run: float | None = None,
    ) -> CronJob:
        """Schedule a job. ``next_run`` overrides the schedule (one-shot times)."""
        now = time.time() if now is None else now
        if self._count_for(conv.key) >= MAX_CRONS_PER_CONVERSATION:
            raise ValueError(
                f"too many scheduled tasks for this conversation "
                f"(limit {MAX_CRONS_PER_CONVERSATION}); remove some with /cron del"
            )
        if next_run is None:
            validate_schedule(schedule)
            next_run = next_fire(schedule, now)
        job = CronJob(
            name=self._unique_name(name),
            schedule=schedule,
            prompt=prompt,
            recipient=conv.recipient,
            group_id=conv.group_id,
            next_run=next_run,
            created=now,
            once=once,
        )
        self.jobs[job.name] = job
        self._save()
        return job

    def _unique_name(self, name: str) -> str:
        if name not in self.jobs:
            return name
        i = 2
        while f"{name}-{i}" in self.jobs:
            i += 1
        return f"{name}-{i}"

    def remove(self, name: str) -> bool:
        if name in self.jobs:
            del self.jobs[name]
            self._claimed.discard(name)
            self._save()
            return True
        return False

    def get(self, name: str) -> CronJob | None:
        return self.jobs.get(name)

    def get_for(self, name: str, key: str) -> CronJob | None:
        """Like :meth:`get`, but only if the cron belongs to conversation ``key``."""
        job = self.jobs.get(name)
        return job if job is not None and job.conversation.key == key else None

    def list(self) -> list[CronJob]:
        return sorted(self.jobs.values(), key=lambda j: j.next_run)

    def list_for(self, key: str) -> list[CronJob]:
        """Crons owned by one conversation, soonest first."""
        return sorted(self._for(key), key=lambda j: j.next_run)

    def _for(self, key: str):
        """Jobs owned by the conversation with this ``Conversation.key``."""
        return (j for j in self.jobs.values() if j.conversation.key == key)

    def _count_for(self, key: str) -> int:
        return sum(1 for _ in self._for(key))

    # --- ticking -------------------------------------------------------
    def due(self, now: float | None = None) -> list[CronJob]:
        """Claim jobs whose time has come and return them.

        Claiming (rather than advancing) means a job stays "due" until
        :meth:`complete` is called once its prompt has actually run and the
        reply was sent. A claimed job is skipped by later ticks, so a slow run
        isn't fired twice; a crash before completion re-fires on restart.

        Must be called from the same thread/event loop as every other method —
        the scheduler holds no lock.
        """
        now = time.time() if now is None else now
        ready: list[CronJob] = []
        for job in list(self.jobs.values()):
            if job.name in self._claimed:
                continue
            if job.next_run <= now:
                self._claimed.add(job.name)
                ready.append(job)
        return ready

    def complete(
        self, name: str, *, created: float | None = None, now: float | None = None
    ) -> None:
        """Finalize a fired job: advance a recurring schedule, drop a one-shot.

        Called after the firing's prompt has run and its reply was delivered.
        Releasing the claim here (not in due()) is what makes an interrupted
        firing retry on the next tick instead of being silently dropped.

        ``created`` pins the finalization to the exact job that fired: if the
        name was deleted and re-added in the meantime, a stale completion is
        ignored so it can't clobber the new job.
        """
        now = time.time() if now is None else now
        if self._is_stale(name, created):
            return  # a different job owns this name now; leave it alone
        self._claimed.discard(name)
        job = self.jobs.get(name)
        if job is None:
            return
        job.last_run = now
        if job.once:
            del self.jobs[name]  # one-shot: don't reschedule
        else:
            job.next_run = next_fire(job.schedule, now)
        self._save()

    def release(self, name: str, *, created: float | None = None) -> None:
        """Release a claim without finalizing, so the firing retries next tick.

        Used when a firing ran but its reply couldn't be delivered: the
        schedule is left untouched and the job becomes due again.
        """
        if self._is_stale(name, created):
            return
        self._claimed.discard(name)

    def _is_stale(self, name: str, created: float | None) -> bool:
        """True if ``name`` now refers to a different job than the one fired."""
        job = self.jobs.get(name)
        return job is not None and created is not None and job.created != created
