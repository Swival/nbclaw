"""Turn a natural-language scheduling request into a concrete schedule.

The cron-expression syntax is precise but unfriendly. Since nbclaw already has a
model wired up, we let it do the translation: given a sentence like "every
weekday at 9am summarize my git log", the model returns a small JSON object that
we validate and hand to the scheduler.

This is a single tool-free completion, not the full agent loop.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from swival import AgentError
from swival import run as swival_run

from .scheduler import fmt_time, next_fire, slugify, validate_schedule

log = logging.getLogger("nbclaw.nl")

_SYSTEM_PROMPT = """\
You convert a natural-language scheduling request into JSON. The current local
date and time is {now}.

Output ONLY a single JSON object — no prose, no code fences — with these fields:
- "name": a short 2-4 word kebab-case label for the task
- "type": "recurring" for a repeating schedule, or "once" for a one-time reminder
- "cron": a standard 5-field cron expression (ONLY when type is "recurring")
- "at": local date-time as "YYYY-MM-DD HH:MM" (ONLY when type is "once")
- "prompt": what to do or say when it fires, phrased as an instruction

Examples:
Request: every weekday at 9am summarize my git log in ~/src/app
{{"name":"git-standup","type":"recurring","cron":"0 9 * * 1-5","prompt":"Summarize today's git log in ~/src/app."}}
Request: remind me to stretch every 2 hours
{{"name":"stretch","type":"recurring","cron":"0 */2 * * *","prompt":"Remind me to stretch."}}
Request: in 10 minutes tell me to check the oven
{{"name":"check-oven","type":"once","at":"{example_once}","prompt":"Tell me to check the oven."}}
"""


@dataclass
class ParsedSchedule:
    name: str
    prompt: str
    once: bool
    schedule: str | None  # cron expression when recurring
    at_epoch: float | None  # epoch seconds when one-shot

    def describe(self) -> str:
        if self.once:
            return f"once at {fmt_time(self.at_epoch)}"
        return self.schedule


class ParseError(ValueError):
    pass


def parse_request(
    text: str, session_kwargs: dict[str, Any], now: float
) -> ParsedSchedule:
    """Parse ``text`` into a :class:`ParsedSchedule` using the configured model."""
    now_dt = datetime.datetime.fromtimestamp(now)
    example_once = (now_dt + datetime.timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M")
    system = _SYSTEM_PROMPT.format(
        now=now_dt.strftime("%A %Y-%m-%d %H:%M"), example_once=example_once
    )

    try:
        answer = swival_run(text, system_prompt=system, **session_kwargs)
    except AgentError as exc:
        raise ParseError("couldn't understand that schedule") from exc

    data = _extract_json(answer)
    if data is None:
        raise ParseError("could not understand that schedule")
    return _build(data, now)


def _extract_json(answer: str) -> dict | None:
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _build(data: dict, now: float) -> ParsedSchedule:
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        raise ParseError("I couldn't tell what the task should be")
    name = slugify(data.get("name") or prompt)

    kind = (data.get("type") or "").lower()
    # One-shot when the model says so, or when it gave a time but no cron expr.
    if kind == "once" or (not data.get("cron") and data.get("at")):
        at_epoch = _parse_at(data.get("at"))
        if at_epoch <= now:
            raise ParseError("that time is in the past")
        return ParsedSchedule(
            name=name, prompt=prompt, once=True, schedule=None, at_epoch=at_epoch
        )

    cron = (data.get("cron") or "").strip()
    if not cron:
        raise ParseError("I couldn't work out when to run that")
    try:
        validate_schedule(cron)
        next_fire(cron, now)  # ensure it actually produces a future time
    except ValueError as exc:
        raise ParseError(f"that schedule didn't make sense ({exc})") from exc
    return ParsedSchedule(
        name=name, prompt=prompt, once=False, schedule=cron, at_epoch=None
    )


def _parse_at(at: str | None) -> float:
    if not at:
        raise ParseError("I couldn't work out the time")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(at.strip(), fmt).timestamp()
        except ValueError:
            continue
    raise ParseError(f"I couldn't read the time {at!r}")
