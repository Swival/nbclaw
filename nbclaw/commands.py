"""Parsing helpers and help text for the slash-command interface.

Anything that doesn't start with ``/`` is treated as a prompt for the agent.
The command dispatch itself lives in :mod:`nbclaw.daemon` because it needs the
daemon's live state (scheduler, sessions, start time).
"""

from __future__ import annotations

HELP_TEXT = """nbclaw — commands

Plain text is sent to the agent. Slash commands:

/help              show this help
/status            model, uptime, active crons
/reset             forget this conversation's context

Scheduling — just say it in plain English after /cron:
  /cron every weekday at 9am summarize my git log in ~/src/app
  /cron remind me to stretch every 2 hours
  /cron tomorrow at 8am say good morning

/cron list         list scheduled tasks
/cron del <name>   cancel a scheduled task
/cron run <name>   run a scheduled task right now

Power-user form (exact cron expression):
  /cron add <name> <schedule> | <prompt>
  schedules: 0 9 * * 1-5  ·  @every 30m  ·  @hourly @daily @weekly @monthly
""".strip()


class CronAddError(ValueError):
    pass


def parse_cron_add(args: str) -> tuple[str, str, str]:
    """Parse the body of ``/cron add`` into (name, schedule, prompt).

    Grammar: ``<name> <schedule...> | <prompt>``
    The ``|`` separates the (space-containing) schedule from the prompt.
    """
    if "|" not in args:
        raise CronAddError("missing '|' separating the schedule from the prompt")
    head, prompt = args.split("|", 1)
    prompt = prompt.strip()
    head_parts = head.split()
    if len(head_parts) < 2:
        raise CronAddError("expected: <name> <schedule> | <prompt>")
    name = head_parts[0]
    schedule = " ".join(head_parts[1:])
    if not prompt:
        raise CronAddError("the prompt is empty")
    return name, schedule, prompt
