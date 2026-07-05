"""Configuration for the nbclaw daemon.

Settings come from three layers, lowest precedence first:

1. Built-in defaults (this file).
2. A TOML config file (``--config nbclaw.toml``).
3. Command-line flags.

The TOML file mirrors the dataclass field names. A ``[swival]`` table is
forwarded verbatim as keyword arguments to ``swival.Session`` so the full
agent can be configured even for options nbclaw doesn't expose directly.
"""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# First line of every AGENTS.md we write. Lets us recognise (and safely
# overwrite) our own file without clobbering one a user put in their workspace.
MANAGED_MARKER = "<!-- managed by nbclaw — edits will be overwritten -->"

# Injected into the agent (as AGENTS.md in the workspace) so it behaves like a
# chat assistant, not a coding agent that "keeps going until the task is done".
# Without this, small local models tool-loop on trivial messages like "Hi".
DEFAULT_INSTRUCTIONS = """\
# nbclaw assistant

You are a personal assistant reachable over Signal text messages. Replies are
read on a phone, so keep them short and conversational — no headings or long
lists unless asked.

## Conversation

Every message is part of one ongoing conversation. Read each new message in the
context of everything said before it, and resolve follow-ups against earlier
turns instead of asking what they refer to. For example, if you just answered
"add 2 + 2" with "4" and the next message is "add 10", that means 4 + 10 = 14.
Only ask for clarification when the conversation genuinely gives you nothing to
go on.

## Style

- For greetings, small talk, and general questions, just answer in a single
  reply. Do NOT read files, run commands, or use any tools for these.
- Use tools only when the user clearly asks you to do something on this computer
  (e.g. "list my files", "what's in ~/notes.txt", "run the tests"). Do exactly
  that, then report the result briefly.
- Never go exploring on your own or keep working past what was asked. One
  message in, one helpful reply out.
"""


@dataclass
class InstructionsWrite:
    """Outcome of :meth:`Config.write_instructions`."""

    path: Path | None
    action: Literal["written", "skipped", "disabled"]


@dataclass
class Config:
    # --- Signal / signal-cli ---
    signal_url: str = "http://127.0.0.1:3080"
    # Senders allowed to drive the agent (E.164 numbers and/or UUIDs).
    allow: list[str] = field(default_factory=list)
    allow_all: bool = False
    # Number to notify when the daemon comes online (optional). Note-to-Self works.
    notify: str | None = None

    # --- Model / provider ---
    provider: str = "lmstudio"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None

    # --- Agent behaviour ---
    # When safe=True the agent is read-only (no shell commands, no edits).
    safe: bool = False
    # Backstop on the agent loop. Kept low: a chat reply rarely needs many turns,
    # and a low cap means even a confused model returns *something* promptly.
    max_turns: int = 20
    # Agent persona/instructions, injected as the workspace AGENTS.md. None uses
    # the built-in chat-assistant framing; set "" to inject nothing.
    instructions: str | None = None
    # MCP servers, in swival's format: {name: {command|url, args?, env?, headers?}}.
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    # Extra kwargs forwarded verbatim to swival.Session (the [swival] TOML table).
    swival: dict[str, Any] = field(default_factory=dict)

    # --- State / workspace ---
    state_dir: Path = field(default_factory=lambda: Path.home() / ".nbclaw")
    workspace_dir: Path | None = None  # defaults to state_dir/workspace

    # Grace period (seconds) for an in-flight job to finish on shutdown before
    # it's abandoned. TOML-configurable; rarely needs changing.
    shutdown_timeout: float = 20.0

    def resolved_workspace(self) -> Path:
        return self.workspace_dir or (self.state_dir / "workspace")

    def crons_path(self) -> Path:
        return self.state_dir / "crons.json"

    def effective_instructions(self) -> str:
        """The AGENTS.md text to inject, or "" to inject nothing."""
        return DEFAULT_INSTRUCTIONS if self.instructions is None else self.instructions

    def _owns_agents_md(self, path: Path) -> bool:
        """True if we may overwrite ``path``.

        The default managed workspace is entirely ours. A custom workspace_dir
        may be a real project, so there we only touch an AGENTS.md that doesn't
        exist yet or that carries our marker (i.e. a previous run wrote it).
        """
        if self.workspace_dir is None or not path.exists():
            return True
        try:
            first_line = path.read_text().split("\n", 1)[0]
        except OSError:
            return False
        return first_line.strip() == MANAGED_MARKER

    def write_instructions(self) -> InstructionsWrite:
        """Materialize the agent instructions as the workspace AGENTS.md.

        swival appends AGENTS.md to its system prompt, so this is how we give the
        agent its chat-assistant persona. To avoid clobbering a user's own
        AGENTS.md when workspace_dir points at a real project, an existing file
        we didn't write is left untouched.
        """
        text = self.effective_instructions().strip()
        if not text:
            return InstructionsWrite(None, "disabled")
        path = self.resolved_workspace() / "AGENTS.md"
        if not self._owns_agents_md(path):
            return InstructionsWrite(path, "skipped")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{MANAGED_MARKER}\n{text}\n")
        return InstructionsWrite(path, "written")

    def session_kwargs(self) -> dict[str, Any]:
        """Build the keyword arguments passed to ``swival.Session``."""
        kwargs: dict[str, Any] = {
            "base_dir": str(self.resolved_workspace()),
            "provider": self.provider,
            "max_turns": self.max_turns,
            # We manage multi-turn context ourselves via long-lived Session objects;
            # swival's own on-disk history/continue isn't what we want for a chat bot.
            "history": False,
            "continue_here": False,
        }
        if self.model:
            kwargs["model"] = self.model
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.mcp_servers:
            kwargs["mcp_servers"] = self.mcp_servers
        if self.safe:
            kwargs.update(commands="none", files="none", read_guard=True)
        else:
            # Capable assistant: it can act when asked. Access is gated by the
            # sender allowlist. We deliberately avoid yolo — its autonomy push
            # makes small models tool-loop on plain chat; the AGENTS.md persona
            # (see Config.write_instructions) keeps replies tool-free by default.
            kwargs.update(commands="all", files="all")
        # The [swival] table overrides anything above.
        kwargs.update(self.swival)
        return kwargs

    def parser_session_kwargs(self) -> dict[str, Any]:
        """Minimal, tool-free Session kwargs for a single JSON parse call."""
        kwargs: dict[str, Any] = {
            "provider": self.provider,
            "commands": "none",
            "files": "none",
            "memory": False,
            "history": False,
            "no_skills": True,
            "continue_here": False,
            "max_turns": 2,
        }
        for key in ("model", "base_url", "api_key"):
            value = getattr(self, key)
            if value:
                kwargs[key] = value
        return kwargs


def build_config(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        prog="nbclaw",
        description="No Bullshit Claw — a 24/7 Signal-driven Swival agent daemon.",
    )
    parser.add_argument("--config", type=Path, help="Path to a TOML config file.")
    parser.add_argument("--signal-url", help="signal-cli HTTP daemon base URL.")
    parser.add_argument(
        "--allow",
        action="append",
        default=None,
        metavar="NUMBER|UUID",
        help="Allow this sender to drive the agent (repeatable).",
    )
    parser.add_argument(
        "--allow-all",
        action="store_true",
        default=None,
        help="Allow ALL senders (dangerous — the agent can run shell commands).",
    )
    parser.add_argument("--notify", help="Number to message when the daemon starts.")
    parser.add_argument("--provider", help="swival provider (lmstudio, generic, ...).")
    parser.add_argument("--model", help="Model id (e.g. ornith-1.0-9b).")
    parser.add_argument("--base-url", help="Override the provider endpoint URL.")
    parser.add_argument("--api-key", help="API key for the provider, if needed.")
    parser.add_argument(
        "--safe",
        action="store_true",
        default=None,
        help="Read-only agent: no shell commands, no file edits.",
    )
    parser.add_argument("--max-turns", type=int, help="Max agent loop iterations.")
    parser.add_argument(
        "--state-dir", type=Path, help="Directory for crons + workspace."
    )
    parser.add_argument("--workspace-dir", type=Path, help="Agent working directory.")

    args = parser.parse_args(argv)

    cfg = Config()
    if args.config:
        with args.config.open("rb") as f:
            data = tomllib.load(f)
        swival_table = data.pop("swival", None)
        for key, value in data.items():
            if not hasattr(cfg, key):
                raise SystemExit(f"Unknown config key: {key!r} in {args.config}")
            if key in ("state_dir", "workspace_dir") and value is not None:
                value = Path(value).expanduser()
            setattr(cfg, key, value)
        if isinstance(swival_table, dict):
            cfg.swival = swival_table

    # CLI overrides (only when explicitly provided).
    if args.signal_url is not None:
        cfg.signal_url = args.signal_url
    if args.allow is not None:
        cfg.allow = args.allow
    if args.allow_all is not None:
        cfg.allow_all = args.allow_all
    if args.notify is not None:
        cfg.notify = args.notify
    if args.provider is not None:
        cfg.provider = args.provider
    if args.model is not None:
        cfg.model = args.model
    if args.base_url is not None:
        cfg.base_url = args.base_url
    if args.api_key is not None:
        cfg.api_key = args.api_key
    if args.safe is not None:
        cfg.safe = args.safe
    if args.max_turns is not None:
        cfg.max_turns = args.max_turns
    if args.state_dir is not None:
        cfg.state_dir = args.state_dir.expanduser()
    if args.workspace_dir is not None:
        cfg.workspace_dir = args.workspace_dir.expanduser()

    return cfg
