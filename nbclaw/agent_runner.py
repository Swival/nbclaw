"""Bridges Signal conversations to the swival agent.

``swival.Session`` is synchronous and CPU/IO heavy, so every call is run in a
thread pool. Because the target here is a single local model, the daemon funnels
all agent work through one worker (see daemon.py); this class is therefore not
trying to be concurrency-safe across many simultaneous runs.

Each conversation keeps its own long-lived ``Session`` so chat context carries
across messages. Cron jobs run as independent one-shots and never touch chat
context.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from swival import AgentError, Result, Session

log = logging.getLogger("nbclaw.agent")


def _setup_and_report_mcp(session: Session) -> None:
    """Force swival's lazy setup and log how many MCP tools loaded.

    swival defers MCP startup to the first run/ask. Doing it eagerly at session
    creation means a misconfigured or unreachable server fails loudly here, in
    the daemon log, instead of silently on someone's first message.
    """
    try:
        session._setup()
        manager = session._mcp_manager
        count = len(manager.list_tools()) if manager is not None else 0
        log.info("MCP servers ready: %d tool(s) available", count)
    except Exception as exc:
        log.error("MCP server startup failed: %s", exc)


def _safe_close(session: Session) -> None:
    try:
        session.close()
    except Exception as exc:  # pragma: no cover - cleanup best effort
        log.debug("session close error: %s", exc)


class AgentRunner:
    def __init__(self, session_kwargs: dict[str, Any]) -> None:
        self._kwargs = session_kwargs
        self._sessions: dict[str, Session] = {}

    def _session_for(self, key: str) -> Session:
        session = self._sessions.get(key)
        if session is None:
            log.info("creating session for %s", key)
            session = Session(**self._kwargs)
            # swival starts MCP servers lazily inside the first run/ask. Trigger
            # it now so any spawn failure surfaces here, and log the tool count
            # as visible confirmation the servers actually came up.
            if self._kwargs.get("mcp_servers"):
                _setup_and_report_mcp(session)
            self._sessions[key] = session
        return session

    def reset(self, key: str) -> bool:
        """Drop a conversation's context. Returns True if there was one."""
        session = self._sessions.pop(key, None)
        if session is None:
            return False
        _safe_close(session)
        return True

    # --- blocking primitives (run inside the executor) -----------------
    def _ask_blocking(self, key: str, prompt: str) -> str:
        session = self._session_for(key)
        result = session.ask(prompt)
        return _answer_text(result)

    def _once_blocking(self, prompt: str) -> str:
        session = Session(**self._kwargs)
        try:
            return _answer_text(session.run(prompt))
        finally:
            _safe_close(session)

    # --- async wrappers ------------------------------------------------
    async def chat(self, key: str, prompt: str) -> str:
        return await asyncio.to_thread(self._ask_blocking, key, prompt)

    async def once(self, prompt: str) -> str:
        return await asyncio.to_thread(self._once_blocking, prompt)

    def close_all(self) -> None:
        for session in self._sessions.values():
            _safe_close(session)
        self._sessions.clear()


def _answer_text(result: Result) -> str:
    if result.answer:
        return result.answer
    if result.exhausted:
        raise AgentError("agent ran out of turns without an answer")
    raise AgentError("agent returned no answer")
