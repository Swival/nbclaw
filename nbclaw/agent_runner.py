"""Bridges Signal conversations to the swival agent.

``swival.Session`` is synchronous and CPU/IO heavy, so every call is run in a
thread pool. Because the target here is a single local model, the daemon funnels
all agent work through one worker (see daemon.py); this class is therefore not
trying to be concurrency-safe across many simultaneous runs. It does have to
survive ``/clear`` arriving on the event loop while the worker is mid-``ask()``.

Each conversation keeps its own long-lived ``Session`` so chat context carries
across messages. Cron jobs run as independent one-shots and never touch chat
context.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from swival import AgentError, Result, Session

log = logging.getLogger("nbclaw.agent")

# Cap on retained chat sessions, keyed by untrusted Signal sender (group id or
# 1:1 number). Past the cap the least-recently-used session is evicted so memory
# and per-session MCP processes can't grow without bound.
MAX_CHAT_SESSIONS = 32


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
        raise


def _safe_close(session: Session) -> None:
    try:
        session.close()
    except Exception as exc:  # pragma: no cover - cleanup best effort
        log.debug("session close error: %s", exc)


class AgentRunner:
    def __init__(self, session_kwargs: dict[str, Any]) -> None:
        self._kwargs = session_kwargs
        self._sessions: dict[str, Session] = {}
        # Checked out by the worker, and dropped while checked out: a session in
        # flight is closed on hand-back, never under the call still using it.
        self._in_use: dict[str, Session] = {}
        self._retired: set[Session] = set()
        # Sessions still being built. reset() drops the key to disown one.
        self._creating: set[str] = set()
        # The containers above are touched by both the event loop and the worker.
        self._lock = threading.Lock()

    def _session_for(self, key: str, *, in_use: bool = False) -> Session:
        """Get or create ``key``'s session, optionally checking it out.

        ``in_use=True`` claims the session for the caller, who then owes it a
        :meth:`_release`. The claim is taken under the lookup's own lock, so
        :meth:`reset` can't slip in between and close it.
        """
        with self._lock:
            session = self._sessions.get(key)
            if session is not None:
                # Most-recently-used goes last; eviction takes from the front.
                self._sessions[key] = self._sessions.pop(key)
                if in_use:
                    self._in_use[key] = session
                return session
            self._creating.add(key)

        log.info("creating session for %s", key)
        # Spawning MCP servers is too slow to hold the lock that /clear needs.
        try:
            session = self._create_session()
        except Exception:
            with self._lock:
                self._creating.discard(key)
            raise

        evicted: list[tuple[str, Session]] = []
        with self._lock:
            if key in self._creating:
                self._creating.discard(key)
                evicted = self._take_evictable()
                self._sessions[key] = session
            else:
                # Cleared while building: serve the call that asked for it from a
                # session nothing else can reach, then close it on hand-back.
                log.info("session for %s was cleared while starting up", key)
                self._retired.add(session)
            if in_use:
                self._in_use[key] = session
        for old_key, old_session in evicted:
            log.info("evicting least-recently-used session for %s", old_key)
            _safe_close(old_session)
        return session

    def _create_session(self) -> Session:
        session = Session(**self._kwargs)
        try:
            # swival starts MCP servers lazily inside the first run/ask. Trigger
            # it now so any spawn failure surfaces here, and log the tool count
            # as visible confirmation the servers actually came up.
            if self._kwargs.get("mcp_servers"):
                _setup_and_report_mcp(session)
        except Exception:
            # A failed setup may already own MCP processes or an event loop. Do
            # not cache that partial session: a later retry would overwrite its
            # managers and make those resources unreachable.
            _safe_close(session)
            raise
        return session

    def _take_evictable(self) -> list[tuple[str, Session]]:
        """Take the least-recently-used idle sessions once we're at the cap.

        The caller holds the lock and closes what comes back. A session in flight
        is never evicted.
        """
        evicted: list[tuple[str, Session]] = []
        while len(self._sessions) >= MAX_CHAT_SESSIONS:
            victim = next((k for k in self._sessions if k not in self._in_use), None)
            if victim is None:
                break  # all in flight: overshoot the cap rather than pull the rug
            evicted.append((victim, self._sessions.pop(victim)))
        return evicted

    def reset(self, key: str) -> bool:
        """Drop a conversation's context. Returns True if there was one.

        Safe to call while the worker is using the session: it leaves the cache
        now, so the next message starts fresh, and is closed once the call using
        it finishes.
        """
        with self._lock:
            session = self._sessions.pop(key, None)
            if session is None:
                had_pending = key in self._creating
                self._creating.discard(key)
                return had_pending
            if self._in_use.get(key) is session:
                self._retired.add(session)
                return True
        _safe_close(session)
        return True

    def _release(self, key: str, session: Session) -> None:
        """Hand back a checked-out session, closing it if it was retired."""
        with self._lock:
            if self._in_use.get(key) is session:
                del self._in_use[key]
            retired = session in self._retired
            self._retired.discard(session)
        if retired:
            log.info("closing session for %s, dropped while it was running", key)
            _safe_close(session)

    # --- blocking primitives (run inside the executor) -----------------
    def _ask_blocking(self, key: str, prompt: str) -> str:
        session = self._session_for(key, in_use=True)
        try:
            result = session.ask(prompt)
        finally:
            self._release(key, session)
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
        """Close every session, in use or not. Shutdown only.

        By then the worker has drained, or its job was abandoned after the grace
        period, so there's nothing left to protect.
        """
        with self._lock:
            sessions = [*self._sessions.values(), *self._retired]
            self._sessions.clear()
            self._retired.clear()
            self._in_use.clear()
            self._creating.clear()  # nothing built after this gets cached
        for session in sessions:
            _safe_close(session)


def _answer_text(result: Result) -> str:
    if result.answer:
        return result.answer
    if result.exhausted:
        raise AgentError("agent ran out of turns without an answer")
    raise AgentError("agent returned no answer")
