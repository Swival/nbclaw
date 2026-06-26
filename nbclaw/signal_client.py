"""Thin async client for the signal-cli HTTP daemon.

signal-cli (started with ``daemon --http``) exposes two things we use:

* ``POST /api/v1/rpc``     — JSON-RPC for sending messages, typing, etc.
* ``GET  /api/v1/events``  — a Server-Sent-Events stream of incoming messages,
  each ``data:`` line being a JSON-RPC ``receive`` notification.

We only care about plain text data-messages here. Receipts, typing
notifications and own-account sync messages are ignored.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger("nbclaw.signal")


@dataclass(frozen=True)
class Conversation:
    """Where a message came from and where replies should go."""

    # Exactly one of these identifies the destination.
    recipient: str | None = None  # E.164 number for a 1:1 chat
    group_id: str | None = None  # base64 group id for a group chat

    @property
    def key(self) -> str:
        return f"group:{self.group_id}" if self.group_id else f"user:{self.recipient}"

    def routing_params(self) -> dict:
        """The signal-cli params that address this conversation."""
        if self.group_id:
            return {"groupId": self.group_id}
        if self.recipient is None:
            raise ValueError("conversation has neither a recipient nor a group_id")
        return {"recipient": [self.recipient]}

    def send_params(self, message: str) -> dict:
        return {"message": message, **self.routing_params()}


@dataclass
class IncomingMessage:
    conversation: Conversation
    text: str
    source: str | None  # sender E.164 number
    source_uuid: str | None
    timestamp: int


class SignalClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.rpc_url = f"{self.base_url}/api/v1/rpc"
        self.events_url = f"{self.base_url}/api/v1/events"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
        self._ids = itertools.count(1)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _rpc(self, method: str, params: dict | None = None) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": next(self._ids),
        }
        if params is not None:
            payload["params"] = params
        resp = await self._client.post(self.rpc_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"signal-cli {method} error: {data['error']}")
        return data.get("result", {})

    async def send(self, conv: Conversation, message: str) -> None:
        await self._rpc("send", conv.send_params(message))

    async def send_reaction(self, msg: IncomingMessage, emoji: str = "👀") -> None:
        """React to an incoming message. Best-effort, like typing indicators."""
        target_author = msg.source or msg.source_uuid
        if not target_author or not msg.timestamp:
            log.debug("cannot react: missing target author or timestamp")
            return
        params = {
            "emoji": emoji,
            "targetAuthor": target_author,
            "targetTimestamp": msg.timestamp,
            **msg.conversation.routing_params(),
        }
        try:
            await self._rpc("sendReaction", params)
        except Exception as exc:  # reactions are best-effort
            log.debug("sendReaction failed: %s", exc)

    async def send_typing(self, conv: Conversation, *, stop: bool = False) -> None:
        params = {"stop": stop, **conv.routing_params()}
        try:
            await self._rpc("sendTyping", params)
        except Exception as exc:  # typing indicators are best-effort
            log.debug("sendTyping failed: %s", exc)

    async def version(self) -> str:
        result = await self._rpc("version")
        return result.get("version", "?")

    async def events(self):
        """Yield :class:`IncomingMessage` objects forever, reconnecting on drop."""
        backoff = 1.0
        while True:
            try:
                async with self._client.stream("GET", self.events_url) as resp:
                    resp.raise_for_status()
                    backoff = 1.0
                    log.info("connected to signal-cli event stream")
                    async for envelope in _parse_sse(resp):
                        msg = _to_message(envelope)
                        if msg is not None:
                            yield msg
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "event stream error (%s); reconnecting in %.0fs", exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)


async def _parse_sse(resp: httpx.Response):
    """Parse an SSE stream, yielding the decoded JSON of each ``data:`` event."""
    data_lines: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if data_lines:
                raw = "\n".join(data_lines)
                data_lines = []
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    log.debug("skipping non-JSON SSE payload: %r", raw[:200])
            continue
        if line.startswith(":"):
            continue  # comment / keepalive
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        # other SSE fields (event:, id:) are ignored


def _to_message(envelope: dict) -> IncomingMessage | None:
    """Extract a usable text message, or None for anything we don't act on.

    Two channels carry a command, depending on how the bot's number is set up:

    * ``dataMessage`` — someone else messaged this account. Reply to the sender.
    * ``syncMessage.sentMessage`` — this account sent a message from one of its
      own devices (your phone) to itself: "Note to Self". This is the channel
      used when nbclaw runs on *your own* Signal number. Reply to Note to Self.

    Sync messages addressed to anyone other than yourself are ignored, so the
    bot never hijacks your normal outgoing conversations.
    """
    # The SSE payload may be a bare {"envelope": ...} or a JSON-RPC
    # {"method":"receive","params":{"envelope": ...}} notification.
    params = envelope.get("params", envelope)
    env = params.get("envelope", params)

    data_message = env.get("dataMessage")
    if data_message and data_message.get("message"):
        group_id = (data_message.get("groupInfo") or {}).get("groupId")
        source = env.get("sourceNumber") or env.get("source")
        conv = (
            Conversation(group_id=group_id)
            if group_id
            else Conversation(recipient=source)
        )
        return _build(env, data_message["message"], conv, env.get("timestamp", 0))

    sent = (env.get("syncMessage") or {}).get("sentMessage")
    if sent and sent.get("message") and _is_note_to_self(env, sent):
        dest = sent.get("destinationNumber") or sent.get("destination")
        return _build(
            env, sent["message"], Conversation(recipient=dest), sent.get("timestamp", 0)
        )

    return None  # receipt, typing, reaction, sync-to-someone-else, etc.


def _is_note_to_self(env: dict, sent: dict) -> bool:
    """True if a synced sent-message was addressed to the account itself."""
    if sent.get("groupInfo"):
        return False  # a group message we sent — not for the bot
    src_uuid = env.get("sourceUuid")
    src_num = env.get("sourceNumber") or env.get("source")
    dst_uuid = sent.get("destinationUuid")
    dst_num = sent.get("destinationNumber") or sent.get("destination")
    return bool((dst_uuid and dst_uuid == src_uuid) or (dst_num and dst_num == src_num))


def _build(env: dict, text: str, conv: Conversation, timestamp: int) -> IncomingMessage:
    return IncomingMessage(
        conversation=conv,
        text=text.strip(),
        source=env.get("sourceNumber") or env.get("source"),
        source_uuid=env.get("sourceUuid"),
        timestamp=timestamp,
    )
