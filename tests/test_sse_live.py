"""Exercise SignalClient.events() against a real HTTP SSE server.

The fake aiter_lines unit test covers parsing; this covers the actual httpx
streaming + reconnect path against a live socket, using a genuine signal-cli
``receive`` notification shape.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from nbclaw.signal_client import SignalClient

ENVELOPE = {
    "jsonrpc": "2.0",
    "method": "receive",
    "params": {
        "account": "+33695226193",
        "envelope": {
            "source": "+15551112222",
            "sourceNumber": "+15551112222",
            "sourceUuid": "abc-123",
            "timestamp": 1782413276619,
            "dataMessage": {"message": "ping from the wire"},
        },
    },
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        payload = json.dumps(ENVELOPE)
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()
        # Keep the connection open briefly so the client can read the event.
        time.sleep(1.0)


async def _consume(base_url):
    client = SignalClient(base_url)
    try:
        async for msg in client.events():
            return msg
    finally:
        await client.aclose()


def test_events_over_real_socket():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        msg = asyncio.run(
            asyncio.wait_for(_consume(f"http://127.0.0.1:{port}"), timeout=10)
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert msg is not None
    assert msg.text == "ping from the wire"
    assert msg.source == "+15551112222"
    assert msg.conversation.recipient == "+15551112222"
