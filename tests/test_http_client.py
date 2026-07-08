"""Tests for the tiny HTTP client (``coreybot.llm.http_client``).

Instead of mocking, we spin up a REAL local HTTP server on 127.0.0.1 in a
background thread and talk to it over a loopback socket. This is a genuine
integration test of the networking code and doubles as executable
documentation of the request/response contract.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator, Tuple

import pytest

from coreybot.core.cancel import CancelToken, CancelledError
from coreybot.llm.http_client import (
    HTTPError,
    apost_json,
    apost_sse,
    post_json,
    post_sse,
)


class _Handler(BaseHTTPRequestHandler):
    """Routes a few paths used by the tests. Silent logging."""

    def log_message(self, *args):  # keep test output clean
        pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_POST(self):
        body = self._read_body()
        if self.path == "/echo":
            # Return the parsed JSON back plus the auth header we saw.
            payload = json.loads(body or b"{}")
            response = {
                "received": payload,
                "auth": self.headers.get("Authorization"),
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/boom":
            data = json.dumps({"error": "bad"}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # A realistic SSE stream: data lines, a comment/keep-alive, blanks.
            for line in [
                "data: hello",
                ": keep-alive",
                "",
                "data: world",
                "data: [DONE]",
            ]:
                self.wfile.write((line + "\n").encode())
            self.wfile.flush()
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture()
def server() -> Iterator[str]:
    """Start a throwaway HTTP server; yield its base URL; shut it down."""
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)  # port 0 = pick a free port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=2)


def test_post_json_sends_and_parses(server):
    result = post_json(
        f"{server}/echo",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer tok"},
    )
    assert result["received"]["model"] == "m"
    assert result["auth"] == "Bearer tok"


def test_post_json_raises_httperror_on_4xx(server):
    with pytest.raises(HTTPError) as excinfo:
        post_json(f"{server}/boom", {})
    assert excinfo.value.status == 400
    assert "bad" in excinfo.value.body


def test_post_json_connection_error_is_runtimeerror():
    # Nothing is listening on this port -> URLError -> RuntimeError.
    with pytest.raises(RuntimeError):
        post_json("http://127.0.0.1:9/none", {}, timeout=1)


def test_post_sse_yields_only_data_lines(server):
    chunks = list(post_sse(f"{server}/sse", {}))
    # Comment (":" line) and blank keep-alive line are filtered out; the raw
    # ``[DONE]`` sentinel is returned to the caller (provider decides to stop).
    assert chunks == ["hello", "world", "[DONE]"]

# --- async variants --------------------------------------------------------
async def test_apost_json_sends_and_parses(server):
    result = await apost_json(
        f"{server}/echo",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer tok"},
    )
    assert result["received"]["model"] == "m"
    assert result["auth"] == "Bearer tok"


async def test_apost_json_raises_httperror_on_4xx(server):
    with pytest.raises(HTTPError) as excinfo:
        await apost_json(f"{server}/boom", {})
    assert excinfo.value.status == 400


async def test_apost_sse_yields_only_data_lines(server):
    chunks = [chunk async for chunk in apost_sse(f"{server}/sse", {})]
    assert chunks == ["hello", "world", "[DONE]"]


async def test_apost_json_pre_cancelled_raises(server):
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        await apost_json(f"{server}/echo", {}, cancel_token=token)
