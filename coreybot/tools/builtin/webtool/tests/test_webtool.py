"""Unit tests for the ``webtool`` builtin (colocated with the tool).

Network is stubbed via a fake ``urlopen`` so the tests are hermetic and fast;
they exercise argument handling, method resolution, size capping, and error
reporting without making real requests.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

from coreybot.security.capabilities import Capability
from coreybot.tools.builtin import webtool as webtool_pkg
from coreybot.tools.builtin.webtool import SPEC, webtool
from coreybot.tools.builtin.webtool.tool import _MAX_BODY_CHARS


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers=None):
        self._body = body
        self.status = status
        self.headers = headers or {"Content-Type": "text/plain"}

    def read(self, amount=None):
        return self._body if amount is None else self._body[:amount]

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake(monkeypatch, captured, response):
    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["data"] = request.data
        captured["timeout"] = timeout
        return response

    monkeypatch.setattr(webtool_pkg.tool.urllib.request, "urlopen", fake_urlopen)


def test_get_returns_status_and_body(monkeypatch):
    captured = {}
    _install_fake(monkeypatch, captured, _FakeResponse(b"hello body"))
    result = webtool("https://example.com/page")
    assert result.ok
    assert captured["method"] == "GET"
    assert "HTTP 200 GET https://example.com/page" in result.output
    assert "hello body" in result.output


def test_data_implies_post(monkeypatch):
    captured = {}
    _install_fake(monkeypatch, captured, _FakeResponse(b"ok"))
    result = webtool("https://example.com/submit", data="a=1")
    assert result.ok
    assert captured["method"] == "POST"
    assert captured["data"] == b"a=1"


def test_rejects_non_http_scheme():
    result = webtool("ftp://example.com/file")
    assert not result.ok
    assert "http" in result.output


def test_empty_url_rejected():
    result = webtool("   ")
    assert not result.ok
    assert "non-empty" in result.output


def test_non_positive_timeout_rejected():
    result = webtool("https://example.com", timeout=0)
    assert not result.ok
    assert "must be positive" in result.output


def test_body_is_truncated(monkeypatch):
    captured = {}
    big = b"x" * (_MAX_BODY_CHARS + 500)
    _install_fake(monkeypatch, captured, _FakeResponse(big))
    result = webtool("https://example.com/big")
    assert result.ok
    assert "body truncated" in result.output


def test_http_error_is_reported(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", hdrs=None, fp=io.BytesIO(b"nope")
        )

    monkeypatch.setattr(webtool_pkg.tool.urllib.request, "urlopen", fake_urlopen)
    result = webtool("https://example.com/missing")
    assert not result.ok
    assert "HTTP 404" in result.output


def test_url_error_is_reported(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(webtool_pkg.tool.urllib.request, "urlopen", fake_urlopen)
    result = webtool("https://example.com")
    assert not result.ok
    assert "request failed" in result.output


def test_webtool_result_carries_execution_log(monkeypatch):
    captured = {}
    _install_fake(monkeypatch, captured, _FakeResponse(b"hello", headers={"Content-Type": "text/plain"}))
    result = webtool("https://example.com/data")
    assert result.ok
    log = result.log
    assert "method: GET" in log
    assert "status: 200" in log
    assert "body:" in log


def test_webtool_spec_declares_network_and_side_effect():
    assert SPEC.name == "webtool"
    assert SPEC.safety.has(Capability.NETWORK)
    assert SPEC.safety.has(Capability.EXTERNAL_SIDE_EFFECT)
    assert SPEC.safety.compensation
