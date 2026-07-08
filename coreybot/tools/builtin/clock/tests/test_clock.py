"""Unit tests for the ``current_time`` builtin (colocated with the tool)."""

from __future__ import annotations

from datetime import datetime

from coreybot.tools.builtin.clock import current_time


def test_current_time_is_parseable_iso8601():
    value = current_time()
    # Round-trips through datetime.fromisoformat -> it is valid ISO-8601.
    parsed = datetime.fromisoformat(value)
    assert isinstance(parsed, datetime)


def test_current_time_has_second_precision():
    # timespec="seconds" -> no microseconds component.
    assert "." not in current_time()


def test_clock_spec_declares_interface():
    from coreybot.tools.builtin.clock import SPEC

    assert SPEC.name == "current_time"
    assert SPEC.parameters == {}
