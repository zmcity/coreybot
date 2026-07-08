"""Unit tests for the ``read_file`` builtin (colocated with the tool)."""

from __future__ import annotations

from coreybot.tools.builtin.read_file import read_file
from coreybot.tools.builtin.read_file.tool import _HARD_MAX_BYTES


def test_reads_full_file(local_tmp_path):
    target = local_tmp_path / "a.txt"
    target.write_text("hello world", encoding="utf-8")
    result = read_file(str(target))
    assert result.ok
    assert result.output == "hello world"


def test_truncates_to_max_bytes(local_tmp_path):
    target = local_tmp_path / "b.txt"
    target.write_text("hello world", encoding="utf-8")
    result = read_file(str(target), max_bytes=5)
    assert result.ok
    assert result.output == "hello"


def test_missing_file_is_reported():
    result = read_file("does/not/exist.txt")
    assert not result.ok
    assert "file not found" in result.output


def test_directory_is_rejected(local_tmp_path):
    result = read_file(str(local_tmp_path))
    assert not result.ok
    assert "not a file" in result.output


def test_non_positive_max_bytes_rejected(local_tmp_path):
    target = local_tmp_path / "c.txt"
    target.write_text("data", encoding="utf-8")
    result = read_file(str(target), max_bytes=0)
    assert not result.ok
    assert "must be positive" in result.output


def test_max_bytes_is_capped_to_hard_limit(local_tmp_path):
    target = local_tmp_path / "d.txt"
    target.write_text("data", encoding="utf-8")
    # Asking for more than the hard cap still succeeds (cap applied silently).
    result = read_file(str(target), max_bytes=_HARD_MAX_BYTES * 10)
    assert result.ok
    assert result.output == "data"


def test_read_file_spec_declares_interface():
    from coreybot.tools.builtin.read_file import SPEC

    assert SPEC.name == "read_file"
    assert set(SPEC.parameters) == {"path", "max_bytes"}
