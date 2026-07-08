"""Unit tests for the ``write_file`` builtin (direct call, no agent)."""

from __future__ import annotations

from pathlib import Path

from coreybot.tools.builtin.write_file import SPEC, write_file
from coreybot.security.capabilities import Capability


def test_write_file_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    result = write_file(path=str(target), content="hello")
    assert result.ok
    assert target.read_text(encoding="utf-8") == "hello"
    assert "created" in result.output


def test_write_file_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("old", encoding="utf-8")
    result = write_file(path=str(target), content="new")
    assert result.ok
    assert target.read_text(encoding="utf-8") == "new"
    assert "overwrote" in result.output


def test_write_file_rejects_non_string_content(tmp_path: Path) -> None:
    result = write_file(path=str(tmp_path / "x.txt"), content=123)  # type: ignore[arg-type]
    assert not result.ok


def test_spec_declares_fs_write_capability_and_paths() -> None:
    assert SPEC.safety.has(Capability.FS_WRITE)
    assert SPEC.safety.paths_for({"path": "/tmp/a.txt"}) == ["/tmp/a.txt"]
