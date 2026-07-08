"""Unit tests for the ``bash`` builtin (colocated with the tool).

These tests only run harmless commands directly (the direct call bypasses the
safety policy); the policy-level gating of dangerous commands is covered by the
safety-layer tests. They stay cross-platform by using ``python -c`` snippets.
"""

from __future__ import annotations

import sys

from coreybot.security.capabilities import Capability
from coreybot.tools.builtin.bash import SPEC, run_bash


def test_runs_command_and_reports_success():
    result = run_bash(f'{sys.executable} -c "print(2 + 3)"')
    assert result.ok
    assert "exit code: 0" in result.output
    assert "5" in result.output


def test_nonzero_exit_is_a_failure():
    result = run_bash(f'{sys.executable} -c "import sys; sys.exit(3)"')
    assert not result.ok
    assert "exit code: 3" in result.output


def test_empty_command_is_rejected():
    result = run_bash("   ")
    assert not result.ok
    assert "non-empty" in result.output


def test_non_positive_timeout_rejected():
    result = run_bash("echo hi", timeout=0)
    assert not result.ok
    assert "must be positive" in result.output


def test_timeout_kills_a_hanging_command():
    result = run_bash(
        f'{sys.executable} -c "import time; time.sleep(5)"', timeout=0.5
    )
    assert not result.ok
    assert "timed out" in result.output


def test_workdir_is_respected(local_tmp_path):
    script = "import os; print(os.path.basename(os.getcwd()))"
    result = run_bash(f'{sys.executable} -c "{script}"', workdir=str(local_tmp_path))
    assert result.ok
    assert local_tmp_path.name in result.output


def test_bash_spec_declares_exec_and_destructive():
    assert SPEC.name == "bash"
    assert SPEC.safety.has(Capability.EXEC)
    assert SPEC.safety.has(Capability.DESTRUCTIVE)
