"""Unit tests for the ``calc`` builtin (colocated with the tool)."""

from __future__ import annotations

from coreybot.tools.builtin.calc import calc
from coreybot.tools.builtin.calc.tool import UnsafeExpressionError, _evaluate
import ast


def _run(expr: str):
    return calc(expr)


def test_calc_basic_arithmetic():
    assert _run("2*(3+4)").output == "14"
    assert _run("10 - 3 - 2").output == "5"


def test_calc_power_and_unary():
    assert _run("2 ** 10").output == "1024"
    assert _run("-2 ** 2").output in {"-4", "-4.0"}


def test_calc_division_by_zero_is_reported():
    result = _run("1/0")
    assert not result.ok
    assert "division by zero" in result.output


def test_calc_rejects_function_calls():
    result = _run("__import__('os').system('echo hi')")
    assert not result.ok
    assert "unsupported syntax" in result.output


def test_calc_rejects_names_and_attributes():
    assert not _run("x + 1").ok
    assert not _run("(2).__class__").ok


def test_calc_rejects_non_numeric_literal():
    result = _run("'a' * 3")
    assert not result.ok


def test_calc_reports_syntax_error():
    result = _run("2 +")
    assert not result.ok
    assert "invalid expression" in result.output


def test_evaluate_helper_raises_on_unsafe_node():
    tree = ast.parse("foo()", mode="eval")
    try:
        _evaluate(tree)
        assert False, "expected UnsafeExpressionError"
    except UnsafeExpressionError:
        pass


def test_calc_spec_declares_interface():
    from coreybot.tools.builtin.calc import SPEC

    assert SPEC.name == "calc"
    assert "arithmetic" in SPEC.description.lower()
    assert set(SPEC.parameters) == {"expression"}
