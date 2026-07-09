"""Unit tests for the ``calc`` builtin (algebra/scientific calculator).

Covers numeric arithmetic, exact/symbolic results, calculus, equation solving,
variable substitution, output modes, and -- most importantly -- the safety
guard that stops SymPy\'s parser from being used for code execution.
"""

from __future__ import annotations

import pytest

from coreybot.tools.builtin.calc import SPEC, calc
from coreybot.tools.builtin.calc.tool import _prevet


def _out(expr, **kw):
    result = calc(expr, **kw)
    assert result.ok, f"expected ok for {expr!r}, got: {result.output}"
    return result.output


# --- numeric / exact arithmetic -------------------------------------------
def test_basic_arithmetic_is_exact():
    assert _out("2 * (3 + 4)") == "14"
    assert _out("10 - 3 - 2") == "5"


def test_power_via_caret_and_star_star():
    assert _out("2^10") == "1024"
    assert _out("2 ** 10") == "1024"


def test_rationals_stay_exact():
    assert _out("2/3 + 1/6") == "5/6"


def test_roots_and_constants():
    assert _out("sqrt(16)") == "4"
    assert _out("sqrt(2)") == "sqrt(2)"
    assert _out("sin(pi/2) + cos(0)") == "2"
    assert _out("log(e^2)") == "2"          # lowercase e is Euler\'s number


def test_implicit_multiplication():
    assert _out("3x + x^2") == "x**2 + 3*x"


def test_factorial_and_large_numbers():
    assert _out("factorial(10)") == "3628800"


def test_complex_numbers():
    assert _out("sqrt(-4)") == "2*I"


# --- algebra / calculus / solving -----------------------------------------
def test_factor_and_expand():
    assert _out("factor(x^2 - 1)") == "(x - 1)*(x + 1)"
    assert _out("expand((x + 1)^3)") == "x**3 + 3*x**2 + 3*x + 1"


def test_simplify():
    assert _out("simplify(sin(x)^2 + cos(x)^2)") == "1"


def test_differentiation_and_integration():
    assert _out("diff(x^3, x)") == "3*x**2"
    assert _out("integrate(x^2, x)") == "x**3/3"


def test_solve_returns_roots():
    assert _out("solve(x^2 - 4, x)") == "[-2, 2]"
    assert _out("solve(Eq(x^2, 4), x)") == "[-2, 2]"


def test_symbolic_expression_is_returned_as_is():
    assert _out("x^2 + 2x + 1") == "x**2 + 2*x + 1"


# --- variable substitution -------------------------------------------------
def test_variable_substitution_numeric():
    assert _out("x^2 + 1", variables={"x": 3}) == "10"


def test_variable_substitution_mixed_and_string_values():
    assert _out("x^2 + y", variables={"x": 2, "y": "3"}) == "7"


def test_variable_substitution_partial_keeps_symbol():
    assert _out("x + y", variables={"x": 1}) == "y + 1"


# --- output modes ----------------------------------------------------------
def test_numeric_mode_forces_decimal():
    out = _out("pi", mode="numeric")
    assert out.startswith("3.14159")


def test_auto_mode_keeps_pi_symbolic():
    assert _out("pi") == "pi"


def test_high_precision_via_N():
    out = _out("N(pi, 25)")
    assert out.startswith("3.14159265358979")
    assert len(out.replace(".", "")) >= 24


def test_invalid_mode_rejected():
    result = calc("1+1", mode="banana")
    assert not result.ok
    assert "mode must be" in result.output


# --- errors & edge cases ---------------------------------------------------
def test_division_by_zero_is_undefined():
    assert _out("1/0") == "undefined"


def test_empty_expression_rejected():
    result = calc("   ")
    assert not result.ok
    assert "non-empty" in result.output


def test_syntax_error_reported():
    result = calc("2 +")
    assert not result.ok
    assert "invalid expression" in result.output


# --- safety: SymPy parser must not become a code-exec vector ---------------
def test_blocks_dunder_import():
    result = calc("__import__(\'os\').system(\'echo hi\')")
    assert not result.ok
    assert "double-underscore" in result.output or "attribute access" in result.output


def test_blocks_attribute_access():
    result = calc("os.system(\'x\')")
    assert not result.ok
    assert "attribute access" in result.output


def test_blocks_class_traversal():
    result = calc("(1).__class__")
    assert not result.ok


def test_prevet_allows_decimals_and_functions():
    assert _prevet("3.5 + .25") is None
    assert _prevet("sin(pi/2)") is None
    assert _prevet("solve(Eq(x^2, 4), x)") is None


def test_prevet_flags_attribute_and_dunder():
    assert _prevet("x.foo") is not None
    assert _prevet("a__b") is not None


def test_expression_length_is_capped():
    result = calc("1+" * 2000 + "1")
    assert not result.ok
    assert "too long" in result.output


# --- interface / safety profile -------------------------------------------
def test_calc_spec_declares_interface():
    assert SPEC.name == "calc"
    assert set(SPEC.parameters) == {"expression", "variables", "mode"}


def test_calc_has_no_side_effect_capabilities():
    # Pure compute -> empty capability set -> risk engine treats it recoverable.
    assert not SPEC.safety.capabilities
