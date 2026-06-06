
from __future__ import annotations

import sympy

from kd.core.expr.simplify import expand_linear_diffs
from kd.core.expr.sympy_bridge import to_sympy


class TestExpandLinearDiffs:

    def test_expands_linear_nested_diff(self) -> None:
        result = expand_linear_diffs("diff_x(add(u_x, diff_x(u)))")

        assert to_sympy(result) == 2 * sympy.Symbol("u_x_x")

    def test_preserves_nonlinear_open_form_diff(self) -> None:
        result = expand_linear_diffs("diff_x(mul(u, u_x))")

        assert result == "diff_x(mul(u, u_x))"

    def test_preserves_function_open_form_diff(self) -> None:
        result = expand_linear_diffs("diff_x(sin(u))")

        assert result == "diff_x(sin(u))"
        expected = sympy.Function("diff_x")(sympy.sin(sympy.Symbol("u")))
        assert to_sympy(result) == expected

    def test_non_strict_returns_raw_on_failure(self) -> None:
        raw = "add(u,"

        assert expand_linear_diffs(raw, strict=False) == raw
