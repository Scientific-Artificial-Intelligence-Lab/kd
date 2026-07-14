
from __future__ import annotations

import math

import pytest
import torch

from kd.core.equation import canonicalize_expression
from kd.core.executor import ExecutionContext
from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import FunctionRegistry
from kd.core.expr.sympy_bridge import from_sympy, to_sympy
from kd.core.safety import safe_div
from kd.data.derivatives import FiniteDiffProvider
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType



_RECIP_ATOMS = (
    "diff2_x(recip(u))",
    "diff_x(mul(recip(n2(u)), u_x))",
    "n2(add(u_xx, mul(recip(x), u_x)))",
)


@pytest.fixture
def grid_context() -> ExecutionContext:
    n_x, n_t = 6, 5
    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt) + 2.0
    dataset = PDEDataset(
        name="recip-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )
    return ExecutionContext(
        dataset=dataset,
        derivative_provider=FiniteDiffProvider(dataset, max_order=3),
    )


class TestRegistration:
    @pytest.mark.unit
    def test_recip_is_registered_unary(self) -> None:
        reg = FunctionRegistry.create_default()
        assert reg.has("recip")
        assert reg.get_arity("recip") == 1


class TestExecutionBitIdentity:
    @pytest.mark.numerical
    def test_recip_fast_path_matches_div_one(
        self, grid_context: ExecutionContext
    ) -> None:
        executor = PythonExecutor(FunctionRegistry.create_default())
        recip_col = executor.execute("recip(u)", grid_context).value
        div_col = executor.execute("div(1.0, u)", grid_context).value

        torch.testing.assert_close(recip_col, div_col, rtol=0, atol=0)
        assert torch.isfinite(recip_col).all()

    @pytest.mark.numerical
    def test_recip_diff_path_matches_div_one(
        self, grid_context: ExecutionContext
    ) -> None:



        executor = PythonExecutor(FunctionRegistry.create_default())
        recip_col = executor.execute(
            "diff2_x(recip(u))", grid_context, force_diff_path=True
        ).value
        div_col = executor.execute(
            "diff2_x(div(1.0, u))", grid_context, force_diff_path=True
        ).value
        torch.testing.assert_close(recip_col, div_col, rtol=0, atol=0)

    @pytest.mark.numerical
    def test_recip_matches_safe_div_at_zero(self) -> None:


        reg = FunctionRegistry.create_default()
        col = torch.tensor([1.0, 0.0, -2.0, 0.5], dtype=torch.float32)
        recip = reg.get_func("recip")(col)
        expected = safe_div(torch.ones_like(col), col)
        torch.testing.assert_close(recip, expected, rtol=0, atol=0)
        assert torch.isfinite(recip).all()


class TestCanonicalization:
    @pytest.mark.unit
    def test_recip_canonicalizes_and_is_stable(self) -> None:
        canonical = canonicalize_expression("recip(u)")
        assert canonical == "recip(u)"
        assert canonicalize_expression(canonical) == canonical

    @pytest.mark.unit
    @pytest.mark.parametrize("atom", _RECIP_ATOMS)
    def test_former_const_atoms_now_canonicalize(self, atom: str) -> None:



        canonical = canonicalize_expression(atom)
        assert canonicalize_expression(canonical) == canonical


class TestSympyRoundTrip:
    @pytest.mark.unit
    def test_to_sympy_parses_recip(self) -> None:


        import sympy

        x = sympy.Symbol("x")
        assert sympy.simplify(to_sympy("recip(x)") - 1 / x) == 0

    @pytest.mark.unit
    def test_from_sympy_emits_recip_for_reciprocal(self) -> None:
        import sympy

        u = sympy.Symbol("u")
        assert from_sympy(u**-1) == "recip(u)"

    @pytest.mark.unit
    def test_from_sympy_negative_power_reuses_named_power(self) -> None:


        import sympy

        u = sympy.Symbol("u")
        assert from_sympy(u**-2) == "recip(n2(u))"
        assert from_sympy(u**-3) == "recip(n3(u))"
