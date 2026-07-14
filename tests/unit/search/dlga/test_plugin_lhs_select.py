
from __future__ import annotations

import dataclasses
import logging
import math
import typing

import pytest
import torch
import torch.nn as nn

from kd.core.evaluator import EvaluationResult, Evaluator
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.dlga.plugin import _INVALID_FITNESS

logger = logging.getLogger(__name__)







class _StubEvaluator:

    def __init__(
        self,
        *,
        mse: float,
        nmse: float,
        complexity: int = 1,
        coefficients: torch.Tensor | None = None,
        r2: float = 0.5,
        is_valid: bool = True,
    ) -> None:
        self._mse = mse
        self._nmse = nmse
        self._complexity = complexity
        self._r2 = r2
        self._is_valid = is_valid
        self._coeff = (
            coefficients
            if coefficients is not None
            else torch.tensor([1.0], dtype=torch.float64)
        )

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        if not self._is_valid:




            return EvaluationResult(
                mse=float("inf"),
                nmse=float("inf"),
                r2=0.0,
                score=float("nan"),
                complexity=self._complexity,
                coefficients=None,
                is_valid=False,
                error_message="stub invalid",
                selected_indices=[],
                residuals=None,
                terms=[expr],
                expression=expr,
                lhs_name=None,
            )
        return EvaluationResult(
            mse=self._mse,
            nmse=self._nmse,
            r2=self._r2,




            score=float("nan"),
            complexity=self._complexity,
            coefficients=self._coeff,
            is_valid=True,
            error_message="",
            selected_indices=list(range(self._complexity)),
            residuals=torch.zeros(2, dtype=torch.float64),
            terms=[expr],
            expression=expr,
            lhs_name=None,
        )


def _make_tiny_dataset() -> PDEDataset:
    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = 1.0 + xg * xg + tg * tg
    return PDEDataset(
        name="dlga-lhs-select-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_prepared_plugin(
    config: DLGAConfig | None = None,
    *,
    surrogate_model: nn.Module | None = None,
) -> DLGAPlugin:
    from kd.core.platform.builder import PlatformBuilder
    from kd.models.field_model import FieldModel

    dataset = _make_tiny_dataset()
    if surrogate_model is None:

        surrogate_model = FieldModel(
            coord_names=["x", "t"],
            field_names=["u"],
            hidden_sizes=[8],
            activation="tanh",
        ).to(dtype=torch.float64)
    plugin = DLGAPlugin(
        config or DLGAConfig(pop_size=2, seed=0, epsilon=0.0),
        surrogate_model=surrogate_model,
    )
    components = PlatformBuilder(dataset, plugin.derivative_requirements).build()
    plugin.prepare(components)
    return plugin


def _install_stub_evaluators(
    plugin: DLGAPlugin,
    *,
    u_t: _StubEvaluator,
    u_tt: _StubEvaluator,
) -> None:
    plugin._evaluators = {
        "u_t": u_t,
        "u_tt": u_tt,
    }







class TestLHSSelectByNMSE:

    @pytest.mark.unit
    def test_lhs_select_uses_nmse_not_raw_mse(self) -> None:
        plugin = _make_prepared_plugin()
        _install_stub_evaluators(
            plugin,
            u_t=_StubEvaluator(mse=0.5, nmse=0.5, complexity=1),
            u_tt=_StubEvaluator(mse=1.0e-7, nmse=10.0, complexity=1),
        )

        result = plugin.evaluate(["u"])[0]

        assert result.is_valid is True
        assert result.lhs_name == "u_t", (
            f"Expected u_t (NMSE 0.5 < 10.0) but got {result.lhs_name!r}. "
            "Plugin still compares raw MSE (1e-7 < 0.5 misleadingly "
            "favours u_tt despite worse normalised fit)."
        )

    @pytest.mark.unit
    def test_zero_lhs_does_not_trivially_win(self) -> None:
        plugin = _make_prepared_plugin()
        _install_stub_evaluators(
            plugin,
            u_t=_StubEvaluator(mse=0.05, nmse=0.05, complexity=2),


            u_tt=_StubEvaluator(mse=1.0e-9, nmse=0.95, complexity=1),
        )

        result = plugin.evaluate(["u"])[0]

        assert result.is_valid is True
        assert result.lhs_name == "u_t", (
            "u_t has structural fit (nmse=0.05) while u_tt has trivial "
            "constant fit (nmse=0.95 even though raw mse=1e-9 looks "
            f"good). Got lhs_name={result.lhs_name!r}; raw-MSE comparator "
            "lets u_tt steal the LHS."
        )

    @pytest.mark.unit
    def test_fitness_uses_nmse_and_length_penalty(self) -> None:
        plugin = _make_prepared_plugin(
            DLGAConfig(
                pop_size=2,
                seed=0,
                epsilon=0.1,
                lhs_auto_select=True,
            ),
        )

        plugin._population = [[[0, 1, 2]]]
        _install_stub_evaluators(
            plugin,
            u_t=_StubEvaluator(
                mse=0.001,
                nmse=0.3,
                complexity=3,
                r2=0.7,
            ),
            u_tt=_StubEvaluator(
                mse=10.0,
                nmse=10.0,
                complexity=3,
                r2=0.0,
            ),
        )

        result = plugin.evaluate(plugin.propose(1))[0]

        assert result.is_valid is True

        assert result.lhs_name == "u_t", (
            f"Expected u_t (nmse 0.3 < 10.0), got {result.lhs_name!r}"
        )



        assert result.score == pytest.approx(0.6, abs=1e-9), (
            f"Expected fitness=0.6 (nmse 0.3 + 0.1*3 length penalty), "
            f"got result.score={result.score!r}. Substitution attacks ruled "
            "out: mse-sub gives 0.301, r2-sub gives 1.0, nmse-only "
            "gives 0.3 — none equal 0.6. Fix: change "
            "``fitness = result.mse + ...`` to ``fitness = result.nmse + ...``."
        )







class TestLHSTieBreak:

    @pytest.mark.unit
    def test_lhs_tie_break_when_nmse_equal_prefers_u_tt(self) -> None:
        plugin = _make_prepared_plugin()
        _install_stub_evaluators(
            plugin,
            u_t=_StubEvaluator(mse=1.0e-3, nmse=0.5, complexity=1),
            u_tt=_StubEvaluator(mse=0.5, nmse=0.5, complexity=1),
        )

        result = plugin.evaluate(["u"])[0]

        assert result.is_valid is True
        assert result.lhs_name == "u_tt", (
            f"Expected u_tt on NMSE tie (the predecessor parity), got "
            f"{result.lhs_name!r}. Today's raw-MSE path picks u_t "
            "(mse=1e-3 < 0.5); after the fix the NMSE tie should "
            "deterministically land on u_tt via "
            "``key=(nmse, 0 if u_tt else 1)``."
        )

    @pytest.mark.unit
    def test_lhs_strict_inequality_in_low_nmse_regime(self) -> None:
        plugin = _make_prepared_plugin()
        _install_stub_evaluators(
            plugin,
            u_t=_StubEvaluator(mse=1.0e-3, nmse=0.5, complexity=1),
            u_tt=_StubEvaluator(mse=0.5, nmse=0.5 - 1.0e-13, complexity=1),
        )

        result = plugin.evaluate(["u"])[0]

        assert result.is_valid is True
        assert result.lhs_name == "u_tt", (
            f"Expected u_tt with NMSE strictly smaller by 1e-13, got "
            f"{result.lhs_name!r}. Today's raw-MSE path picks u_t "
            "(mse=1e-3 ≪ 0.5); after the fix NMSE comparison gives "
            "u_tt the win (its nmse is strictly smaller)."
        )







class TestWaveRegressionGuard:

    @pytest.mark.unit
    def test_wave_pde_picks_u_tt(self) -> None:
        plugin = _make_prepared_plugin()
        _install_stub_evaluators(
            plugin,




            u_t=_StubEvaluator(mse=1.0e-4, nmse=0.95, complexity=1),


            u_tt=_StubEvaluator(
                mse=0.5,
                nmse=1.0e-3,
                complexity=1,
                coefficients=torch.tensor([1.0], dtype=torch.float64),
            ),
        )

        result = plugin.evaluate(["u_xx"])[0]

        assert result.is_valid is True
        assert result.lhs_name == "u_tt", (
            f"Wave regression guard: u_tt should win (nmse=1e-3 vs "
            f"u_t nmse=0.95), got {result.lhs_name!r}. Today's raw-MSE "
            "comparator picks u_t (mse=1e-4 < 0.5) — Wave PDE recovery "
            "inverts. The fix (NMSE comparison) preserves Wave."
        )



        assert result.nmse < 1.0e-2, (
            f"Wave NMSE should be tiny (good fit), got nmse={result.nmse}"
        )







class TestLHSAutoSelectDisabled:

    @pytest.mark.unit
    def test_lhs_auto_select_off_only_uses_u_t(self) -> None:
        plugin = _make_prepared_plugin(
            DLGAConfig(
                pop_size=2,
                seed=0,
                epsilon=0.1,
                lhs_auto_select=False,
            ),
        )


        assert set(plugin._evaluators) == {"u_t"}, (
            f"With lhs_auto_select=False, plugin._evaluators must only "
            f"contain 'u_t', got {set(plugin._evaluators)!r}"
        )


        plugin._population = [[[0, 1, 2]]]
        plugin._evaluators = {
            "u_t": _StubEvaluator(
                mse=0.001,
                nmse=0.5,
                complexity=3,
                r2=0.7,
            ),
        }

        result = plugin.evaluate(plugin.propose(1))[0]

        assert result.is_valid is True
        assert result.lhs_name == "u_t"



        assert result.score == pytest.approx(0.8, abs=1e-9), (
            f"Single-evaluator fitness must be nmse + epsilon * length "
            f"= 0.5 + 0.1*3 = 0.8, got result.score={result.score!r}. "
            "The plugin still computes mse + epsilon * length = "
            "0.001 + 0.3 = 0.301 (mse-substitution attack) instead of "
            "using nmse."
        )







class TestSchemaInvariants:

    @pytest.mark.smoke
    def test_evaluation_result_nmse_field_present(self) -> None:
        field_names = {f.name for f in dataclasses.fields(EvaluationResult)}
        assert "nmse" in field_names, (
            f"EvaluationResult schema missing 'nmse' field; got "
            f"{sorted(field_names)}. The fix relies on per-branch NMSE "
            "being available on the result object."
        )




        hints = typing.get_type_hints(EvaluationResult)
        assert hints["nmse"] is float, (
            f"EvaluationResult.nmse must be typed ``float``, got {hints['nmse']!r}"
        )

    @pytest.mark.smoke
    def test_evaluator_caches_lhs_var(self) -> None:
        from kd.core.executor.context import ExecutionContext
        from kd.core.expr.executor import PythonExecutor
        from kd.core.expr.registry import FunctionRegistry
        from kd.core.linear_solve import LeastSquaresSolver

        registry = FunctionRegistry.create_default()
        executor = PythonExecutor(registry)
        dataset = _make_tiny_dataset()


        from kd.data.derivatives.finite_diff import FiniteDiffProvider

        provider = FiniteDiffProvider(dataset, max_order=2)
        context = ExecutionContext(dataset=dataset, derivative_provider=provider)
        lhs = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dtype=torch.float64,
        )
        evaluator = Evaluator(
            executor=executor,
            solver=LeastSquaresSolver(),
            context=context,
            lhs=lhs,
        )

        assert hasattr(evaluator, "_lhs_var"), (
            "Evaluator must cache _lhs_var for NMSE normalisation"
        )
        expected_var = lhs.flatten().var(correction=0).item()
        assert evaluator._lhs_var == pytest.approx(expected_var, rel=1e-12), (
            f"Expected _lhs_var={expected_var}, got {evaluator._lhs_var}"
        )







class TestBurgersIntegrationRegression:

    @pytest.mark.integration
    def test_burgers_synthetic_short_expr_does_not_drift_to_u_tt(self) -> None:
        from kd.core.platform.builder import PlatformBuilder
        from kd.data.synthetic import load_burgers
        from kd.models.field_model import FieldModel

        try:
            dataset = load_burgers()
        except FileNotFoundError:
            pytest.skip(
                "Burgers reference data not available "
                "(data/Burgers_equation.mat missing)"
            )




        u_full = dataset.get_field("u")
        x_full = dataset.axes["x"].values
        t_full = dataset.axes["t"].values
        x_stride = max(1, u_full.shape[0] // 32)
        t_stride = max(1, u_full.shape[1] // 32)
        x = x_full[::x_stride].clone()
        t = t_full[::t_stride].clone()
        u = u_full[::x_stride, ::t_stride].clone().to(dtype=torch.float64)
        small_dataset = PDEDataset(
            name="burgers-tiny",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
            axis_order=["x", "t"],
            fields={"u": FieldData("u", u)},
            lhs_field="u",
            lhs_axis="t",
        )




        torch.manual_seed(0)
        surrogate = FieldModel(
            coord_names=["x", "t"],
            field_names=["u"],
            hidden_sizes=[8, 8],
            activation="tanh",
        ).to(dtype=torch.float64)

        from kd.models.trainer import FieldModelTrainer

        coords = {
            "x": torch.meshgrid(x, t, indexing="ij")[0]
            .flatten()
            .to(dtype=torch.float64),
            "t": torch.meshgrid(x, t, indexing="ij")[1]
            .flatten()
            .to(dtype=torch.float64),
        }
        trainer = FieldModelTrainer(surrogate, lr=1e-2)
        trainer.fit(
            coords,
            {"u": u.flatten().to(dtype=torch.float64)},
            max_epochs=200,
            patience=None,
        )

        plugin = DLGAPlugin(
            DLGAConfig(
                pop_size=2,
                seed=0,
                epsilon=0.0,
                lhs_auto_select=True,
                surrogate_hidden_sizes=[8, 8],
                surrogate_activation="tanh",
                surrogate_max_epochs=1,
            ),
            surrogate_model=surrogate,
        )
        components = PlatformBuilder(
            small_dataset, plugin.derivative_requirements
        ).build()
        plugin.prepare(components)






        var_u_t = plugin._lhs_targets["u_t"].var().item()
        var_u_tt = plugin._lhs_targets["u_tt"].var().item()
        ratio = var_u_t / max(var_u_tt, 1e-30)
        logger.info(
            "Burgers tiny: var(u_t)=%.3e, var(u_tt)=%.3e, ratio=%.3e",
            var_u_t,
            var_u_tt,
            ratio,
        )
        if var_u_t < 100 * var_u_tt:
            pytest.skip(
                f"Test data not in bug regime: var(u_t)={var_u_t:.3e}, "
                f"var(u_tt)={var_u_tt:.3e}, ratio={ratio:.1f} < 100. "
                "Pre-fix selection cannot drift to u_tt unless "
                "var(u_tt) << var(u_t). Try a larger surrogate or "
                "different seed."
            )



        result = plugin._evaluate_one("u", genome=[[0]])




        assert math.isfinite(result.nmse), f"nmse must be finite, got {result.nmse}"









        assert result.is_valid is True
        assert result.lhs_name == "u_t", (
            f"Burgers regression: 1-token expression ``u`` drifted to "
            f"{result.lhs_name!r} (should be 'u_t'). "
            f"var(u_t)={var_u_t:.3e}, var(u_tt)={var_u_tt:.3e}; raw MSE "
            "is incommensurable across these branches."
        )







class TestInvalidEvaluators:

    @pytest.mark.unit
    def test_one_invalid_one_valid_picks_valid(self) -> None:
        plugin = _make_prepared_plugin()
        _install_stub_evaluators(
            plugin,
            u_t=_StubEvaluator(mse=0.001, nmse=0.3, complexity=2),
            u_tt=_StubEvaluator(
                mse=float("inf"),
                nmse=float("inf"),
                complexity=1,
                is_valid=False,
            ),
        )

        result = plugin.evaluate(["u"])[0]

        assert result.is_valid is True, (
            "Plugin must surface the valid branch even when the other branch failed."
        )
        assert result.lhs_name == "u_t"

        assert result.nmse == pytest.approx(0.3, abs=1e-12)

    @pytest.mark.unit
    def test_both_invalid_returns_first_with_invalid_fitness(self) -> None:
        plugin = _make_prepared_plugin()
        _install_stub_evaluators(
            plugin,
            u_t=_StubEvaluator(
                mse=float("inf"),
                nmse=float("inf"),
                complexity=2,
                is_valid=False,
            ),
            u_tt=_StubEvaluator(
                mse=float("inf"),
                nmse=float("inf"),
                complexity=1,
                is_valid=False,
            ),
        )

        result = plugin.evaluate(["u"])[0]

        assert result.is_valid is False, (
            "Plugin must propagate is_valid=False when no branch fits."
        )


        assert result.lhs_name == "u_t", (
            f"Both-invalid fallback must return u_t (insertion-order "
            f"first), got {result.lhs_name!r}"
        )
        assert result.score == _INVALID_FITNESS, (
            f"Both-invalid fallback must apply _INVALID_FITNESS sentinel, "
            f"got result.score={result.score!r}"
        )

    @pytest.mark.unit
    def test_invalid_branch_aic_overrides_stub_leftover(self) -> None:
        plugin = _make_prepared_plugin()




        class _StubLeftoverAic:
            def evaluate_expression(self, expr: str) -> EvaluationResult:
                return EvaluationResult(
                    mse=float("inf"),
                    nmse=float("inf"),
                    r2=0.0,
                    score=42.0,
                    complexity=1,
                    coefficients=None,
                    is_valid=False,
                    error_message="leftover aic",
                    selected_indices=[],
                    residuals=None,
                    terms=[expr],
                    expression=expr,
                    lhs_name=None,
                )

        plugin._evaluators = {
            "u_t": _StubLeftoverAic(),
            "u_tt": _StubLeftoverAic(),
        }

        result = plugin.evaluate(["u"])[0]

        assert result.is_valid is False
        assert result.score == _INVALID_FITNESS, (
            f"Invalid-branch fallback must apply _INVALID_FITNESS, not "
            f"surface the leftover score=42.0 from the stub. Got "
            f"result.score={result.score!r}."
        )

        assert result.score != 42.0
