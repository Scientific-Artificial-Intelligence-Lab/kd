
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from kd.core.integrator import IntegrationResult, integrate_pde
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.result import ExperimentResult
from kd.viz.engine import VizEngine


def _make_smooth_dataset() -> PDEDataset:
    nx, nt = 16, 8
    x = torch.linspace(0.0, 1.0, nx)
    t = torch.linspace(0.0, 0.2, nt)
    u = (torch.sin(2 * torch.pi * x)[:, None] * torch.exp(-t)[None,:]).to(
        torch.float64
    )
    return PDEDataset(
        name="ir-assembly-smooth-1d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _with_terms(
    base: ExperimentResult,
    terms: list[str],
    coefficients: list[float],
    selected_indices: list[int] | None,
) -> ExperimentResult:
    final_eval = replace(
        base.final_eval,
        terms=terms,
        coefficients=torch.tensor(coefficients, dtype=torch.float64),
        selected_indices=selected_indices,
    )
    return replace(base, final_eval=final_eval)


def _capture_assembled_rhs(
    engine: VizEngine,
    result: ExperimentResult,
    dataset: PDEDataset,
) -> tuple[object, list[str]]:
    captured: list[object] = []

    def _stub(rhs: object, ds: PDEDataset, **kwargs: object) -> IntegrationResult:
        captured.append(rhs)
        assert ds is dataset, "engine must forward the dataset unchanged"
        return IntegrationResult(
            success=True,
            predicted_field=torch.zeros(dataset.get_shape(), dtype=torch.float64),
        )



    with patch("kd.core.integrator.integrate_pde", side_effect=_stub):
        _integration, notes = engine._get_integration_result(result, dataset)

    assert len(captured) == 1, "integrate_pde must be called exactly once"
    return captured[0], notes


class TestIrRhsAssembly:

    def test_rhs_assembled_from_terms_and_coeffs_via_repr(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        dataset = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        c0, c1 = 1.0 / 3.0, -0.5
        result = _with_terms(
            mock_experiment_result,
            ["u_xx", "mul(u, u_x)"],
            [c0, c1],
            None,
        )

        rhs, notes = _capture_assembled_rhs(engine, result, dataset)

        assert isinstance(rhs, str), (
            f"integrate_pde must receive an IR string, got {type(rhs)!r}"
        )
        assert rhs == f"({c0!r})*(u_xx) + ({c1!r})*(mul(u, u_x))"
        assert notes == []

    def test_selected_indices_honored(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        dataset = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(
            mock_experiment_result,
            ["u_xx", "t", "v"],
            [0.1, 2.0, 3.0],
            [0, 2],
        )

        rhs, notes = _capture_assembled_rhs(engine, result, dataset)

        assert isinstance(rhs, str), (
            f"integrate_pde must receive an IR string, got {type(rhs)!r}"
        )
        assert rhs == "(0.1)*(u_xx) + (3.0)*(v)"
        assert notes == []

    def test_near_zero_terms_pruned_from_rhs_string(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        dataset = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(
            mock_experiment_result,
            ["u_xx", "t"],
            [0.1, -9.5e-16],
            None,
        )

        rhs, notes = _capture_assembled_rhs(engine, result, dataset)

        assert isinstance(rhs, str), (
            f"integrate_pde must receive an IR string, got {type(rhs)!r}"
        )
        assert rhs == "(0.1)*(u_xx)"
        assert len(notes) == 1
        assert "'t'" in notes[0]

    def test_protected_semantics_note_for_log_rhs(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        dataset = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(mock_experiment_result, ["log(u)"], [0.5], None)

        rhs, notes = _capture_assembled_rhs(engine, result, dataset)

        assert rhs == "(0.5)*(log(u))"
        assert len(notes) == 1
        assert "safe_log" in notes[0] or "protected" in notes[0].lower()

    def test_protected_semantics_note_for_nested_exp_rhs(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        dataset = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(
            mock_experiment_result, ["mul(u, exp(u))"], [1.0], None
        )

        _rhs, notes = _capture_assembled_rhs(engine, result, dataset)

        assert len(notes) == 1
        assert "safe_exp" in notes[0] or "protected" in notes[0].lower()

    def test_no_protected_note_without_exp_or_log(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        dataset = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(mock_experiment_result, ["expanded_u_xx"], [0.5], None)

        _rhs, notes = _capture_assembled_rhs(engine, result, dataset)

        assert notes == []

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_coefficient_fails_loud_with_accurate_message(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        bad: float,
    ) -> None:
        dataset = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(
            mock_experiment_result,
            ["u", "u_xx"],
            [bad, 0.1],
            None,
        )

        with pytest.raises(ValueError, match="non-finite coefficient") as excinfo:
            engine._get_integration_result(result, dataset)



        assert "'u'" in str(excinfo.value)
        assert "unrecognised" not in str(excinfo.value)

    def test_deselected_non_finite_coefficient_does_not_raise(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        dataset = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(
            mock_experiment_result,
            ["u", "u_xx"],
            [float("nan"), 0.1],
            [1],
        )

        rhs, notes = _capture_assembled_rhs(engine, result, dataset)

        assert rhs == "(0.1)*(u_xx)"
        assert notes == []

    def test_all_pruned_yields_zero_rhs_and_constant_field(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        dataset = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(
            mock_experiment_result,
            ["u_xx", "t"],
            [0.1, 2.0],
            [],
        )

        captured: list[object] = []
        real_integrate: Callable[..., IntegrationResult] = integrate_pde

        def _spy(rhs: object, ds: PDEDataset, **kwargs: object) -> IntegrationResult:
            captured.append(rhs)
            return real_integrate(rhs, ds, **kwargs)

        with patch("kd.core.integrator.integrate_pde", side_effect=_spy):
            integration, notes = engine._get_integration_result(result, dataset)

        assert len(captured) == 1
        assert isinstance(captured[0], str), (
            f"integrate_pde must receive an IR string, got {type(captured[0])!r}"
        )
        assert captured[0] == "0"
        assert notes == []
        assert integration.success is True, integration.warning
        assert integration.predicted_field is not None
        ic = dataset.get_field("u")[:, 0]
        for t_idx in range(integration.predicted_field.shape[-1]):
            torch.testing.assert_close(
                integration.predicted_field[:, t_idx],
                ic,
                rtol=1e-4,
                atol=1e-6,
            )
