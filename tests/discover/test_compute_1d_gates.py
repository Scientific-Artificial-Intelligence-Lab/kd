
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest





import kd.search.discover.tokens.prior

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "discover" / "analysis" / "compute_1d_gates.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_compute_1d_gates", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gates() -> ModuleType:
    return _load_script_module()







@pytest.mark.unit
class TestPDETableAC2D:
    def test_allen_cahn_2d_in_table(self, gates: ModuleType) -> None:
        assert "allen_cahn_2d" in gates._PDE_TABLE

    def test_allen_cahn_2d_gt_terms(self, gates: ModuleType) -> None:
        gt, _threshold = gates._PDE_TABLE["allen_cahn_2d"]
        assert set(gt.keys()) == {"diff2_x(u)", "diff2_y(u)", "u", "n3(u)"}
        assert gt["diff2_x(u)"] == pytest.approx(0.001)
        assert gt["diff2_y(u)"] == pytest.approx(0.001)
        assert gt["u"] == pytest.approx(1.0)
        assert gt["n3(u)"] == pytest.approx(-1.0)

    def test_allen_cahn_2d_g2_threshold(self, gates: ModuleType) -> None:
        _gt, threshold = gates._PDE_TABLE["allen_cahn_2d"]

        assert 0.05 < threshold <= 0.15







def _ac2d_payload(
    expression: str, terms: list[str], coefs: list[float], seed: int = 0
) -> dict[str, Any]:
    return {
        "seed": seed,
        "result": {
            "best_expression": expression,
            "best_terms": terms,
            "best_coefficients": coefs,
        },
    }


@pytest.mark.unit
class TestEvaluateAC2DCoefficientGate:
    def test_correct_form_passes(self, gates: ModuleType) -> None:
        payload = _ac2d_payload(
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            ["diff2_x(u)", "diff2_y(u)", "u", "n3(u)"],
            [0.001, 0.001, 1.0, -1.0],
        )
        row = gates._evaluate(payload, "allen_cahn_2d", "test.json")
        assert row["g1"] is True
        assert row["g2"] is True
        assert row["pass"] is True

    def test_wrong_coefficients_fail_g2(self, gates: ModuleType) -> None:
        payload = _ac2d_payload(
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            ["diff2_x(u)", "diff2_y(u)", "u", "n3(u)"],
            [0.005, 0.005, 1.0, -1.0],
        )
        row = gates._evaluate(payload, "allen_cahn_2d", "test.json")
        assert row["g1"] is True
        assert row["g2"] is False
        assert row["pass"] is False







@pytest.mark.unit
class TestStructuralSignFlipFlag:

    def test_correct_form_structural_ok_true(self, gates: ModuleType) -> None:
        payload = _ac2d_payload(
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            ["diff2_x(u)", "diff2_y(u)", "u", "n3(u)"],
            [0.001, 0.001, 1.0, -1.0],
        )
        row = gates._evaluate(payload, "allen_cahn_2d", "test.json")
        assert row.get("structural_ok") is True

    def test_v10_paper_ic_form_structural_ok_false(
        self, gates: ModuleType
    ) -> None:
        payload = _ac2d_payload(
            "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",

            ["diff2_x(u)", "neg(diff2_y(u))", "neg(u)", "neg(n3(u))"],
            [0.000999, -0.000999, -0.998, 0.997],
        )
        row = gates._evaluate(payload, "allen_cahn_2d", "test.json")

        assert row.get("structural_ok") is False

        assert row["pass"] is False

    def test_structural_flag_absent_for_burgers(self, gates: ModuleType) -> None:
        payload = {
            "seed": 0,
            "result": {
                "best_expression": "sub(mul(neg(u), diff_x(u)), mul(c, diff2_x(u)))",
                "best_terms": ["mul(u, diff_x(u))", "diff2_x(u)"],
                "best_coefficients": [-1.0, 0.1],
            },
        }
        row = gates._evaluate(payload, "burgers", "test.json")

        assert row.get("structural_ok") is True







def _ac2d_task3_payload(
    expression: str, terms: list[str], coefs: list[float], seed: int = 0
) -> dict[str, Any]:
    return {
        "seed": seed,
        "data_path": "data/allen_cahn_2d_paper.npz",
        "n_iterations": 100,
        "n_points": 4000,
        "mode1_run": {
            "best_expression": expression,
            "best_terms": terms,
            "best_coefficients": coefs,
        },
        "ground_truth_fit": {
            "max_rel_coef_error": 0.003,
            "l1_ratio_error": 0.002,
        },
    }


@pytest.mark.unit
class TestEvaluateAcceptsTask3Schema:

    def test_correct_form_passes_on_task3_schema(
        self, gates: ModuleType
    ) -> None:
        payload = _ac2d_task3_payload(
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            ["diff2_x(u)", "diff2_y(u)", "u", "n3(u)"],
            [0.001, 0.001, 1.0, -1.0],
        )
        row = gates._evaluate(payload, "allen_cahn_2d", "task3.json")
        assert row["g1"] is True
        assert row["g2"] is True
        assert row.get("structural_ok") is True
        assert row["pass"] is True

    def test_v10_cheating_form_flagged_on_task3_schema(
        self, gates: ModuleType
    ) -> None:
        payload = _ac2d_task3_payload(
            "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
            ["diff2_x(u)", "neg(diff2_y(u))", "neg(u)", "neg(n3(u))"],
            [0.000999, -0.000999, -0.998, 0.997],
        )
        row = gates._evaluate(payload, "allen_cahn_2d", "task3.json")
        assert row.get("structural_ok") is False
        assert row["pass"] is False

    def test_real_task3_artifact_loadable(self, gates: ModuleType) -> None:
        import json

        artifact = (
            _PROJECT_ROOT
            / ".session"
            / "phase3-research"
            / "task3-multiseed"
            / "seed_00.json"
        )
        if not artifact.exists():
            pytest.skip(f"Task 3 artifact missing at {artifact}")
        payload = json.loads(artifact.read_text())

        row = gates._evaluate(payload, "allen_cahn_2d", "seed_00.json")

        assert row.get("structural_ok") is False







@pytest.mark.unit
def test_filename_with_allen_cahn_2d_detected(gates: ModuleType) -> None:
    name = "mode2_allen_cahn_2d_seed0.json"
    detected = next(
        (k for k in gates._PDE_TABLE if k in name.lower()), None
    )
    assert detected == "allen_cahn_2d"
