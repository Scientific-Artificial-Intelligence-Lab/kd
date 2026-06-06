
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "discover" / "inspect_mode2_burgers_results.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_inspect_mode2_burgers_results",
        _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inspector() -> ModuleType:
    return _load_script_module()


@pytest.mark.unit
def test_classifies_known_td086_expression_as_no_derivative(
    inspector: ModuleType,
) -> None:
    structure = inspector.classify_burgers_expression(
        "sub(u,mul(div(u,t),div(sin(x),x)))",
    )

    assert structure.status == "NO_DERIVATIVE"
    assert structure.has_trig is True
    assert structure.has_diff_x is False
    assert structure.has_diff2_x is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "expression",
    [
        "add(diff2_x(u),mul(diff_x(u),u))",
        "sub(diff2_x(u),mul(u,diff_x(u)))",
    ],
)
def test_classifies_clean_burgers_structure(
    inspector: ModuleType,
    expression: str,
) -> None:
    structure = inspector.classify_burgers_expression(expression)

    assert structure.status == "BURGERS_STRUCTURE"
    assert structure.has_diff_x is True
    assert structure.has_diff2_x is True
    assert structure.has_advection is True


@pytest.mark.unit
def test_loads_nested_script_artifact_payload(
    inspector: ModuleType,
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps(
            {
                "tier": "fast",
                "seed": 42,
                "result": {
                    "best_reward": 0.5,
                    "best_expression": "diff2_x(u)",
                    "best_terms": ["diff2_x(u)"],
                },
            },
        ),
        encoding="utf-8",
    )

    summary = inspector.load_artifact_summary(path)

    assert summary.tier == "fast"
    assert summary.seed == 42
    assert summary.reward == 0.5
    assert summary.structure.status == "DIFFUSION_NO_ADVECTION"


@pytest.mark.unit
def test_format_summary_includes_status_and_expression(
    inspector: ModuleType,
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps(
            {
                "tier": "aligned",
                "seed": 123,
                "result": {
                    "best_reward": 0.75,
                    "best_expression": "mul(u,diff_x(u))",
                },
            },
        ),
        encoding="utf-8",
    )

    line = inspector.format_summary(inspector.load_artifact_summary(path))

    assert "tier=aligned" in line
    assert "seed=123" in line
    assert "reward=0.750000" in line
    assert "status=DERIVATIVE_NO_DIFFUSION" in line
    assert "expr=mul(u,diff_x(u))" in line


@pytest.mark.unit
def test_uses_preset_when_tier_is_missing(
    inspector: ModuleType,
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps(
            {
                "preset": "pilot",
                "seed": 42,
                "result": {"best_expression": "diff_x(u)"},
            },
        ),
        encoding="utf-8",
    )

    summary = inspector.load_artifact_summary(path)

    assert summary.tier == "pilot"
