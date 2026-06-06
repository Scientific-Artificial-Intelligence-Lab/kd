
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "discover" / "research" / "run_td086_vocab_ablation.py"


def _load_script_module() -> ModuleType:
    scripts_dir = str(_SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "test_run_td086_vocab_ablation",
        _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ablation() -> ModuleType:
    return _load_script_module()


@pytest.mark.unit
def test_e2e_vocab_keeps_trig_tokens(ablation: ModuleType) -> None:
    operators = ablation.operators_for_vocab("e2e")

    assert "sin" in operators
    assert "cos" in operators
    assert "n2" not in operators
    assert "n3" not in operators


@pytest.mark.unit
def test_script_vocab_removes_trig_tokens(ablation: ModuleType) -> None:
    operators = ablation.operators_for_vocab("script")

    assert "sin" not in operators
    assert "cos" not in operators
    assert "n2" in operators
    assert "n3" in operators


@pytest.mark.unit
def test_e2e_preset_matches_legacy_test_budget(ablation: ModuleType) -> None:
    preset = ablation.PRESETS["e2e"]

    assert preset.n_iterations == 200
    assert preset.batch_size == 64
    assert preset.max_length == 20
    assert preset.number_layer == 4
    assert preset.pretrain_epoch == 1_000
    assert preset.pinn_epoch == 50
    assert preset.n_cycles == 2
    assert preset.n_collocation == 10_000


@pytest.mark.unit
def test_default_output_path_is_session_scoped(ablation: ModuleType) -> None:
    path = ablation.default_output_path("pilot", "script", 42)

    assert path.name == "mode2_burgers_pilot_script_seed42.json"
