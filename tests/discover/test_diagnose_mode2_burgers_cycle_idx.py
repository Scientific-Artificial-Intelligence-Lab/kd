from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "discover" / "research" / "diagnose_mode2_burgers.py"


def _collect_run_cycle_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_cycle"
        ):
            calls.append(node)
    return calls


def _call_passes_cycle_idx(call: ast.Call) -> bool:
    return any(kw.arg == "cycle_idx" for kw in call.keywords)


@pytest.fixture(scope="module")
def diagnose_script_source() -> str:
    assert _SCRIPT_PATH.exists(), f"expected script at {_SCRIPT_PATH}"
    return _SCRIPT_PATH.read_text(encoding="utf-8")












@pytest.mark.unit
def test_diagnose_script_has_run_cycle_calls(
    diagnose_script_source: str,
) -> None:
    calls = _collect_run_cycle_calls(diagnose_script_source)
    assert len(calls) >= 2, (
        "expected at least the per-cycle loop call and the final-search "
        f"call; got {len(calls)}. Script: {_SCRIPT_PATH}"
    )


@pytest.mark.unit
def test_every_run_cycle_call_passes_cycle_idx(
    diagnose_script_source: str,
) -> None:
    calls = _collect_run_cycle_calls(diagnose_script_source)
    offenders = [
        call.lineno for call in calls if not _call_passes_cycle_idx(call)
    ]
    assert offenders == [], (
        "run_cycle call(s) missing cycle_idx= in "
        f"{_SCRIPT_PATH.name}: lines {offenders}. "
        "Each site must forward either the loop variable (per-cycle) "
        "or pinn_config.n_cycles (final search)."
    )
