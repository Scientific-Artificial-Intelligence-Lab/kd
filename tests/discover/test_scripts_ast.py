from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = [
    _PROJECT_ROOT / "scripts" / "discover" / "run_burgers_comparison.py",
    _PROJECT_ROOT / "scripts" / "discover" / "run_chafee_comparison.py",
]


def _find_validator_calls(tree: ast.AST) -> list[ast.Call]:
    hits: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target_name: str | None = None
        if isinstance(func, ast.Name):
            target_name = func.id
        elif isinstance(func, ast.Attribute):
            target_name = func.attr
        if target_name == "CandidateValidator":
            hits.append(node)
    return hits


@pytest.mark.unit
@pytest.mark.parametrize("script_path", _SCRIPTS, ids=lambda p: p.name)
def test_script_validator_passes_min_length(script_path: Path) -> None:
    assert script_path.exists(), f"script missing: {script_path}"
    tree = ast.parse(script_path.read_text())
    calls = _find_validator_calls(tree)
    assert calls, f"No CandidateValidator(...) call found in {script_path.name}"
    for call in calls:
        keyword_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "min_length" in keyword_names, (
            f"{script_path.name} line {call.lineno}: "
            f"CandidateValidator must pass min_length= (got kwargs: "
            f"{sorted(k for k in keyword_names if k is not None)})"
        )
        assert "max_length" in keyword_names, (
            f"{script_path.name} line {call.lineno}: "
            f"CandidateValidator must pass max_length="
        )


@pytest.mark.unit
@pytest.mark.parametrize("script_path", _SCRIPTS, ids=lambda p: p.name)
def test_script_validator_kwargs_use_config(script_path: Path) -> None:
    tree = ast.parse(script_path.read_text())
    calls = _find_validator_calls(tree)
    assert calls, f"No CandidateValidator(...) call found in {script_path.name}"
    for call in calls:
        for kw in call.keywords:
            if kw.arg not in {"min_length", "max_length"}:
                continue
            assert isinstance(kw.value, ast.Attribute), (
                f"{script_path.name} line {kw.value.lineno}: "
                f"{kw.arg} must come from config.*, not a literal"
            )
            assert isinstance(kw.value.value, ast.Name), (
                f"{script_path.name} line {kw.value.lineno}: "
                f"{kw.arg} must be an attribute of a Name (e.g. config.{kw.arg})"
            )
            assert kw.value.value.id == "config", (
                f"{script_path.name} line {kw.value.lineno}: "
                f"{kw.arg} must reference `config.{kw.arg}`, not "
                f"`{kw.value.value.id}.{kw.value.attr}`"
            )
            assert kw.value.attr == kw.arg, (
                f"{script_path.name} line {kw.value.lineno}: "
                f"{kw.arg} must be bound to config.{kw.arg}, not "
                f"config.{kw.value.attr}"
            )
