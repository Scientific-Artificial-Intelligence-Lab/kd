
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"








_DENY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^from kd\.search\.\w+\.\w+ import", re.MULTILINE),
        "plugin submodule import (kd.search.<plugin>.<module>) bypasses that "
        "plugin's own facade kd.search.<plugin> -- e.g. "
        "kd.search.eqgpt.backend -> kd.search.eqgpt",
    ),
    (
        re.compile(r"^from kd\.search\.dlga import", re.MULTILINE),
        "DLGAConfig is in kd.__all__ -- import from kd directly",
    ),
    (
        re.compile(r"^from kd\.data\.regression import", re.MULTILINE),
        "load_tlc_cc is in kd.__all__ -- import from kd directly",
    ),
    (
        re.compile(r"^from kd\.core\.evaluator import", re.MULTILINE),
        "EvaluationResult is in kd.__all__ -- import from kd directly",
    ),
    (
        re.compile(r"^from kd\.data\.schema import", re.MULTILINE),
        "AxisInfo/DataTopology/FieldData/PDEDataset/TaskType are in "
        "kd.__all__ -- import from kd directly",
    ),
    (
        re.compile(r"^from kd\.search\.result import", re.MULTILINE),
        "ExperimentResult is in kd.__all__ -- import from kd directly",
    ),
    (
        re.compile(r"^from kd\.search\.recorder import", re.MULTILINE),
        "VizRecorder is in kd.search.__all__ -- import from kd.search directly",
    ),
    (
        re.compile(r"^from kd\.viz\.plots\.animation import", re.MULTILINE),
        "plot_field_animation is in kd.viz.plots.__all__ -- import from "
        "kd.viz.plots directly",
    ),
]







_ALLOWED_DEEP_IMPORT_LINES: frozenset[str] = frozenset()


def _example_files() -> list[Path]:
    return sorted(_EXAMPLES_DIR.glob("*.py"))


def test_examples_directory_has_the_expected_scripts() -> None:
    names = {p.name for p in _example_files()}
    assert "01_quickstart.py" in names


    assert len(names) >= 15


def test_no_example_deep_imports_bypass_a_facade() -> None:
    violations: list[str] = []
    for path in _example_files():
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for pattern, reason in _DENY_PATTERNS:
            for match in pattern.finditer(source):
                line_no = source.count("\n", 0, match.start()) + 1
                line = lines[line_no - 1]
                if line.strip() in _ALLOWED_DEEP_IMPORT_LINES:
                    continue
                violations.append(f"{path.name}:{line_no}: {line.strip()} ({reason})")
    assert not violations, (
        "facade-bypassing imports found in examples/ (F03 -- rewrite to the "
        "shallowest public surface that has the name; top-level kd > "
        "kd.data / kd.search.<plugin> / kd.viz.plots):\n"
        + "\n".join(violations)
    )


def _notebook_code_sources(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources: list[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        raw = cell["source"]
        sources.append(raw if isinstance(raw, str) else "".join(raw))
    return sources


def test_no_notebook_deep_imports_bypass_a_facade() -> None:
    notebooks = sorted((_EXAMPLES_DIR / "notebooks").glob("*.ipynb"))
    assert notebooks, "examples/notebooks/ lost its committed notebooks"
    violations: list[str] = []
    for path in notebooks:
        for index, source in enumerate(_notebook_code_sources(path)):
            for pattern, reason in _DENY_PATTERNS:
                for match in pattern.finditer(source):
                    line = source[: match.end()].splitlines()[-1]
                    if line.strip() in _ALLOWED_DEEP_IMPORT_LINES:
                        continue
                    violations.append(
                        f"{path.name} code cell {index}: {line.strip()} "
                        f"({reason})"
                    )
    assert not violations, (
        "facade-bypassing imports found in examples/notebooks/ (same F03 "
        "contract as the scripts):\n" + "\n".join(violations)
    )
