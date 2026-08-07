
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _REPO_ROOT / "examples" / "notebooks" / "getting_started.ipynb"




_PRIVATE_KEY = "_best_score"


_FACADE_IMPORT_RE = re.compile(
    r"from\s+kd\.viz\.plots\s+import\s+[^\n]*\bplot_convergence\b"
)

_PLOT_CALL_RE = re.compile(r"\bplot_convergence\(")


def _load_notebook() -> dict[str, object]:
    return json.loads(_NOTEBOOK.read_text(encoding="utf-8"))


def _cell_sources(nb: dict[str, object]) -> list[str]:
    cells = nb["cells"]
    assert isinstance(cells, list)
    return ["".join(cell.get("source", [])) for cell in cells]


def _convergence_cell(nb: dict[str, object]) -> dict[str, object]:
    cells = nb["cells"]
    assert isinstance(cells, list)
    seen_heading = False
    for cell in cells:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and "## Visualize" in source:
            seen_heading = True
            continue
        if seen_heading and cell.get("cell_type") == "code":
            return cell
    raise AssertionError("no code cell found after the '## Visualize' heading")


def _has_image_output(cell: dict[str, object]) -> bool:
    for out in cell.get("outputs", []):
        if out.get("output_type") in {"display_data", "execute_result"} and (
            "image/png" in out.get("data", {})
        ):
            return True
    return False


def test_convergence_cell_calls_public_plot_convergence() -> None:
    source = "".join(_convergence_cell(_load_notebook()).get("source", []))
    assert _FACADE_IMPORT_RE.search(source), (
        "convergence cell must import plot_convergence from kd.viz.plots "
        "(the sanctioned single-plot facade), got:\n" + source
    )
    assert _PLOT_CALL_RE.search(source), (
        "convergence cell must actually call plot_convergence(...), not merely "
        "import it, got:\n" + source
    )


def test_notebook_never_names_private_best_score_key() -> None:
    offenders = [
        source for source in _cell_sources(_load_notebook()) if _PRIVATE_KEY in source
    ]
    assert not offenders, (
        f"notebook must not name the private recorder key {_PRIVATE_KEY!r}; "
        "read the series through kd.viz.plots.plot_convergence instead. "
        "Offending cells:\n" + "\n---\n".join(offenders)
    )


def test_convergence_cell_drops_ad_hoc_probe() -> None:
    source = "".join(_convergence_cell(_load_notebook()).get("source", []))
    assert "_find_score_history" not in source, (
        "convergence cell must not probe ad-hoc attribute names via "
        "_find_score_history; call plot_convergence instead."
    )


def test_convergence_cell_has_executed_figure() -> None:
    cell = _convergence_cell(_load_notebook())
    assert _has_image_output(cell), (
        "convergence cell must have an executed figure (image/png) output "
        "after regeneration."
    )
