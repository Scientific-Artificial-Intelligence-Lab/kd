
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _REPO_ROOT / "examples" / "notebooks" / "getting_started.ipynb"




_RECORDER_CALL_RE = re.compile(r"""recorder\.get\(\s*["']_best_score["']""")


def _load_notebook() -> dict[str, object]:
    return json.loads(_NOTEBOOK.read_text(encoding="utf-8"))


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


def test_convergence_cell_reads_recorder_best_score() -> None:
    source = "".join(_convergence_cell(_load_notebook()).get("source", []))
    assert _RECORDER_CALL_RE.search(source), (
        'convergence cell must call recorder.get("_best_score") to read the '
        "score series, got:\n" + source
    )


def test_convergence_cell_drops_ad_hoc_probe() -> None:
    source = "".join(_convergence_cell(_load_notebook()).get("source", []))
    assert "_find_score_history" not in source, (
        "convergence cell must not probe ad-hoc attribute names via "
        "_find_score_history; read the recorder instead."
    )


def test_convergence_cell_has_executed_figure() -> None:
    cell = _convergence_cell(_load_notebook())
    assert _has_image_output(cell), (
        "convergence cell must have an executed figure (image/png) output "
        "after regeneration."
    )
