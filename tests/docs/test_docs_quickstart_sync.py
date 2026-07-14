
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_README = _REPO_ROOT / "README.md"
_PUBLIC_README = _REPO_ROOT / "scripts" / "public-assets" / "README.md"
_NOTEBOOK = _REPO_ROOT / "examples" / "notebooks" / "getting_started.ipynb"


CANONICAL_EXPR = "u_t = -1*mul(u_x, u) + 0.1002*diff2_x(u)"

STALE_EXPR_FRAGMENT = "diff2_x(add(u, x))"

_MODEL_SEED_RE = re.compile(r"kd\.Model\([^)]*seed=0")


def _first_python_block_after(text: str, heading: str) -> str:
    idx = text.find(heading)
    assert idx != -1, f"heading {heading!r} not found"
    fence = text.find("```python", idx)
    assert fence != -1, f"no python code block after {heading!r}"
    body_start = text.index("\n", fence) + 1
    body_end = text.find("```", body_start)
    assert body_end != -1, "unterminated python code block"
    return text[body_start:body_end]


def _load_notebook() -> dict[str, object]:
    return json.loads(_NOTEBOOK.read_text(encoding="utf-8"))


def _fit_cell(nb: dict[str, object]) -> dict[str, object]:
    cells = nb["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if ".fit(" in source and "best_expr_" in source:
            return cell
    raise AssertionError("no fit cell (contains '.fit(' and 'best_expr_') found")


def _stream_text(cell: dict[str, object]) -> str:
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            return "".join(out.get("text", []))
    return ""


def test_readme_quickstart_is_seeded() -> None:
    block = _first_python_block_after(
        _README.read_text(encoding="utf-8"), "## Quick start"
    )
    assert _MODEL_SEED_RE.search(block), (
        "README quick-start kd.Model(...) call must bind seed=0 so the printed "
        f"output is reproducible. Block:\n{block}"
    )


def test_public_readme_quickstart_is_seeded() -> None:
    block = _first_python_block_after(
        _PUBLIC_README.read_text(encoding="utf-8"), "quick start"
    )
    assert _MODEL_SEED_RE.search(block), (
        "public-assets README quick-start kd.Model(...) call must bind seed=0. "
        f"Block:\n{block}"
    )


def test_readme_expected_output_comment_matches_canonical() -> None:
    block = _first_python_block_after(
        _README.read_text(encoding="utf-8"), "## Quick start"
    )
    assert f"# {CANONICAL_EXPR}" in block, (
        "README quick-start expected-output comment must equal "
        f"'# {CANONICAL_EXPR}'. Block:\n{block}"
    )
    assert STALE_EXPR_FRAGMENT not in block, (
        f"the stale expression {STALE_EXPR_FRAGMENT!r} must be removed from the "
        f"README quick-start block, not left alongside the corrected line."
    )


def test_public_readme_expected_output_comment_matches_canonical() -> None:
    block = _first_python_block_after(
        _PUBLIC_README.read_text(encoding="utf-8"), "quick start"
    )
    assert f"# {CANONICAL_EXPR}" in block, (
        "public-assets README quick-start expected-output comment must equal "
        f"'# {CANONICAL_EXPR}'. Block:\n{block}"
    )
    assert STALE_EXPR_FRAGMENT not in block, (
        f"the stale expression {STALE_EXPR_FRAGMENT!r} must be removed from the "
        f"public-assets README quick-start block."
    )


def test_notebook_fit_cell_is_seeded_and_matches_canonical() -> None:
    cell = _fit_cell(_load_notebook())
    source = "".join(cell.get("source", []))
    assert _MODEL_SEED_RE.search(source), (
        f"notebook fit-cell kd.Model(...) call must bind seed=0. Source:\n{source}"
    )
    stdout = _stream_text(cell)
    assert stdout.strip(), "fit cell has no captured stdout stream output"
    first_line = stdout.splitlines()[0].strip()
    assert first_line == CANONICAL_EXPR, (
        f"notebook fit-cell stdout first line {first_line!r} != canonical "
        f"{CANONICAL_EXPR!r}; regenerate the notebook."
    )
