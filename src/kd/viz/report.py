
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

from kd.core.jsonsafe import JSON_INDENT_SPACES
from kd.viz.equation_display import EquationDisplay, latex_display

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class ReportResult:

    figures: list[Path] = field(default_factory=list)
    data_files: list[Path] = field(default_factory=list)
    report: Path | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FigureSpec:

    path: Path
    title: str | None = None
    description: str | None = None


@dataclass
class _FigureEntry:

    title: str
    data_uri: str
    description: str = ""


def _figure_title_from_path(path: Path) -> str:
    stem = path.stem
    return stem.replace("_", " ").replace("-", " ").title()


def _build_figure_entries(
    figures: Sequence[Path | FigureSpec],
) -> list[_FigureEntry]:
    entries: list[_FigureEntry] = []
    for figure in figures:
        spec = figure if isinstance(figure, FigureSpec) else FigureSpec(path=figure)
        data_uri = _encode_figure(spec.path)
        if data_uri is None:
            continue
        entries.append(
            _FigureEntry(
                title=spec.title or _figure_title_from_path(spec.path),
                data_uri=data_uri,
                description=spec.description or "",
            )
        )
    return entries


def _encode_figure(path: Path) -> str | None:
    if not path.exists():
        logger.warning("Figure file not found: %s", path)
        return None

    suffix = path.suffix.lower()
    raw = path.read_bytes()

    if suffix == ".svg":
        mime = "image/svg+xml"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "application/octet-stream"

    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_json_summary(result: ExperimentResult) -> str:
    data = result.to_dict()

    for key in ("actual", "predicted"):
        if key in data:
            arr = data[key]
            if isinstance(arr, list) and len(arr) > 20:
                data[key] = f"[{len(arr)} elements]"

    final_eval = data.get("final_eval")
    if isinstance(final_eval, dict):
        for key in ("residuals", "coefficients"):
            arr = final_eval.get(key)
            if isinstance(arr, list) and len(arr) > 20:
                final_eval[key] = f"[{len(arr)} elements]"
    return json.dumps(data, indent=JSON_INDENT_SPACES, default=str)


def _best_expression_display(result: ExperimentResult) -> EquationDisplay:
    return latex_display(result, label=result.algorithm_name)


def generate_report(
    result: ExperimentResult,
    figures: Sequence[Path | FigureSpec],
    output_path: Path,
    *,
    plugin_figures: Sequence[Path | FigureSpec] | None = None,
    warnings: list[str] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_entries = _build_figure_entries(figures)
    plugin_entries = _build_figure_entries(plugin_figures or [])






    json_summary = _build_json_summary(result)
    expression = _best_expression_display(result)


    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")

    html = template.render(
        dataset_name=result.dataset_name,
        algorithm_name=result.algorithm_name,



        best_expression_latex=expression.text,
        best_expression_is_math=expression.is_math,
        best_expression_note=expression.note,
        best_score=f"{result.best_score:.6g}",


        score_label=result.score_kind,
        r2=f"{result.final_eval.r2:.6f}",
        nmse=f"{result.final_eval.nmse:.4g}",
        iterations=result.iterations,
        early_stopped=result.early_stopped,
        config=result.config,
        figures=fig_entries,
        plugin_figures=plugin_entries,
        warnings=warnings or [],
        json_summary=json_summary,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.debug("Generated HTML report at %s", output_path)
    return output_path
