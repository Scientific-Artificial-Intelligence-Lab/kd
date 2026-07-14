
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from kd.core.expr.sympy_bridge import format_pde, to_latex

if TYPE_CHECKING:
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_JSON_INDENT = 2


@dataclass
class ReportResult:

    figures: list[Path] = field(default_factory=list)
    data_files: list[Path] = field(default_factory=list)
    report: Path | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class _FigureEntry:

    title: str
    data_uri: str


def _figure_title_from_path(path: Path) -> str:
    stem = path.stem
    return stem.replace("_", " ").replace("-", " ").title()


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
    return json.dumps(data, indent=_JSON_INDENT, default=str)


def _best_expression_latex(result: ExperimentResult) -> str:
    final_eval = result.final_eval
    if final_eval.terms is not None and final_eval.coefficients is not None:
        try:
            return format_pde(
                final_eval.terms,
                final_eval.coefficients,
                lhs=result.lhs_label,
                selected_indices=final_eval.selected_indices,
            ).latex
        except ValueError:
            logger.exception("Failed to format report equation as full PDE")
    return to_latex(result.best_expression, strict=False)


def generate_report(
    result: ExperimentResult,
    figures: list[Path],
    output_path: Path,
    *,
    plugin_figures: list[Path] | None = None,
    warnings: list[str] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)


    fig_entries: list[_FigureEntry] = []
    for fig_path in figures:
        data_uri = _encode_figure(fig_path)
        if data_uri is None:
            continue
        fig_entries.append(
            _FigureEntry(
                title=_figure_title_from_path(fig_path),
                data_uri=data_uri,
            )
        )


    plugin_entries: list[_FigureEntry] = []
    if plugin_figures:
        for fig_path in plugin_figures:
            data_uri = _encode_figure(fig_path)
            if data_uri is None:
                continue
            plugin_entries.append(
                _FigureEntry(
                    title=_figure_title_from_path(fig_path),
                    data_uri=data_uri,
                )
            )


    json_summary = Markup(_build_json_summary(result))


    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")

    html = template.render(
        dataset_name=result.dataset_name,
        algorithm_name=result.algorithm_name,
        best_expression=result.best_expression,
        best_expression_latex=_best_expression_latex(result),
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
