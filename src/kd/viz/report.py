
from __future__ import annotations

import base64
import html
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

from kd.core.expr.sympy_bridge import to_latex
from kd.core.jsonsafe import JSON_INDENT_SPACES
from kd.viz._result_data import _sketch_fit_note
from kd.viz.equation_display import (
    STRUCTURE_ONLY_NOTE,
    EquationDisplay,
    expression_display,
    latex_display,
)
from kd.viz.plots._comparison_cells import _fitted_formatted
from kd.viz.plots.equation_card import _truth_comparison

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


_FIGURE_DESCRIPTIONS = {
    "convergence": ("Read the best native score reached at each search iteration."),
    "term_presence": (
        "Track active fitted terms among the recorded top-k distinct structures. "
        "Targets remain separate; gray columns have no eligible candidates."
    ),
    "search_score_distribution": (
        "Read the median, interquartile range and min-max native scores of the "
        "best representatives of the selected distinct structures. These bands "
        "describe the top-k selection, not the entire population."
    ),
    "parity": (
        "Compare predicted and observed targets against the diagonal of exact "
        "agreement."
    ),
    "equation": (
        "Read the published equation with its fitted coefficients and active terms."
    ),
    "equation_tree": (
        "Read the operators and derivatives that compose the published equation."
    ),
    "coefficient_bar": (
        "Compare the magnitudes and signs of the active fitted coefficients."
    ),
    "residual": (
        "Read the distribution and spatial pattern of predicted minus "
        "observed target values."
    ),
    "field_comparison": (
        "Compare the measured field with forward integration of the "
        "discovered equation and their difference."
    ),
    "pde_residual_field": (
        "Compare the fitted derivative target with the equation prediction "
        "and their residual, without forward integration."
    ),
    "time_slices": (
        "Compare measured and integrated fields at the labeled observation times."
    ),
    "error_heatmap": (
        "Locate the signed field error from forward integration at the "
        "labeled physical coordinates."
    ),
    "pareto_front_table": (
        "Compare candidate complexity, loss and fitted outer scale alongside "
        "each expression."
    ),
}


@dataclass
class ReportResult:

    figures: list[Path] = field(default_factory=list)
    data_files: list[Path] = field(default_factory=list)
    report: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def _repr_html_(self) -> str:
        if self.report is None:
            items = "".join(
                f"<li><code>{html.escape(str(p))}</code></li>" for p in self.figures
            )
            return f"<p>Figures rendered, no report page:</p><ul>{items}</ul>"
        page = html.escape(self.report.read_text(encoding="utf-8"), quote=True)
        frame_style = "width:100%;height:820px;border:0"
        return (
            f'<iframe srcdoc="{page}" style="{frame_style}"></iframe>'
            f"<p>Report: <code>{html.escape(str(self.report))}</code></p>"
        )


@dataclass(frozen=True)
class FigureSpec:

    path: Path
    title: str | None = None
    description: str | None = None


def _universal_figure_spec(path: Path) -> FigureSpec:
    return FigureSpec(path, description=_FIGURE_DESCRIPTIONS.get(path.stem))


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


def _display_algorithm(result: ExperimentResult) -> str:
    return str(result.config.get("algorithm") or result.algorithm_name)


def _best_expression_display(result: ExperimentResult) -> EquationDisplay:
    return latex_display(result, label=_display_algorithm(result))



_PROVIDER_WORDS = {
    "finite_diff": "finite differences",
    "autograd": "autograd derivatives",
}


@dataclass(frozen=True)
class _MathText:

    text: str
    tex: str | None


@dataclass(frozen=True)
class _TermRow:

    term: _MathText
    published: str
    truth: str


def _run_sentence(result: ExperimentResult) -> str:
    parts = [f"{_display_algorithm(result)} on {result.dataset_name}"]
    seed = result.config.get("seed")
    if seed is not None:
        parts.append(f"seed {seed}")
    plural = "" if result.iterations == 1 else "s"
    length = f"{result.iterations} iteration{plural}"
    parts.append(f"{length}, stopped early" if result.early_stopped else length)
    provider = _PROVIDER_WORDS.get(str(result.config.get("provider_kind")))
    if provider is not None:
        parts.append(provider)
    return ", ".join(parts) + "."


def _law_text(result: ExperimentResult, display: EquationDisplay) -> _MathText:
    if not display.is_math:
        return _MathText(display.text, None)
    if display.note == STRUCTURE_ONLY_NOTE:
        return _MathText(expression_display(result.best_expression).text, display.text)
    formatted = _fitted_formatted(result)
    assert formatted is not None
    return _MathText(f"{formatted.lhs} = {formatted.rhs}", display.text)


def _truth_text(truth: str) -> _MathText:
    display = expression_display(truth)
    if display.degraded:
        return _MathText(truth, None)
    tex = " = ".join(to_latex(part.strip()) for part in truth.split("=", 1))
    return _MathText(display.text, tex)


def _term_rows(
    result: ExperimentResult, truth: str | None
) -> tuple[list[_TermRow], list[str]]:
    rows, normalization, notes = _truth_comparison(result, truth)
    if normalization is not None:
        notes = [*notes, normalization]
    return [
        _TermRow(
            term=_MathText(expression_display(key).text, to_latex(key)),
            published=f"{published:.6g}",
            truth=f"{expected:.6g}",
        )
        for key, published, expected in rows
    ], notes


def _display_config(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: _display_config(item, key=name) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_display_config(item, key=key) for item in value]
    if isinstance(value, str) and key in {"library", "terms", "term_ir", "expression"}:
        display = expression_display(value)
        return (
            display.text if display.note is None else f"{display.text} ({display.note})"
        )
    return value


def generate_report(
    result: ExperimentResult,
    figures: Sequence[Path | FigureSpec],
    output_path: Path,
    *,
    plugin_figures: Sequence[Path | FigureSpec] | None = None,
    warnings: list[str] | None = None,
    ground_truth: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)






    json_summary = _build_json_summary(result)
    expression = _best_expression_display(result)
    term_rows, term_notes = _term_rows(result, ground_truth)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")
    config = _display_config(result.config)

    html = template.render(
        dataset_name=result.dataset_name,
        algorithm_name=_display_algorithm(result),
        run_sentence=_run_sentence(result),



        law=_law_text(result, expression),
        law_note=expression.note,
        truth=None if ground_truth is None else _truth_text(ground_truth),
        term_rows=term_rows,
        term_notes=term_notes,
        best_score=f"{result.best_score:.6g}",


        score_label=result.score_kind,
        lhs_label=result.lhs_label,
        r2=f"{result.final_eval.r2:.6f}",
        nmse=f"{result.final_eval.nmse:.4g}",
        fit_note=_sketch_fit_note(result),
        iterations=result.iterations,
        early_stopped=result.early_stopped,
        config=config,
        figures=_build_figure_entries(figures),
        plugin_figures=_build_figure_entries(plugin_figures or []),
        warnings=warnings or [],
        json_summary=json_summary,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.debug("Generated HTML report at %s", output_path)
    return output_path
