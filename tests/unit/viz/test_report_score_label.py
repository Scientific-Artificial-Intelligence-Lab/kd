
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots.convergence import plot_convergence
from kd.viz.report import generate_report






@pytest.fixture()
def make_mock_result():

    def _make(
        algorithm: str | None,
        scores: list[float] | None = None,
        algorithm_name: str = "TestPlugin",
    ) -> ExperimentResult:






        if scores is None:
            scores = [1.0, 0.5, 0.25]
        n_samples = 10
        residuals = torch.zeros(n_samples)
        eval_result = EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=0.99,
            aic=-50.0,
            complexity=2,
            coefficients=torch.tensor([1.0, 0.5]),
            is_valid=True,
            error_message="",
            selected_indices=[0, 1],
            residuals=residuals,
            terms=["u", "u_x"],
            expression="add(u, u_x)",
        )
        recorder = VizRecorder()
        for score in scores:
            recorder.log("_best_score", score)
            recorder.log("_best_expr", "expr")
            recorder.log("_n_candidates", 5)
        actual = torch.sin(torch.linspace(0, 1, n_samples))
        predicted = actual + torch.randn(n_samples) * 0.01




        config: dict[str, object] = (
            {} if algorithm is None else {"algorithm": algorithm}
        )
        return ExperimentResult(
            best_expression="add(u, u_x)",
            best_score=0.25,
            iterations=len(scores),
            early_stopped=False,
            final_eval=eval_result,
            actual=actual,
            predicted=predicted,
            dataset_name="mock_dataset",
            algorithm_name=algorithm_name,



            config=config,
            recorder=recorder,
        )

    return _make


@pytest.fixture()
def svg_file(tmp_path: Path) -> Path:
    svg = tmp_path / "fig.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        '<rect width="10" height="10" fill="red"/></svg>'
    )
    return svg







@pytest.mark.unit
def test_score_label_returns_aic_for_sga() -> None:
    from kd.viz._labels import score_label

    assert score_label("sga") == "AIC"


@pytest.mark.unit
def test_score_label_returns_dlga_fitness_for_dlga() -> None:
    from kd.viz._labels import score_label

    assert score_label("dlga") == "DLGA fitness"
    assert score_label("dlga") != "AIC"


@pytest.mark.unit
def test_score_label_returns_reward_for_discover() -> None:
    from kd.viz._labels import score_label

    assert score_label("discover") == "reward"


@pytest.mark.unit
@pytest.mark.parametrize("unknown", ["nonexistent", "", "SGA", "Discover"])
def test_score_label_returns_score_for_unknown_or_empty(unknown: str) -> None:
    from kd.viz._labels import score_label

    assert score_label(unknown) == "Score"







@pytest.mark.unit
@pytest.mark.parametrize(
    ("algorithm", "expected_label"),
    [
        ("sga", "Best AIC"),
        ("dlga", "Best DLGA fitness"),
        ("discover", "Best reward"),
    ],
)
def test_generate_report_renders_algorithm_specific_label(
    tmp_path: Path,
    svg_file: Path,
    make_mock_result,
    algorithm: str,
    expected_label: str,
) -> None:
    result = make_mock_result(algorithm)
    output = tmp_path / "report.html"
    generate_report(result, [svg_file], output)

    content = output.read_text()
    expected_row = f"<th>{expected_label}</th>"
    default_row = "<th>Best Score</th>"

    assert expected_row in content, (
        f"HTML must contain {expected_row!r} for algorithm={algorithm!r}. "
        "Snippet of lines mentioning 'Best': "
        f"{[ln.strip() for ln in content.split(chr(10)) if 'Best' in ln]!r}"
    )
    assert default_row not in content, (
        f"HTML must NOT contain {default_row!r} when algorithm={algorithm!r} "
        f"is known — the fallback row must be replaced, not coexist."
    )


@pytest.mark.unit
def test_generate_report_falls_back_for_unknown_algorithm(
    tmp_path: Path,
    svg_file: Path,
    make_mock_result,
) -> None:
    result = make_mock_result("totally_unknown_algorithm")
    output = tmp_path / "report.html"
    generate_report(result, [svg_file], output)

    content = output.read_text()
    default_row = "<th>Best Score</th>"
    assert default_row in content, (
        f"Unknown algorithm must fall back to the {default_row!r} row. "
        "Snippet of lines mentioning 'Best': "
        f"{[ln.strip() for ln in content.split(chr(10)) if 'Best' in ln]!r}"
    )







@pytest.mark.unit
@pytest.mark.parametrize(
    ("algorithm", "expected_ylabel"),
    [
        ("sga", "Best AIC"),
        ("dlga", "Best DLGA fitness"),
        ("discover", "Best reward"),
    ],
)
def test_plot_convergence_ylabel_is_algorithm_aware(
    make_mock_result,
    algorithm: str,
    expected_ylabel: str,
) -> None:
    result = make_mock_result(algorithm, scores=[0.1, 0.2, 0.3])
    fig, ax = plt.subplots()
    try:
        plot_convergence(result, ax)
        assert ax.get_ylabel() == expected_ylabel, (
            f"plot_convergence ylabel for algorithm={algorithm!r} should be "
            f"{expected_ylabel!r}, got {ax.get_ylabel()!r}."
        )
    finally:
        plt.close(fig)


@pytest.mark.unit
def test_plot_convergence_ylabel_unknown_falls_back(make_mock_result) -> None:
    result = make_mock_result("unrecognized", scores=[0.5, 0.4, 0.3])
    fig, ax = plt.subplots()
    try:
        plot_convergence(result, ax)
        assert ax.get_ylabel() == "Best Score", (
            f"Unknown algorithm ylabel should be 'Best Score', got {ax.get_ylabel()!r}."
        )
    finally:
        plt.close(fig)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("algorithm", "expected_ylabel"),
    [
        ("sga", "Best AIC"),
        ("dlga", "Best DLGA fitness"),
        ("discover", "Best reward"),
    ],
)
def test_plot_convergence_empty_recorder_uses_algorithm_label(
    make_mock_result,
    algorithm: str,
    expected_ylabel: str,
) -> None:
    result = make_mock_result(algorithm, scores=[])


    assert result.recorder.get("_best_score") == [], (
        "make_mock_result(scores=[]) must yield an empty recorder; the "
        "default series leaked through the fixture's fallback path."
    )

    fig, ax = plt.subplots()
    try:
        warnings = plot_convergence(result, ax)

        assert len(warnings) > 0
        assert ax.get_ylabel() == expected_ylabel, (
            f"Empty recorder + algorithm={algorithm!r} must produce the "
            f"{expected_ylabel!r} ylabel; got {ax.get_ylabel()!r}."
        )
    finally:
        plt.close(fig)







@pytest.mark.unit
def test_generate_report_handles_missing_algorithm_key(
    tmp_path: Path,
    svg_file: Path,
    make_mock_result,
) -> None:
    result = make_mock_result(algorithm=None)

    assert "algorithm" not in result.config, (
        "make_mock_result(algorithm=None) must omit the 'algorithm' key "
        "entirely so we exercise the dict.get() fallback path."
    )

    output = tmp_path / "report.html"
    generate_report(result, [svg_file], output)

    content = output.read_text()
    default_row = "<th>Best Score</th>"
    assert default_row in content, (
        f"Missing 'algorithm' key must fall back to {default_row!r}. "
        "Snippet of lines mentioning 'Best': "
        f"{[ln.strip() for ln in content.split(chr(10)) if 'Best' in ln]!r}"
    )



    assert "<th>Best None</th>" not in content
    assert "<th>Best </th>" not in content







@pytest.mark.unit
@pytest.mark.parametrize(
    ("algorithm", "expected_ylabel"),
    [
        ("sga", "Best AIC"),
        ("dlga", "Best DLGA fitness"),
        ("discover", "Best reward"),
    ],
)
def test_render_overlaid_convergence_ylabel_uniform_algorithm(
    make_mock_result,
    algorithm: str,
    expected_ylabel: str,
) -> None:
    from kd.viz.plots.comparison import render_overlaid_convergence

    r1 = make_mock_result(algorithm, scores=[0.1, 0.3, 0.5])
    r2 = make_mock_result(algorithm, scores=[0.2, 0.4, 0.6])
    fig, ax = plt.subplots()
    try:
        render_overlaid_convergence([r1, r2], ax)
        assert ax.get_ylabel() == expected_ylabel, (
            f"render_overlaid_convergence ylabel for uniform "
            f"algorithm={algorithm!r} should be {expected_ylabel!r}, "
            f"got {ax.get_ylabel()!r}."
        )
    finally:
        plt.close(fig)


@pytest.mark.unit
def test_render_overlaid_convergence_ylabel_mixed_algorithms_falls_back(
    make_mock_result,
) -> None:
    from kd.viz.plots.comparison import render_overlaid_convergence

    r_sga = make_mock_result("sga", scores=[0.5, 0.3, 0.1])
    r_discover = make_mock_result("discover", scores=[0.1, 0.3, 0.5])
    fig, ax = plt.subplots()
    try:
        render_overlaid_convergence([r_sga, r_discover], ax)
        assert ax.get_ylabel() == "Best Score", (
            "Mixed-algorithm overlay must fall back to 'Best Score', "
            f"got {ax.get_ylabel()!r}."
        )
    finally:
        plt.close(fig)


@pytest.mark.unit
def test_render_overlaid_convergence_ylabel_empty_result_list_falls_back() -> None:
    from kd.viz.plots.comparison import render_overlaid_convergence

    fig, ax = plt.subplots()
    try:
        render_overlaid_convergence([], ax)
        assert ax.get_ylabel() == "Best Score", (
            "Empty results overlay must fall back to 'Best Score', "
            f"got {ax.get_ylabel()!r}."
        )
    finally:
        plt.close(fig)


@pytest.mark.unit
def test_render_overlaid_convergence_ylabel_sga_dlga_not_shared(
    make_mock_result,
) -> None:
    from kd.viz.plots.comparison import render_overlaid_convergence

    r_sga = make_mock_result("sga", scores=[0.5, 0.3, 0.1])
    r_dlga = make_mock_result("dlga", scores=[0.4, 0.2, 0.05])
    fig, ax = plt.subplots()
    try:
        render_overlaid_convergence([r_sga, r_dlga], ax)
        assert ax.get_ylabel() == "Best Score", (
            "SGA (AIC) + DLGA (fitness) are different metrics; ylabel must fall "
            f"back to 'Best Score', got {ax.get_ylabel()!r}."
        )
    finally:
        plt.close(fig)







@pytest.mark.unit
def test_score_label_covers_all_supported_algorithms() -> None:
    from kd.api import _SUPPORTED_ALGORITHMS
    from kd.viz._labels import score_label

    fallback_label = "Score"
    missing_labels: list[str] = []
    for algo in _SUPPORTED_ALGORITHMS:
        label = score_label(algo)
        if label == fallback_label:
            missing_labels.append(algo)

    assert not missing_labels, (
        f"Algorithms in _SUPPORTED_ALGORITHMS without an explicit "
        f"score_label entry (falling back to 'Score'): {missing_labels}. "
        f"Add their label to viz/_labels.py:score_label so HTML reports "
        f"and stdout pick up the correct algorithm-aware metric name."
    )
