
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kd.search.eqgpt import steady_viz as eqgpt_viz

pytestmark = pytest.mark.unit


@pytest.fixture
def ax():
    fig, axis = plt.subplots()
    try:
        yield axis
    finally:
        plt.close(fig)


@pytest.fixture
def ax3d():
    fig = plt.figure()
    axis = fig.add_subplot(111, projection="3d")
    try:
        yield axis
    finally:
        plt.close(fig)


def _texts(ax) -> list[str]:
    return [text.get_text() for text in ax.texts]


def _assert_reproduction_title(ax) -> None:
    title = ax.get_title().lower()
    assert "reproduces eqgpt" in title
    assert "kd discovers" not in title
    assert "non-memorized" not in title
    assert "genuine discovery" not in title


def test_residual_domain_draws_one_honest_scatter_on_caller_axes(ax3d) -> None:
    x = np.array([-1.0, -0.25, 0.5, 1.25])
    y = np.array([0.2, 0.4, 0.8, 1.6])
    residual = np.array([-3.0, 0.5, -0.25, 2.0])

    returned = eqgpt_viz.render_steady_residual_domain(ax3d, x, y, residual)

    assert returned is None
    assert len(ax3d.collections) == 1
    scatter = ax3d.collections[0]
    scatter_x, scatter_y, scatter_z = scatter._offsets3d
    np.testing.assert_allclose(scatter_x, x)
    np.testing.assert_allclose(scatter_y, y)
    np.testing.assert_allclose(scatter_z, np.abs(residual))
    np.testing.assert_allclose(scatter.get_array(), np.abs(residual))
    assert not ax3d.images
    _assert_reproduction_title(ax3d)


@pytest.mark.parametrize(
    ("x", "y", "residual", "reason"),
    [
        (None, None, None, "no data"),
        (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([np.nan, np.inf]),
            "non-finite",
        ),
    ],
)
def test_residual_domain_degrades_explicitly_without_fabricating_points(
    ax3d, x, y, residual, reason
) -> None:
    eqgpt_viz.render_steady_residual_domain(ax3d, x, y, residual)

    assert not ax3d.collections
    assert not ax3d.images
    assert any(reason in text.lower() for text in _texts(ax3d))
    _assert_reproduction_title(ax3d)


def test_term_balance_plots_column_rms_and_highlights_pivot(ax) -> None:
    matrix = np.array(
        [
            [3.0, 0.0, 1.0],
            [4.0, 2.0, 1.0],
            [0.0, -2.0, 1.0],
        ],
        dtype=np.float64,
    )
    terms = ["u_xx", "u_yy", "one"]

    returned = eqgpt_viz.render_steady_term_balance(
        ax, matrix, terms, pivot_index=1
    )

    assert returned is None
    assert len(ax.patches) == len(terms)
    heights = np.array([patch.get_height() for patch in ax.patches])
    np.testing.assert_allclose(heights, np.sqrt(np.mean(matrix**2, axis=0)))
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert labels == terms
    pivot_face = np.asarray(ax.patches[1].get_facecolor())
    other_faces = [np.asarray(ax.patches[i].get_facecolor()) for i in (0, 2)]
    assert any(not np.allclose(pivot_face, face) for face in other_faces)
    assert "pivot" in (ax.patches[1].get_label() + " " + " ".join(_texts(ax))).lower()
    _assert_reproduction_title(ax)


@pytest.mark.parametrize(
    ("matrix", "terms", "reason"),
    [
        (None, None, "no data"),
        (np.full((3, 2), np.nan), ["u_xx", "u_yy"], "non-finite"),
    ],
)
def test_term_balance_degrades_explicitly_without_fake_bars(
    ax, matrix, terms, reason
) -> None:
    eqgpt_viz.render_steady_term_balance(ax, matrix, terms, pivot_index=0)

    assert not ax.patches
    assert any(reason in text.lower() for text in _texts(ax))
    _assert_reproduction_title(ax)


def test_surrogate_fit_draws_observed_predicted_scatter_and_r2(ax) -> None:
    observed = np.array([-2.0, -0.5, 1.0, 3.0], dtype=np.float64)
    predicted = np.array([-1.8, -0.6, 1.1, 2.9], dtype=np.float64)
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    expected_r2 = 1.0 - ss_res / ss_tot

    returned = eqgpt_viz.render_steady_surrogate_fit(ax, observed, predicted)

    assert returned is None
    assert len(ax.collections) == 1
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(), np.column_stack((observed, predicted))
    )
    assert len(ax.lines) == 1
    annotations = " ".join(_texts(ax)).lower()
    assert "r²" in annotations or "r^2" in annotations
    assert f"{expected_r2:.4f}" in annotations
    _assert_reproduction_title(ax)


@pytest.mark.parametrize(
    ("observed", "predicted", "reason"),
    [
        (None, None, "no data"),
        (
            np.array([np.nan, np.inf]),
            np.array([np.inf, np.nan]),
            "non-finite",
        ),
    ],
)
def test_surrogate_fit_degrades_explicitly_without_fake_fit(
    ax, observed, predicted, reason
) -> None:
    eqgpt_viz.render_steady_surrogate_fit(ax, observed, predicted)

    assert not ax.collections
    assert not ax.lines
    assert any(reason in text.lower() for text in _texts(ax))
    _assert_reproduction_title(ax)
