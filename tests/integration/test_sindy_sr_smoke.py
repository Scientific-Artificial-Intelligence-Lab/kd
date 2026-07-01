
from __future__ import annotations

import numpy as np
import pytest
from kd.search.sindy.sr import SINDyRegressor


def _make_noisy_problem(
    n: int = 400,
    noise: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    clean = 1.5 * x1 - 2.0 * np.sin(x2) + 0.8 * (x1 * x3)
    y = clean + noise * rng.normal(size=n)
    ground_truth = {"x1": 1.5, "sin(x2)": -2.0, "mul(x1, x3)": 0.8}
    return x, y, ground_truth



_RICH_LIBRARY: list[str] = [
    "x1",
    "x2",
    "x3",
    "sin(x1)",
    "sin(x2)",
    "sin(x3)",
    "cos(x1)",
    "cos(x3)",
    "mul(x1, x2)",
    "mul(x1, x3)",
    "mul(x2, x3)",
    "mul(x2, x2)",
]


@pytest.mark.smoke
def test_sindy_recovers_known_sparse_combo_with_noise() -> None:
    x, y, ground_truth = _make_noisy_problem(n=400, noise=0.02, seed=0)
    reg = SINDyRegressor()
    reg.fit(x, y, _RICH_LIBRARY)

    assert set(reg.selected_terms_) == set(ground_truth)

    coeffs = np.asarray(reg.coefficients_, dtype=float).reshape(-1)
    recovered = dict(zip(reg.selected_terms_, coeffs.tolist(), strict=True))
    for term, coef in ground_truth.items():

        assert recovered[term] == pytest.approx(coef, abs=0.1)


    assert np.isfinite(reg.nmse_)
    assert reg.nmse_ < 0.05


@pytest.mark.smoke
def test_sindy_predict_round_trip_under_noise() -> None:
    x_train, y_train, _ = _make_noisy_problem(n=400, noise=0.02, seed=1)
    reg = SINDyRegressor()
    reg.fit(x_train, y_train, _RICH_LIBRARY)

    rng = np.random.default_rng(123)
    x_new = rng.normal(size=(200, 3))
    x1, x2, x3 = x_new[:, 0], x_new[:, 1], x_new[:, 2]
    y_new_clean = 1.5 * x1 - 2.0 * np.sin(x2) + 0.8 * (x1 * x3)

    y_hat = reg.predict(x_new)
    assert y_hat.shape == (200,)

    np.testing.assert_allclose(y_hat, y_new_clean, rtol=0.05, atol=0.1)


@pytest.mark.smoke
def test_sindy_bad_term_fails_loud_end_to_end() -> None:
    x, y, _ = _make_noisy_problem(n=200, seed=2)
    reg = SINDyRegressor()
    bad = "sin(x2"
    with pytest.raises((ValueError, SyntaxError)) as excinfo:
        reg.fit(x, y, ["x1", bad, "mul(x1, x3)"])
    assert bad in str(excinfo.value)
