
from __future__ import annotations

import numpy as np
import pytest

from kd.search.discover.stability import _solve_lstsq


@pytest.mark.unit
def test_matches_numpy_on_full_rank_overdetermined() -> None:
    rng = np.random.default_rng(0)
    theta = rng.standard_normal((40, 5))
    lhs = rng.standard_normal(40)
    expected, *_ = np.linalg.lstsq(theta, lhs, rcond=None)
    np.testing.assert_allclose(_solve_lstsq(theta, lhs), expected, atol=1e-12)


@pytest.mark.unit
def test_matches_numpy_on_near_singular_spectrum() -> None:
    rng = np.random.default_rng(1)
    m, n = 15, 7
    eps = float(np.finfo(np.float64).eps)
    u, _ = np.linalg.qr(rng.standard_normal((m, n)))
    v, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.array([1.0, 0.5, 0.1, 1e-3, 1e-6, 5 * eps, 2 * eps])
    theta = u @ np.diag(s) @ v.T
    lhs = rng.standard_normal(m)
    expected, *_ = np.linalg.lstsq(theta, lhs, rcond=None)
    got = _solve_lstsq(theta, lhs)


    np.testing.assert_allclose(got, expected, rtol=1e-9)


@pytest.mark.unit
def test_matches_numpy_on_bootstrap_duplicated_rows() -> None:
    rng = np.random.default_rng(2)
    base = rng.standard_normal(30)
    theta_full = np.column_stack([base**k for k in range(1, 8)])
    idx = rng.choice(30, 15, replace=True)
    theta = theta_full[idx]
    lhs = (theta_full @ np.ones(7))[idx]
    expected, *_ = np.linalg.lstsq(theta, lhs, rcond=None)
    np.testing.assert_allclose(_solve_lstsq(theta, lhs), expected, rtol=1e-9)
