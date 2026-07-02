
from __future__ import annotations

import pytest

import kd


@pytest.mark.slow
def test_same_seed_fits_are_identical() -> None:
    data = kd.load_burgers()

    results: list[tuple[str, float]] = []
    for _ in range(2):
        model = kd.Model(algorithm="sga", generations=5, seed=0)
        model.fit(data)
        results.append((model.best_expr_, model.best_score_))

    assert results[0][0] == results[1][0], (
        f"same seed, different equations: {results[0][0]!r} vs {results[1][0]!r}"
    )
    assert results[0][1] == results[1][1]
