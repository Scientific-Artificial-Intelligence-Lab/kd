
from __future__ import annotations

import pytest

import kd
from kd.search.dlga import DLGAConfig


@pytest.mark.integration
def test_facade_dlga_trains_default_surrogate_without_crash() -> None:
    dataset = kd.generate_burgers_data(nx=16, nt=8, nu=0.1, seed=0)
    model = kd.Model(
        algorithm="dlga",
        generations=1,
        verbose=False,
        config=DLGAConfig(pop_size=4, seed=0, surrogate_max_epochs=5),
    )


    model.fit(dataset)

    assert model.result_ is not None


    assert model.best_expr_, "DLGA facade should recover a non-empty expression"
