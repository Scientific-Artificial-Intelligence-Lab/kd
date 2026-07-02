
from __future__ import annotations

import math

import numpy as np
import pytest

from kd.data.regression import load_tlc_cc
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.sr import PySRSymbolicRegressor

pytestmark = pytest.mark.slow

_VAR_NAMES = ["R_F", "r"]
_N_SAMPLES = 74


def test_pysr_symbolic_regressor_real_tlc_cc_smoke() -> None:
    pytest.importorskip("pysr")
    dataset = load_tlc_cc(target="start")
    x, y = dataset.X, dataset.y
    config = PySRConfig(
        niterations=5,
        population_size=20,
        populations=2,
        maxsize=10,
        seed=0,
    )
    model = PySRSymbolicRegressor(config=config)

    model.fit(x, y, var_names=_VAR_NAMES)

    assert isinstance(model.best_score_, float)
    assert math.isfinite(model.best_score_)
    assert isinstance(model.best_expr_, str)
    assert model.best_expr_
    symbol_names = {symbol.name for symbol in model.best_sympy_.free_symbols}
    assert symbol_names <= set(_VAR_NAMES)

    y_hat = model.predict(x)
    assert y_hat.shape == (_N_SAMPLES,)
    assert np.all(np.isfinite(y_hat))
