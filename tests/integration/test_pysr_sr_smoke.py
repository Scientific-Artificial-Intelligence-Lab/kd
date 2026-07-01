
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from kd.search.pysr.config import PySRConfig
from kd.search.pysr.sr import PySRSymbolicRegressor

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TLC_CC_PATH = _REPO_ROOT / "tests" / "fixtures" / "pysr" / "tlc_cc_Rf_t1.npy"
_VAR_NAMES = ["R_F", "r"]
_N_SAMPLES = 74


def _load_tlc_cc() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(_TLC_CC_PATH)
    return data[:, [0, 1]], data[:, 2]


def test_pysr_symbolic_regressor_real_tlc_cc_smoke() -> None:
    pytest.importorskip("pysr")
    x, y = _load_tlc_cc()
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
