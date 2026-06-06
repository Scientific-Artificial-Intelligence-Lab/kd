
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clean_diagnostic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCOVER_ENABLE_DIAGNOSTICS", raising=False)


@pytest.fixture(autouse=True)
def _reset_scaffold_warn_cache() -> None:
    from kd.search.discover.tokens.scaffold_prior import ScaffoldPrior
    ScaffoldPrior.reset_warn_cache()


@pytest.fixture
def burgers_fixture() -> dict[str, np.ndarray]:
    data = np.load(FIXTURES_DIR / "library_burgers.npz", allow_pickle=True)
    return dict(data)


@pytest.fixture
def tree_fixture() -> dict[str, np.ndarray]:
    data = np.load(FIXTURES_DIR / "tree_model.npz", allow_pickle=True)
    return dict(data)


@pytest.fixture
def tree_state_fixture() -> dict[str, np.ndarray]:
    data = np.load(FIXTURES_DIR / "tree_state.npz", allow_pickle=True)
    return dict(data)


@pytest.fixture
def ir_fixture() -> dict[str, Any]:
    import json

    with open(FIXTURES_DIR / "ir_conversion.json") as f:
        return cast("dict[str, Any]", json.load(f))


@pytest.fixture
def prior_fixture() -> dict[str, np.ndarray]:
    data = np.load(FIXTURES_DIR / "prior_system.npz", allow_pickle=True)
    return dict(data)


@pytest.fixture
def repeat_fixture() -> dict[str, np.ndarray]:
    data = np.load(FIXTURES_DIR / "prior_repeat.npz", allow_pickle=True)
    return dict(data)


@pytest.fixture
def relational_fixture() -> dict[str, np.ndarray]:
    data = np.load(FIXTURES_DIR / "prior_relational.npz", allow_pickle=True)
    return dict(data)


@pytest.fixture
def validator_fixture() -> dict[str, np.ndarray]:
    data = np.load(FIXTURES_DIR / "validator.npz", allow_pickle=True)
    return dict(data)
