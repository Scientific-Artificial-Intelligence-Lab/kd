
from __future__ import annotations

import pytest

from kd.api import _PLUGIN_CLASS_BY_ALGORITHM, _SUPPORTED_ALGORITHMS, Model




_EXPECTED_CONFIG_CLASS_NAMES = (
    "SGAConfig",
    "DLGAConfig",
    "DiscoverConfig",
    "PySRConfig",
    "EqGPTConfig",
    "Llm4edConfig",
    "PySINDyConfig",
)


@pytest.mark.parametrize("algorithm", _SUPPORTED_ALGORITHMS)
def test_class_docstring_names_every_algorithm(algorithm: str) -> None:
    doc = Model.__doc__ or ""
    assert algorithm in doc, (
        f"Model class docstring does not mention supported algorithm "
        f"{algorithm!r}; every entry in _SUPPORTED_ALGORITHMS must appear in "
        f"the Args section."
    )


@pytest.mark.parametrize("algorithm", _SUPPORTED_ALGORITHMS)
def test_best_score_docstring_names_every_algorithm(algorithm: str) -> None:
    best_score_doc = Model.best_score_.__doc__ or ""
    assert algorithm in best_score_doc, (
        f"Model.best_score_ docstring does not document the score direction "
        f"for supported algorithm {algorithm!r}."
    )


def test_class_docstring_names_every_config_class() -> None:


    assert len(_EXPECTED_CONFIG_CLASS_NAMES) == len(_PLUGIN_CLASS_BY_ALGORITHM)
    doc = Model.__doc__ or ""
    missing = [name for name in _EXPECTED_CONFIG_CLASS_NAMES if name not in doc]
    assert not missing, (
        f"Model class docstring omits plugin config class(es) {missing} from "
        f"the ``config`` argument enumeration."
    )
