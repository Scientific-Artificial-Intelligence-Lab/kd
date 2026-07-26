
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kd.api import (
    _PLUGIN_CLASS_BY_ALGORITHM,
    _SUPPORTED_ALGORITHMS,
    Model,
)
from kd.search.eqgpt.config import EqGPTConfig

if TYPE_CHECKING:
    from kd.search.discover import DiscoverConfig
    from kd.search.dlga import DLGAConfig
    from kd.search.pysindy.config import PySINDyConfig
    from kd.search.pysr.config import PySRConfig
    from kd.search.sga import SGAConfig

    AnyConfig = (
        SGAConfig
        | DLGAConfig
        | DiscoverConfig
        | PySRConfig
        | PySINDyConfig
        | EqGPTConfig
    )




















_CONFIG_BUILDER_BY_ALGORITHM: dict[str, str] = {
    "sga": "_build_config",
    "dlga": "_build_dlga_config",
    "discover": "_build_discover_config",
    "pysr": "_build_pysr_config",
    "eqgpt": "_build_eqgpt_config",
    "llm4ed": "_build_llm4ed_config",
    "pysindy": "_build_pysindy_config",
}


def _build_config_for(algorithm: str, *, seed: int) -> AnyConfig:
    if algorithm == "eqgpt":






        model = Model(
            algorithm=algorithm,
            config=EqGPTConfig(seed=seed, sparsity_alpha=0.02),
            verbose=False,
        )
    else:
        model = Model(algorithm=algorithm, seed=seed, verbose=False)
    builder_name = _CONFIG_BUILDER_BY_ALGORITHM[algorithm]
    builder = getattr(model, builder_name)
    return builder()

















_SEED_VALUES: tuple[int, ...] = (42, 0)


def _seed_reader(config: AnyConfig) -> int:
    return int(config.seed)


@pytest.mark.parametrize("algorithm", _SUPPORTED_ALGORITHMS)
class TestEveryAlgorithmHasASeed:

    def test_every_config_has_a_seed_field(self, algorithm: str) -> None:
        import dataclasses

        config = _build_config_for(algorithm, seed=42)
        field_names = {f.name for f in dataclasses.fields(config)}
        assert "seed" in field_names, (
            f"{type(config).__name__} (algorithm={algorithm!r}) has no 'seed' "
            "field. If seed is threaded by a documented alternative path, "
            "special-case this algorithm in the passthrough table with a "
            "comment — do not let the cross-algorithm net silently skip it."
        )


class TestConfigBuilderSeamCompleteness:

    def test_every_algorithm_has_a_config_builder_seam(self) -> None:
        missing = set(_SUPPORTED_ALGORITHMS) - set(_CONFIG_BUILDER_BY_ALGORITHM)
        assert not missing, (
            f"Algorithms registered in _PLUGIN_CLASS_BY_ALGORITHM but missing a "
            f"config-builder seam in this test: {sorted(missing)}. Add the "
            "builder method name to _CONFIG_BUILDER_BY_ALGORITHM so the "
            "param-passthrough net covers it."
        )



        assert set(_SUPPORTED_ALGORITHMS) == set(_PLUGIN_CLASS_BY_ALGORITHM)


@pytest.mark.parametrize("algorithm", _SUPPORTED_ALGORITHMS)
@pytest.mark.parametrize("seed", _SEED_VALUES)
class TestFacadeParamPassthrough:

    def test_seed_reaches_config(self, algorithm: str, seed: int) -> None:
        config = _build_config_for(algorithm, seed=seed)
        assert _seed_reader(config) == seed, (
            f"Model(algorithm={algorithm!r}, seed={seed}) did not thread the "
            f"facade seed into {type(config).__name__}.seed "
            f"(got {config.seed!r}). The facade builder is dropping the seed — "
            "this is the recurring silent-drop bug."
        )
