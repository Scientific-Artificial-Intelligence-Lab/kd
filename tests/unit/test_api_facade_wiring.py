
from __future__ import annotations

import pytest

from kd.api import _PLUGIN_CLASS_BY_ALGORITHM, Model
from kd.search.discover import DiscoverConfig
from kd.search.dlga import DLGAConfig
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.llm4ed.config import Llm4edConfig
from kd.search.pysr import PySRConfig
from kd.search.sga import SGAConfig

pytestmark = pytest.mark.unit

_EXPECTED_ORDER = ("sga", "dlga", "discover", "pysr", "eqgpt", "llm4ed")


_EXPECTED_CONFIG_CLS = {
    "sga": SGAConfig,
    "dlga": DLGAConfig,
    "discover": DiscoverConfig,
    "pysr": PySRConfig,
    "eqgpt": EqGPTConfig,
    "llm4ed": Llm4edConfig,
}




_EXPECTED_BATCH = {
    "sga": SGAConfig().num,
    "dlga": DLGAConfig().pop_size,
    "discover": DiscoverConfig().batch_size,
    "pysr": 1,
    "eqgpt": EqGPTConfig(sparsity_alpha=0.02).samples_per_epoch,
    "llm4ed": Llm4edConfig().samples_per_epoch,
}


def _model_for_algorithm(algorithm: str) -> Model:
    if algorithm == "eqgpt":
        return Model(
            algorithm="eqgpt",
            config=EqGPTConfig(seed=0, sparsity_alpha=0.02),
        )
    return Model(algorithm=algorithm)







class TestPluginClassVarDeclarations:

    def test_expected_order_covers_registry_keys(self) -> None:
        assert set(_EXPECTED_ORDER) == set(_PLUGIN_CLASS_BY_ALGORITHM)

    @pytest.mark.parametrize("algorithm", _EXPECTED_ORDER)
    def test_config_cls_matches_expected_map(self, algorithm: str) -> None:
        plugin_cls = _PLUGIN_CLASS_BY_ALGORITHM[algorithm]
        config_cls = plugin_cls.config_cls
        assert isinstance(config_cls, type), (
            f"{algorithm}: config_cls must be a class, got {config_cls!r}"
        )
        assert config_cls is _EXPECTED_CONFIG_CLS[algorithm]

    @pytest.mark.parametrize("algorithm", _EXPECTED_ORDER)
    def test_one_shot_is_bool_true_iff_pysr(self, algorithm: str) -> None:
        plugin_cls = _PLUGIN_CLASS_BY_ALGORITHM[algorithm]
        one_shot = plugin_cls.one_shot
        assert isinstance(one_shot, bool), (
            f"{algorithm}: one_shot must be a bool, got {one_shot!r}"
        )
        assert one_shot is (algorithm == "pysr")

    def test_every_registry_entry_declares_both_classvars(self) -> None:
        missing: list[str] = []
        for name, cls in _PLUGIN_CLASS_BY_ALGORITHM.items():
            if not hasattr(cls, "config_cls") or not hasattr(cls, "one_shot"):
                missing.append(name)
        assert not missing, (
            f"Registry entries missing config_cls/one_shot declarations: "
            f"{missing}. Declare them on the plugin class (see SGAPlugin)."
        )

    def test_config_builder_dispatch_covers_exactly_the_registry(self) -> None:
        from kd.api import Model

        assert set(Model._CONFIG_BUILDER_BY_ALGORITHM) == set(
            _PLUGIN_CLASS_BY_ALGORITHM
        )

        for method_name in Model._CONFIG_BUILDER_BY_ALGORITHM.values():
            assert callable(getattr(Model, method_name))







class TestRunnerBatchSizeProperty:

    @pytest.mark.parametrize("algorithm", _EXPECTED_ORDER)
    def test_agrees_with_build_plugin_and_config_default(
        self, algorithm: str
    ) -> None:
        model = _model_for_algorithm(algorithm)
        plugin, batch_size = model._build_plugin()

        runner_batch = plugin.runner_batch_size
        expected = _EXPECTED_BATCH[algorithm]

        assert isinstance(runner_batch, int) and runner_batch > 0, (
            f"{algorithm}: runner_batch_size must be a positive int, "
            f"got {runner_batch!r}"
        )
        assert runner_batch == batch_size, (
            f"{algorithm}: runner_batch_size ({runner_batch}) must match the "
            f"batch size returned by _build_plugin() ({batch_size})"
        )
        assert runner_batch == expected, (
            f"{algorithm}: runner_batch_size ({runner_batch}) must equal the "
            f"config-default field ({expected})"
        )







class TestFacadeWiringContractProtocol:

    def test_exists_and_extends_score_contract(self) -> None:
        from kd.search.protocol import FacadeWiringContract, ScoreContract

        assert ScoreContract in FacadeWiringContract.__mro__, (
            "FacadeWiringContract must subclass ScoreContract "
            "(inherit score_kind/score_direction)"
        )
        annotations = getattr(FacadeWiringContract, "__annotations__", {})
        assert "config_cls" in annotations, "config_cls ClassVar not declared"
        assert "one_shot" in annotations, "one_shot ClassVar not declared"
        assert hasattr(FacadeWiringContract, "runner_batch_size"), (
            "runner_batch_size property not declared"
        )
