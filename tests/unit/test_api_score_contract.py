
from __future__ import annotations

from typing import Any, ClassVar, Literal

import pytest

from kd.api import (
    _PLUGIN_CLASS_BY_ALGORITHM,
    _SUPPORTED_ALGORITHMS,
    Model,
    _score_label,
)
from kd.search.callbacks import EarlyStoppingCallback
from kd.search.discover import DISCOVERPlugin
from kd.search.dlga import DLGAPlugin
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.protocol import ScoreContract
from kd.search.pysr import PySRPlugin
from kd.search.sga import SGAPlugin

_EXPECTED_ORDER = ("sga", "dlga", "discover", "pysr", "eqgpt", "llm4ed")







class TestPluginRegistry:

    def test_expected_order_covers_registry_keys(self) -> None:
        assert set(_EXPECTED_ORDER) == set(_PLUGIN_CLASS_BY_ALGORITHM)

    def test_registry_keys_and_order(self) -> None:
        assert tuple(_PLUGIN_CLASS_BY_ALGORITHM) == _EXPECTED_ORDER

    def test_registry_maps_names_to_the_plugin_classes(self) -> None:
        assert _PLUGIN_CLASS_BY_ALGORITHM["sga"] is SGAPlugin
        assert _PLUGIN_CLASS_BY_ALGORITHM["dlga"] is DLGAPlugin
        assert _PLUGIN_CLASS_BY_ALGORITHM["discover"] is DISCOVERPlugin
        assert _PLUGIN_CLASS_BY_ALGORITHM["pysr"] is PySRPlugin
        assert _PLUGIN_CLASS_BY_ALGORITHM["eqgpt"] is EqGPTPlugin

    def test_supported_algorithms_derived_from_registry(self) -> None:
        assert tuple(_PLUGIN_CLASS_BY_ALGORITHM) == _SUPPORTED_ALGORITHMS


class TestRegistryStructuralPrevention:

    def test_every_registry_entry_implements_score_contract(self) -> None:
        undeclared = [
            name
            for name, cls in _PLUGIN_CLASS_BY_ALGORITHM.items()
            if not isinstance(cls, ScoreContract)
        ]
        assert not undeclared, (
            f"Registry entries without a ScoreContract declaration: "
            f"{undeclared}. Declare class-level score_kind/score_direction "
            f"on the plugin class (see SGAPlugin)."
        )

    def test_no_entry_declares_placeholder_or_invalid_values(self) -> None:
        for name, cls in _PLUGIN_CLASS_BY_ALGORITHM.items():
            kind = cls.score_kind
            direction = cls.score_direction
            assert isinstance(kind, str) and kind.strip(), (
                f"{name}: score_kind must be a non-empty str, got {kind!r}"
            )
            assert kind != "Score", (
                f"{name}: 'Score' is the undeclared-fallback label; declare "
                f"the real metric name"
            )
            assert direction in ("min", "max"), (
                f"{name}: score_direction must be 'min'|'max', got {direction!r}"
            )







def _opposite(direction: str) -> Literal["min", "max"]:
    return "max" if direction == "min" else "min"


class TestEarlyStopGuardFollowsDeclaration:

    @pytest.mark.parametrize("algorithm", _EXPECTED_ORDER)
    def test_mismatched_mode_raises_typeerror_at_init(self, algorithm: str) -> None:
        declared = _PLUGIN_CLASS_BY_ALGORITHM[algorithm].score_direction
        wrong_mode = _opposite(declared)
        cb = EarlyStoppingCallback(mode=wrong_mode, patience=5)

        with pytest.raises(TypeError) as excinfo:
            Model(algorithm=algorithm, callbacks=[cb])

        msg = str(excinfo.value)
        assert "mode" in msg, f"error must name the mode; got {msg!r}"
        assert algorithm in msg, f"error must name the algorithm; got {msg!r}"

    @pytest.mark.parametrize("algorithm", _EXPECTED_ORDER)
    def test_matched_mode_accepted_at_init(self, algorithm: str) -> None:
        declared = _PLUGIN_CLASS_BY_ALGORITHM[algorithm].score_direction
        cb = EarlyStoppingCallback(mode=declared, patience=5)

        model = Model(algorithm=algorithm, callbacks=[cb])

        assert model.algorithm == algorithm







class _CustomKindStub:

    score_kind: ClassVar[str] = "CUSTOM"
    score_direction: ClassVar[Literal["min", "max"]] = "min"
    config: ClassVar[dict[str, Any]] = {"algorithm": "sga"}


class _UndeclaredStub:

    config: ClassVar[dict[str, Any]] = {"algorithm": "sga"}


class TestScoreLabelFromInstance:

    def test_reads_declared_kind_over_lookup_table(self) -> None:
        assert _score_label(_CustomKindStub()) == "CUSTOM"

    def test_falls_back_to_generic_label_without_declaration(self) -> None:
        assert _score_label(_UndeclaredStub()) == "Score"

    @pytest.mark.parametrize(
        ("algorithm", "expected"),
        [
            pytest.param("sga", "AIC", id="sga"),
            pytest.param("dlga", "DLGA fitness", id="dlga"),
            pytest.param("discover", "reward", id="discover"),
            pytest.param("pysr", "NMSE", id="pysr"),
        ],
    )
    def test_real_plugin_instances_label_as_declared(
        self, algorithm: str, expected: str
    ) -> None:
        plugin = _PLUGIN_CLASS_BY_ALGORITHM[algorithm]()
        assert _score_label(plugin) == expected
