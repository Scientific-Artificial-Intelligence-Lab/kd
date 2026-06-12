
from __future__ import annotations

from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import pytest

from kd.search.discover import DISCOVERPlugin
from kd.search.dlga import DLGAPlugin
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    ScoreContract,
    SearchAlgorithm,
)
from kd.search.pysr import PySRPlugin
from kd.search.sga import SGAPlugin
from tests.unit.search._runner_mocks import (
    IterativeRecordingAlgorithm,
    RecordingAlgorithm,
)






_EXPECTED_CONTRACTS = [
    pytest.param(SGAPlugin, "AIC", "min", id="sga"),
    pytest.param(DLGAPlugin, "DLGA fitness", "min", id="dlga"),
    pytest.param(DISCOVERPlugin, "reward", "max", id="discover"),
    pytest.param(PySRPlugin, "NMSE", "min", id="pysr"),
]









@runtime_checkable
class _LocalDataProtocol(Protocol):

    kind: ClassVar[str]
    direction: ClassVar[Literal["min", "max"]]


class _LocalConforming:
    kind: ClassVar[str] = "demo-kind"
    direction: ClassVar[Literal["min", "max"]] = "max"


class _LocalNonConforming:
    pass


@pytest.mark.smoke
class TestRuntimeCheckableDataMemberSanity:

    def test_instance_with_classvar_members_passes_isinstance(self) -> None:
        assert isinstance(_LocalConforming(), _LocalDataProtocol)

    def test_instance_without_members_fails_isinstance(self) -> None:
        assert not isinstance(_LocalNonConforming(), _LocalDataProtocol)

    def test_class_level_read_without_instantiation(self) -> None:
        assert _LocalConforming.direction == "max"
        assert _LocalConforming.kind == "demo-kind"

    def test_issubclass_unsupported_for_data_member_protocols(self) -> None:
        with pytest.raises(TypeError):
            issubclass(_LocalConforming, _LocalDataProtocol)







class _DeclaringFake:

    score_kind: ClassVar[str] = "fake-metric"
    score_direction: ClassVar[Literal["min", "max"]] = "max"


class TestScoreContractProtocol:

    def test_conforming_instance_passes_isinstance(self) -> None:
        assert isinstance(_DeclaringFake(), ScoreContract)

    def test_undeclared_instance_fails_isinstance(self) -> None:
        assert not isinstance(RecordingAlgorithm(), ScoreContract)

    def test_object_without_members_fails_isinstance(self) -> None:
        assert not isinstance(object(), ScoreContract)


class TestSearchAlgorithmIsinstanceRipple:

    def test_undeclared_fake_still_satisfies_search_algorithm(self) -> None:
        algo = RecordingAlgorithm()
        assert not isinstance(algo, ScoreContract)
        assert isinstance(algo, SearchAlgorithm)

    def test_undeclared_iterative_fake_still_satisfies_subprotocol(self) -> None:
        algo = IterativeRecordingAlgorithm()
        assert not isinstance(algo, ScoreContract)
        assert isinstance(algo, IterativeSearchAlgorithm)

    def test_declaring_contract_does_not_grant_search_algorithm(self) -> None:
        fake = _DeclaringFake()
        assert isinstance(fake, ScoreContract)
        assert not isinstance(fake, SearchAlgorithm)








@pytest.mark.smoke
def test_sga_direction_readable_from_class() -> None:
    assert SGAPlugin.score_direction == "min"


class TestPluginScoreContractValues:

    @pytest.mark.parametrize(("cls", "kind", "direction"), _EXPECTED_CONTRACTS)
    def test_plugin_instance_implements_score_contract(
        self, cls: type[Any], kind: str, direction: str
    ) -> None:
        assert isinstance(cls(), ScoreContract), (
            f"{cls.__name__} must declare score_kind/score_direction (ScoreContract)"
        )

    @pytest.mark.parametrize(("cls", "kind", "direction"), _EXPECTED_CONTRACTS)
    def test_plugin_score_kind_class_level_value(
        self, cls: type[Any], kind: str, direction: str
    ) -> None:
        assert cls.score_kind == kind

    @pytest.mark.parametrize(("cls", "kind", "direction"), _EXPECTED_CONTRACTS)
    def test_plugin_score_direction_class_level_value(
        self, cls: type[Any], kind: str, direction: str
    ) -> None:
        assert cls.score_direction == direction

    def test_builtin_score_kinds_are_pairwise_distinct(self) -> None:
        kinds = [
            SGAPlugin.score_kind,
            DLGAPlugin.score_kind,
            DISCOVERPlugin.score_kind,
            PySRPlugin.score_kind,
        ]
        assert len(set(kinds)) == len(kinds), f"duplicate score_kind in {kinds}"

    def test_no_builtin_uses_the_generic_fallback_kind(self) -> None:
        for cls in (SGAPlugin, DLGAPlugin, DISCOVERPlugin, PySRPlugin):
            assert cls.score_kind != "Score"
            assert cls.score_kind.strip() != ""
