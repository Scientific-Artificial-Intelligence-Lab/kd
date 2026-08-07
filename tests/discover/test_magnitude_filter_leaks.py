
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.nn import Parameter

from kd.core.evaluator import EvaluationResult
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.core.batch import Batch
from kd.search.discover.engine import DiscoverEngine
from kd.search.discover.engine_types import EngineState
from kd.search.discover.evaluation.magnitude import (
    MAGNITUDE_FILTER_MAX,
    apply_magnitude_filter,
)
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.validator import CandidateValidator
from kd.search.discover.training.strategy import RSPGStrategy





LIB_CONFIG = LibraryConfig(
    operators=["add", "mul", "diff_x", "diff2_x"],
    state_vars=["u"],
    coord_vars=["x", "t"],
)

MAX_LENGTH = 15
BATCH_SIZE = 4

X_TOKENS: tuple[str, ...] = ("diff_x", "u")
X_IR = "diff_x(u)"


COEFF_IN_BOUNDS = 2.5
NMSE_IN_BOUNDS = 0.05
REWARD_IN_BOUNDS = 0.9



COEFF_OUT_OF_BOUNDS = 1.6649e4
NMSE_OUT_OF_BOUNDS = 0.02
REWARD_OUT_OF_BOUNDS = 0.95







class ScriptedGenerator:

    def __init__(
        self,
        library: Library,
        scripts: Sequence[Sequence[str]],
    ) -> None:
        if not scripts:
            raise ValueError("scripts must contain at least one token row.")
        self._library = library
        self._scripts: list[list[int]] = [
            [library.names.index(name) for name in row] for row in scripts
        ]
        self._call_idx = 0
        self._param = Parameter(torch.zeros(1))
        self._state: dict[str, Any] = {"scripted": True}

    @property
    def library(self) -> Library:
        return self._library

    def sample(self, batch_size: int) -> Batch:
        row = self._scripts[min(self._call_idx, len(self._scripts) - 1)]
        self._call_idx += 1
        length = len(row)
        n_tokens = len(self._library.tokens)
        actions = np.tile(np.asarray(row, dtype=np.int32), (batch_size, 1))
        obs = np.zeros((batch_size, 4, length), dtype=np.float32)
        priors = np.ones((batch_size, length, n_tokens), dtype=np.float32)
        lengths = np.full(batch_size, length, dtype=np.int32)
        return Batch(actions=actions, obs=obs, priors=priors, lengths=lengths)

    def make_neglogp_and_entropy(
        self,
        batch: Batch,
        entropy_gamma: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        batch_len = batch.actions.shape[0]
        anchor = self._param.sum()
        return torch.zeros(batch_len) + anchor, torch.zeros(batch_len) + anchor

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def parameters(self) -> Iterator[Parameter]:
        yield self._param

    def state_dict(self) -> dict[str, Any]:
        return dict(self._state)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._state = dict(state_dict)

    def train(self, mode: bool = True) -> None:
        pass


@dataclass(frozen=True)
class EvalSpec:

    reward: float = 0.0
    nmse: float = 0.5
    coefficient: float = 1.0
    valid: bool = True
    error: str = ""


class TableEvaluator:

    def __init__(self, table: dict[str, EvalSpec]) -> None:
        self._table = dict(table)
        self.calls: list[str] = []

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        self.calls.append(expr)
        spec = self._table.get(expr)
        if spec is None:
            raise AssertionError(
                f"TableEvaluator got unexpected expression {expr!r}; "
                f"known: {sorted(self._table)}"
            )
        if not spec.valid:
            return EvaluationResult(
                mse=0.0,
                nmse=0.0,
                r2=0.0,
                complexity=0,
                is_valid=False,
                error_message=spec.error,
                expression=expr,
            )
        return EvaluationResult(
            mse=spec.nmse,
            nmse=spec.nmse,
            r2=spec.reward,
            complexity=1,
            is_valid=True,
            expression=expr,
            terms=[expr],
            coefficients=torch.tensor([spec.coefficient]),
            selected_indices=None,
        )


def passthrough_reward(result: EvaluationResult) -> float:
    if not result.is_valid:
        return 0.0
    return float(result.r2)







@pytest.fixture
def lib() -> Library:
    return Library.from_config(LIB_CONFIG)


def _make_gated_engine(
    lib: Library,
    scripts: Sequence[Sequence[str]],
) -> DiscoverEngine:
    generator = ScriptedGenerator(lib, scripts)
    return DiscoverEngine(
        generator=generator,
        strategy=RSPGStrategy(epsilon=0.5, baseline="R_e", entropy_weight=0.005),
        reward_adapter=passthrough_reward,
        result_filter=apply_magnitude_filter,
        validator=CandidateValidator(lib, max_length=MAX_LENGTH),
        deduplicator=None,
        batch_size=BATCH_SIZE,
    )


def _seed_in_bounds_champion(engine: DiscoverEngine) -> None:
    eval_a = TableEvaluator(
        {
            X_IR: EvalSpec(
                reward=REWARD_IN_BOUNDS,
                nmse=NMSE_IN_BOUNDS,
                coefficient=COEFF_IN_BOUNDS,
            ),
        }
    )
    engine.run_cycle(eval_a, n_iterations=1)
    assert engine.best_expression == X_IR, "precondition: X must win cycle A"
    best = engine.best_result
    assert best is not None and best.is_valid, (
        "precondition: champion must be a valid in-bounds scale-A evaluation"
    )


def _rebase_into_out_of_bounds(engine: DiscoverEngine) -> TableEvaluator:
    eval_b = TableEvaluator(
        {
            X_IR: EvalSpec(
                reward=REWARD_OUT_OF_BOUNDS,
                nmse=NMSE_OUT_OF_BOUNDS,
                coefficient=COEFF_OUT_OF_BOUNDS,
            ),
        }
    )
    engine.run_cycle(eval_b, n_iterations=0)
    return eval_b







@pytest.mark.unit
class TestStateGetterGateLeak:

    def test_precondition_rebase_rejects_out_of_bounds_champion(
        self, lib: Library
    ) -> None:
        engine = _make_gated_engine(lib, scripts=[X_TOKENS])
        _seed_in_bounds_champion(engine)
        _rebase_into_out_of_bounds(engine)

        best = engine.best_result
        assert best is not None
        assert best.is_valid is False, (
            "rebase onto out-of-bounds coefficients must reject the champion"
        )
        assert "large_coe" in best.error_message, (
            "rejection must name the offending bound (large_coe)"
        )
        assert engine.best_expression == X_IR, (
            "DSO keeps the champion expression even when gate-rejected"
        )
        assert best.coefficients is not None
        assert float(best.coefficients.abs().max()) > MAGNITUDE_FILTER_MAX, (
            "precondition: the stored coefficients really are out of bounds"
        )

    def test_state_carries_best_result_validity_flag(self, lib: Library) -> None:
        engine = _make_gated_engine(lib, scripts=[X_TOKENS])
        _seed_in_bounds_champion(engine)

        valid_state = engine.state
        assert valid_state.best_result_is_valid is True, (
            "an in-bounds valid champion must report best_result_is_valid=True"
        )

        _rebase_into_out_of_bounds(engine)
        rejected_state = engine.state
        assert rejected_state.best_result_is_valid is False, (
            "a gate-rejected champion must report best_result_is_valid=False "
            "so consumers can drop its out-of-bounds coefficients"
        )

    def test_legacy_engine_state_defaults_valid(self, lib: Library) -> None:
        engine = _make_gated_engine(lib, scripts=[X_TOKENS])
        _seed_in_bounds_champion(engine)
        assert engine.state.best_result_is_valid is True







@pytest.mark.integration
class TestCheckpointRoundTripValidityFlag:

    def _state(self, *, is_valid: bool) -> EngineState:
        from kd.search.discover.training.strategy import BaselineState

        return EngineState(

            controller_state_dict={"w": torch.zeros(1)},
            baseline_state=BaselineState(),
            best_reward=REWARD_IN_BOUNDS,
            best_expression=X_IR,
            best_result_terms=[X_IR],
            best_result_coefficients=[COEFF_IN_BOUNDS],
            best_result_is_valid=is_valid,
        )

    def _payload_for(self, *, is_valid: bool) -> dict[str, Any]:
        from kd.search.discover.plugin import DISCOVERPlugin

        plugin = DISCOVERPlugin()
        plugin._pending_state = self._state(is_valid=is_valid)
        return plugin.state

    def test_invalid_flag_survives_serialize_parse_roundtrip(self) -> None:
        from kd.search.discover.plugin import _parse_state_payload

        payload = self._payload_for(is_valid=False)
        parsed = _parse_state_payload(payload)
        assert parsed.best_result_is_valid is False, (
            "the gate-rejected validity flag must survive the checkpoint chain"
        )

    def test_valid_flag_survives_serialize_parse_roundtrip(self) -> None:
        from kd.search.discover.plugin import _parse_state_payload

        payload = self._payload_for(is_valid=True)
        parsed = _parse_state_payload(payload)
        assert parsed.best_result_is_valid is True

    def test_legacy_payload_without_flag_parses_as_valid(self) -> None:
        from kd.search.discover.plugin import (
            BEST_RESULT_IS_VALID_KEY,
            _parse_state_payload,
        )

        payload = self._payload_for(is_valid=True)
        engine_state = payload["engine_state"]
        engine_state.pop(BEST_RESULT_IS_VALID_KEY, None)
        parsed = _parse_state_payload(payload)
        assert parsed.best_result_is_valid is True, (
            "legacy payloads lack the key; default must be True"
        )













@pytest.mark.unit
class TestRestoreChainPreservesGateRejection:

    def _invalid_state(self) -> EngineState:
        from kd.search.discover.training.strategy import BaselineState

        return EngineState(
            controller_state_dict={"scripted": True},
            baseline_state=BaselineState(),
            best_reward=0.0,
            best_expression=X_IR,
            best_result_terms=[X_IR],
            best_result_coefficients=[COEFF_OUT_OF_BOUNDS],
            best_result_is_valid=False,
        )

    def test_restore_then_resave_keeps_flag_false(self, lib: Library) -> None:
        engine = _make_gated_engine(lib, scripts=[X_TOKENS])
        engine.state = self._invalid_state()

        resaved = engine.state
        assert resaved.best_result_is_valid is False, (
            "a gate-rejected best must stay untrustworthy across save/restore; "
            "the restore must not launder it into a trustworthy placeholder"
        )

    def test_restored_invalid_best_is_refused_by_finalize(self, lib: Library) -> None:
        from kd.search.discover.pinn.cycle import _honest_best_terms

        engine = _make_gated_engine(lib, scripts=[X_TOKENS])
        engine.state = self._invalid_state()

        terms, coefficients = _honest_best_terms(engine.state)
        assert terms is None and coefficients is None, (
            "a restored gate-invalid champion must not be reportable as the "
            "final equation"
        )

    def test_valid_restore_stays_byte_identical(self, lib: Library) -> None:
        from kd.search.discover.training.strategy import BaselineState

        valid_state = EngineState(
            controller_state_dict={"scripted": True},
            baseline_state=BaselineState(),
            best_reward=REWARD_IN_BOUNDS,
            best_expression=X_IR,
            best_result_terms=[X_IR],
            best_result_coefficients=[COEFF_IN_BOUNDS],
            best_result_is_valid=True,
        )
        engine = _make_gated_engine(lib, scripts=[X_TOKENS])
        engine.state = valid_state

        resaved = engine.state
        assert resaved.best_result_is_valid is True
        assert resaved.best_result_terms == [X_IR]
        assert resaved.best_result_coefficients is not None
        assert resaved.best_result_coefficients == pytest.approx([COEFF_IN_BOUNDS])

    def test_repricing_heals_restored_invalid_best(self, lib: Library) -> None:
        engine = _make_gated_engine(lib, scripts=[X_TOKENS])
        engine.state = self._invalid_state()
        assert engine.state.best_result_is_valid is False, (
            "precondition: restored champion is still untrustworthy"
        )


        healed_eval = TableEvaluator(
            {
                X_IR: EvalSpec(
                    reward=REWARD_IN_BOUNDS,
                    nmse=NMSE_IN_BOUNDS,
                    coefficient=COEFF_IN_BOUNDS,
                ),
            }
        )
        engine.run_cycle(healed_eval, n_iterations=0)

        best = engine.best_result
        assert best is not None and best.is_valid, (
            "repricing on the current scale must replace the restored "
            "placeholder with a valid evaluation"
        )
        assert engine.state.best_result_is_valid is True, (
            "a champion that becomes legal on the current evaluator must "
            "recover trustworthiness (restore must not permanently poison it)"
        )


@pytest.mark.integration
class TestPluginRestoreChainPreservesGateRejection:

    def _prepared_plugin(self) -> Any:
        from unittest.mock import MagicMock

        from kd.search.discover.plugin import DISCOVERPlugin
        from kd.search.protocol import PlatformComponents

        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=_OutOfBoundsEvaluator(),
            context=MagicMock(training_result=None),
            registry=MagicMock(),
            recorder=None,
        )
        plugin = DISCOVERPlugin(DiscoverConfig(magnitude_filter=True))
        plugin.prepare(components)
        return plugin

    def _gate_invalid_payload(self, plugin: Any) -> dict[str, Any]:
        from kd.search.discover.plugin import (
            BEST_RESULT_COEFFICIENTS_KEY,
            BEST_RESULT_IS_VALID_KEY,
            BEST_RESULT_TERMS_KEY,
        )

        engine_state = plugin.state["engine_state"]
        return {
            "algorithm": "discover",
            "engine_state": {
                "controller_state_dict": engine_state["controller_state_dict"],
                "baseline_state": engine_state["baseline_state"],
                "best_reward": 0.0,
                "best_expression": X_IR,
                BEST_RESULT_TERMS_KEY: [X_IR],
                BEST_RESULT_COEFFICIENTS_KEY: [COEFF_OUT_OF_BOUNDS],
                BEST_RESULT_IS_VALID_KEY: False,
            },
        }

    def test_plugin_restore_keeps_flag_false(self) -> None:
        plugin = self._prepared_plugin()
        plugin.state = self._gate_invalid_payload(plugin)

        resaved = plugin.state["engine_state"]
        from kd.search.discover.plugin import BEST_RESULT_IS_VALID_KEY

        assert resaved.get(BEST_RESULT_IS_VALID_KEY) is False, (
            "the plugin restore chain must thread best_result_is_valid=False "
            "all the way into _rebuild_best_result"
        )

        from kd.search.discover.plugin import BEST_RESULT_TERMS_KEY

        assert resaved.get(BEST_RESULT_TERMS_KEY) == [X_IR]







def _invalid_final_state() -> EngineState:
    from kd.search.discover.training.strategy import BaselineState

    return EngineState(
        controller_state_dict={"w": torch.zeros(1)},
        baseline_state=BaselineState(),
        best_reward=0.0,
        best_expression=X_IR,
        best_result_terms=[X_IR],
        best_result_coefficients=[COEFF_OUT_OF_BOUNDS],
        best_result_is_valid=False,
    )


def _valid_final_state() -> EngineState:
    from kd.search.discover.training.strategy import BaselineState

    return EngineState(
        controller_state_dict={"w": torch.zeros(1)},
        baseline_state=BaselineState(),
        best_reward=REWARD_IN_BOUNDS,
        best_expression=X_IR,
        best_result_terms=[X_IR],
        best_result_coefficients=[COEFF_IN_BOUNDS],
        best_result_is_valid=True,
    )


@pytest.mark.unit
class TestHonestBestTerms:

    def test_invalid_best_yields_none_terms(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from kd.search.discover.pinn.cycle import _honest_best_terms

        with caplog.at_level("WARNING"):
            terms, coefficients = _honest_best_terms(_invalid_final_state())
        assert terms is None, "gate-invalid best must not surface its terms"
        assert coefficients is None, (
            "gate-invalid best must not surface its out-of-bounds coefficients"
        )
        assert any("invalid" in record.message.lower() for record in caplog.records), (
            "dropping a gate-invalid equation must be logged"
        )

    def test_valid_best_passes_terms_through(self) -> None:
        from kd.search.discover.pinn.cycle import _honest_best_terms

        terms, coefficients = _honest_best_terms(_valid_final_state())
        assert terms == [X_IR]
        assert coefficients == [COEFF_IN_BOUNDS]


@pytest.mark.unit
class TestFinalizeDropsGateInvalidEquation:

    def _finalize(self, final_state: EngineState) -> EngineState:
        from types import SimpleNamespace

        from kd.search.discover.pinn.cycle import PINNCycleRunner

        config = DiscoverConfig(stability_selection=0, magnitude_filter=True)
        stub = SimpleNamespace(_config=config)


        return PINNCycleRunner._finalize_with_stability_selection(
            stub, final_state, evaluator=None
        )

    def test_invalid_best_dropped_from_final_state(self) -> None:
        out = self._finalize(_invalid_final_state())
        assert out.best_result_terms is None, (
            "a gate-invalid champion must not be reported as the final equation"
        )
        assert out.best_result_coefficients is None

        assert out.best_expression == X_IR

    def test_valid_best_preserved_in_final_state(self) -> None:
        out = self._finalize(_valid_final_state())
        assert out.best_result_terms == [X_IR]
        assert out.best_result_coefficients == [COEFF_IN_BOUNDS]







class _OutOfBoundsEvaluator:

    def __init__(self, coefficient: float = COEFF_OUT_OF_BOUNDS) -> None:
        self._coefficient = coefficient
        self.calls: list[str] = []

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        self.calls.append(expr)
        return EvaluationResult(
            mse=NMSE_OUT_OF_BOUNDS,
            nmse=NMSE_OUT_OF_BOUNDS,
            r2=REWARD_OUT_OF_BOUNDS,
            complexity=1,
            is_valid=True,
            expression=expr,
            terms=[expr or X_IR],
            coefficients=torch.tensor([self._coefficient]),
            selected_indices=None,
        )


def _make_components(evaluator: object) -> Any:
    from unittest.mock import MagicMock

    from kd.search.protocol import PlatformComponents

    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=evaluator,
        context=MagicMock(training_result=None),
        registry=MagicMock(),
        recorder=None,
    )


def _prepared_plugin(*, magnitude_filter: bool, evaluator: object) -> Any:
    from kd.search.discover.plugin import DISCOVERPlugin

    plugin = DISCOVERPlugin(DiscoverConfig(magnitude_filter=magnitude_filter))
    plugin.prepare(_make_components(evaluator))
    return plugin


@pytest.mark.integration
class TestBuildFinalResultGate:

    def test_gate_on_rejects_out_of_bounds_final_result(self) -> None:
        plugin = _prepared_plugin(
            magnitude_filter=True,
            evaluator=_OutOfBoundsEvaluator(),
        )
        result = plugin.build_final_result()
        assert result.is_valid is False, (
            "build_final_result must apply the magnitude gate when enabled; "
            "an out-of-bounds coefficient must not pass as a valid final result"
        )
        assert "large_coe" in result.error_message

    def test_gate_off_leaves_out_of_bounds_final_result_valid(self) -> None:
        plugin = _prepared_plugin(
            magnitude_filter=False,
            evaluator=_OutOfBoundsEvaluator(),
        )
        result = plugin.build_final_result()
        assert result.is_valid is True, (
            "with the gate off build_final_result must be unchanged "
            "(the out-of-bounds fit stays valid as before)"
        )

    def test_gate_on_keeps_in_bounds_final_result_valid(self) -> None:
        plugin = _prepared_plugin(
            magnitude_filter=True,
            evaluator=_OutOfBoundsEvaluator(coefficient=COEFF_IN_BOUNDS),
        )
        result = plugin.build_final_result()
        assert result.is_valid is True


@pytest.mark.unit
class TestEvaluateSelectedCandidateGate:

    def _runner_stub(self, *, magnitude_filter: bool) -> Any:
        from types import SimpleNamespace

        config = DiscoverConfig(magnitude_filter=magnitude_filter)
        return SimpleNamespace(_config=config)

    def test_gate_on_rejects_out_of_bounds_selected_candidate(self) -> None:
        from kd.search.discover.pinn.cycle import PINNCycleRunner

        stub = self._runner_stub(magnitude_filter=True)
        with pytest.raises(ValueError):
            PINNCycleRunner._evaluate_selected_candidate(
                stub, _OutOfBoundsEvaluator(), X_IR
            )

    def test_gate_off_returns_out_of_bounds_selected_candidate(self) -> None:
        from kd.search.discover.pinn.cycle import PINNCycleRunner

        stub = self._runner_stub(magnitude_filter=False)
        terms, coefficients = PINNCycleRunner._evaluate_selected_candidate(
            stub, _OutOfBoundsEvaluator(), X_IR
        )
        assert terms == [X_IR]
        assert coefficients == [COEFF_OUT_OF_BOUNDS]







class _RecordingPINNModel:

    def __init__(self) -> None:
        self.trained_coefficients: list[list[float]] = []

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def train_pinn(self, *, coefficients: list[float], **_: Any) -> Any:
        from types import SimpleNamespace

        self.trained_coefficients.append(list(coefficients))
        return SimpleNamespace(data_loss=0.0, physics_loss=0.0, total_loss=0.0)


@pytest.mark.unit
class TestPINNTrainingFreshResultGate:

    def _stub(self, *, magnitude_filter: bool, model: _RecordingPINNModel) -> Any:
        from types import SimpleNamespace

        valid_best = EvaluationResult(
            mse=NMSE_IN_BOUNDS,
            nmse=NMSE_IN_BOUNDS,
            r2=REWARD_IN_BOUNDS,
            complexity=1,
            is_valid=True,
            expression=X_IR,
            terms=[X_IR],
            coefficients=torch.tensor([COEFF_IN_BOUNDS]),
            selected_indices=None,
        )
        engine = SimpleNamespace(
            best_reward=REWARD_IN_BOUNDS,
            best_expression=X_IR,
            best_result=valid_best,
        )
        return SimpleNamespace(
            _config=DiscoverConfig(magnitude_filter=magnitude_filter),
            _engine=engine,
            _pinn_model=model,
            _pinn_executor=None,
            _obs_coords={},
            _obs_targets={},
            _colloc_coords={},
            _dataset_metadata=None,
            _pinn_config=None,
            _make_local_coords=lambda _cycle_idx: None,
        )

    def test_gate_on_skips_training_on_out_of_bounds_fresh_fit(self) -> None:
        from kd.search.discover.pinn.cycle import PINNCycleRunner

        model = _RecordingPINNModel()
        stub = self._stub(magnitude_filter=True, model=model)
        evaluator = _OutOfBoundsEvaluator()
        _metrics, pinn_ok = PINNCycleRunner._run_pinn_phase(stub, 0, evaluator)
        assert pinn_ok is False, (
            "an out-of-bounds fresh re-fit must skip PINN training when gated"
        )
        assert model.trained_coefficients == [], (
            "the PINN must never train on gate-rejected out-of-bounds coefficients"
        )

    def test_gate_off_trains_on_fresh_fit_as_today(self) -> None:
        from kd.search.discover.pinn.cycle import PINNCycleRunner

        model = _RecordingPINNModel()
        stub = self._stub(magnitude_filter=False, model=model)
        evaluator = _OutOfBoundsEvaluator()
        _metrics, pinn_ok = PINNCycleRunner._run_pinn_phase(stub, 0, evaluator)
        assert pinn_ok is True, "with the gate off, training proceeds as before"
        assert model.trained_coefficients == [[COEFF_OUT_OF_BOUNDS]], (
            "gate-off path is byte-identical: it still trains on the fresh fit"
        )


__all__: list[str] = []
