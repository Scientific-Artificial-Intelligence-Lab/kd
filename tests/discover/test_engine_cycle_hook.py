from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.nn import Parameter

from kd.search.discover.core.batch import Batch
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import PriorContext, PriorSystem, ScaffoldPrior

_BURGERS_CONFIG = LibraryConfig(
    coord_vars=["x", "t"],
    state_vars=["u"],
    operators=["add", "mul", "sub", "div", "sin", "cos", "diff_x", "diff2_x"],
)


class _SpyPrior:

    def __init__(self, library: Library) -> None:
        self.library = library
        self.n_choices = len(library.tokens)
        self.calls: list[int] = []

    def on_cycle_start(self, cycle_idx: int) -> None:
        self.calls.append(int(cycle_idx))


    def initial_adjustment(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.n_choices), dtype=np.float32)

    def __call__(self, ctx: PriorContext) -> np.ndarray:
        return np.zeros(
            (ctx.actions.shape[0], self.n_choices), dtype=np.float32,
        )


class _MockGenerator:

    def __init__(self, library: Library, prior_system: Any) -> None:
        self._library = library
        self._param = Parameter(torch.zeros(1))
        self._state: dict[str, Any] = {"mock": True}
        self.prior_system = prior_system

    @property
    def library(self) -> Library:
        return self._library

    def sample(self, batch_size: int) -> Batch:
        n_tokens = len(self._library.tokens)
        return Batch(
            actions=np.full((batch_size, 1), 0, dtype=np.int32),
            obs=np.zeros((batch_size, 4, 1), dtype=np.float32),
            priors=np.ones((batch_size, 1, n_tokens), dtype=np.float32),
            lengths=np.ones(batch_size, dtype=np.int32),
        )

    def make_neglogp_and_entropy(
        self,
        batch: Batch,
        entropy_gamma: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        bs = batch.actions.shape[0]
        return torch.zeros(bs), torch.zeros(bs)

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


def _make_engine_with_spy() -> tuple[Any, _SpyPrior]:
    from kd.search.discover.engine import DiscoverEngine
    from kd.search.discover.evaluation.reward import compute_reward
    from kd.search.discover.tokens.validator import CandidateValidator
    from kd.search.discover.training.strategy import RSPGStrategy

    library = Library.from_config(_BURGERS_CONFIG)
    spy = _SpyPrior(library)
    prior_system = MagicMock(spec=PriorSystem)
    prior_system.priors = [spy]

    generator = _MockGenerator(library, prior_system)

    strategy = RSPGStrategy(epsilon=0.5, baseline="R_e", entropy_weight=0.0)
    validator = CandidateValidator(library, max_length=15)

    engine = DiscoverEngine(
        generator=generator,
        strategy=strategy,
        reward_adapter=compute_reward,
        validator=validator,
        batch_size=4,
    )
    return engine, spy


@pytest.mark.unit
class TestRunCycleFiresHook:

    def test_run_cycle_without_cycle_idx_does_not_notify(self) -> None:
        engine, spy = _make_engine_with_spy()
        engine.run_cycle(MagicMock(), n_iterations=0)
        assert spy.calls == []

    def test_run_cycle_with_cycle_idx_zero_fires_once(self) -> None:
        engine, spy = _make_engine_with_spy()
        engine.run_cycle(MagicMock(), n_iterations=0, cycle_idx=0)
        assert spy.calls == [0]

    def test_run_cycle_with_cycle_idx_deactivates_on_nonzero(self) -> None:
        engine, spy = _make_engine_with_spy()
        engine.run_cycle(MagicMock(), n_iterations=0, cycle_idx=3)
        assert spy.calls == [3]

    def test_run_delegates_cycle_idx_to_run_cycle(self) -> None:
        engine, spy = _make_engine_with_spy()
        engine.run(MagicMock(), n_iterations=0, n_cycles=3)
        assert spy.calls == [0, 1, 2]

    def test_run_cycle_rejects_negative_cycle_idx(self) -> None:
        engine, _spy = _make_engine_with_spy()
        with pytest.raises(ValueError, match="non-negative"):
            engine.run_cycle(MagicMock(), n_iterations=0, cycle_idx=-1)

    def test_run_cycle_rejects_non_int_cycle_idx(self) -> None:
        engine, _spy = _make_engine_with_spy()
        with pytest.raises(TypeError):
            engine.run_cycle(
                MagicMock(), n_iterations=0, cycle_idx=1.5,
            )


@pytest.mark.unit
class TestScaffoldPriorOnCycleStartValidation:

    def _make_prior(self) -> ScaffoldPrior:
        lib = Library.from_config(_BURGERS_CONFIG)
        return ScaffoldPrior(
            lib,
            diffusion_tokens=["diff2_x"],
            reaction_tokens=["u", "n3"],
        )

    def test_rejects_bool(self) -> None:
        prior = self._make_prior()
        with pytest.raises(TypeError, match="not bool"):
            prior.on_cycle_start(True)

    def test_rejects_float(self) -> None:
        prior = self._make_prior()
        with pytest.raises(TypeError):
            prior.on_cycle_start(0.0)

    def test_rejects_none(self) -> None:
        prior = self._make_prior()
        with pytest.raises(TypeError):
            prior.on_cycle_start(None)

    def test_rejects_negative(self) -> None:
        prior = self._make_prior()
        with pytest.raises(ValueError):
            prior.on_cycle_start(-1)

    def test_accepts_numpy_int(self) -> None:
        prior = self._make_prior()

        prior.on_cycle_start(np.int32(0))
        assert prior._active is True

    def test_idempotent_reactivation(self) -> None:
        prior = self._make_prior()
        prior.on_cycle_start(0)
        assert prior._active is True
        prior.on_cycle_start(0)
        assert prior._active is True


@pytest.mark.unit
class TestPINNCycleRunnerForwardsCycleIdx:

    def test_pinn_cycle_observability_passes_cycle_idx(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import logging

        import kd.search.discover.runners.pinn_cycle_observability as obs

        runner = MagicMock()
        runner._cycle_iterations.return_value = 0

        monkeypatch.setattr(
            obs, "_write_stage_checkpoint", lambda *args, **kwargs: None,
        )
        obs._run_search_stage(
            runner,
            evaluator=MagicMock(),
            cycle_idx=0,
            checkpoint_dir=tmp_path,
            heartbeat_iterations=0,
            start=0.0,
            run_logger=logging.getLogger("test_p21"),
        )
        call = runner._engine.run_cycle.call_args
        assert call.kwargs.get("cycle_idx") == 0, (
            f"cycle_idx missing from engine.run_cycle call: {call}"
        )

    def test_pinn_cycle_observability_final_stage_forwards_cycle_idx(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import logging

        import kd.search.discover.runners.pinn_cycle_observability as obs

        runner = MagicMock()
        runner._pinn_config.n_cycles = 3
        runner._cycle_iterations.return_value = 0
        monkeypatch.setattr(
            obs, "_write_stage_checkpoint", lambda *args, **kwargs: None,
        )
        obs._run_final_search(
            runner,
            evaluator=MagicMock(),
            checkpoint_dir=tmp_path,
            heartbeat_iterations=0,
            start=0.0,
            run_logger=logging.getLogger("test_p21"),
        )
        call = runner._engine.run_cycle.call_args
        assert call.kwargs.get("cycle_idx") == 3


@pytest.mark.unit
class TestScaffoldFlipThroughRunCycle:

    def test_scaffold_active_on_cycle_zero_inactive_on_one(self) -> None:
        from kd.search.discover.engine import DiscoverEngine
        from kd.search.discover.evaluation.reward import compute_reward
        from kd.search.discover.tokens.validator import CandidateValidator
        from kd.search.discover.training.strategy import RSPGStrategy

        library = Library.from_config(
            LibraryConfig(
                coord_vars=["x", "t"],
                state_vars=["u"],
                operators=[
                    "add", "sub", "mul", "sin", "n3", "diff_x", "diff2_x",
                ],
            )
        )
        scaffold = ScaffoldPrior(
            library,
            diffusion_tokens=["diff2_x"],
            reaction_tokens=["u", "n3"],
        )
        assert scaffold._active is False
        prior_system = MagicMock(spec=PriorSystem)
        prior_system.priors = [scaffold]

        generator = _MockGenerator(library, prior_system)

        strategy = RSPGStrategy(
            epsilon=0.5, baseline="R_e", entropy_weight=0.0,
        )
        engine = DiscoverEngine(
            generator=generator,
            strategy=strategy,
            reward_adapter=compute_reward,
            validator=CandidateValidator(library, max_length=15),
            batch_size=4,
        )

        engine.run_cycle(MagicMock(), n_iterations=0, cycle_idx=0)
        assert scaffold._active is True, (
            "scaffold must be active during cycle 0 via run_cycle"
        )

        engine.run_cycle(MagicMock(), n_iterations=0, cycle_idx=1)
        assert scaffold._active is False, (
            "scaffold must deactivate on cycle >= 1"
        )
