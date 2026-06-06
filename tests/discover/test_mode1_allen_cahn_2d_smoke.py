
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from kd.core.evaluator import Evaluator
    from kd.search.discover.config import DiscoverConfig







_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_PATH: Path = _PROJECT_ROOT / "data" / "allen_cahn_2d_smoke.npz"




_MIN_DATA_BYTES: int = 50_000
_SKIP_REASON = (
    f"Allen-Cahn 2D smoke data not found (or smaller than "
    f"{_MIN_DATA_BYTES} bytes) at {_DATA_PATH}. "
    "Regenerate with: uv run python scripts/generate_allen_cahn_2d.py "
    f"--seed 42 --out {_DATA_PATH}"
)


def _smoke_data_ready() -> bool:
    return _DATA_PATH.exists() and _DATA_PATH.stat().st_size > _MIN_DATA_BYTES


pytestmark = pytest.mark.skipif(not _smoke_data_ready(), reason=_SKIP_REASON)






SEED: int = 42

SMOKE_ITERATIONS: int = 20
SMOKE_BATCH_SIZE: int = 32


SMOKE_MAX_LENGTH: int = 30





ALLEN_CAHN_OPERATORS: list[str] = [
    "add", "sub", "mul", "div", "n2", "n3", "diff2_x", "diff2_y",
]
STATE_VARS: list[str] = ["u"]
COORD_VARS: list[str] = ["x", "y", "t"]




GROUND_TRUTH_IR: str = "add(add(diff2_x(u),diff2_y(u)),sub(u,n3(u)))"



GROUND_TRUTH_TOKENS: list[str] = [
    "add", "add", "diff2_x", "u", "diff2_y", "u", "sub", "u", "n3", "u",
]



TWO_D_DIFF_TOKENS: tuple[str, ...] = ("diff2_x", "diff2_y")







@pytest.fixture(scope="module")
def allen_cahn_components() -> PlatformBundle:
    from kd.core.evaluator import Evaluator
    from kd.core.executor.context import (
        ExecutionContext,
    )
    from kd.core.expr import (
        FunctionRegistry,
        PythonExecutor,
    )
    from kd.core.linear_solve.least_squares import (
        LeastSquaresSolver,
    )
    from kd.data.derivatives.finite_diff import (
        FiniteDiffProvider,
    )
    from kd.search.discover.data.allen_cahn_2d import load_allen_cahn_2d

    dataset = load_allen_cahn_2d(_DATA_PATH)
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(
        dataset=dataset, derivative_provider=provider,
    )
    registry = FunctionRegistry.create_default()
    u_t = provider.get_derivative(
        dataset.lhs_field, dataset.lhs_axis, order=1,
    ).flatten()
    evaluator = Evaluator(
        PythonExecutor(registry),
        LeastSquaresSolver(),
        context,
        lhs=u_t,
    )
    return PlatformBundle(dataset=dataset, evaluator=evaluator)


@pytest.fixture
def smoke_config() -> DiscoverConfig:
    from kd.search.discover.config import DiscoverConfig
    from kd.search.discover.tokens.library import LibraryConfig

    return DiscoverConfig(
        n_iterations=SMOKE_ITERATIONS,
        batch_size=SMOKE_BATCH_SIZE,
        max_length=SMOKE_MAX_LENGTH,
        library=LibraryConfig(
            operators=ALLEN_CAHN_OPERATORS,
            state_vars=STATE_VARS,
            coord_vars=COORD_VARS,
        ),
        num_units=16,
        num_layers=1,
        embedding_dim=4,
        epsilon=0.05,
        entropy_weight=0.005,




        stability_selection=1,
        stability_queue_capacity=100,
    )







class PlatformBundle:

    __slots__ = ("dataset", "evaluator")

    def __init__(self, dataset: object, evaluator: Evaluator) -> None:
        self.dataset = dataset
        self.evaluator = evaluator







class TestMode1AllenCahn2dSmoke:

    @pytest.mark.smoke
    def test_engine_completes_without_raising(
        self,
        allen_cahn_components: PlatformBundle,
        smoke_config: DiscoverConfig,
    ) -> None:
        from kd.search.discover.builder import build_engine

        torch.manual_seed(SEED)
        engine = build_engine(smoke_config)



        engine.run(
            allen_cahn_components.evaluator,
            n_iterations=SMOKE_ITERATIONS,
        )

    @pytest.mark.smoke
    def test_best_reward_positive(
        self,
        allen_cahn_components: PlatformBundle,
        smoke_config: DiscoverConfig,
    ) -> None:
        from kd.search.discover.builder import build_engine

        torch.manual_seed(SEED)
        engine = build_engine(smoke_config)
        state = engine.run(
            allen_cahn_components.evaluator,
            n_iterations=SMOKE_ITERATIONS,
        )
        assert state.best_reward > 0.0, (
            f"best_reward={state.best_reward} is not positive after "
            f"{SMOKE_ITERATIONS} iterations — no valid candidate found"
        )

    @pytest.mark.smoke
    def test_best_expression_contains_2d_operator(
        self,
        allen_cahn_components: PlatformBundle,
        smoke_config: DiscoverConfig,
    ) -> None:
        from kd.search.discover.builder import build_engine

        torch.manual_seed(SEED)
        engine = build_engine(smoke_config)
        state = engine.run(
            allen_cahn_components.evaluator,
            n_iterations=SMOKE_ITERATIONS,
        )
        assert state.best_expression, (
            "best_expression is empty — controller produced no valid "
            f"candidate after {SMOKE_ITERATIONS} iterations"
        )
        expressions = {state.best_expression}
        expressions.update(
            snapshot.expression for snapshot in engine.cycle_top_candidates
        )
        two_d_calls = tuple(f"{token}(" for token in TWO_D_DIFF_TOKENS)
        assert any(
            any(call in expr for call in two_d_calls) for expr in expressions
        ), (
            f"no valid candidate across {len(expressions)} expressions "
            f"(best + cycle hall-of-fame) used a 2D diff operator call "
            f"from {two_d_calls}; controller may be stuck on a 1D "
            f"shadow of the space. Sampled expressions: {sorted(expressions)}"
        )

    @pytest.mark.smoke
    def test_ground_truth_expression_is_reachable(
        self,
        allen_cahn_components: PlatformBundle,
        smoke_config: DiscoverConfig,
    ) -> None:
        from kd.search.discover.builder import build_library, build_prior_system
        from kd.search.discover.controller.tree_state import BatchTracker
        from kd.search.discover.tokens.validator import CandidateValidator

        library = build_library(smoke_config)
        prior_system = build_prior_system(library, smoke_config)
        validator = CandidateValidator(
            library,
            max_length=smoke_config.max_length,
            max_diff_order=smoke_config.max_diff_order,
            min_length=smoke_config.min_length,
        )

        token_ids = np.asarray(
            [library.name_to_index(name) for name in GROUND_TRUTH_TOKENS],
            dtype=np.int32,
        )


        assert validator.validate_single(token_ids), (
            f"CandidateValidator rejected ground-truth preorder "
            f"{GROUND_TRUTH_TOKENS}; check max_length / max_diff_order / "
            f"min_length config"
        )





        actions = token_ids[np.newaxis,:]
        obs = BatchTracker(library).compute_obs(actions)



        priors = prior_system.compute_batch(actions, obs)

        for step_idx, tok_id in enumerate(token_ids.tolist()):
            logit_adjust = float(priors[0, step_idx, int(tok_id)])
            assert np.isfinite(logit_adjust), (
                f"PriorSystem forbids ground-truth token "
                f"'{GROUND_TRUTH_TOKENS[step_idx]}' at step {step_idx} "
                f"(logit adjustment = -inf); the ground-truth expression "
                f"is unreachable under the current prior configuration"
            )
