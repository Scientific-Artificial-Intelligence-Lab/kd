
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

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

if TYPE_CHECKING:
    from kd.search.discover.config import DiscoverConfig

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_PATH: Path = (
    _PROJECT_ROOT
    / "refs" / "discover" / "dso" / "dso" / "task" / "pde"
    / "data_new" / "PDE_divide.npy"
)


_MIN_DATA_BYTES: int = 50_000
_SKIP_REASON = (
    f"PDE_divide paper data not found (or under {_MIN_DATA_BYTES} "
    f"bytes) at {_DATA_PATH}."
)


def _data_ready() -> bool:
    return _DATA_PATH.exists() and _DATA_PATH.stat().st_size > _MIN_DATA_BYTES


pytestmark = pytest.mark.skipif(not _data_ready(), reason=_SKIP_REASON)






SEED: int = 42
SANITY_ITERATIONS: int = 20
SANITY_BATCH_SIZE: int = 500
SANITY_MAX_LENGTH: int = 30
SANITY_MIN_REWARD: float = 0.5



PDE_DIVIDE_OPERATORS: list[str] = [
    "add", "mul", "sub", "div", "n2", "n3", "diff_x", "diff2_x",
]
STATE_VARS: list[str] = ["u"]
COORD_VARS: list[str] = ["x", "t"]




GROUND_TRUTH_TOKENS: list[str] = [
    "add", "div", "diff_x", "u", "x", "diff2_x", "u",
]





GT_TOKEN_CALLS: tuple[str, ...] = ("div(", "diff2_x(")







class _PlatformBundle:
    __slots__ = ("dataset", "evaluator")

    def __init__(self, dataset: object, evaluator: Evaluator) -> None:
        self.dataset = dataset
        self.evaluator = evaluator


@pytest.fixture(scope="module")
def pde_divide_components() -> _PlatformBundle:
    from kd.search.discover.data.loader import load_pde_divide_npy

    dataset = load_pde_divide_npy(_DATA_PATH)
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
    return _PlatformBundle(dataset=dataset, evaluator=evaluator)


@pytest.fixture
def sanity_config() -> DiscoverConfig:
    from kd.search.discover.config import DiscoverConfig
    from kd.search.discover.tokens.library import LibraryConfig

    return DiscoverConfig(
        n_iterations=SANITY_ITERATIONS,
        batch_size=SANITY_BATCH_SIZE,
        max_length=SANITY_MAX_LENGTH,
        library=LibraryConfig(
            operators=PDE_DIVIDE_OPERATORS,
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







@pytest.mark.smoke
def test_engine_completes_without_raising(
    pde_divide_components: _PlatformBundle,
    sanity_config: DiscoverConfig,
) -> None:
    from kd.search.discover.builder import build_engine

    torch.manual_seed(SEED)
    engine = build_engine(sanity_config)
    engine.run(
        pde_divide_components.evaluator,
        n_iterations=SANITY_ITERATIONS,
    )


@pytest.mark.smoke
def test_best_reward_above_sanity_threshold(
    pde_divide_components: _PlatformBundle,
    sanity_config: DiscoverConfig,
) -> None:
    from kd.search.discover.builder import build_engine

    torch.manual_seed(SEED)
    engine = build_engine(sanity_config)
    state = engine.run(
        pde_divide_components.evaluator,
        n_iterations=SANITY_ITERATIONS,
    )
    assert state.best_reward > SANITY_MIN_REWARD, (
        f"best_reward={state.best_reward:.4f} <= sanity threshold "
        f"{SANITY_MIN_REWARD}; pipeline either does not converge on "
        f"PDE_divide at 20 iter or the reward calc is broken."
    )


@pytest.mark.smoke
def test_controller_explores_ground_truth_tokens(
    pde_divide_components: _PlatformBundle,
    sanity_config: DiscoverConfig,
) -> None:
    from kd.search.discover.builder import build_engine

    torch.manual_seed(SEED)
    engine = build_engine(sanity_config)
    state = engine.run(
        pde_divide_components.evaluator,
        n_iterations=SANITY_ITERATIONS,
    )
    assert state.best_expression, (
        "best_expression empty — controller produced no valid candidate."
    )
    expressions = {state.best_expression}
    expressions.update(
        snapshot.expression for snapshot in engine.cycle_top_candidates
    )
    assert any(
        any(call in expr for call in GT_TOKEN_CALLS) for expr in expressions
    ), (
        f"No candidate across {len(expressions)} expressions used a "
        f"PDE_divide GT-relevant call from {GT_TOKEN_CALLS}; sample: "
        f"{sorted(expressions)[:5]}"
    )


@pytest.mark.unit
def test_ground_truth_expression_is_reachable(
    sanity_config: DiscoverConfig,
) -> None:
    from kd.search.discover.builder import build_library, build_prior_system
    from kd.search.discover.controller.tree_state import BatchTracker
    from kd.search.discover.tokens.validator import CandidateValidator

    library = build_library(sanity_config)
    prior_system = build_prior_system(library, sanity_config)
    validator = CandidateValidator(
        library,
        max_length=sanity_config.max_length,
        max_diff_order=sanity_config.max_diff_order,
        min_length=sanity_config.min_length,
    )

    token_ids = np.asarray(
        [library.name_to_index(name) for name in GROUND_TRUTH_TOKENS],
        dtype=np.int32,
    )
    assert validator.validate_single(token_ids), (
        f"CandidateValidator rejected PDE_divide GT preorder "
        f"{GROUND_TRUTH_TOKENS}."
    )
    actions = token_ids[np.newaxis,:]
    obs = BatchTracker(library).compute_obs(actions)
    priors = prior_system.compute_batch(actions, obs)
    for step_idx, tok_id in enumerate(token_ids.tolist()):
        logit_adjust = float(priors[0, step_idx, int(tok_id)])
        assert np.isfinite(logit_adjust), (
            f"PriorSystem forbids GT token "
            f"'{GROUND_TRUTH_TOKENS[step_idx]}' at step {step_idx}."
        )
