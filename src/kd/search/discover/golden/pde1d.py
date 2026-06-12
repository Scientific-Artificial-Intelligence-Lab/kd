
from __future__ import annotations

from pathlib import Path
from typing import Any

from kd.search.discover.golden.constants import (
    BATCH_SIZE_1D,
    BURGERS_OPERATORS,
    CHAFEE_ENTROPY_GAMMA,
    CHAFEE_OPERATORS,
    DEFAULT_BURGERS_DATA,
    DEFAULT_CHAFEE_DATA_DIR,
    DEFAULT_REWARD_ALPHA,
    EMBEDDING_DIM,
    ENTROPY_WEIGHT,
    EPSILON,
    GAMMA,
    MAX_LENGTH_1D,
    MIN_LENGTH_1D,
    N_ITERATIONS_1D,
    NUM_LAYERS,
    NUM_UNITS,
    project_relative,
    resolve_project_path,
)
from kd.search.discover.golden.summarise import (
    GoldenRunResult,
    run_engine_and_summarise,
    seed_all,
)


def run_burgers_mode1(
    seed: int,
    *,
    data_path: Path | None,
) -> tuple[GoldenRunResult, dict[str, Any]]:
    from kd.search.discover.data.loader import load_burgers_mat

    resolved = _resolve_data_path(data_path, DEFAULT_BURGERS_DATA)
    if not resolved.exists():
        raise FileNotFoundError(f"Burgers data not found: {resolved}")
    dataset = load_burgers_mat(resolved)
    return _run_1d_pde(
        pde="burgers",
        seed=seed,
        dataset=dataset,
        operators=list(BURGERS_OPERATORS),
        entropy_gamma=1.0,
        data_path=resolved,
    )


def run_chafee_mode1(
    seed: int,
    *,
    data_path: Path | None,
) -> tuple[GoldenRunResult, dict[str, Any]]:
    from kd.search.discover.data.loader import load_chafee_infante_npy

    resolved = _resolve_data_path(data_path, DEFAULT_CHAFEE_DATA_DIR)
    if not resolved.exists():
        raise FileNotFoundError(f"Chafee data dir not found: {resolved}")
    dataset = load_chafee_infante_npy(resolved)
    return _run_1d_pde(
        pde="chafee",
        seed=seed,
        dataset=dataset,
        operators=list(CHAFEE_OPERATORS),
        entropy_gamma=CHAFEE_ENTROPY_GAMMA,
        data_path=resolved,
    )


def _run_1d_pde(
    *,
    pde: str,
    seed: int,
    dataset: Any,
    operators: list[str],
    entropy_gamma: float,
    data_path: Path,
) -> tuple[GoldenRunResult, dict[str, Any]]:
    from kd.search.discover.config import DiscoverConfig
    from kd.search.discover.tokens.library import LibraryConfig

    config = DiscoverConfig(
        n_iterations=N_ITERATIONS_1D,
        batch_size=BATCH_SIZE_1D,
        max_length=MAX_LENGTH_1D,
        min_length=MIN_LENGTH_1D,
        library=LibraryConfig(
            operators=operators,
            state_vars=["u"],
            coord_vars=["x", "t"],
        ),
        num_units=NUM_UNITS,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        epsilon=EPSILON,
        entropy_weight=ENTROPY_WEIGHT,
        gamma=GAMMA,
        reward_alpha=DEFAULT_REWARD_ALPHA,
    )

    evaluator = _build_1d_evaluator(dataset)
    seed_all(seed)
    engine = _build_1d_engine(config=config, entropy_gamma=entropy_gamma)
    result = run_engine_and_summarise(
        engine=engine,
        evaluator=evaluator,
        n_iterations=config.n_iterations,
    )
    config_dict = _build_1d_config_dict(
        pde=pde,
        seed=seed,
        operators=operators,
        config=config,
        entropy_gamma=entropy_gamma,
        data_path=data_path,
    )
    return result, config_dict


def _build_1d_evaluator(dataset: Any) -> Any:
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

    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    u_t = provider.get_derivative("u", "t", order=1).flatten()
    return Evaluator(
        executor=PythonExecutor(registry),
        solver=LeastSquaresSolver(),
        context=context,
        lhs=u_t,
    )


def _build_1d_engine(*, config: Any, entropy_gamma: float) -> Any:
    from kd.search.discover.builder import (
        _make_magnitude_filter,
        _make_reward_adapter,
    )
    from kd.search.discover.engine import DiscoverEngine
    from kd.search.discover.evaluation.dedup import Deduplicator
    from kd.search.discover.tokens.library import Library
    from kd.search.discover.tokens.validator import CandidateValidator

    library = Library.from_config(config.library)
    prior_system = _build_1d_prior(library, config)
    controller = _build_1d_controller(library, prior_system, config)
    strategy = _build_1d_strategy(config, entropy_gamma)
    validator = CandidateValidator(
        library,
        max_length=config.max_length,
        min_length=config.min_length,
    )
    return DiscoverEngine(
        generator=controller,
        strategy=strategy,
        reward_adapter=_make_reward_adapter(config.reward_alpha),
        result_filter=_make_magnitude_filter(enabled=config.magnitude_filter),
        validator=validator,
        deduplicator=Deduplicator(library),
        batch_size=config.batch_size,
    )


def _build_1d_prior(library: Any, config: Any) -> Any:
    from kd.search.discover.tokens.prior import (
        DiffChildConstraint,
        LengthConstraint,
        PriorSystem,
    )

    return PriorSystem(
        library,
        [
            LengthConstraint(library, min_=config.min_length, max_=config.max_length),
            DiffChildConstraint(library),
        ],
    )


def _build_1d_controller(library: Any, prior_system: Any, config: Any) -> Any:
    from kd.search.discover.controller.lstm import LSTMController

    return LSTMController(
        library=library,
        prior_system=prior_system,
        num_units=config.num_units,
        num_layers=config.num_layers,
        embedding_dim=config.embedding_dim,
    )


def _build_1d_strategy(config: Any, entropy_gamma: float) -> Any:
    from kd.search.discover.training.strategy import RSPGStrategy

    return RSPGStrategy(
        epsilon=config.epsilon,
        baseline=config.baseline,
        entropy_weight=config.entropy_weight,
        gamma=config.gamma,
        entropy_gamma=entropy_gamma,
    )


def _build_1d_config_dict(
    *,
    pde: str,
    seed: int,
    operators: list[str],
    config: Any,
    entropy_gamma: float,
    data_path: Path,
) -> dict[str, Any]:
    return {
        "pde": pde,
        "mode": "mode1",
        "seed": seed,
        "noise_level": 0.0,
        "tier": "comparison",
        "n_iterations": config.n_iterations,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "min_length": config.min_length,
        "operators": list(operators),
        "state_vars": list(config.library.state_vars),
        "coord_vars": list(config.library.coord_vars),
        "epsilon": EPSILON,
        "entropy_weight": ENTROPY_WEIGHT,
        "entropy_gamma": entropy_gamma,
        "gamma": GAMMA,
        "num_units": NUM_UNITS,
        "num_layers": NUM_LAYERS,
        "embedding_dim": EMBEDDING_DIM,
        "reward_alpha": DEFAULT_REWARD_ALPHA,
        "data_path": project_relative(data_path),
    }


def _resolve_data_path(data_path: Path | None, default: Path) -> Path:
    if data_path is None:
        return default
    return resolve_project_path(data_path)


__all__ = ["run_burgers_mode1", "run_chafee_mode1"]
