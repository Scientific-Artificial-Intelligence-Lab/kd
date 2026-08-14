
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
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
from kd.search.discover.builder import build_engine
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.data.loader import load_burgers_mat
from kd.search.discover.paths import REFERENCE_DATA_DIR

logger = logging.getLogger(__name__)


_DEFAULT_DATA_PATH = REFERENCE_DATA_DIR / "burgers.mat"

SEED = 42
MAX_DERIV_ORDER = 2
DERIV_VAR = "u"
DERIV_COORD = "t"
DERIV_ORDER = 1
LOG_INTERVAL = 10


def _build_evaluator(data_path: Path) -> Evaluator:
    dataset = load_burgers_mat(data_path)
    provider = FiniteDiffProvider(dataset, max_order=MAX_DERIV_ORDER)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()
    u_t = provider.get_derivative(DERIV_VAR, DERIV_COORD, order=DERIV_ORDER).flatten()
    return Evaluator(
        executor=executor,
        solver=solver,
        context=context,
        lhs=u_t,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not _DEFAULT_DATA_PATH.exists():
        logger.error("Burgers data not found at %s", _DEFAULT_DATA_PATH)
        sys.exit(1)

    config = DiscoverConfig()
    logger.info(
        "Starting DISCOVER: %d iterations, batch=%d, max_length=%d",
        config.n_iterations,
        config.batch_size,
        config.max_length,
    )

    torch.manual_seed(SEED)
    evaluator = _build_evaluator(_DEFAULT_DATA_PATH)
    engine = build_engine(config)

    for i in range(config.n_iterations):
        metrics = engine.run_iteration(evaluator)
        if (i + 1) % LOG_INTERVAL == 0:
            logger.info(
                "iter %4d | reward_max=%.4f best=%.4f n_valid=%d n_unique=%d",
                i + 1,
                metrics.get("reward_max", 0.0),
                metrics.get("best_reward", 0.0),
                int(metrics.get("n_valid", 0)),
                int(metrics.get("n_unique", 0)),
            )

    logger.info("Best reward: %.6f", engine.best_reward)
    logger.info("Best expression: %s", engine.best_expression)


if __name__ == "__main__":
    main()
