

from __future__ import annotations

import logging
import math
from pathlib import Path

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.synthetic import generate_burgers_data
from kd.search.protocol import PlatformComponents
from kd.search.result import ExperimentResult
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin
from kd.viz.engine import VizEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("examples.05")
logger.setLevel(logging.INFO)

SEED = 42
GENERATIONS = 30
POPULATION = 10
SAVE_PATH = Path(__file__).parent / "out" / "burgers_result.json"
RELOAD_REPORT_DIR = Path(__file__).parent / "out" / "burgers_loaded"


def main() -> None:
    logger.info("=== kd: Save / Load ExperimentResult ===")

    dataset = generate_burgers_data(nx=96, nt=48, nu=0.1, seed=SEED)
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()
    u_t = provider.get_derivative("u", "t", 1).flatten()
    evaluator = Evaluator(executor=executor, solver=solver, context=context, lhs=u_t)
    components = PlatformComponents(
        dataset=dataset,
        executor=executor,
        evaluator=evaluator,
        context=context,
        registry=registry,
    )

    plugin = SGAPlugin(SGAConfig(num=POPULATION, depth=4, width=5, seed=SEED))
    runner = ExperimentRunner(
        algorithm=plugin, max_iterations=GENERATIONS, batch_size=POPULATION
    )

    logger.info("Running SGA (%d generations) ...", GENERATIONS)
    result = runner.run(components)
    logger.info("Best expression: %s", result.best_expression)
    logger.info("Best AIC: %.4f", result.best_score)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving result to %s", SAVE_PATH)
    result.save(SAVE_PATH)
    logger.info("Saved %d bytes", SAVE_PATH.stat().st_size)

    logger.info("Reloading from disk ...")
    loaded = ExperimentResult.load(SAVE_PATH)

    logger.info("--- Round-trip check ---")
    logger.info(
        "expression match: %s", loaded.best_expression == result.best_expression
    )


    aic_match = (
        math.isinf(loaded.best_score) and math.isinf(result.best_score)
    ) or abs(loaded.best_score - result.best_score) < 1e-9
    logger.info("aic match: %s", aic_match)
    logger.info("iterations match: %s", loaded.iterations == result.iterations)
    logger.info("dataset_name: %s", loaded.dataset_name)
    logger.info("algorithm_name: %s", loaded.algorithm_name)



    logger.info("Re-rendering HTML report from loaded result ...")
    RELOAD_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    viz = VizEngine(output_dir=RELOAD_REPORT_DIR)
    report = viz.render_all(loaded, algorithm=None, dataset=dataset)
    logger.info("HTML report: %s", report.report)
    logger.info(
        "Figures (%d): %s", len(report.figures), [f.name for f in report.figures]
    )
    if report.warnings:
        logger.warning("Viz warnings: %s", report.warnings)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
