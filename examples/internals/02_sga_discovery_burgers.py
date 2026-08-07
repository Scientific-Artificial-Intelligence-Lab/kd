#!/usr/bin/env python3
"""Example 02: End-to-end SGA discovery on 1D Burgers + visualization.

What this demonstrates
----------------------
1. Build a `PDEDataset` (synthetic Burgers data with known ground truth).
2. Wire `PlatformComponents` (executor, evaluator, derivative provider).
3. Run `SGAPlugin` through `ExperimentRunner` for N generations.
4. Inspect `ExperimentResult`: best expression, AIC, selected coefficients.
5. Render a full HTML report via `VizEngine` (5+ figures, auto-saved).

Equation
--------
Burgers: u_t = -u * u_x + nu * u_xx (here nu = 0.1)

Expected outcome
----------------
On a clean 128x64 grid with 80 generations and population=15, SGA typically
finds a best expression containing both ``u * u_x`` (advection) and
``u_xx`` (diffusion) with effective coefficients close to (-1, +0.1).
Best AIC typically reaches negative double digits (concrete value depends
on seed, ``aic_ratio``, and tree size). Run-to-run variance is normal —
try a few seeds before drawing conclusions.

Usage
-----
    python examples/internals/02_sga_discovery_burgers.py
    open examples/out/burgers/report.html
"""

from __future__ import annotations

import logging
from pathlib import Path

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.synthetic import generate_burgers_data
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin
from kd.viz.engine import VizEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("examples.02")
logger.setLevel(logging.INFO)

# Tunables (all small enough for ~1 minute on CPU)
NX, NT = 128, 64
NU = 0.1
SEED = 42
GENERATIONS = 80
POPULATION = 15
OUT_DIR = Path(__file__).parent / "out" / "burgers"


def build_components(dataset) -> PlatformComponents:
    """Wire up the standard FD-based platform stack for a PDE dataset."""
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()
    u_t = provider.get_derivative("u", "t", 1).flatten()
    evaluator = Evaluator(executor=executor, solver=solver, context=context, lhs=u_t)
    return PlatformComponents(
        dataset=dataset,
        executor=executor,
        evaluator=evaluator,
        context=context,
        registry=registry,
    )


def main() -> None:
    logger.info("=== kd SGA discovery: Burgers 1D ===")
    logger.info("Generating dataset (nx=%d, nt=%d, nu=%.2f)", NX, NT, NU)
    dataset = generate_burgers_data(nx=NX, nt=NT, nu=NU, seed=SEED)
    logger.info("Ground truth: %s", dataset.ground_truth)

    components = build_components(dataset)
    plugin = SGAPlugin(
        SGAConfig(
            num=POPULATION,
            depth=4,
            width=5,
            seed=SEED,
        )
    )
    runner = ExperimentRunner(
        algorithm=plugin,
        max_iterations=GENERATIONS,
        batch_size=POPULATION,
    )

    logger.info(
        "Running SGA: pop=%d, generations=%d, seed=%d",
        POPULATION,
        GENERATIONS,
        SEED,
    )
    result = runner.run(components)

    # ---- Inspect result ---------------------------------------------------
    logger.info("--- Result ---")
    logger.info("Best expression: %s", result.best_expression)
    logger.info("Best AIC: %.4f", result.best_score)
    logger.info("Iterations: %d", result.iterations)
    logger.info("Early stopped: %s", result.early_stopped)

    final = result.final_eval
    logger.info("MSE: %.6e", final.mse)
    if final.r2 is not None:
        logger.info("R²: %.6f", final.r2)
    if final.coefficients is not None:
        coeffs = final.coefficients.tolist()
        logger.info("Coefficients: %s", [f"{c:+.4f}" for c in coeffs])
    if final.selected_indices is not None:
        logger.info("Selected idx: %s", final.selected_indices)

    # ---- Render visualization ---------------------------------------------
    logger.info("Rendering HTML report to %s", OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    viz = VizEngine(output_dir=OUT_DIR)
    report = viz.render_all(result, algorithm=plugin, dataset=dataset)
    logger.info("HTML report: %s", report.report)
    logger.info(
        "Figures (%d): %s", len(report.figures), [f.name for f in report.figures]
    )
    if report.warnings:
        logger.warning("Viz warnings: %s", report.warnings)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
