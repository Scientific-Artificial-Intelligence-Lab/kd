

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin
from kd.viz.engine import VizEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("examples.03")
logger.setLevel(logging.INFO)

ALPHA = 0.1
NX, NY, NT = 32, 32, 15
SEED = 42
GENERATIONS = 80
POPULATION = 15
MODES: tuple[tuple[int, int, float], ...] = (
    (1, 2, 1.0),
    (3, 1, 1.0),
    (2, 3, 0.8),
)
OUT_DIR = Path(__file__).parent / "out" / "diffusion_2d"


def build_three_mode_diffusion_2d() -> PDEDataset:
    dtype = torch.float64
    x = torch.linspace(0.0, 2.0 * math.pi, NX + 1, dtype=dtype)[:-1]
    y = torch.linspace(0.0, 2.0 * math.pi, NY + 1, dtype=dtype)[:-1]
    t = torch.linspace(0.0, 1.0, NT, dtype=dtype)
    gx, gy, gt = torch.meshgrid(x, y, t, indexing="ij")
    u = torch.zeros_like(gx)
    for kx, ky, amp in MODES:
        decay = ALPHA * (kx * kx + ky * ky)
        u = u + amp * torch.sin(kx * gx) * torch.sin(ky * gy) * torch.exp(-decay * gt)
    return PDEDataset(
        name="diffusion-2d-three-mode",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "y": AxisInfo(name="y", values=y, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
        ground_truth=f"u_t = {ALPHA} * (u_xx + u_yy)",
    )


def build_components(dataset: PDEDataset) -> PlatformComponents:
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
    logger.info("=== kd SGA discovery: Diffusion 2D ===")
    dataset = build_three_mode_diffusion_2d()
    logger.info("Ground truth: %s", dataset.ground_truth)
    logger.info("Modes (kx, ky, amp): %s", MODES)


    provider = FiniteDiffProvider(dataset, max_order=2)
    u = dataset.fields["u"].values.flatten()
    u_t = provider.get_derivative("u", "t", 1).flatten()
    u_xx = provider.get_derivative("u", "x", 2).flatten()
    u_yy = provider.get_derivative("u", "y", 2).flatten()

    def fit_residual(cols: list[torch.Tensor]) -> float:
        theta = torch.stack(cols, dim=1)
        sol = torch.linalg.lstsq(theta, u_t.unsqueeze(1)).solution
        pred = (theta @ sol).squeeze()
        return float((pred - u_t).norm() / (u_t.norm() + 1e-30))

    cheat_x = fit_residual([u, u_xx])
    cheat_y = fit_residual([u, u_yy])
    logger.info("Cheat residual ||u_t - lstsq([u, u_xx])||/||u_t|| = %.4f", cheat_x)
    logger.info("Cheat residual ||u_t - lstsq([u, u_yy])||/||u_t|| = %.4f", cheat_y)



    assert cheat_x > 0.05, (
        f"Data degenerated: u + u_xx fits u_t with residual {cheat_x:.4f} "
        f"(<= 0.05). SGA could skip the y axis. Check MODES."
    )
    assert cheat_y > 0.05, (
        f"Data degenerated: u + u_yy fits u_t with residual {cheat_y:.4f} "
        f"(<= 0.05). SGA could skip the x axis. Check MODES."
    )
    logger.info("Both > 0.05 — single-axis shortcut blocked.")

    components = build_components(dataset)
    plugin = SGAPlugin(SGAConfig(num=POPULATION, depth=4, width=5, seed=SEED))
    runner = ExperimentRunner(
        algorithm=plugin, max_iterations=GENERATIONS, batch_size=POPULATION
    )

    logger.info(
        "Running SGA: pop=%d, generations=%d, seed=%d",
        POPULATION,
        GENERATIONS,
        SEED,
    )
    result = runner.run(components)

    logger.info("--- Result ---")
    logger.info("Best expression: %s", result.best_expression)
    logger.info("Best AIC: %.4f", result.best_score)
    final = result.final_eval
    logger.info("MSE: %.6e", final.mse)
    if final.r2 is not None:
        logger.info("R²: %.6f", final.r2)
    if final.coefficients is not None:
        coeffs = final.coefficients.tolist()
        logger.info("Coefficients: %s", [f"{c:+.4f}" for c in coeffs])
    if final.selected_indices is not None:
        logger.info("Selected idx: %s", final.selected_indices)
        n_pde = sum(1 for i in final.selected_indices if i >= 1)
        logger.info(
            "Active PDE terms: %d %s",
            n_pde,
            "(>=2 with both x and y axes = real 2D discovery)"
            if n_pde >= 2
            else "(only 1 — possible single-axis shortcut)",
        )

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
