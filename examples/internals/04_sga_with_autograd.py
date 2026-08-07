#!/usr/bin/env python3
"""Example 04: SGA with NN + autograd derivative mode.

What this demonstrates
----------------------
1. Run SGA twice on the SAME noisy 1D Burgers data:
   - Once with default FD derivatives.
   - Once with ``use_autograd=True`` — kd auto-trains a ``FieldModel``
     surrogate and routes the SGA terminal derivatives (``u_x``, ``u_t``)
     through ``AutogradProvider``.
2. Compare best AIC, MSE, and recovered coefficients side by side.

Why bother
----------
On clean analytic data, FD derivatives are usually sufficient. For
noisy data (instrumental error, simulation jitter, low resolution)
finite differences amplify noise — a smooth NN surrogate can produce
cleaner derivatives, and SGA discovery becomes more stable.

Where the autograd switch applies
---------------------------------
SGA touches derivatives in three places:

- raw ``u`` leaf: always raw dataset values
- terminal ``u_x``/``u_t``: FD provider OR AutogradProvider (this switch)
- tree-internal ``d``/``d^2``: always SGA-internal finite-difference kernel

So ``use_autograd=True`` only swaps the terminal layer. Tree-internal
``d`` operators still go through finite differences — the autograd
surrogate is only used to clean up the derivatives that feed into the
linear regression step.

Expected outcome
----------------
On the noisy Burgers below (noise_level=0.05), use_autograd=True *may*
produce a slightly cleaner expression with lower MSE than FD alone.
Whether AD wins depends strongly on noise level, FieldModel training
budget (``autograd_train_epochs``), and learning rate. ``autograd_train_epochs``
is an HONEST fixed-step budget: the surrogate trains exactly that many epochs
(no silent early stop), so runtime scales linearly with it. With the demo's
800-epoch training, AD does not always beat FD on a single seed — try
larger noise (e.g. 0.1), more epochs (3000+), or a few seeds before
concluding. Training overhead scales with data size AND epoch budget; for the
demo's 96x48 grid + 800 epochs it is typically under 5s, but a 256x128 grid
+ 5000 epochs can add 30-90s. (To stop early once the surrogate plateaus, set
``autograd_train_patience`` + ``autograd_train_val_ratio``.)

Usage
-----
    python examples/internals/04_sga_with_autograd.py
"""

from __future__ import annotations

import logging
import time

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.synthetic import generate_burgers_data
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("examples.04")
logger.setLevel(logging.INFO)

NX, NT = 96, 48
NU = 0.1
NOISE_LEVEL = 0.05
SEED = 42
GENERATIONS = 60
POPULATION = 12


def build_components(dataset) -> PlatformComponents:
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


def run_one(label: str, use_autograd: bool, dataset) -> dict:
    """Run a single SGA pass and return summary metrics."""
    components = build_components(dataset)
    plugin = SGAPlugin(
        SGAConfig(
            num=POPULATION,
            depth=4,
            width=5,
            seed=SEED,
            use_autograd=use_autograd,
            autograd_train_epochs=800, # quick training for demo
            autograd_train_lr=1e-3,
        )
    )
    runner = ExperimentRunner(
        algorithm=plugin, max_iterations=GENERATIONS, batch_size=POPULATION
    )
    t0 = time.perf_counter()
    result = runner.run(components)
    elapsed = time.perf_counter() - t0
    final = result.final_eval
    return {
        "label": label,
        "use_autograd": use_autograd,
        "best_expression": result.best_expression,
        "best_aic": result.best_score,
        "mse": final.mse,
        "r2": final.r2,
        "coefficients": (
            [round(c, 4) for c in final.coefficients.tolist()]
            if final.coefficients is not None
            else None
        ),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    logger.info("=== kd SGA: FD vs use_autograd on noisy Burgers ===")
    logger.info(
        "Generating noisy Burgers (nx=%d, nt=%d, nu=%.2f, noise=%.2f)",
        NX,
        NT,
        NU,
        NOISE_LEVEL,
    )
    dataset = generate_burgers_data(
        nx=NX, nt=NT, nu=NU, noise_level=NOISE_LEVEL, seed=SEED
    )
    logger.info("Ground truth: %s", dataset.ground_truth)

    logger.info("--- Pass 1: FD provider (default) ---")
    fd_summary = run_one("FD", use_autograd=False, dataset=dataset)
    logger.info("FD result: %s", fd_summary)

    logger.info("--- Pass 2: use_autograd=True (auto-train FieldModel surrogate) ---")
    ad_summary = run_one("AD", use_autograd=True, dataset=dataset)
    logger.info("AD result: %s", ad_summary)

    logger.info("--- Comparison ---")
    logger.info(
        "%-3s | AIC %10s | MSE %10s | elapsed %s | expression",
        "mode",
        "",
        "",
        "(s)",
    )
    for s in (fd_summary, ad_summary):
        logger.info(
            "%-3s | %14.4f | %14.6e | %7.1f | %s",
            s["label"],
            s["best_aic"],
            s["mse"],
            s["elapsed_s"],
            s["best_expression"][:80],
        )
        logger.info(" coefficients: %s", s["coefficients"])
    logger.info(
        "AD trades ~%.0fs of training overhead for derivative smoothing.",
        ad_summary["elapsed_s"] - fd_summary["elapsed_s"],
    )
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
