
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from kd.core.evaluator import (
    EvaluationResult,
    Evaluator,
)
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
from kd.data.schema import PDEDataset
from kd.search.discover.builder import build_engine
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.data.allen_cahn_2d import load_allen_cahn_2d
from kd.search.discover.runners.expression_check import (
    check_diffusion_sign_consistency,
)
from kd.search.discover.runners.sampled_evaluator import SampledEvaluator
from kd.search.discover.tokens.library import LibraryConfig

DEFAULT_DATA_PATH = Path("data/allen_cahn_2d_paper.npz")
MAX_DIFF_ORDER = 2
GROUND_TRUTH_TERMS: tuple[str, ...] = (
    "diff2_x(u)",
    "diff2_y(u)",
    "u",
    "n3(u)",
)
TERM_LABELS: tuple[str, ...] = ("diff2_x", "diff2_y", "u", "n3_u")
GROUND_TRUTH_COEFFS: tuple[float, ...] = (0.001, 0.001, 1.0, -1.0)
REL_ERROR_DENOM_FLOOR = 1e-30
L1_RATIO_DENOM_FLOOR = 1e-30
ALLEN_CAHN_OPERATORS = [
    "add",
    "sub",
    "mul",
    "div",
    "n2",
    "n3",
    "diff2_x",
    "diff2_y",
]
LOG_INTERVAL = 10

V11_EXTENSION_BOUNDARY_PASS = 1
V11_EXTENSION_BOUNDARY_TOTAL = 3
V11_EXTENSION_COUNT = 2




TIME_SLICE_INTERIOR_MARGIN = 2
TIME_SLICE_N_SHORT = 3
MID_3_T_INDICES = (48, 49, 50)
LATE_3_T_INDICES = (96, 97, 98)
SPREAD_3_T_INDICES = (10, 50, 90)

logger = logging.getLogger("multiseed")







def compute_pass_rate(
    per_seed_results: list[dict[str, Any]],
    threshold: float,
) -> tuple[int, int]:
    n_total = len(per_seed_results)
    n_pass = sum(
        1
        for r in per_seed_results
        if r["ground_truth_fit"]["max_rel_coef_error"] <= threshold
    )
    return (n_pass, n_total)


def compute_structural_pass_rate(
    per_seed_results: list[dict[str, Any]],
) -> tuple[int, int]:
    n_total = len(per_seed_results)
    n_pass = 0
    for result in per_seed_results:
        mode1_run = result.get("mode1_run") or {}
        expression = mode1_run.get("best_expression")
        if not isinstance(expression, str) or not expression.strip():
            continue
        ok, _reason = check_diffusion_sign_consistency(expression)
        if ok:
            n_pass += 1
    return (n_pass, n_total)


def compute_combined_pass_rate(
    per_seed_results: list[dict[str, Any]],
    threshold: float,
) -> tuple[int, int]:
    n_total = len(per_seed_results)
    n_pass = 0
    for result in per_seed_results:

        coef_err = result.get("ground_truth_fit", {}).get("max_rel_coef_error")
        if coef_err is None or coef_err > threshold:
            continue

        mode1_run = result.get("mode1_run") or {}
        expression = mode1_run.get("best_expression")
        if not isinstance(expression, str) or not expression.strip():
            continue
        ok, _reason = check_diffusion_sign_consistency(expression)
        if not ok:
            continue
        n_pass += 1
    return (n_pass, n_total)


@dataclass(frozen=True, slots=True)
class ReleaseVerdict:

    passed: bool
    n_pass: int
    n_total: int
    strict: bool


def compute_release_decision(
    per_seed_results: list[dict[str, Any]],
    threshold: float,
    *,
    min_pass_rate: float,
    strict: bool = True,
) -> tuple[bool, int, int]:
    if strict:
        n_pass, n_total = compute_combined_pass_rate(per_seed_results, threshold)
    else:
        n_pass, n_total = compute_pass_rate(per_seed_results, threshold)
    if n_total == 0:
        return (False, 0, 0)
    passed = (n_pass / n_total) >= min_pass_rate
    return (passed, n_pass, n_total)


def should_extend_v11(n_pass: int, n_total: int) -> bool:
    return (
        n_pass == V11_EXTENSION_BOUNDARY_PASS
        and n_total == V11_EXTENSION_BOUNDARY_TOTAL
    )


def extension_seeds(initial: list[int]) -> list[int]:
    if not initial:
        raise ValueError("initial seeds list cannot be empty")
    base = max(initial) + 1
    return [base + i for i in range(V11_EXTENSION_COUNT)]


def aggregate_metrics(per_seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_seed_results:
        raise ValueError("per_seed_results cannot be empty")
    max_rels = [
        float(r["ground_truth_fit"]["max_rel_coef_error"]) for r in per_seed_results
    ]
    l1_ratios = [
        float(r["ground_truth_fit"]["l1_ratio_error"]) for r in per_seed_results
    ]
    return {
        "max_rel": _summary(max_rels),
        "l1_ratio": _summary(l1_ratios),
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }







def _resolve_t_keep(time_slice: str, nt: int) -> list[int]:
    if time_slice == "all":
        return list(range(nt))
    if time_slice == "first-3":
        return [0, 1, 2]
    if time_slice == "middle-3":
        return [2, 3, 4]
    if time_slice == "interior-3":
        lo, hi = TIME_SLICE_INTERIOR_MARGIN, nt - (TIME_SLICE_INTERIOR_MARGIN + 1)
        if hi - lo + 1 < TIME_SLICE_N_SHORT:
            min_nt = 2 * TIME_SLICE_INTERIOR_MARGIN + TIME_SLICE_N_SHORT
            raise ValueError(f"interior-3 requires nt >= {min_nt}; got nt={nt}")




        step = (hi - lo) / (TIME_SLICE_N_SHORT - 1)
        return [int(lo + i * step) for i in range(TIME_SLICE_N_SHORT)]
    if time_slice == "mid-3":
        return list(MID_3_T_INDICES)
    if time_slice == "late-3":
        return list(LATE_3_T_INDICES)
    if time_slice == "spread-3":
        return list(SPREAD_3_T_INDICES)
    raise ValueError(f"unknown time_slice: {time_slice!r}")


def time_slice_indices(
    time_slice: str,
    *,
    nx: int,
    ny: int,
    nt: int,
) -> Tensor:
    t_keep = _resolve_t_keep(time_slice, nt)

    for t in t_keep:
        if t < 0 or t >= nt:
            raise ValueError(f"time_slice {time_slice!r} requests t={t} but nt={nt}")


    x_range = torch.arange(nx, dtype=torch.int64)
    y_range = torch.arange(ny, dtype=torch.int64)
    t_tensor = torch.tensor(t_keep, dtype=torch.int64)

    x_b = x_range.view(-1, 1, 1)
    y_b = y_range.view(1, -1, 1)
    t_b = t_tensor.view(1, 1, -1)
    flat = (x_b * (ny * nt) + y_b * nt + t_b).flatten()


    return torch.sort(flat).values







def run_single_seed(
    seed: int,
    *,
    n_iterations: int,
    n_points: int,
    batch_size: int,
    data_path: Path | str = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    dataset = load_allen_cahn_2d(data_path, dtype=torch.float64)
    provider = FiniteDiffProvider(dataset, max_order=MAX_DIFF_ORDER)
    evaluator = _build_base_evaluator(dataset, provider)
    sample_indices = _sample_indices(seed, evaluator.lhs_target.shape[0], n_points)
    sampled_evaluator = SampledEvaluator(evaluator, sample_indices, rank_check=True)
    config = _multiseed_config(n_iterations, batch_size)

    engine_result = _run_mode1(seed, config, sampled_evaluator)
    fit_result = sampled_evaluator.evaluate_terms(list(GROUND_TRUTH_TERMS))
    if not fit_result.is_valid or fit_result.coefficients is None:
        raise RuntimeError(f"Ground-truth fit failed: {fit_result.error_message}")

    return {
        "seed": seed,
        "data_path": str(data_path),
        "n_points": n_points,
        "n_iterations": n_iterations,
        "ground_truth_fit": _fit_summary(fit_result),
        "mode1_run": engine_result,
    }


def run_single_seed_time_sliced(
    seed: int,
    *,
    n_iterations: int,
    batch_size: int,
    time_slice: str,
    data_path: Path | str = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    dataset = load_allen_cahn_2d(data_path, dtype=torch.float64)
    field_shape = dataset.get_field(dataset.lhs_field).shape
    if len(field_shape) != 3:
        raise ValueError(
            f"Expected 3-D lhs field (Nx, Ny, Nt); got shape {tuple(field_shape)}"
        )
    nx, ny, nt = (int(s) for s in field_shape)

    provider = FiniteDiffProvider(dataset, max_order=MAX_DIFF_ORDER)
    evaluator = _build_base_evaluator(dataset, provider)
    sample_indices = time_slice_indices(time_slice, nx=nx, ny=ny, nt=nt)
    sampled_evaluator = SampledEvaluator(evaluator, sample_indices, rank_check=True)
    config = _multiseed_config(n_iterations, batch_size)

    engine_result = _run_mode1(seed, config, sampled_evaluator)
    fit_result = sampled_evaluator.evaluate_terms(list(GROUND_TRUTH_TERMS))
    if not fit_result.is_valid or fit_result.coefficients is None:
        raise RuntimeError(f"Ground-truth fit failed: {fit_result.error_message}")

    return {
        "seed": seed,
        "data_path": str(data_path),
        "time_slice": time_slice,
        "n_points": int(sample_indices.shape[0]),
        "n_iterations": n_iterations,
        "ground_truth_fit": _fit_summary(fit_result),
        "mode1_run": engine_result,
    }







def _build_base_evaluator(dataset: PDEDataset, provider: Any) -> Evaluator:
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    lhs = provider.get_derivative(
        dataset.lhs_field, dataset.lhs_axis, order=1
    ).flatten()
    return Evaluator(
        executor=PythonExecutor(registry),
        solver=LeastSquaresSolver(),
        context=context,
        lhs=lhs,
    )


def _sample_indices(seed: int, total_points: int, n_points: int) -> Tensor:
    if n_points > total_points:
        raise ValueError(f"n_points={n_points} exceeds total points={total_points}")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total_points, generator=generator)[:n_points]


def _multiseed_config(n_iterations: int, batch_size: int) -> DiscoverConfig:
    return DiscoverConfig.burgers_preset(
        n_iterations=n_iterations,
        batch_size=batch_size,
        library=LibraryConfig(
            operators=ALLEN_CAHN_OPERATORS,
            state_vars=["u"],
            coord_vars=["x", "y", "t"],
        ),
        max_diff_order=MAX_DIFF_ORDER,
    )


def _run_mode1(
    seed: int,
    config: DiscoverConfig,
    evaluator: SampledEvaluator,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    engine = build_engine(config)
    start = time.time()
    last_metrics: dict[str, float] = {}
    for iteration in range(config.n_iterations):
        last_metrics = engine.run_iteration(evaluator)
        if (iteration + 1) % LOG_INTERVAL == 0:
            logger.info(
                "seed=%d iter=%d best_reward=%.6f valid=%d unique=%d expr=%s",
                seed,
                iteration + 1,
                engine.best_reward,
                int(last_metrics.get("n_valid", 0.0)),
                int(last_metrics.get("n_unique", 0.0)),
                engine.best_expression,
            )
    state = engine.state
    return {
        "elapsed_seconds": time.time() - start,
        "best_reward": float(engine.best_reward),
        "best_expression": engine.best_expression,
        "best_terms": state.best_result_terms,
        "best_coefficients": state.best_result_coefficients,
        "last_metrics": last_metrics,
    }


def _fit_summary(result: EvaluationResult) -> dict[str, Any]:
    if result.coefficients is None:
        raise ValueError("fit result has no coefficients")
    coefficients = result.coefficients.detach().cpu().to(torch.float64)
    true = torch.tensor(GROUND_TRUTH_COEFFS, dtype=torch.float64)
    rel = (coefficients - true).abs() / true.abs().clamp_min(REL_ERROR_DENOM_FLOOR)
    l1_ratio = float(
        (coefficients - true).abs().sum().item()
        / max(float(true.abs().sum().item()), L1_RATIO_DENOM_FLOOR)
    )
    return {
        "terms": list(GROUND_TRUTH_TERMS),
        "coefficients": _label_mapping(coefficients),
        "ground_truth": dict(zip(TERM_LABELS, GROUND_TRUTH_COEFFS, strict=True)),
        "per_term_rel_error": _label_mapping(rel),
        "max_rel_coef_error": float(rel.max().item()),
        "l1_ratio_error": l1_ratio,
        "mse": result.mse,
        "nmse": result.nmse,
        "r2": result.r2,
    }


def _label_mapping(values: Tensor) -> dict[str, float]:
    return {
        label: float(value)
        for label, value in zip(
            TERM_LABELS,
            values.detach().cpu().tolist(),
            strict=True,
        )
    }
