
from __future__ import annotations

import argparse
import ast
import logging
import math
import time
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
from kd.search.discover.runners.multiseed import (
    ALLEN_CAHN_OPERATORS,
    GROUND_TRUTH_COEFFS,
    GROUND_TRUTH_TERMS,
    MAX_DIFF_ORDER,
    TERM_LABELS,
)
from kd.search.discover.runners.sampled_evaluator import SampledEvaluator
from kd.search.discover.tokens.library import LibraryConfig
from kd.search.discover.utils.math import safe_div

_LOGGER = logging.getLogger(__name__)





NOISE_LEVEL = 0.01
PRETRAIN_LOSS_GATE = 0.5
STRETCH_TARGET = 0.05

_NRMSE_EPS = 1e-12
_STAT_EPS = 1e-12
_SKEW_ABS_MAX = 0.2
_EXCESS_KURT_ABS_MAX = 0.5
_AUTOCORR_ABS_MAX = 0.1
_REL_ERROR_DENOM_FLOOR = 1e-30
_L1_RATIO_DENOM_FLOOR = 1e-30
_MODE1_LOG_INTERVAL = 10


def assert_noise_statistics(
    noisy: PDEDataset,
    clean: PDEDataset,
    level: float,
    *,
    field: str = "u",
    rtol: float = 0.05,
) -> dict[str, float]:
    clean_values = _field_values(clean, field)
    noisy_values = _field_values(noisy, field)
    if clean_values.shape != noisy_values.shape:
        raise ValueError(
            f"field '{field}' shape mismatch: "
            f"{tuple(noisy_values.shape)} != {tuple(clean_values.shape)}"
        )

    residual = (noisy_values - clean_values).detach().to(torch.float64).flatten()
    expected_std = level * float(clean_values.detach().abs().max().item())
    centered = residual - residual.mean()
    observed = {
        "mean": float(residual.mean().item()),
        "std": float(residual.std(correction=0).item()),
        "skew": _standardized_moment(centered, 3),
        "kurt": _standardized_moment(centered, 4) - 3.0,
        "max_abs_autocorr_lag1": abs(_lag1_autocorr(residual)),
    }
    _assert_noise_bounds(observed, expected_std, rtol, n_samples=residual.numel())
    return observed


def compute_paired_nrmse(
    u_pred: Tensor,
    u_ref: Tensor,
    *,
    per_channel: bool = True,
) -> float:
    if u_pred.shape != u_ref.shape:
        raise ValueError(
            f"u_pred shape {tuple(u_pred.shape)} must match "
            f"u_ref shape {tuple(u_ref.shape)}"
        )
    _ = per_channel
    pred = u_pred.detach().to(torch.float64)
    ref = u_ref.detach().to(torch.float64)
    rmse = torch.sqrt((pred - ref).pow(2).mean())
    denom = ref.std(correction=0).clamp_min(_NRMSE_EPS)
    return float(safe_div(rmse, denom).item())


def detect_structure_hit(
    expression: str,
    ground_truth_terms: list[str],
    *,
    registry: Any | None = None,
) -> bool:
    if not expression.strip() or not ground_truth_terms:
        return False

    from kd.core.expr.terms import split_terms

    active_registry = registry or FunctionRegistry.create_default()
    try:
        terms = split_terms(expression, active_registry)
    except Exception:
        return False

    observed = {_canonical_term(term) for term in terms}
    expected = {_canonical_term(term) for term in ground_truth_terms}
    return observed == expected


def run_mode1_on_noise(
    noisy: PDEDataset,
    *,
    seed: int,
    n_points: int,
    batch_size: int,
    n_iterations: int,
) -> dict[str, Any]:
    _validate_mode1_args(n_points, batch_size, n_iterations)
    provider = FiniteDiffProvider(noisy, max_order=MAX_DIFF_ORDER)
    evaluator = _build_base_evaluator(noisy, provider)
    indices = _sample_indices(seed, evaluator.lhs_target.shape[0], n_points)
    sampled = SampledEvaluator(evaluator, indices)
    config = _mode1_config(n_iterations, batch_size)
    mode1_result = _run_mode1_iterations(seed, config, sampled)
    fit_result = sampled.evaluate_terms(list(GROUND_TRUTH_TERMS))
    if not fit_result.is_valid or fit_result.coefficients is None:
        raise RuntimeError(
            f"Ground-truth fit on noisy data failed: {fit_result.error_message}"
        )
    return {
        "seed": seed,
        "n_points": n_points,
        "n_iterations": n_iterations,
        "ground_truth_fit": build_fit_summary(fit_result),
        "mode1_run": mode1_result,
    }


def _field_values(dataset: PDEDataset, field: str) -> Tensor:
    fields = dataset.fields
    if fields is None or field not in fields:
        raise KeyError(f"field '{field}' not present in dataset")
    return torch.as_tensor(fields[field].values)


def _standardized_moment(centered: Tensor, order: int) -> float:
    std = centered.std(correction=0).clamp_min(_STAT_EPS)
    moment = centered.pow(order).mean()
    denom = std.pow(order)
    return float(safe_div(moment, denom).item())


def _lag1_autocorr(values: Tensor) -> float:
    if values.numel() < 2:
        return 0.0
    left = values[:-1] - values[:-1].mean()
    right = values[1:] - values[1:].mean()
    denom = left.std(correction=0) * right.std(correction=0)
    if float(denom.item()) <= _STAT_EPS:
        return 0.0
    return float(safe_div((left * right).mean(), denom).item())


def _assert_noise_bounds(
    observed: dict[str, float],
    expected_std: float,
    rtol: float,
    *,
    n_samples: int,
) -> None:




    sem = expected_std / math.sqrt(max(n_samples, 1))
    mean_tol = max(rtol * expected_std + sem, _STAT_EPS)
    if abs(observed["mean"]) > mean_tol:
        raise AssertionError(
            f"noise mean {observed['mean']:.6g} exceeds tolerance {mean_tol:.6g}"
        )
    if not math.isclose(observed["std"], expected_std, rel_tol=rtol):
        raise AssertionError(
            f"noise std {observed['std']:.6g} != expected {expected_std:.6g}"
        )
    if abs(observed["skew"]) > _SKEW_ABS_MAX:
        raise AssertionError(f"noise skew {observed['skew']:.6g} too large")
    if abs(observed["kurt"]) > _EXCESS_KURT_ABS_MAX:
        raise AssertionError(f"noise excess kurtosis {observed['kurt']:.6g} too large")
    if observed["max_abs_autocorr_lag1"] > _AUTOCORR_ABS_MAX:
        raise AssertionError(
            "noise lag-1 autocorrelation "
            f"{observed['max_abs_autocorr_lag1']:.6g} too large"
        )


def _canonical_term(term: str) -> str:
    try:
        tree = ast.parse(term, mode="eval")
    except SyntaxError:
        return "".join(term.split())
    node = _strip_sign_and_scalar(tree.body)
    return "".join(ast.unparse(node).split())


def _strip_sign_and_scalar(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Call) and _is_call(node, "neg", 1):
        return _strip_sign_and_scalar(node.args[0])
    if isinstance(node, ast.Call) and _is_call(node, "mul", 2):
        return _strip_mul_scalar(node)
    if (
        isinstance(node, ast.Call)
        and _is_call(node, "div", 2)
        and _is_numeric_scalar(node.args[1])
    ):
        return _strip_sign_and_scalar(node.args[0])
    return node


def _strip_mul_scalar(node: ast.Call) -> ast.expr:
    left, right = node.args
    if _is_numeric_scalar(left):
        return _strip_sign_and_scalar(right)
    if _is_numeric_scalar(right):
        return _strip_sign_and_scalar(left)
    return node


def _is_call(node: ast.expr, name: str, arity: int) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and len(node.args) == arity
    )


def _is_numeric_scalar(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(
            node.value, bool
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return _is_numeric_scalar(node.operand)
    return False


def _validate_mode1_args(
    n_points: int,
    batch_size: int,
    n_iterations: int,
) -> None:
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")


def _build_base_evaluator(dataset: PDEDataset, provider: Any) -> Evaluator:
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    lhs = provider.get_derivative(dataset.lhs_field, dataset.lhs_axis, order=1)
    return Evaluator(
        executor=PythonExecutor(registry),
        solver=LeastSquaresSolver(),
        context=context,
        lhs=lhs.flatten(),
    )


def _sample_indices(seed: int, total_points: int, n_points: int) -> Tensor:
    if n_points > total_points:
        raise ValueError(
            f"n_points={n_points} exceeds total points={total_points}"
        )
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total_points, generator=generator)[:n_points]


def _mode1_config(n_iterations: int, batch_size: int) -> DiscoverConfig:
    return DiscoverConfig.burgers_preset(
        n_iterations=n_iterations,
        batch_size=batch_size,
        library=LibraryConfig(
            operators=list(ALLEN_CAHN_OPERATORS),
            state_vars=["u"],
            coord_vars=["x", "y", "t"],
        ),
        max_diff_order=MAX_DIFF_ORDER,
    )


def _run_mode1_iterations(
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
        _log_mode1_progress(seed, iteration, config.n_iterations, engine)
    state = engine.state
    return {
        "elapsed_seconds": time.time() - start,
        "best_reward": float(engine.best_reward),
        "best_expression": engine.best_expression,
        "best_terms": state.best_result_terms,
        "best_coefficients": state.best_result_coefficients,
        "last_metrics": last_metrics,
    }


def _log_mode1_progress(
    seed: int,
    iteration: int,
    n_iterations: int,
    engine: Any,
) -> None:
    if (iteration + 1) % _MODE1_LOG_INTERVAL != 0 and iteration + 1 != n_iterations:
        return
    _LOGGER.info(
        "mode1-noise seed=%d iter=%d best_reward=%.6f expr=%s",
        seed,
        iteration + 1,
        engine.best_reward,
        engine.best_expression,
    )


def build_fit_summary(result: EvaluationResult) -> dict[str, Any]:
    if result.coefficients is None:
        raise ValueError("fit result has no coefficients")
    coefficients = result.coefficients.detach().cpu().to(torch.float64)
    true = torch.tensor(GROUND_TRUTH_COEFFS, dtype=torch.float64)
    rel = safe_div(
        (coefficients - true).abs(),
        true.abs().clamp_min(_REL_ERROR_DENOM_FLOOR),
    )
    l1_ratio = float(
        (coefficients - true).abs().sum().item()
        / max(float(true.abs().sum().item()), _L1_RATIO_DENOM_FLOOR)
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







def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device for PINN model + coords/collocation tensors: "
            "'auto' picks cuda > cpu (MPS skipped — float64 not supported); "
            "or force 'cuda' / 'cpu' / 'mps'."
        ),
    )
