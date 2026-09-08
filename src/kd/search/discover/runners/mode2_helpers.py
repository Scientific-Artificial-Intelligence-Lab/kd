
from __future__ import annotations

import argparse
import math
from typing import Any

import torch
from torch import Tensor

from kd.core.expr import (
    FunctionRegistry,
)
from kd.core.expr.term_key import structure_term_key
from kd.core.interrupt import SearchInterrupted
from kd.core.safety import safe_div
from kd.data.derivatives.finite_diff import (
    FiniteDiffProvider,
)
from kd.data.schema import PDEDataset
from kd.search.discover.runners.multiseed import (
    GROUND_TRUTH_TERMS,
    MAX_DIFF_ORDER,
    _build_base_evaluator,
    _multiseed_config,
    _run_mode1,
    _sample_indices,
    build_fit_summary,
)
from kd.search.discover.runners.sampled_evaluator import SampledEvaluator





NOISE_LEVEL = 0.01
PRETRAIN_LOSS_GATE = 0.5
STRETCH_TARGET = 0.05

_NRMSE_EPS = 1e-12
_STAT_EPS = 1e-12
_SKEW_ABS_MAX = 0.2
_EXCESS_KURT_ABS_MAX = 0.5
_AUTOCORR_ABS_MAX = 0.1


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
    except SearchInterrupted:


        raise
    except Exception:
        return False

    observed = {structure_term_key(term) for term in terms}
    expected = {structure_term_key(term) for term in ground_truth_terms}
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
    config = _multiseed_config(n_iterations, batch_size)
    mode1_result = _run_mode1(seed, config, sampled)
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
