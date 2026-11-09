
from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import kd
from kdui.api.runs import UnknownRunError, resolve_run
from kdui.jobstore import JobContext
from kdui.models import (
    AblationResult,
    AblationRow,
    BlockDispersionResult,
    FitRequest,
    ForwardErrorResult,
    StoredJob,
    VerificationCheck,
    VerifyRequest,
    VerifyResult,
)
from kdui.ops.inspect import load_dataset
from kdui.ops.preprocess import apply_preprocessing
from kdui.ops.result import expression_latex, sealed_record

if TYPE_CHECKING:
    from kd import PDEDataset, TabularDataset

FORWARD_ERROR_DEFINITION = (
    "Relative L2 error of the forward solve from the first time slice: "
    "||pred - u||_2 / ||u||_2 over the full grid, with u the left-hand-side field."
)

_REFUSALS = (
    ValueError,
    TypeError,
    NotImplementedError,
    kd.InvalidTermsError,
    kd.EvaluationFailedError,
)


def _ablation(
    dataset: PDEDataset | TabularDataset, terms: list[str], lhs_order: int | None
) -> AblationResult:
    result = kd.ablate_terms(dataset, terms, lhs_order=lhs_order)
    baseline = result.baseline
    return AblationResult(
        baseline_mse=float(baseline.mse),
        baseline_nmse=float(baseline.nmse),
        rows=[
            AblationRow(
                dropped_latex=expression_latex(row.dropped_term),
                mse=row.mse,
                nmse=row.nmse,
                delta_mse=row.delta_mse,
                relative_rise=row.relative_rise,
            )
            for row in result.rows
        ],
    )


def _block_dispersion(
    dataset: PDEDataset | TabularDataset, terms: list[str], lhs_order: int | None
) -> BlockDispersionResult:
    n_blocks = 4
    value = kd.block_dispersion(
        dataset,
        terms,
        n_blocks=n_blocks,
        lhs_order=lhs_order,
    )
    axis = dataset.spatial_axes[0]
    return BlockDispersionResult(value=value, n_blocks=n_blocks, axis=axis)


def _forward_error(
    dataset: PDEDataset | TabularDataset,
    terms: list[str],
    coefficients: list[float],
    max_seconds: float | None,
) -> ForwardErrorResult:
    start = time.monotonic()
    result = kd.forward_error(
        dataset,
        terms,
        coefficients,
        max_seconds=max_seconds,
    )
    return ForwardErrorResult(
        value=result.value,
        definition=FORWARD_ERROR_DEFINITION,
        reason=result.reason,
        warning=result.warning or None,
        seconds=time.monotonic() - start,
        figure_artifact_id=None,
    )


def check_run(
    run_dir: Path,
    dataset: PDEDataset | TabularDataset,
    checks: Sequence[VerificationCheck],
    *,
    run_id: str,
    artifacts_dir: Path,
    max_seconds: float | None = None,
) -> VerifyResult:
    record = sealed_record(run_dir)
    support = record.evidence.support if record is not None else None
    if record is None or not support:
        return unavailable(run_id, checks, "the run has no sealed equation to check")
    coefficients = record.evidence.coefficients or []
    fit = record.evidence.catalog_fit
    lhs_order = int(fit["lhs_spec"]["order"]) if fit and fit.get("lhs_spec") else None
    result = VerifyResult(
        run_id=run_id,
        forward_error=None,
        ablation=None,
        block_dispersion=None,
        unavailable={},
    )
    for check in checks:
        try:
            if check is VerificationCheck.ABLATION:
                result.ablation = _ablation(dataset, support, lhs_order)
            elif check is VerificationCheck.BLOCK_DISPERSION:
                result.block_dispersion = _block_dispersion(dataset, support, lhs_order)
            elif None in coefficients:
                result.unavailable[check] = (
                    "a coefficient of the equation is not finite"
                )
            else:
                result.forward_error = _forward_error(
                    dataset,
                    support,
                    [float(value) for value in coefficients if value is not None],
                    max_seconds,
                )
        except _REFUSALS as error:
            result.unavailable[check] = str(error)
    return result


def unavailable(
    run_id: str, checks: Sequence[VerificationCheck], reason: str
) -> VerifyResult:
    return VerifyResult(
        run_id=run_id,
        forward_error=None,
        ablation=None,
        block_dispersion=None,
        unavailable=dict.fromkeys(checks, reason),
    )


def run_verify(request: VerifyRequest, job: JobContext) -> VerifyResult:
    try:
        run = resolve_run(job.workspace, request.run_id)
    except UnknownRunError as error:
        raise ValueError(f"no run {request.run_id}") from error
    if run.job_dir is None:
        return unavailable(
            request.run_id,
            request.checks,
            "the run was started outside the UI, so the UI has no registered "
            "dataset to check it on",
        )
    stored = StoredJob.model_validate_json(
        (Path(run.job_dir) / "request.json").read_text(encoding="utf-8")
    )
    fit = stored.request
    assert isinstance(fit, FitRequest)
    dataset = apply_preprocessing(
        load_dataset(job.workspace, fit.dataset_id, with_reference=False),
        fit.preprocessing,
    )
    return check_run(
        Path(run.run_dir),
        dataset,
        request.checks,
        run_id=request.run_id,
        artifacts_dir=job.artifacts_dir,
        max_seconds=request.max_seconds,
    )
