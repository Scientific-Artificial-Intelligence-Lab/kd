
from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

import kd
import kd.search
from kd.data import savgol_smooth
from kd.harness import check_memory_cap_available
from kdui.models import (
    AlgorithmInfo,
    CapabilitiesRequest,
    CapabilitySnapshot,
    EnvironmentInfo,
    FacadeParameter,
    GpuInfo,
    LowpassSmoothing,
    MemoryCapKind,
    ModeInfo,
    ParameterField,
    PreflightCall,
    PreflightIssue,
    PreflightRequest,
    PreflightResult,
    Segmentation,
)
from kdui.ops.inspect import load_dataset
from kdui.ops.preprocess import apply_preprocessing
from kdui.settings import Workspace

logger = logging.getLogger(__name__)

_TERM_PARAMS = ("terms", "library")
_TERM_MAX_ORDER = 3
_KD_REFUSALS = (TypeError, ValueError, KeyError, NotImplementedError)
_NAMED_FIELDS = re.compile(r"fields=\['([^']+)'")


def project_capabilities(request: CapabilitiesRequest) -> CapabilitySnapshot:
    cuda = torch.cuda.is_available()
    return CapabilitySnapshot(
        refreshed_at=datetime.now(UTC),
        kd_version=kd.__version__,
        schema_digest=kd.instrument_schema_digest(),
        algorithms=[
            _algorithm(schema, cuda=cuda) for schema in kd.instrument_schemas()
        ],
        environment=_environment(cuda),
    )


def environment_warnings(schema: Mapping[str, Any], *, cuda: bool) -> list[str]:
    requirements = schema["requirements"]
    extras = requirements["extras"]
    warnings = [
        f"The Python package {name} is not installed"
        + (f' (pip install "sail-kd[{extras[0]}]").' if extras else ".")
        for name in requirements["import_names"]
        if importlib.util.find_spec(name) is None
    ]
    warnings += [
        f"The environment variable {variable['name']} is not set."
        for variable in requirements["environment"]
        if variable["required"] and not os.environ.get(variable["name"])
    ]
    gpu_modes = [
        mode["name"]
        for mode in schema["modes"]
        if (mode["resources"] or {}).get("preferred_device") == "cuda"
    ]
    if gpu_modes and not cuda:
        warnings.append(
            f"No CUDA device: mode {', '.join(gpu_modes)} prefers a GPU and "
            "runs on the CPU here, more slowly."
        )
    return warnings


def _parameter(row: Mapping[str, Any]) -> ParameterField:
    subs = row["object_fields"]
    return ParameterField(
        name=row["name"],
        json_kind=row["json_kind"],
        default=row["default"],
        literal_values=row["literal_values"],
        nullable=row["nullable"],
        required=row["required"],
        settable_from=row["settable_from"],
        facade_param=row["facade_param"],
        resume_tier=row["resume_tier"],
        knob=row["knob"],
        description=row["description"],
        object_fields=None if subs is None else [_parameter(sub) for sub in subs],
    )


def _mode(row: Mapping[str, Any]) -> ModeInfo:
    return ModeInfo(
        name=row["name"],
        description=row["description"],
        forms=row["forms"],
        topologies=row["topologies"],
        lhs_orders=row["lhs_orders"],
        provider_kind=row["provider_kind"],
        derivative_providers=row["derivative_providers"],
        selected_by=row["selected_by"],
        required_params=row["required_params"],
        refused_fields=row["refused_fields"],
        default_overrides=row["default_overrides"],
        sketch=row["sketch"],
        selectable_from_data=row["selectable_from_data"],
    )


def _algorithm(schema: Mapping[str, Any], *, cuda: bool) -> AlgorithmInfo:
    return AlgorithmInfo(
        algorithm=schema["algorithm"],
        summary=schema["summary"],
        cost_class=schema["cost_class"],
        score_kind=schema["score_kind"],
        score_direction=schema["score_direction"],
        one_shot=schema["one_shot"],
        segmentation=Segmentation(**schema["segmentation"]),
        modes=[_mode(mode) for mode in schema["modes"]],
        fields=[_parameter(row) for row in schema["fields"]],
        facade_params=[
            FacadeParameter(
                name=row["name"],
                kind=row["kind"],
                default=row["default"],
                literal_values=row["literal_values"],
                config_field=row["config_field"],
                effect=row["effect"],
            )
            for row in schema["facade_params"]
        ],
        warnings=environment_warnings(schema, cuda=cuda),
    )


def _environment(cuda: bool) -> EnvironmentInfo:
    gpus = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                GpuInfo(
                    index=index,
                    name=properties.name,
                    memory_bytes=properties.total_memory,
                )
            )
    kernel_kind = (
        MemoryCapKind.JOB_OBJECT if sys.platform == "win32" else MemoryCapKind.CGROUP
    )
    try:
        check_memory_cap_available()
    except (RuntimeError, OSError) as error:


        return EnvironmentInfo(
            cuda_available=cuda,
            gpus=gpus,
            memory_cap_kind=MemoryCapKind.SAMPLED,
            memory_cap_reason=_sampled_reason(error),
        )
    return EnvironmentInfo(
        cuda_available=cuda,
        gpus=gpus,
        memory_cap_kind=kernel_kind,
        memory_cap_reason=None,
    )


def _sampled_reason(error: Exception) -> str:
    if sys.platform == "win32":
        hint = (
            "Windows refused a memory-limited Job Object here; start kd-ui from a "
            "terminal that does not run inside a restricting Job."
        )
    else:
        hint = (
            "A memory-capped systemd user scope cannot be started here; enable "
            "systemd (on WSL: [boot] systemd=true in /etc/wsl.conf) to get a cap "
            "the kernel enforces."
        )
    return f"{hint} ({error})"







def preflight(request: PreflightCall) -> PreflightResult:
    fit = request.request
    schemas = {schema["algorithm"]: schema for schema in kd.instrument_schemas()}
    refusals: list[PreflightIssue] = []
    warnings: list[str] = []
    if fit.schema_digest != kd.instrument_schema_digest():
        warnings.append(
            "The form was built from an older list of kd's parameters; "
            "refresh the algorithms to get the current one."
        )
    if fit.algorithm not in schemas:
        message = f"kd has no algorithm {fit.algorithm!r}; it has {sorted(schemas)}."
        refusals.append(PreflightIssue(message=message, field="algorithm"))
        return PreflightResult(
            ok=False, refusals=refusals, warnings=warnings, resolved_mode=None
        )
    schema = schemas[fit.algorithm]
    cuda = torch.cuda.is_available()
    warnings += environment_warnings(schema, cuda=cuda)
    refusals += _construct(fit, cuda=cuda)

    dataset = load_dataset(
        Workspace(Path(request.workspace)), fit.dataset_id, with_reference=False
    )
    dataset, data_refusals, data_warnings = _prepared(dataset, fit)
    refusals += data_refusals
    warnings += data_warnings

    inputs = kd.search.mode_inputs(kd.preview_report(dataset).to_dict())
    try:
        resolved = kd.execution_mode_for_request(
            fit.algorithm,
            kd.DataTopology(inputs["topology"]),
            fit.params,
            lhs_order=inputs["lhs_order"],
        )
    except _KD_REFUSALS as error:
        refusals.append(PreflightIssue(message=str(error), field="mode"))
        return PreflightResult(
            ok=False, refusals=refusals, warnings=warnings, resolved_mode=None
        )
    if fit.mode is not None and fit.mode != resolved:
        warnings.append(
            f"kd selects mode {resolved} for these parameters and this data, "
            f"not {fit.mode}."
        )
    mode = next(row for row in schema["modes"] if row["name"] == resolved)
    refusals += _mode_refusals(schema, mode, inputs, fit.params)
    refusals += _sketch_refusals(fit, mode, dataset)
    refusals += _term_refusals(fit.params, dataset)
    return PreflightResult(
        ok=not refusals, refusals=refusals, warnings=warnings, resolved_mode=resolved
    )


def _field_of(message: str, params: Mapping[str, Any]) -> str | None:
    named = _NAMED_FIELDS.search(message)
    if named is not None:
        return f"params.{named.group(1)}"
    for name in params:
        if re.search(rf"\b{re.escape(name)}\b", message):
            return f"params.{name}"
    return None


def _construct(fit: PreflightRequest, *, cuda: bool) -> list[PreflightIssue]:
    refusals = []
    params: dict[str, Any] = dict(fit.params)
    try:
        kd.Model(algorithm=fit.algorithm, verbose=False, device=fit.device, **params)
    except _KD_REFUSALS as error:
        message = str(error)
        refusals.append(
            PreflightIssue(message=message, field=_field_of(message, fit.params))
        )
    if fit.device.startswith("cuda") and not cuda:
        refusals.append(
            PreflightIssue(
                message=(
                    f"This machine has no CUDA device; a fit on {fit.device} fails."
                ),
                field="device",
            )
        )
    return refusals


def _prepared(
    dataset: Any, fit: PreflightRequest
) -> tuple[Any, list[PreflightIssue], list[str]]:
    smoothing = fit.preprocessing.smoothing
    if smoothing is not None and not isinstance(smoothing, LowpassSmoothing):
        return (
            dataset,
            [],
            [
                "The network smoothing is trained inside the run; "
                "the check used the data as loaded."
            ],
        )
    refusals = []
    try:
        dataset = apply_preprocessing(dataset, fit.preprocessing)
    except _KD_REFUSALS as error:
        refusals.append(
            PreflightIssue(message=str(error), field="preprocessing.smoothing")
        )
    window = fit.params.get("savgol_window", 0)
    if isinstance(window, int) and window > 0:
        try:
            dataset = savgol_smooth(dataset, window)
        except _KD_REFUSALS as error:
            refusals.append(
                PreflightIssue(message=str(error), field="params.savgol_window")
            )
    return dataset, refusals, []


def _mode_refusals(
    schema: Mapping[str, Any],
    mode: Mapping[str, Any],
    inputs: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[PreflightIssue]:
    verdict = kd.mode_accepts(mode, **inputs)
    if not verdict.accepts:
        return [PreflightIssue(message=str(verdict.reason), field=None)]
    refusals = [
        PreflightIssue(
            message=(
                f"Mode {mode['name']} of {schema['algorithm']} needs {name}: {value}"
            ),
            field=f"params.{name}",
        )
        for name, value in verdict.params.items()
        if isinstance(value, str) and value.startswith("<") and name not in params
    ]
    for row in schema["fields"]:
        given = [key for key in (row["name"], row["facade_param"]) if key in params]
        if (
            row["name"] in mode["refused_fields"]
            and given
            and params[given[0]] != row["default"]
        ):
            refusals.append(
                PreflightIssue(
                    message=(
                        f"Mode {mode['name']} of {schema['algorithm']} refuses a "
                        f"value of {given[0]} other than its default "
                        f"{row['default']!r}."
                    ),
                    field=f"params.{given[0]}",
                )
            )
    return refusals


def _sketch_refusals(
    fit: PreflightRequest, mode: Mapping[str, Any], dataset: Any
) -> list[PreflightIssue]:
    if fit.sketch is None:
        return []
    if all(level == "unsupported" for level in mode["sketch"].values()):
        return [
            PreflightIssue(
                message=f"Mode {mode['name']} of {fit.algorithm} takes no sketch.",
                field="sketch",
            )
        ]
    try:
        kd.load_sketch(fit.sketch, dataset=dataset)
    except _KD_REFUSALS as error:
        return [PreflightIssue(message=str(error), field="sketch")]
    return []


def _term_refusals(params: Mapping[str, Any], dataset: Any) -> list[PreflightIssue]:
    if not isinstance(dataset, kd.PDEDataset):
        return []
    refusals = []
    for key in _TERM_PARAMS:
        terms = params.get(key)
        if not isinstance(terms, list) or not terms:
            continue
        report = kd.validate_terms(dataset, terms, max_order=_TERM_MAX_ORDER)
        refusals += [
            PreflightIssue(
                message=f"{rejection.term}: {rejection.reason}", field=f"params.{key}"
            )
            for rejection in report.rejected
        ]
    return refusals
