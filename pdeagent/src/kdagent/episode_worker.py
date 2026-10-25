
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import signal
import sys
import tempfile
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from types import FrameType

    from collections.abc import Callable

    from kd.harness import EpisodeOutcome, PlanEntry
    from kd.models import TrainingResult
    from kd.search import RunRecord

    from kdagent.platform_eval import PlatformEvaluation

logger = logging.getLogger(__name__)

_WORKER_MODULE: Final = "kdagent.episode_worker"






TERM_SLACK_SECONDS: Final = 120.0
PINN_WRITE_SECONDS: Final = 30.0
KILL_GRACE_SECONDS: Final = 45.0

_LIVE_GROUPS: set[int] = set()

TASK_KEY: Final = "task"
TRAIN_TASK: Final = "train_surrogate"
REFINE_TASK: Final = "refine_candidate"
SMOOTH_TASK: Final = "smooth_input"
PINN_TASK: Final = "refine_pinn"
NETWORK_TASK: Final = "network_field"
SELECT_LOWPASS_TASK: Final = "select_lowpass"
ABLATE_TASK: Final = "ablate_terms"
SCORE_TASK: Final = "score_term_sets"
VERIFY_TASK: Final = "verify_candidate"
SKETCH_TASK: Final = "make_sketch"
FORWARD_TASK: Final = "selection_forward"
SYSTEM_TASK: Final = "verify_system"

FORWARD_SECONDS: Final = 10.0

FORWARD_WORKER_SECONDS: Final = 30.0

FORWARD_METHOD: Final = "Radau"

SYSTEM_RTOL: Final = 1e-6
SYSTEM_ATOL: Final = 1e-9

MEASURE_MAX_ORDER: Final = 3

DISPERSION_BLOCKS: Final = 4

FORWARD_FLOOR_SECONDS: Final = 0.001

RESULT_FILENAME: Final = "result.json"

STORED_SURROGATE_KEY: Final = "surrogate_model"

SPECTRUM_FRACTIONS: Final = (0.99, 0.999)

KNEE_FLOOR_DIVISOR: Final = 8
KNEE_FLOOR_FACTOR: Final = 1.5


def _modes_at_knee(kept: Any) -> int:
    import numpy as np

    remainder = 1.0 - np.asarray(kept, dtype=np.float64)
    floor = remainder[max(1, remainder.size // KNEE_FLOOR_DIVISOR) - 1]
    return int(np.argmax(remainder <= KNEE_FLOOR_FACTOR * floor)) + 1



_ERROR_MESSAGE_MAX_CHARS: Final = 2000


def _traceback_message(error: Any) -> str:
    return str(error)[-_ERROR_MESSAGE_MAX_CHARS:]


WALL_CLOCK_STOP_REASONS: Final = frozenset(
    {"wall_clock_budget", "wall_clock_budget_before_search"}
)

EVALUATION_STOP_REASON: Final = "evaluation_budget"


def _cap_verdicts(record: RunRecord | None) -> tuple[bool, bool]:
    stop_reason = None if record is None else record.cost.stop_reason
    return stop_reason in WALL_CLOCK_STOP_REASONS, stop_reason == EVALUATION_STOP_REASON







def _platform_evaluation(dataset: Any, outcome: Any) -> PlatformEvaluation:
    from kd import SearchInterrupted

    from kdagent.platform_eval import (
        PlatformEvaluation,
        lhs_order_of,
        platform_evaluation,
    )

    record = outcome.record
    if record is None or not record.evidence.is_valid or not record.evidence.support:
        return PlatformEvaluation(None, None, None, None)
    interrupted = False
    interrupted_message = (
        "SIGTERM received: the parent asked this platform evaluation to stop"
    )

    def _latch_sigterm(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        raise SearchInterrupted(interrupted_message)

    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _latch_sigterm)
    try:
        evaluation = platform_evaluation(
            dataset,
            record.evidence.support,
            lhs_order_of(record.evidence.catalog_fit),
        )
        if interrupted:





            raise SearchInterrupted(interrupted_message)
        return evaluation
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _on_device(dataset: Any, device: str | None) -> Any:
    import dataclasses

    import kd

    if device is None or not isinstance(dataset, kd.PDEDataset):
        return dataset
    axes = None
    if dataset.axes is not None:
        axes = {
            name: dataclasses.replace(axis, values=axis.values.to(device))
            for name, axis in dataset.axes.items()
        }
    fields = None
    if dataset.fields is not None:
        fields = {
            name: dataclasses.replace(field, values=field.values.to(device))
            for name, field in dataset.fields.items()
        }
    return dataclasses.replace(dataset, axes=axes, fields=fields)


def _device_resources() -> dict[str, Any]:
    import torch


    initialized: bool = torch.cuda.is_initialized()
    peak = int(torch.cuda.max_memory_allocated()) if initialized else None
    return {
        "device": "cuda" if peak else "cpu",
        "cuda_max_memory_allocated_bytes": peak,
    }


def _run(request: dict[str, Any], *, started_at: float) -> dict[str, Any]:
    import kd
    from kd.harness import PlanEntry, RecordingOptions, run_episode
    from kd.search import EvaluationBudgetCallback

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))

    device = request.get("device")
    entry = PlanEntry(
        instrument=request["algorithm"],
        dataset_ref=request["dataset_id"],
        seed=request["seed"],
        model_kwargs=request["model_kwargs"],
    )





    extra: dict[str, Any] = {}
    surrogate = request.get("surrogate")
    build = kd.Model if surrogate is None else _injecting_factory(surrogate)
    built: list[Any] = []

    def keeping_factory(**kwargs: Any) -> Any:
        model = build(**kwargs)
        built.append(model)
        return model

    extra["model_factory"] = keeping_factory
    sketch = request.get("sketch")
    if sketch is not None:




        extra["sketch"] = kd.load_sketch(sketch, dataset=_grid(dataset))


        verify = request.get("verify")
        if verify is not None:
            extra["verify"] = kd.VerifyPolicy(nmse_max=verify["nmse_max"])
    time_cap_seconds = request["time_cap_seconds"]
    max_seconds: float | None = None
    if time_cap_seconds is not None:








        max_seconds = max(0.001, time_cap_seconds - (time.perf_counter() - started_at))






    evaluation_cap = request.get("evaluation_cap")
    extra_callbacks = (
        None
        if evaluation_cap is None
        else [EvaluationBudgetCallback(max_evaluations=int(evaluation_cap))]
    )
    checkpoint_every = request["checkpoint_every"]
    recording = (
        RecordingOptions()
        if checkpoint_every is None
        else RecordingOptions(
            checkpoint_every=checkpoint_every,
            checkpoint_keep_last=request["checkpoint_keep_last"],
        )
    )
    outcome = run_episode(
        entry=entry,
        entry_index=0,
        dataset=_on_device(dataset, device),
        device=device,
        run_dir=Path(request["run_dir"]),




        recording=recording,
        resume_from=request["resume_from"],



        reseed=request.get("reseed", False),
        max_seconds=max_seconds,
        extra_callbacks=extra_callbacks,
        **extra,
    )
    evaluation = _platform_evaluation(dataset, outcome)





    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if outcome.status == "completed":

        built[0].result_.save(Path(request["run_dir"]) / RESULT_FILENAME)
    network, network_error = _saved_network(built, request.get("surrogate_target"))
    stopped_by_time_cap, stopped_by_evaluation_cap = _cap_verdicts(outcome.record)
    return {
        "status": outcome.status,
        "error_type": outcome.error_type,
        "error_message": outcome.error_message,
        "wallclock_seconds": outcome.wallclock_seconds,
        "run_id": outcome.run_id,
        "run_dir": None if outcome.run_dir is None else str(outcome.run_dir),
        "lineage": outcome.lineage,
        "record": None if outcome.record is None else outcome.record.to_dict(),
        "stopped_by_time_cap": stopped_by_time_cap,
        "stopped_by_evaluation_cap": stopped_by_evaluation_cap,
        "platform_nmse": evaluation.nmse,
        "platform_coefficients": evaluation.coefficients,
        "platform_eval_error": evaluation.error,
        "platform_eval_dropped": evaluation.dropped,
        "unavailable_reason": evaluation.unavailable_reason,
        "resources": _device_resources(),
        "trained_surrogate": network,
        "trained_surrogate_error": network_error,
    }


def _saved_network(
    built: list[Any], target: str | None
) -> tuple[dict[str, Any] | None, str | None]:
    from kd.models import load_stored_surrogate

    if target is None:
        return None, None
    if not built:
        return None, "the episode built no model, so it trained no network"
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        training = built[0].save_surrogate(path)
        identity = load_stored_surrogate(path).identity
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:_ERROR_MESSAGE_MAX_CHARS]


    trained = cast("TrainingResult", training)
    return {
        "file": str(path),
        "identity": identity,
        "train_seconds": trained.elapsed_seconds,
        "epochs": trained.epochs_run,
        "final_loss": trained.final_loss,
    }, None


def _injecting_factory(surrogate: dict[str, Any]) -> Callable[..., Any]:
    import kd
    from kd.models import load_field_model, load_stored_surrogate

    path = Path(surrogate["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != surrogate["sha256"]:
        raise RuntimeError(
            f"surrogate file {path} hashes to {digest}, not the registered "
            f"{surrogate['sha256']}; it changed after the parent checked it"
        )
    key = surrogate["key"]
    injected: Any
    trained_under: dict[str, Any] = {}
    if key == STORED_SURROGATE_KEY:
        injected = load_stored_surrogate(path)


        trained_under = cast("dict[str, Any]", injected.identity)["surrogate"]
    else:
        injected = load_field_model(path)

    def factory(**kwargs: Any) -> Any:
        merged: dict[str, Any] = {**trained_under, **kwargs, key: injected}
        return kd.Model(**merged)

    return factory


def _train(request: dict[str, Any]) -> dict[str, Any]:
    import kd
    from kd.models import FieldModel, load_stored_surrogate, save_field_model

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))

    device = request.get("device")
    model = kd.Model(
        algorithm=request["algorithm"],
        seed=request["seed"],
        verbose=False,
        device=device,
        **request["model_kwargs"],
    )
    output_path = Path(request["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stored = request["key"] == STORED_SURROGATE_KEY
    module, training = model.train_surrogate(
        _on_device(dataset, device), save_to=output_path if stored else None
    )




    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    identity: dict[str, Any] | None = None
    if stored:
        identity = load_stored_surrogate(output_path).identity
    else:



        save_field_model(cast("FieldModel", module), output_path)
    return {
        "file": str(output_path),
        "key": request["key"],
        "train_seconds": training.elapsed_seconds,
        "epochs": training.epochs_run,
        "final_loss": training.final_loss,
        "identity": identity,
        "resources": _device_resources(),
    }


def _prepare(request: dict[str, Any]) -> dict[str, Any]:
    import kd

    from kdagent.data_source import load_dataset, pde_task_kind, prepare_input

    if "dataset_id" in request:
        dataset_id = request["dataset_id"]

        grid = cast("kd.PDEDataset", load_dataset(dataset_id, None))
        return _input_summary(grid, dataset_id, pde_task_kind(grid), None)
    path = Path(request["path"])
    report = kd.inspect_file(path)
    inventory = {
        "container": report.container,


        "arrays": [
            {"key": row.key, "shape": list(row.shape), "dtype": row.dtype}
            for row in report.arrays
        ],
    }

    dataset, description = prepare_input(
        path,
        request["load_options"],
        task_kind=request.get("task_kind"),
        target_state=request.get("target_state"),
        recognized=report.layouts,
    )
    summary = _input_summary(
        dataset, description["ref"], description["task_kind"], inventory
    )
    if description["projection"] is not None:


        summary["projection"] = description["projection"]
        summary["trajectories"] = description["trajectories"]
    description["summary"] = summary
    return description


def _input_summary(
    dataset: Any,
    dataset_id: str,
    task_kind: str,
    inventory: dict[str, Any] | None,
) -> dict[str, Any]:
    import kd

    preview = kd.preview_report(dataset).to_dict()
    if isinstance(dataset, kd.TabularDataset):
        target: str | None = dataset.target_name
    else:
        target = None if dataset.lhs_order == 0 else preview["lhs_label"]
    return {
        "id": dataset_id,
        "task_kind": task_kind,
        "target": target,
        "report": preview,
        "inventory": inventory,
    }


def _refine(request: dict[str, Any]) -> dict[str, Any]:
    import kd
    from kd.data import savgol_smooth

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))
    terms = request["terms"]
    refit: dict[str, Any]
    try:
        if isinstance(dataset, kd.TabularDataset):
            constant_fit = kd.refit_constants(dataset, terms)
            refit = {
                "refit_kind": "constants",
                "terms": list(constant_fit.terms),
                "coefficients": list(constant_fit.coefficients),
                "nmse": constant_fit.nmse,
                "constants": dict(constant_fit.constants),
                "initial": dict(constant_fit.initial),
            }
        else:
            if request["savgol_window"] > 0:
                dataset = savgol_smooth(dataset, request["savgol_window"])


            if request.get("lowpass_modes", 0):
                from kd import lowpass_smooth

                dataset = lowpass_smooth(
                    dataset,
                    modes=request["lowpass_modes"],
                    time_window=request.get("time_window", 0),
                )
            fit = kd.refit_terms(
                dataset,
                terms,
                request["coefficients"],
                small_term_rtol=request["small_term_rtol"],
                lhs_order=request["lhs_order"],
                **_frame_arguments(request),
            )
            refit = {"refit_kind": "terms", **_fit_json(fit)}
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {
            "refit_error": {
                "code": _refit_failure_code(exc),
                "message": f"{type(exc).__name__}: {exc}",
            }
        }
    return {"refit": refit}


def _refit_failure_code(exc: Exception) -> str:
    import kd
    import torch

    if isinstance(exc, MemoryError | torch.cuda.OutOfMemoryError):
        return "out_of_memory"
    if isinstance(exc, kd.InvalidTermsError):
        return "invalid_term"
    if isinstance(exc, NotImplementedError):
        return "unsupported_form"
    if isinstance(exc, ValueError | TypeError):
        return "invalid_argument"
    return type(exc).__name__


def finite_tree(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite_tree(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [finite_tree(item) for item in value]
    return value


def _kd_failure(exc: Exception) -> dict[str, str]:
    return {"code": _refit_failure_code(exc), "message": f"{type(exc).__name__}: {exc}"}


def _stored_surrogate(spec: dict[str, Any]) -> Any:
    from kd.models import load_stored_surrogate

    path = Path(spec["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError(
            f"model file {path} hashes to {digest}, not the registered "
            f"{spec['sha256']}; it changed after the parent checked it"
        )
    return load_stored_surrogate(path)


def _frame_arguments(request: dict[str, Any]) -> dict[str, Any]:
    import kd

    weak = request.get("weak")
    surrogate = request.get("surrogate")
    return {
        "method": request.get("method", "finite_diff"),
        "trim": request.get("trim", 0),
        "weak": None if weak is None else kd.WeakFormOptions(**weak),
        "surrogate": None if surrogate is None else _stored_surrogate(surrogate),
    }


def _grid(dataset: Any) -> Any:
    import kd

    if not isinstance(dataset, kd.PDEDataset):
        raise ValueError(
            "a table has no field to measure terms on; this is a grid verb"
        )
    return dataset


def _fit_json(fit: Any) -> dict[str, Any]:
    return {
        "terms": list(fit.terms),
        "coefficients": list(fit.coefficients),
        "dropped": [[term, value] for term, value in fit.dropped],
        "nmse": fit.nmse,
        "mse": fit.mse,
        "r2": fit.r2,
        "condition_number": fit.condition_number,
        "derivative_source": fit.derivative_source,
        "frame": dataclasses.asdict(fit.frame),
        "weak": None if fit.weak is None else dataclasses.asdict(fit.weak),
    }


def _smooth(request: dict[str, Any]) -> dict[str, Any]:
    import kd
    import numpy as np
    from kd import lowpass_smooth, lowpass_spectrum

    from kdagent.data_source import load_dataset

    parent_id = request["dataset_id"]
    parent = load_dataset(parent_id, request["input_source"])
    modes = request["lowpass_modes"]
    window = request["time_window"]
    try:
        if not isinstance(parent, kd.PDEDataset):
            raise ValueError("a table has no axes to low-pass")
        smoothed = lowpass_smooth(parent, modes=modes, time_window=window)
        assert smoothed.axis_order is not None and smoothed.axes is not None
        assert smoothed.fields is not None
        smoothed_axes = (
            [axis for axis in smoothed.axis_order if axis != smoothed.lhs_axis]
            if isinstance(modes, int)
            else [axis for axis in smoothed.axis_order if axis in modes]
        )
        spectrum: dict[str, dict[str, Any]] = {}
        for axis in smoothed_axes:
            kept = np.asarray(lowpass_spectrum(parent, parent.lhs_field, axis))
            count = modes if isinstance(modes, int) else modes[axis]
            spectrum[axis] = {
                "n": int(kept.size),
                "energy_kept": float(kept[count - 1]),
                **{
                    f"modes_at_{fraction}": int(np.argmax(kept >= fraction)) + 1
                    for fraction in SPECTRUM_FRACTIONS
                },
                "modes_at_knee": _modes_at_knee(kept),
            }
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {
            "smooth_error": {
                "code": _refit_failure_code(exc),
                "message": f"{type(exc).__name__}: {exc}",
            }
        }
    description = _write_derived_input(
        smoothed,
        Path(request["output_path"]),
        derived={
            "from": parent_id,
            "lowpass_modes": modes,
            "time_window": window,
            "parent_sha256": request["input_source"]["source"]["sha256"],
            "spectrum": spectrum,
        },
    )
    return {"description": description}


def _write_derived_input(
    derived_dataset: Any, output_path: Path, *, derived: dict[str, Any]
) -> dict[str, Any]:
    import kd
    import numpy as np

    from kdagent.data_source import prepare_derived_input

    axis_order = derived_dataset.axis_order
    arrays: dict[str, Any] = {"axis_order": np.array(list(axis_order))}
    for name in axis_order:
        arrays[name] = derived_dataset.axes[name].values.detach().cpu().numpy()
    for name, field in derived_dataset.fields.items():
        arrays[name] = field.values.detach().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with open(tmp_path, "wb") as stream:
        np.savez(stream, **arrays)
    os.replace(tmp_path, output_path)
    options = {
        "layout": "kd-npz",
        "lhs": kd.render_lhs_label(
            kd.LhsSpec(
                derived_dataset.lhs_field,
                derived_dataset.lhs_axis,
                derived_dataset.lhs_order,
            )
        ),
        "periodic": [
            name for name in axis_order if derived_dataset.axes[name].is_periodic
        ],
    }
    dataset, description = prepare_derived_input(output_path, options, derived=derived)
    report = kd.inspect_file(output_path)
    inventory = {
        "container": report.container,
        "arrays": [
            {"key": row.key, "shape": list(row.shape), "dtype": row.dtype}
            for row in report.arrays
        ],
    }
    description["summary"] = {
        **_input_summary(
            dataset, description["ref"], description["task_kind"], inventory
        ),
        "derived_from": derived["from"],
    }
    return description


def _pinn(request: dict[str, Any]) -> dict[str, Any]:
    import kd

    from kdagent.data_source import load_dataset

    parent_id = request["dataset_id"]
    parent = load_dataset(parent_id, request["input_source"])
    pinn = request["pinn"]
    try:
        if not isinstance(parent, kd.PDEDataset):
            raise ValueError("a table has no grid to fit a network on")
        result = kd.refine_pinn(
            parent,
            pinn["terms"],
            pinn["coefficients"],
            epochs=pinn["epochs"],
            pretrain_epochs=pinn["pretrain_epochs"],
            residual_weight=pinn["residual_weight"],
            seed=pinn["seed"],
            device=request.get("device"),
            max_seconds=request["max_seconds"],
        )
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {
            "refit_error": {
                "code": _refit_failure_code(exc),
                "message": f"{type(exc).__name__}: {exc}",
            },
            "resources": _device_resources(),
        }
    outcome = {
        "coefficients": list(result.coefficients),
        "initial_data_loss": result.initial_data_loss,
        "initial_residual_loss": result.initial_residual_loss,
        "data_loss": result.data_loss,
        "residual_loss": result.residual_loss,
        "epochs_run": result.epochs_run,
        "stopped_at_deadline": result.stopped_at_deadline,
        "pretrain_epochs_run": result.pretrain_epochs_run,
        "pretrain_stopped_early": result.pretrain_stopped_early,
        "seconds": result.seconds,
        "device": result.device,
    }
    description = _write_derived_input(
        result.dataset,
        Path(request["output_path"]),
        derived={
            "from": parent_id,
            "lowpass_modes": None,
            "time_window": None,
            "parent_sha256": request["input_source"]["source"]["sha256"],
            "pinn": dict(pinn),
            "pinn_result": outcome,
        },
    )
    return {"description": description, "resources": _device_resources()}


def _network_field(request: dict[str, Any]) -> dict[str, Any]:
    import kd
    import kd.data
    from kd.models import FieldRecipe, save_field_model

    from kdagent.data_source import load_dataset

    parent_id = request["dataset_id"]
    parent = load_dataset(parent_id, request["input_source"])
    recipe = request.get("recipe")
    try:
        if not isinstance(parent, kd.PDEDataset):
            raise ValueError("a table has no grid field to fit a network to")
        if recipe is not None:
            smoothing = kd.data.network_smooth(
                parent,
                recipe=FieldRecipe(
                    **{**recipe, "hidden_sizes": tuple(recipe["hidden_sizes"])}
                ),
                seed=request["seed"],
                device=request.get("device"),
            )
            field = smoothing.dataset
        else:
            stored = _stored_surrogate(request["surrogate"])
            field = kd.surrogate_field(parent, stored.module)
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {"task_error": _kd_failure(exc), "resources": _device_resources()}


    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    network = dict(request["network"])
    training: dict[str, Any] | None = None
    if recipe is not None:
        identity = smoothing.identity
        Path(request["model_path"]).parent.mkdir(parents=True, exist_ok=True)
        save_field_model(
            smoothing.model,
            Path(request["model_path"]),
            identity={
                **identity,
                "target_sha256": identity["field_sha256"][parent.lhs_field],
            },
        )
        network["preparation"] = identity
        network["quality"] = finite_tree(dataclasses.asdict(smoothing.quality))
        training = {
            "train_seconds": smoothing.training.elapsed_seconds,
            "epochs": smoothing.training.epochs_run,
            "final_loss": smoothing.training.final_loss,
        }
    network["artifact_sha256"] = hashlib.sha256(
        Path(request["model_path"]).read_bytes()
    ).hexdigest()
    description = _write_derived_input(
        field,
        Path(request["output_path"]),
        derived={
            "from": parent_id,
            "lowpass_modes": None,
            "time_window": None,
            "parent_sha256": request["input_source"]["source"]["sha256"],
            "network": network,
        },
    )
    return {
        "value": {"description": description, "training": finite_tree(training)},
        "resources": _device_resources(),
    }


def _select_lowpass(request: dict[str, Any]) -> dict[str, Any]:
    import functools

    import kd

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request["input_source"])
    try:
        if not isinstance(dataset, kd.PDEDataset):
            raise ValueError("a table has no axes to low-pass")
        selection = kd.select_lowpass_modes(
            dataset,
            dataset.lhs_field,
            terms=request["terms"],
            refit=functools.partial(kd.refit_terms, lhs_order=dataset.lhs_order),
            time_window=request["time_window"],
            tolerance=request["tolerance"],
            max_counts=request["max_counts"],
        )
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {"task_error": _kd_failure(exc)}
    return {"value": dataclasses.asdict(selection)}


def _ablate(request: dict[str, Any]) -> dict[str, Any]:
    import kd

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))
    try:
        result = kd.ablate_terms(
            _grid(dataset),
            request["terms"],
            lhs_order=request["lhs_order"],
            **_frame_arguments(request),
        )
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {"task_error": _kd_failure(exc)}
    return {
        "value": {
            "baseline": _fit_json(result.baseline),
            "rows": [dataclasses.asdict(row) for row in result.rows],
            "relative_rise_reason": result.relative_rise_reason,
        }
    }


def _score(request: dict[str, Any]) -> dict[str, Any]:
    import kd

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))
    try:
        scores = kd.score_term_sets(
            _grid(dataset),
            request["term_sets"],
            lhs_order=request["lhs_order"],
            **_frame_arguments(request),
        )
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {"task_error": _kd_failure(exc)}
    return {
        "value": {
            "frame": dataclasses.asdict(scores.frame),
            "entries": [
                {
                    "terms": list(entry.terms),
                    "fit": None if entry.fit is None else _fit_json(entry.fit),
                    "failure": entry.failure,
                    "failure_kind": entry.failure_kind,
                }
                for entry in scores.entries
            ],
        }
    }


def _verify(request: dict[str, Any], *, started_at: float) -> dict[str, Any]:
    import kd

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))
    terms, coefficients = request["terms"], request["coefficients"]
    lhs_order = request["lhs_order"]
    failures: list[dict[str, str]] = []

    def measured(name: str, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except kd.SearchInterrupted:
            raise
        except Exception as exc:
            failures.append({"measurement": name, **_kd_failure(exc)})
            return None

    report = measured(
        "validation",
        lambda: kd.validate_terms(
            _grid(dataset), terms, max_order=MEASURE_MAX_ORDER, lhs_order=lhs_order
        ),
    )
    dispersion = measured(
        "dispersion",
        lambda: kd.block_dispersion(
            _grid(dataset),
            terms,
            n_blocks=DISPERSION_BLOCKS,
            max_order=MEASURE_MAX_ORDER,
            lhs_order=lhs_order,
        ),
    )
    budget = max(
        FORWARD_FLOOR_SECONDS,
        request["max_seconds"] - (time.perf_counter() - started_at),
    )
    forward = measured(
        "forward",
        lambda: kd.forward_error(
            _forward_input(_grid(dataset), request["lhs"]),
            terms,
            coefficients,
            max_seconds=budget,
        ),
    )
    return {
        "value": {
            "validation": (
                None
                if report is None
                else {
                    "ok": report.ok,
                    "results": [dataclasses.asdict(row) for row in report.results],
                    "max_order": MEASURE_MAX_ORDER,
                }
            ),
            "dispersion": (
                None
                if dispersion is None
                else {
                    "coefficient_of_variation": dispersion,
                    "n_blocks": DISPERSION_BLOCKS,
                    "max_order": MEASURE_MAX_ORDER,
                }
            ),
            "forward": None if forward is None else dataclasses.asdict(forward),
            "forward_max_seconds": budget,
            "failures": failures,
        }
    }


def _forward_input(dataset: Any, lhs: dict[str, Any] | None) -> Any:
    import kd

    if lhs is None or (dataset.lhs_field, dataset.lhs_axis, dataset.lhs_order) == (
        lhs["field"],
        lhs["axis"],
        lhs["order"],
    ):
        return dataset
    return dataset.retarget(kd.render_lhs_label(kd.LhsSpec(**lhs)))


def _selection_forward(request: dict[str, Any], *, started_at: float) -> dict[str, Any]:
    import kd

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))
    loaded = time.perf_counter()
    timing = {"load_seconds": loaded - started_at, "forward_seconds": None}
    if not isinstance(dataset, kd.PDEDataset):
        return {"value": {"inapplicable": "tabular", **timing}}
    if dataset.topology is not kd.DataTopology.GRID:
        return {"value": {"inapplicable": dataset.topology.value, **timing}}
    budget = min(FORWARD_SECONDS, request["max_seconds"] - (loaded - started_at))
    if budget <= 0.0:
        return {"value": {"forward": None, "forward_max_seconds": budget, **timing}}
    try:
        forward = kd.forward_error(
            _forward_input(dataset, request["lhs"]),
            request["terms"],
            request["coefficients"],
            method=FORWARD_METHOD,
            max_step=None,
            max_seconds=budget,
        )
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {"task_error": _kd_failure(exc), "resources": _device_resources()}
    timing["forward_seconds"] = time.perf_counter() - loaded
    return {
        "value": {
            "forward": finite_tree(dataclasses.asdict(forward)),
            "forward_max_seconds": budget,
            **timing,
        },
        "resources": _device_resources(),
    }


_SYSTEM_FIELDS: Final = (
    "value",
    "reason",
    "warning",
    "per_target",
    "per_trajectory",
    "intervals",
    "solver",
    "input_identity",
)


def _verify_system(request: dict[str, Any], *, started_at: float) -> dict[str, Any]:
    import kd

    from kdagent.data_source import load_trajectories

    raw = load_trajectories(request["dataset_id"], request["input_source"])
    loaded = time.perf_counter()
    timing = {"load_seconds": loaded - started_at, "forward_seconds": None}
    budget = min(FORWARD_SECONDS, request["max_seconds"] - (loaded - started_at))
    if budget <= 0.0:
        return {"value": {"forward": None, "forward_max_seconds": budget, **timing}}
    laws = {
        target: (list(terms), [float(value) for value in coefficients])
        for target, (terms, coefficients) in request["laws"].items()
    }
    try:
        forward = kd.forward_error_trajectories(
            raw,
            laws,
            method=FORWARD_METHOD,
            max_step=None,
            rtol=SYSTEM_RTOL,
            atol=SYSTEM_ATOL,
            max_seconds=budget,
        )
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {"task_error": _kd_failure(exc)}
    timing["forward_seconds"] = time.perf_counter() - loaded
    return {
        "value": {
            "forward": finite_tree(
                {name: getattr(forward, name) for name in _SYSTEM_FIELDS}
            ),
            "forward_max_seconds": budget,
            **timing,
        }
    }


def _bound_sketch(grid: Any, request: dict[str, Any]) -> Any:
    import kd

    if "payload" in request:
        return kd.load_sketch(request["payload"], dataset=grid)
    clauses = request["clauses"]
    spec = kd.Sketch(
        pinned=clauses["pinned"],
        anchored=clauses["anchored"],
        holes=[kd.Hole(**hole) for hole in clauses["holes"]],
    )
    return spec.resolve(grid)


def _make_sketch(request: dict[str, Any]) -> dict[str, Any]:
    import kd

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))
    try:
        if "anchors" not in request:
            sketch = _bound_sketch(_grid(dataset), request)
            return {"value": {"sketch": dict(kd.sketch_to_dict(sketch))}}
        anchors = request["anchors"]
        report = kd.validate_terms(_grid(dataset), anchors, max_order=MEASURE_MAX_ORDER)
        if not report.ok:
            rejected = "; ".join(
                f"{row.term!r}: {row.reason}" for row in report.results if not row.ok
            )
            return {
                "task_error": {
                    "code": "invalid_term",
                    "message": f"anchors this input cannot evaluate: {rejected}",
                }
            }
        sketch = kd.sketch_from_anchors(
            _grid(dataset),
            anchors,
            min_additional_terms=request["min_additional_terms"],
            max_additional_terms=request["max_additional_terms"],
            max_derivative_order=request["max_derivative_order"],
        )
    except kd.SearchInterrupted:
        raise
    except Exception as exc:
        return {"task_error": _kd_failure(exc)}
    return {"value": {"sketch": dict(kd.sketch_to_dict(sketch))}}


def _write_result(result: dict[str, Any], result_path: str) -> None:
    tmp_path = f"{result_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(tmp_path, result_path)


def main(argv: list[str]) -> None:
    started_at = time.perf_counter()
    request_path, result_path = argv






    from kd import SearchInterrupted

    def _raise_search_interrupted(signum: int, frame: FrameType | None) -> None:
        raise SearchInterrupted(
            "SIGTERM received: the parent asked this search to stop"
        )

    signal.signal(signal.SIGTERM, _raise_search_interrupted)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


    with open(request_path, encoding="utf-8") as f:
        request = json.load(f)
    result: dict[str, Any]
    try:
        if request.get(TASK_KEY) == "prepare_input":
            result = _prepare(request)
        elif request.get(TASK_KEY) == TRAIN_TASK:
            result = _train(request)
        elif request.get(TASK_KEY) == REFINE_TASK:
            result = _refine(request)
        elif request.get(TASK_KEY) == SMOOTH_TASK:
            result = _smooth(request)
        elif request.get(TASK_KEY) == PINN_TASK:
            result = _pinn(request)
        elif request.get(TASK_KEY) == NETWORK_TASK:
            result = _network_field(request)
        elif request.get(TASK_KEY) == SELECT_LOWPASS_TASK:
            result = _select_lowpass(request)
        elif request.get(TASK_KEY) == ABLATE_TASK:
            result = _ablate(request)
        elif request.get(TASK_KEY) == SCORE_TASK:
            result = _score(request)
        elif request.get(TASK_KEY) == VERIFY_TASK:
            result = _verify(request, started_at=started_at)
        elif request.get(TASK_KEY) == SKETCH_TASK:
            result = _make_sketch(request)
        elif request.get(TASK_KEY) == FORWARD_TASK:
            result = _selection_forward(request, started_at=started_at)
        elif request.get(TASK_KEY) == SYSTEM_TASK:
            result = _verify_system(request, started_at=started_at)
        else:
            result = _run(request, started_at=started_at)
    except Exception:




        result = {"error": traceback.format_exc(), "resources": _device_resources()}
    _write_result(result, result_path)







@dataclass(frozen=True)
class IsolatedEpisode:

    outcome: EpisodeOutcome
    stopped_by_time_cap: bool
    post_seal_error: str | None = None
    platform_nmse: float | None = None
    platform_coefficients: list[float] | None = None
    platform_eval_error: str | None = None
    platform_eval_dropped: int | None = None
    unavailable_reason: str | None = None
    stopped_by_evaluation_cap: bool = False
    resources: dict[str, Any] | None = None
    trained_surrogate: TrainedNetwork | None = None
    trained_surrogate_error: str | None = None


@dataclass(frozen=True)
class TrainedNetwork:

    file: Path
    file_sha256: str
    identity: dict[str, Any]
    train_seconds: float
    epochs: int
    final_loss: float


def run_isolated_episode(
    entry: PlanEntry,
    *,
    run_dir: Path,
    time_cap_seconds: float | None,
    resume_from: Path | str | None = None,
    reseed: bool = False,
    checkpoint_every: int | None = None,
    checkpoint_keep_last: int | None = None,
    sketch: dict[str, Any] | None = None,
    surrogate: dict[str, Any] | None = None,
    evaluation_cap: int | None = None,
    input_source: dict[str, Any] | None = None,
    device: str = "cpu",
    verify: dict[str, float] | None = None,
    llm4ed_api_key: str | None = None,
    surrogate_target: Path | None = None,
) -> IsolatedEpisode:
    request: dict[str, Any] = {
        "dataset_id": entry.dataset_ref,
        "algorithm": entry.instrument,
        "seed": entry.seed,
        "model_kwargs": entry.model_kwargs,
        "run_dir": str(run_dir),
        "time_cap_seconds": time_cap_seconds,
        "resume_from": (
            None if resume_from is None else str(Path(resume_from).resolve())
        ),
        "checkpoint_every": checkpoint_every,
        "checkpoint_keep_last": checkpoint_keep_last,
    }
    if input_source is not None:
        request["input_source"] = input_source
    if reseed:
        request["reseed"] = True
    if sketch is not None:
        request["sketch"] = sketch
    if surrogate is not None:
        request["surrogate"] = surrogate
    if verify is not None:
        request["verify"] = verify
    if surrogate_target is not None:
        request["surrogate_target"] = str(surrogate_target)
    if evaluation_cap is not None:


        request["evaluation_cap"] = int(evaluation_cap)
    _add_device(request, device)
    logger.info("launching episode worker for %s", run_dir.name)
    result, sent_sigterm, returncode, elapsed = _supervise(
        request,
        time_cap_seconds=time_cap_seconds,



        term_slack=TERM_SLACK_SECONDS,
        llm4ed_api_key=llm4ed_api_key,
    )
    return _episode_from(
        entry,
        run_dir=run_dir,
        result=result,
        sent_sigterm=sent_sigterm,
        returncode=returncode,
        elapsed=elapsed,
        time_cap_seconds=time_cap_seconds,
    )


def _add_device(request: dict[str, Any], device: str) -> None:
    if device == "cuda":
        request["device"] = "cuda"
    elif device != "cpu":
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")


@dataclass(frozen=True)
class IsolatedTraining:

    status: str
    error_type: str | None
    error_message: str | None
    file: Path | None
    file_sha256: str | None
    key: str | None
    train_seconds: float | None
    epochs: int | None
    final_loss: float | None
    identity: dict[str, Any] | None
    resources: dict[str, Any] | None = None


def run_isolated_training(
    *,
    dataset_id: str,
    algorithm: str,
    seed: int,
    model_kwargs: dict[str, Any],
    output_path: Path,
    key: str,
    time_cap_seconds: float | None,
    input_source: dict[str, Any] | None = None,
    device: str = "cpu",
) -> IsolatedTraining:
    request: dict[str, Any] = {
        TASK_KEY: TRAIN_TASK,
        "dataset_id": dataset_id,
        "algorithm": algorithm,
        "seed": seed,
        "model_kwargs": model_kwargs,
        "output_path": str(output_path),
        "key": key,
        "time_cap_seconds": time_cap_seconds,
    }
    if input_source is not None:
        request["input_source"] = input_source
    _add_device(request, device)
    logger.info("launching surrogate training for %s", output_path.name)
    result, sent_sigterm, returncode, elapsed = _supervise(
        request, time_cap_seconds=time_cap_seconds, term_slack=0.0
    )
    return _training_from(
        result, sent_sigterm=sent_sigterm, returncode=returncode, elapsed=elapsed
    )


@dataclass(frozen=True)
class IsolatedRefine:

    refit: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    wallclock_seconds: float


def run_isolated_refine(
    *,
    dataset_id: str,
    input_source: dict[str, Any] | None,
    terms: list[str],
    coefficients: list[float] | None,
    lhs_order: int,
    small_term_rtol: float,
    savgol_window: int,
    lowpass_modes: int | dict[str, int],
    time_window: int,
    method: str,
    trim: int | dict[str, int] | str,
    weak: dict[str, Any] | None,
    surrogate: dict[str, Any] | None,
    time_cap_seconds: float | None,
) -> IsolatedRefine:
    request: dict[str, Any] = {
        TASK_KEY: REFINE_TASK,
        "dataset_id": dataset_id,
        "terms": terms,
        "coefficients": coefficients,
        "lhs_order": lhs_order,
        "small_term_rtol": small_term_rtol,
        "savgol_window": savgol_window,
        "lowpass_modes": lowpass_modes,
        "time_window": time_window,
        **frame_request(method=method, trim=trim, weak=weak, surrogate=surrogate),
    }
    if input_source is not None:
        request["input_source"] = input_source
    logger.info("launching refit worker for %s", dataset_id)
    result, sent_sigterm, returncode, elapsed = _supervise(
        request, time_cap_seconds=time_cap_seconds, term_slack=0.0
    )
    return _refine_from(
        result, sent_sigterm=sent_sigterm, returncode=returncode, elapsed=elapsed
    )


@dataclass(frozen=True)
class IsolatedSmooth:

    description: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    wallclock_seconds: float
    resources: dict[str, Any] | None = None


def run_isolated_smooth(
    *,
    dataset_id: str,
    input_source: dict[str, Any],
    lowpass_modes: int | dict[str, int],
    time_window: int,
    output_path: Path,
    time_cap_seconds: float | None,
) -> IsolatedSmooth:
    request: dict[str, Any] = {
        TASK_KEY: SMOOTH_TASK,
        "dataset_id": dataset_id,
        "input_source": input_source,
        "lowpass_modes": lowpass_modes,
        "time_window": time_window,
        "output_path": str(output_path),
    }
    logger.info("launching smoothing worker for %s", dataset_id)
    result, sent_sigterm, returncode, elapsed = _supervise(
        request, time_cap_seconds=time_cap_seconds, term_slack=0.0
    )
    if result is not None and "description" in result:
        return IsolatedSmooth(result["description"], None, None, elapsed)
    if sent_sigterm:
        error_code = "SearchTimeout"
        message = (
            "The smoothing was terminated at its wall-clock cap; it has no "
            "cooperative stop, so nothing was kept."
        )
    elif result is not None and "smooth_error" in result:
        error_code = result["smooth_error"]["code"]
        message = result["smooth_error"]["message"]
    elif result is not None:
        error_code = "SearchWorkerError"
        message = _traceback_message(result["error"])
    else:
        error_code = "SearchWorkerDied"
        message = (
            f"The smoothing subprocess died on its own with return code "
            f"{returncode} (a negative value is a signal number); nothing was "
            "kept."
        )
    return IsolatedSmooth(None, error_code, message, elapsed)


def run_isolated_pinn(
    *,
    dataset_id: str,
    input_source: dict[str, Any],
    pinn: dict[str, Any],
    output_path: Path,
    device: str,
    time_cap_seconds: float | None,
) -> IsolatedSmooth:
    request: dict[str, Any] = {
        TASK_KEY: PINN_TASK,
        "dataset_id": dataset_id,
        "input_source": input_source,
        "pinn": pinn,
        "output_path": str(output_path),
        "max_seconds": (
            None
            if time_cap_seconds is None
            else max(0.0, time_cap_seconds - PINN_WRITE_SECONDS)
        ),
    }
    _add_device(request, device)
    logger.info("launching PINN worker for %s", dataset_id)
    result, sent_sigterm, returncode, elapsed = _supervise(
        request, time_cap_seconds=time_cap_seconds, term_slack=TERM_SLACK_SECONDS
    )
    if result is not None and "description" in result:
        return IsolatedSmooth(
            result["description"], None, None, elapsed, result["resources"]
        )
    if sent_sigterm:
        error_code = "SearchTimeout"
        message = (
            "The PINN fit was terminated at its wall-clock cap after its own "
            "deadline had passed; nothing was kept."
        )
    elif result is not None and "refit_error" in result:
        error_code = result["refit_error"]["code"]
        message = result["refit_error"]["message"]
    elif result is not None:
        error_code = "SearchWorkerError"
        message = _traceback_message(result["error"])
    else:
        error_code = "SearchWorkerDied"
        message = (
            f"The PINN subprocess died on its own with return code "
            f"{returncode} (a negative value is a signal number); nothing was "
            "kept."
        )
    return IsolatedSmooth(
        None,
        error_code,
        message,
        elapsed,
        None if result is None else result["resources"],
    )


def frame_request(
    *,
    method: str,
    trim: int | dict[str, int] | str,
    weak: dict[str, Any] | None,
    surrogate: dict[str, Any] | None,
) -> dict[str, Any]:
    keys: dict[str, Any] = {}
    if method != "finite_diff":
        keys["method"] = method
    if trim != 0:
        keys["trim"] = trim
    if weak is not None:
        keys["weak"] = weak
    if surrogate is not None:
        keys["surrogate"] = surrogate
    return keys


@dataclass(frozen=True)
class IsolatedTask:

    value: Any
    error_code: str | None
    error_message: str | None
    wallclock_seconds: float
    resources: dict[str, Any] | None = None


def run_isolated_task(
    request: dict[str, Any],
    *,
    time_cap_seconds: float | None,
    cooperative: bool,
) -> IsolatedTask:
    term_slack = TERM_SLACK_SECONDS if cooperative else 0.0
    logger.info("launching %s worker for %s", request[TASK_KEY], request["dataset_id"])
    result, sent_sigterm, returncode, elapsed = _supervise(
        request, time_cap_seconds=time_cap_seconds, term_slack=term_slack
    )
    resources = None if result is None else result.get("resources")
    if result is not None and "value" in result:



        return IsolatedTask(result["value"], None, None, elapsed, resources)
    if sent_sigterm:
        error_code = "SearchTimeout"
        message = (
            f"The {request[TASK_KEY]} call was stopped at its hard limit "
            f"({time_cap_seconds} s plus {term_slack} s); nothing was kept."
        )
    elif result is not None and "task_error" in result:
        error_code = result["task_error"]["code"]
        message = result["task_error"]["message"]
    elif result is not None:
        error_code = "SearchWorkerError"
        message = _traceback_message(result["error"])
    else:
        error_code = "SearchWorkerDied"
        message = (
            f"The {request[TASK_KEY]} subprocess died on its own with return "
            f"code {returncode} (a negative value is a signal number); nothing "
            "was kept."
        )
    return IsolatedTask(None, error_code, message, elapsed, resources)


_WORKER_FAILURES: Final = frozenset(
    {"SearchTimeout", "SearchWorkerError", "SearchWorkerDied"}
)


def _forward_precheck(
    law: dict[str, Any], input_source: dict[str, Any] | None
) -> str | None:
    from kdagent.data_source import table_task
    from kdagent.lineage import lhs_inapplicable

    inapplicable = lhs_inapplicable(law["lhs"])
    if inapplicable is not None:
        return inapplicable
    task_kind = table_task(input_source)
    if task_kind is not None:
        return "ode" if task_kind == "ode" else "tabular"
    return None


def measure_selection_forward(
    *,
    run_id: str,
    law: dict[str, Any] | None,
    record_hash: str | None,
    dataset_id: str,
    input_source: dict[str, Any] | None,
    input_record: dict[str, Any],
    remaining_seconds: float | None,
) -> dict[str, Any]:
    record = _forward_record(run_id, law, record_hash, input_record)
    unmeasurable = _unmeasurable(law, input_source, remaining_seconds)
    if unmeasurable is not None:
        status, reason = unmeasurable
        return {**record, "status": status, "reason": reason}
    cap = (
        FORWARD_WORKER_SECONDS
        if remaining_seconds is None
        else min(FORWARD_WORKER_SECONDS, remaining_seconds)
    )
    return _forward_task(
        record, cast(dict[str, Any], law), dataset_id, input_source, cap
    )


def _forward_record(
    run_id: str,
    law: dict[str, Any] | None,
    record_hash: str | None,
    input_record: dict[str, Any],
) -> dict[str, Any]:
    from kdagent.lineage import FORWARD_FRAME

    return {
        "frame": FORWARD_FRAME,
        "run_id": run_id,
        "record_hash": record_hash,
        "law": deepcopy(law),
        "input": deepcopy(input_record),
        "input_identity": None,
        "method": FORWARD_METHOD,
        "max_step": None,
        "budget": {"forward_seconds": None, "worker_seconds": None},
        "status": "not_measured",
        "value": None,
        "reason": None,
        "warning": "",
        "interval": None,
        "boundary": None,
        "seconds": {"load": None, "forward": None, "worker": None},
        "error": None,
        "resources": None,
    }


def _unmeasurable(
    law: dict[str, Any] | None,
    input_source: dict[str, Any] | None,
    remaining_seconds: float | None,
) -> tuple[str, str] | None:
    if law is None:
        return "inapplicable", "no_law"
    inapplicable = _forward_precheck(law, input_source)
    if inapplicable is not None:
        return "inapplicable", inapplicable
    coefficients = law["coefficients"]
    if coefficients is None or any(value is None for value in coefficients):
        return "not_measured", "unknown_coefficient"
    if remaining_seconds is not None and remaining_seconds <= 0.0:
        return "not_measured", "budget_exhausted"
    return None


def _forward_task(
    record: dict[str, Any],
    law: dict[str, Any],
    dataset_id: str,
    input_source: dict[str, Any] | None,
    cap: float,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        TASK_KEY: FORWARD_TASK,
        "dataset_id": dataset_id,
        "terms": list(law["support"]),
        "coefficients": [float(value) for value in law["coefficients"]],
        "lhs": law["lhs"],
        "max_seconds": cap,
    }
    if input_source is not None:
        request["input_source"] = input_source
    task = run_isolated_task(request, time_cap_seconds=cap, cooperative=False)
    record["budget"]["worker_seconds"] = cap
    record["seconds"]["worker"] = task.wallclock_seconds
    record["resources"] = task.resources
    if task.value is None:
        failed = task.error_code in _WORKER_FAILURES
        return {
            **record,
            "reason": "worker_failed" if failed else "raised",
            "error": {"code": task.error_code, "message": task.error_message},
        }
    return _forward_outcome(record, task.value)


def _forward_outcome(record: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    record["seconds"]["load"] = value["load_seconds"]
    record["seconds"]["forward"] = value["forward_seconds"]
    if "inapplicable" in value:
        return {**record, "status": "inapplicable", "reason": value["inapplicable"]}
    record["budget"]["forward_seconds"] = value["forward_max_seconds"]
    forward = value["forward"]
    if forward is None:
        return {**record, "reason": "timeout"}
    return {
        **record,
        "status": "measured",
        "value": forward["value"],
        "reason": forward["reason"],
        "warning": forward["warning"],
        "input_identity": forward["input_identity"],
        "interval": forward["interval"],
        "boundary": forward["boundary"],
    }


def prepare_file(
    path: Path,
    load_options: dict[str, Any] | None,
    *,
    task_kind: str | None = None,
    target_state: str | None = None,
    time_cap_seconds: float | None,
) -> dict[str, Any]:
    return _prepare_isolated(
        {
            TASK_KEY: "prepare_input",
            "path": str(path.resolve()),
            "load_options": load_options,
            "task_kind": task_kind,
            "target_state": target_state,
        },
        time_cap_seconds,
    )


def prepare_catalog(dataset_id: str, time_cap_seconds: float | None) -> dict[str, Any]:
    return _prepare_isolated(
        {TASK_KEY: "prepare_input", "dataset_id": dataset_id}, time_cap_seconds
    )


def _prepare_isolated(
    request: dict[str, Any], time_cap_seconds: float | None
) -> dict[str, Any]:
    result, sent_sigterm, returncode, _elapsed = _supervise(
        request, time_cap_seconds=time_cap_seconds, term_slack=0.0
    )
    if result is None:
        raise ValueError(
            f"input preparation produced no result (exit={returncode}, "
            f"terminated={sent_sigterm})"
        )
    if "error" in result:
        raise ValueError(result["error"])
    return result


def _supervise(
    request: dict[str, Any],
    *,
    time_cap_seconds: float | None,
    term_slack: float,
    llm4ed_api_key: str | None = None,
) -> tuple[dict[str, Any] | None, bool, int | None, float]:
    from kd.harness import ProcessSupervisor







    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    if llm4ed_api_key is not None and request.get("algorithm") == "llm4ed":
        env["OPENAI_API_KEY"] = llm4ed_api_key
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="kdagent-episode-") as tmp:
        request_path = os.path.join(tmp, "request.json")
        result_path = os.path.join(tmp, "result.json")
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(request, f)





        worker = ProcessSupervisor(
            [sys.executable, "-m", _WORKER_MODULE, request_path, result_path],
            env=env,
            hard_seconds=(
                None if time_cap_seconds is None else time_cap_seconds + term_slack
            ),
            grace_seconds=KILL_GRACE_SECONDS,
        )
        _LIVE_GROUPS.add(worker.pid)
        try:
            supervised = worker.wait()
        finally:
            _LIVE_GROUPS.discard(worker.pid)
        elapsed = time.perf_counter() - start
        result = _read_result(result_path)
    return result, supervised.stop_reason == "timeout", supervised.returncode, elapsed


def _group_alive(pgid: int) -> bool:
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pgid, os.WNOHANG)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_live_workers(grace_seconds: float) -> None:
    groups = set(_LIVE_GROUPS)
    for pgid in groups:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while groups and time.monotonic() < deadline:
        time.sleep(0.1)
        groups = {pgid for pgid in groups if _group_alive(pgid)}
    for pgid in groups:
        logger.warning("SIGKILL to worker group %d", pgid)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)


def _training_from(
    result: dict[str, Any] | None,
    *,
    sent_sigterm: bool,
    returncode: int | None,
    elapsed: float,
) -> IsolatedTraining:
    if result is not None and "file" in result:
        file = Path(result["file"])
        return IsolatedTraining(
            status="completed",
            error_type=None,
            error_message=None,
            file=file,
            file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            key=result["key"],
            train_seconds=result["train_seconds"],
            epochs=result["epochs"],
            final_loss=result["final_loss"],
            identity=result["identity"],
            resources=result["resources"],
        )
    if sent_sigterm:
        error_type = "SearchTimeout"
        message = (
            "The training was terminated at its wall-clock cap; a surrogate "
            "training has no cooperative stop, so nothing was saved."
        )
    elif result is not None:
        error_type = "SearchWorkerError"
        message = _traceback_message(result["error"])
    else:
        error_type = "SearchWorkerDied"
        message = (
            f"The training subprocess died on its own with return code "
            f"{returncode} (a negative value is a signal number); nothing was "
            "saved."
        )
    return IsolatedTraining(
        status="raised",
        error_type=error_type,
        error_message=message,
        file=None,
        file_sha256=None,
        key=None,
        train_seconds=None,
        epochs=None,
        final_loss=None,
        identity=None,
        resources=None if result is None else result["resources"],
    )


def _refine_from(
    result: dict[str, Any] | None,
    *,
    sent_sigterm: bool,
    returncode: int | None,
    elapsed: float,
) -> IsolatedRefine:
    if result is not None and "refit" in result:
        return IsolatedRefine(result["refit"], None, None, elapsed)
    if sent_sigterm:
        error_code = "SearchTimeout"
        message = (
            "The refit was terminated at its wall-clock cap; a refit has no "
            "cooperative stop, so nothing was kept."
        )
    elif result is not None and "refit_error" in result:
        error_code = result["refit_error"]["code"]
        message = result["refit_error"]["message"]
    elif result is not None:
        error_code = "SearchWorkerError"
        message = _traceback_message(result["error"])
    else:
        error_code = "SearchWorkerDied"
        message = (
            f"The refit subprocess died on its own with return code "
            f"{returncode} (a negative value is a signal number); nothing was "
            "kept."
        )
    return IsolatedRefine(None, error_code, message, elapsed)


def _read_result(result_path: str) -> dict[str, Any] | None:
    try:
        with open(result_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except FileNotFoundError:
        return None
    return data


def _episode_from(
    entry: PlanEntry,
    *,
    run_dir: Path,
    result: dict[str, Any] | None,
    sent_sigterm: bool,
    returncode: int | None,
    elapsed: float,
    time_cap_seconds: float | None,
) -> IsolatedEpisode:
    from kd.harness import EpisodeOutcome
    from kd.search import RunRecord

    if result is not None and "status" in result:
        error_type = result["error_type"]
        error_message = result["error_message"]
        if sent_sigterm and error_type == "SearchInterrupted":






            error_type = "SearchTimeout"
            error_message = (
                f"The search was terminated after exceeding the wall-clock cap "
                f"of {time_cap_seconds} s; the run directory is sealed "
                f"(status=raised). {_best_so_far_clause(run_dir)}"
            )
        outcome = EpisodeOutcome(
            entry_index=0,
            entry=entry,
            status=result["status"],
            record=(
                None
                if result["record"] is None
                else RunRecord.from_dict(result["record"])
            ),
            error_type=error_type,
            error_message=error_message,
            wallclock_seconds=result["wallclock_seconds"],
            run_id=result["run_id"],
            run_dir=None if result["run_dir"] is None else Path(result["run_dir"]),
            lineage=result["lineage"],
        )
        network, network_error = _trained_network(result.get("trained_surrogate"))
        return IsolatedEpisode(
            outcome=outcome,
            stopped_by_time_cap=result["stopped_by_time_cap"],
            stopped_by_evaluation_cap=result.get("stopped_by_evaluation_cap", False),
            platform_nmse=result.get("platform_nmse"),
            platform_coefficients=result.get("platform_coefficients"),
            platform_eval_error=result.get("platform_eval_error"),
            platform_eval_dropped=result.get("platform_eval_dropped"),
            unavailable_reason=result.get("unavailable_reason"),
            resources=result["resources"],
            trained_surrogate=network,
            trained_surrogate_error=(
                result.get("trained_surrogate_error")
                if network_error is None
                else network_error
            ),
        )
    if sent_sigterm or result is None:





        sealed = _outcome_from_sealed_dir(
            entry,
            run_dir,
            elapsed=elapsed,
            sent_sigterm=sent_sigterm,
            returncode=returncode,
            time_cap_seconds=time_cap_seconds,
        )
        if sealed is not None:
            stopped_by_time_cap, stopped_by_evaluation_cap = _cap_verdicts(
                sealed.record
            )
            return IsolatedEpisode(
                outcome=sealed,
                stopped_by_time_cap=sent_sigterm or stopped_by_time_cap,
                stopped_by_evaluation_cap=stopped_by_evaluation_cap,
                resources=None if result is None else result["resources"],
            )
    if result is not None and not sent_sigterm:













        sealed = _outcome_from_sealed_dir(
            entry,
            run_dir,
            elapsed=elapsed,
            sent_sigterm=False,
            returncode=returncode,
            time_cap_seconds=time_cap_seconds,
        )
        if sealed is not None and sealed.status == "completed":
            stopped_by_time_cap, stopped_by_evaluation_cap = _cap_verdicts(
                sealed.record
            )
            return IsolatedEpisode(
                outcome=sealed,
                stopped_by_time_cap=stopped_by_time_cap,
                stopped_by_evaluation_cap=stopped_by_evaluation_cap,
                post_seal_error=_traceback_message(result["error"]),
                resources=result["resources"],
            )




    dir_exists = run_dir.is_dir()
    if sent_sigterm:



        error_type = "SearchTimeout"
        message = (
            f"The search was terminated after exceeding the wall-clock cap of "
            f"{time_cap_seconds} s; "
        ) + (
            f"the run directory was not sealed. {_best_so_far_clause(run_dir)}"
            if dir_exists
            else "no run directory had been created yet."
        )
    elif result is not None:



        error_type = "SearchWorkerError"
        message = _traceback_message(result["error"])
    else:
        error_type = "SearchWorkerDied"
        message = (
            f"The search subprocess died on its own with return code "
            f"{returncode} (a negative value is a signal number); "
        ) + (
            f"the run directory was not sealed. {_best_so_far_clause(run_dir)}"
            if dir_exists
            else "no run directory had been created yet."
        )
    outcome = EpisodeOutcome(
        entry_index=0,
        entry=entry,
        status="raised",
        record=None,
        error_type=error_type,
        error_message=message,


        error_traceback=None if result is None else str(result["error"]),
        wallclock_seconds=elapsed,








        run_id=run_dir.name if dir_exists else None,
        run_dir=run_dir if dir_exists else None,
        lineage=None,
    )
    return IsolatedEpisode(
        outcome=outcome,
        stopped_by_time_cap=False,
        resources=None if result is None else result["resources"],
    )


def _trained_network(
    network: dict[str, Any] | None,
) -> tuple[TrainedNetwork | None, str | None]:
    if network is None:
        return None, None
    file = Path(network["file"])
    try:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"[:_ERROR_MESSAGE_MAX_CHARS]
    return TrainedNetwork(
        file=file,
        file_sha256=digest,
        identity=network["identity"],
        train_seconds=network["train_seconds"],
        epochs=network["epochs"],
        final_loss=network["final_loss"],
    ), None


def _outcome_from_sealed_dir(
    entry: PlanEntry,
    run_dir: Path,
    *,
    elapsed: float,
    sent_sigterm: bool,
    returncode: int | None,
    time_cap_seconds: float | None,
) -> EpisodeOutcome | None:
    from kd.harness import EpisodeOutcome
    from kd.search import RunDirPaths, RunRecord

    paths = RunDirPaths(root=run_dir)
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    status: str = manifest["status"]
    record = None
    error_type: str | None = None
    error_message: str | None = None
    if status == "completed":
        record = RunRecord.from_dict(
            json.loads(paths.record.read_text(encoding="utf-8"))
        )
    elif status == "raised":
        error_type = "SearchTimeout" if sent_sigterm else "SearchWorkerDied"
        cause = (
            f"was terminated after exceeding the wall-clock cap of {time_cap_seconds} s"
            if sent_sigterm
            else f"subprocess died on its own (return code {returncode})"
        )
        error_message = (
            f"The search {cause}; the run directory is sealed (status=raised) "
            f"and the error detail was lost with the result file. "
            f"{_best_so_far_clause(run_dir)}"
        )
    return EpisodeOutcome(
        entry_index=0,
        entry=entry,
        status=status,
        record=record,
        error_type=error_type,
        error_message=error_message,
        wallclock_seconds=elapsed,
        run_id=manifest["run_id"],
        run_dir=run_dir,
        lineage=manifest["lineage"],
    )


def _best_so_far_clause(run_dir: Path) -> str:
    best = _best_from_events(run_dir)
    if best is None:
        return "Nothing usable came out of this run: events records no completed iteration."
    expression, score = best
    return (
        f"Best result of the last completed iteration in events.jsonl: "
        f"{expression} (score={score}) -- an expression string only, not a "
        f"structured law, so it cannot be passed to submit_answer."
    )


def _best_from_events(run_dir: Path) -> tuple[str, float | None] | None:
    from kd.search import RunDirPaths

    events = RunDirPaths(root=run_dir).events
    try:
        lines = events.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines[-2:]):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        expression = event["best_expression"]
        if expression is None:
            return None
        return expression, event["best_score"]
    return None


if __name__ == "__main__":
    main(sys.argv[1:])
