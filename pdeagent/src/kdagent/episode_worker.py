
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from types import FrameType

    from collections.abc import Callable

    from kd.harness import EpisodeOutcome, PlanEntry

    from kdagent.platform_eval import PlatformEvaluation

logger = logging.getLogger(__name__)

_WORKER_MODULE: Final = "kdagent.episode_worker"






TERM_SLACK_SECONDS: Final = 120.0
KILL_GRACE_SECONDS: Final = 45.0

TASK_KEY: Final = "task"
TRAIN_TASK: Final = "train_surrogate"


_ERROR_MESSAGE_MAX_CHARS: Final = 2000







class _CapCallback:

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.fired = False

    @property
    def should_stop(self) -> bool:
        if self._inner.should_stop:
            self.fired = True
            return True
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        self._inner.on_experiment_start(algorithm)

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        self._inner.on_iteration_start(iteration, algorithm)

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        self._inner.on_iteration_end(iteration, algorithm, candidates, results)

    def on_experiment_end(self, algorithm: Any) -> None:
        self._inner.on_experiment_end(algorithm)


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


def _run(request: dict[str, Any], *, started_at: float) -> dict[str, Any]:
    from kd.core.equation import sketch_from_dict
    from kd.harness import PlanEntry, RecordingOptions, run_episode
    from kd.search import EvaluationBudgetCallback, WallClockBudgetCallback

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))
    entry = PlanEntry(
        instrument=request["algorithm"],
        dataset_ref=request["dataset_id"],
        seed=request["seed"],
        model_kwargs=request["model_kwargs"],
    )




    extra: dict[str, Any] = {}
    surrogate = request.get("surrogate")
    if surrogate is not None:
        extra["model_factory"] = _injecting_factory(surrogate)
    sketch = request.get("sketch")
    if sketch is not None:
        extra["sketch"] = sketch_from_dict(sketch)
    time_cap_seconds = request["time_cap_seconds"]
    cap: _CapCallback | None = None
    if time_cap_seconds is not None:













        remaining = max(0.001, time_cap_seconds - (time.perf_counter() - started_at))
        cap = _CapCallback(WallClockBudgetCallback(max_seconds=remaining))





    evaluation_cap = request.get("evaluation_cap")
    eval_cap: _CapCallback | None = None
    if evaluation_cap is not None:
        eval_cap = _CapCallback(
            EvaluationBudgetCallback(max_evaluations=int(evaluation_cap))
        )
    extra_callbacks = [cb for cb in (cap, eval_cap) if cb is not None]
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
        dataset=dataset,
        run_dir=Path(request["run_dir"]),




        recording=recording,
        resume_from=request["resume_from"],



        reseed=request.get("reseed", False),
        extra_callbacks=extra_callbacks or None,
        **extra,
    )
    evaluation = _platform_evaluation(dataset, outcome)





    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    return {
        "status": outcome.status,
        "error_type": outcome.error_type,
        "error_message": outcome.error_message,
        "wallclock_seconds": outcome.wallclock_seconds,
        "run_id": outcome.run_id,
        "run_dir": None if outcome.run_dir is None else str(outcome.run_dir),
        "lineage": outcome.lineage,
        "record": None if outcome.record is None else outcome.record.to_dict(),
        "stopped_by_time_cap": cap.fired if cap is not None else False,
        "stopped_by_evaluation_cap": (
            eval_cap.fired if eval_cap is not None else False
        ),
        "platform_nmse": evaluation.nmse,
        "platform_coefficients": evaluation.coefficients,
        "platform_eval_error": evaluation.error,
        "platform_eval_dropped": evaluation.dropped,
    }


def _injecting_factory(surrogate: dict[str, Any]) -> Callable[..., Any]:
    import kd
    from kd.models import load_field_model

    path = Path(surrogate["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != surrogate["sha256"]:
        raise RuntimeError(
            f"surrogate file {path} hashes to {digest}, not the registered "
            f"{surrogate['sha256']}; it changed after the parent checked it"
        )
    module = load_field_model(path)
    key = surrogate["key"]

    def factory(**kwargs: Any) -> Any:
        merged: dict[str, Any] = {**kwargs, key: module}
        return kd.Model(**merged)

    return factory


def _train(request: dict[str, Any]) -> dict[str, Any]:
    import kd
    from kd.models import FieldModel, save_field_model

    from kdagent.data_source import load_dataset

    dataset = load_dataset(request["dataset_id"], request.get("input_source"))
    model = kd.Model(
        algorithm=request["algorithm"],
        seed=request["seed"],
        verbose=False,
        **request["model_kwargs"],
    )
    module, training = model.train_surrogate(dataset)



    field_model = cast("FieldModel", module)




    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    output_path = Path(request["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_field_model(field_model, output_path)
    return {
        "file": str(output_path),
        "key": request["key"],
        "train_seconds": training.elapsed_seconds,
        "epochs": training.epochs_run,
        "final_loss": training.final_loss,
    }


def _prepare(request: dict[str, Any]) -> dict[str, Any]:
    import kd

    from kdagent.data_source import prepare_input

    path = Path(request["path"])
    inventory = None


    if path.suffix.lower() != ".xlsx":
        report = kd.inspect_file(path)
        inventory = {
            "container": report.container,


            "arrays": [
                {"key": row.key, "shape": list(row.shape), "dtype": row.dtype}
                for row in report.arrays
            ],
        }
    dataset, description = prepare_input(path, request["load_options"])
    preview = kd.preview_report(dataset).to_dict()
    description["summary"] = {
        "id": description["ref"],
        "axes": [axis["name"] for axis in preview["axes"]],
        "lhs": preview["lhs_label"],
        "layout": preview["topology"],
        "report": preview,
        "inventory": inventory,
    }
    return description


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
        else:
            result = _run(request, started_at=started_at)
    except Exception:




        result = {"error": traceback.format_exc()}
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
    stopped_by_evaluation_cap: bool = False


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
    if evaluation_cap is not None:


        request["evaluation_cap"] = int(evaluation_cap)
    logger.info("launching episode worker for %s", run_dir.name)
    result, sent_sigterm, returncode, elapsed = _supervise(
        request,
        time_cap_seconds=time_cap_seconds,



        term_slack=TERM_SLACK_SECONDS,
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
    logger.info("launching surrogate training for %s", output_path.name)
    result, sent_sigterm, returncode, elapsed = _supervise(
        request, time_cap_seconds=time_cap_seconds, term_slack=0.0
    )
    return _training_from(
        result, sent_sigterm=sent_sigterm, returncode=returncode, elapsed=elapsed
    )


def prepare_file(
    path: Path,
    load_options: dict[str, Any] | None,
    *,
    time_cap_seconds: float | None,
) -> dict[str, Any]:
    result, sent_sigterm, returncode, _elapsed = _supervise(
        {
            TASK_KEY: "prepare_input",
            "path": str(path.resolve()),
            "load_options": load_options,
        },
        time_cap_seconds=time_cap_seconds,
        term_slack=0.0,
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
) -> tuple[dict[str, Any] | None, bool, int | None, float]:



    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="kdagent-episode-") as tmp:
        request_path = os.path.join(tmp, "request.json")
        result_path = os.path.join(tmp, "result.json")
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(request, f)




        proc = subprocess.Popen(
            [sys.executable, "-m", _WORKER_MODULE, request_path, result_path],
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        sent_sigterm = False
        try:
            if time_cap_seconds is None:
                proc.wait()
            else:
                try:
                    proc.wait(timeout=time_cap_seconds + term_slack)
                except subprocess.TimeoutExpired:
                    sent_sigterm = True
                    _signal_group(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=KILL_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        _signal_group(proc.pid, signal.SIGKILL)
                        proc.wait()
        finally:





            _signal_group(proc.pid, signal.SIGKILL)
        elapsed = time.perf_counter() - start
        result = _read_result(result_path)
    return result, sent_sigterm, proc.returncode, elapsed


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
        )
    if sent_sigterm:
        error_type = "SearchTimeout"
        message = (
            "The training was terminated at its wall-clock cap; a surrogate "
            "training has no cooperative stop, so nothing was saved."
        )
    elif result is not None:
        error_type = "SearchWorkerError"
        message = str(result["error"])[:_ERROR_MESSAGE_MAX_CHARS]
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
    )


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


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
        return IsolatedEpisode(
            outcome=outcome,
            stopped_by_time_cap=result["stopped_by_time_cap"],
            stopped_by_evaluation_cap=result.get("stopped_by_evaluation_cap", False),
            platform_nmse=result.get("platform_nmse"),
            platform_coefficients=result.get("platform_coefficients"),
            platform_eval_error=result.get("platform_eval_error"),
            platform_eval_dropped=result.get("platform_eval_dropped"),
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
            return IsolatedEpisode(
                outcome=sealed,
                stopped_by_time_cap=sent_sigterm,
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
            return IsolatedEpisode(
                outcome=sealed,
                stopped_by_time_cap=False,
                post_seal_error=str(result["error"])[:_ERROR_MESSAGE_MAX_CHARS],
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
        message = str(result["error"])[:_ERROR_MESSAGE_MAX_CHARS]
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
        wallclock_seconds=elapsed,








        run_id=run_dir.name if dir_exists else None,
        run_dir=run_dir if dir_exists else None,
        lineage=None,
    )
    return IsolatedEpisode(outcome=outcome, stopped_by_time_cap=False)


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
