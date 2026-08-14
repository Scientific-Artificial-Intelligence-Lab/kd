
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from kd.search.discover.engine_types import SearchProgressCallback
from kd.search.discover.pinn.cycle import PINNCycleResult, PINNCycleRunner

DEFAULT_HEARTBEAT_ITERATIONS = 10
MAX_LOG_EXPRESSION_LENGTH = 120
HEARTBEAT_FILENAME = "search_heartbeat.jsonl"

logger = logging.getLogger(__name__)


def run_pinn_cycle_with_observability(
    runner: PINNCycleRunner,
    *,
    checkpoint_dir: Path,
    heartbeat_iterations: int = DEFAULT_HEARTBEAT_ITERATIONS,
    run_logger: logging.Logger | None = None,
) -> PINNCycleResult:
    if heartbeat_iterations < 0:
        raise ValueError("heartbeat_iterations must be non-negative.")
    active_logger = run_logger or logger
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pretrain_result = _run_pretrain_with_checkpoint(runner, checkpoint_dir, start)
    cycle_metrics: list[dict[str, float]] = []
    evaluator = runner._rebuild_after_pinn()
    for cycle_idx in range(runner._pinn_config.n_cycles):
        _run_search_stage(
            runner,
            evaluator,
            cycle_idx,
            checkpoint_dir,
            heartbeat_iterations,
            start,
            active_logger,
        )
        metrics, pinn_ok = runner._run_pinn_phase(cycle_idx, evaluator)
        cycle_metrics.append(metrics)
        _write_pinn_checkpoint(checkpoint_dir, cycle_idx, metrics, pinn_ok, start)
        if pinn_ok:
            evaluator = runner._rebuild_after_pinn()

    final_state = _run_final_search(
        runner,
        evaluator,
        checkpoint_dir,
        heartbeat_iterations,
        start,
        active_logger,
    )
    final_state = runner._finalize_run(final_state, evaluator)
    _write_stage_checkpoint(
        checkpoint_dir,
        "999_final_stability",
        _state_checkpoint_payload("final_stability", final_state, start),
    )
    return PINNCycleResult(final_state, cycle_metrics, pretrain_result)


def _run_pretrain_with_checkpoint(
    runner: PINNCycleRunner,
    checkpoint_dir: Path,
    start: float,
) -> Any:
    pretrain_result = runner._pretrain()
    _write_stage_checkpoint(
        checkpoint_dir,
        "000_pretrain",
        _pretrain_checkpoint_payload(pretrain_result, start),
    )
    return pretrain_result


def _run_search_stage(
    runner: PINNCycleRunner,
    evaluator: Any,
    cycle_idx: int,
    checkpoint_dir: Path,
    heartbeat_iterations: int,
    start: float,
    run_logger: logging.Logger,
) -> Any:
    stage = f"cycle_{cycle_idx:02d}_search"
    n_iterations = runner._cycle_iterations(cycle_idx)
    run_logger.info("MODE2 %s start n_iterations=%d", stage, n_iterations)


    state = runner._engine.run_cycle(
        evaluator,
        n_iterations,
        progress_callback=_make_progress_callback(
            stage,
            n_iterations,
            checkpoint_dir,
            heartbeat_iterations,
            start,
            run_logger,
        ),
        cycle_idx=cycle_idx,
    )
    checkpoint_name = f"{cycle_idx + 1:03d}_{stage}"
    _write_stage_checkpoint(
        checkpoint_dir,
        checkpoint_name,
        _state_checkpoint_payload(stage, state, start),
    )
    run_logger.info("MODE2 %s done best_reward=%.6f", stage, state.best_reward)
    return state


def _run_final_search(
    runner: PINNCycleRunner,
    evaluator: Any,
    checkpoint_dir: Path,
    heartbeat_iterations: int,
    start: float,
    run_logger: logging.Logger,
) -> Any:
    cycle_idx = runner._pinn_config.n_cycles
    n_iterations = runner._cycle_iterations(cycle_idx)
    run_logger.info("MODE2 final_search start n_iterations=%d", n_iterations)


    state = runner._engine.run_cycle(
        evaluator,
        n_iterations,
        progress_callback=_make_progress_callback(
            "final_search",
            n_iterations,
            checkpoint_dir,
            heartbeat_iterations,
            start,
            run_logger,
        ),
        cycle_idx=cycle_idx,
    )
    _write_stage_checkpoint(
        checkpoint_dir,
        "900_final_search",
        _state_checkpoint_payload("final_search", state, start),
    )
    run_logger.info("MODE2 final_search done best_reward=%.6f", state.best_reward)
    return state


def _make_progress_callback(
    stage: str,
    n_iterations: int,
    checkpoint_dir: Path,
    heartbeat_iterations: int,
    start: float,
    run_logger: logging.Logger,
) -> SearchProgressCallback:
    def _callback(iteration: int, metrics: dict[str, float], engine: Any) -> None:
        if not should_log_search_iteration(
            iteration,
            n_iterations,
            heartbeat_iterations,
        ):
            return
        payload = search_heartbeat_payload(
            stage,
            iteration,
            n_iterations,
            metrics,
            engine,
            start,
        )
        _log_search_heartbeat(run_logger, payload)
        append_jsonl(checkpoint_dir / HEARTBEAT_FILENAME, payload)

    return _callback


def should_log_search_iteration(
    iteration: int,
    n_iterations: int,
    heartbeat_iterations: int,
) -> bool:
    if heartbeat_iterations <= 0:
        return False
    return (
        iteration == 1
        or iteration == n_iterations
        or iteration % heartbeat_iterations == 0
    )


def search_heartbeat_payload(
    stage: str,
    iteration: int,
    n_iterations: int,
    metrics: dict[str, float],
    engine: Any,
    start: float,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "iteration": iteration,
        "n_iterations": n_iterations,
        "elapsed_seconds": time.time() - start,
        "metrics": _float_metrics(metrics),
        "global_best": {
            "reward": float(engine.best_reward),
            "expression": engine.best_expression,
        },
    }


def write_stage_checkpoint(
    checkpoint_dir: Path,
    name: str,
    payload: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{name}.json"


    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _pretrain_checkpoint_payload(pretrain: Any, start: float) -> dict[str, Any]:
    return {
        "stage": "pretrain",
        "elapsed_seconds": time.time() - start,
        "pretrain": {
            "train_loss": float(pretrain.train_loss),
            "val_loss": float(pretrain.val_loss),
            "epochs_run": int(pretrain.epochs_run),
            "stopped_early": bool(pretrain.stopped_early),
        },
    }


def _state_checkpoint_payload(stage: str, state: Any, start: float) -> dict[str, Any]:
    return {
        "stage": stage,
        "elapsed_seconds": time.time() - start,
        "best_reward": float(state.best_reward),
        "best_expression": state.best_expression,
        "best_terms": state.best_result_terms,
        "best_coefficients": state.best_result_coefficients,
        "stability_selection": (state.extras or {}).get("stability_selection"),
    }


def _write_pinn_checkpoint(
    checkpoint_dir: Path,
    cycle_idx: int,
    metrics: dict[str, float],
    pinn_ok: bool,
    start: float,
) -> None:
    write_stage_checkpoint(
        checkpoint_dir,
        f"{cycle_idx + 1:03d}_cycle_{cycle_idx:02d}_pinn",
        {
            "stage": f"cycle_{cycle_idx:02d}_pinn",
            "elapsed_seconds": time.time() - start,
            "pinn_ok": pinn_ok,
            "metrics": _float_metrics(metrics),
        },
    )


def _write_stage_checkpoint(
    checkpoint_dir: Path,
    name: str,
    payload: dict[str, Any],
) -> None:
    write_stage_checkpoint(checkpoint_dir, name, payload)


def _log_search_heartbeat(
    run_logger: logging.Logger,
    payload: dict[str, Any],
) -> None:
    metrics = payload["metrics"]
    global_best = payload["global_best"]
    run_logger.info(
        "MODE2 %s %d/%d reward_max=%.6f best=%.6f valid=%.0f "
        "unique=%.0f expr=%s",
        payload["stage"],
        payload["iteration"],
        payload["n_iterations"],
        _metric_value(metrics, "reward_max"),
        global_best["reward"],
        _metric_value(metrics, "n_eval_valid"),
        _metric_value(metrics, "n_unique"),
        _short_expression(global_best["expression"]),
    )


def _float_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _metric_value(metrics: dict[str, float], key: str) -> float:
    return float(metrics.get(key, math.nan))


def _short_expression(expression: str) -> str:
    if len(expression) <= MAX_LOG_EXPRESSION_LENGTH:
        return expression
    return f"{expression[:MAX_LOG_EXPRESSION_LENGTH]}..."


__all__ = [
    "DEFAULT_HEARTBEAT_ITERATIONS",
    "HEARTBEAT_FILENAME",
    "append_jsonl",
    "run_pinn_cycle_with_observability",
    "search_heartbeat_payload",
    "should_log_search_iteration",
    "write_stage_checkpoint",
]
