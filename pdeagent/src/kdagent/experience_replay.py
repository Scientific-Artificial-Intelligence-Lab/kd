
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast

from kdagent.experience import _subset_calls, check_conditioning, library_terms

MENU_FORMAT = "kd-experience-pipelines-v1"
_MENU = Path(__file__).parent / "_assets" / "experience" / "pipelines.json"


def load_menu(path: Path | None = None) -> list[dict[str, Any]]:
    document = json.loads((path or _MENU).read_text(encoding="utf-8"))
    if set(document) != {"format", "pipelines"} or document["format"] != MENU_FORMAT:
        raise ValueError(f"pipeline menu must use {MENU_FORMAT}")
    rows = document["pipelines"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("pipelines must be a non-empty list")
    for row in rows:
        _check_recipe(row)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("pipeline ids must be unique")
    return rows


def _check_recipe(row: dict[str, Any]) -> None:
    if set(row) != {"id", "pipeline", "preparation", "frame", "selection"}:
        raise ValueError("pipeline recipe has missing or unknown fields")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError("pipeline id must be a non-empty string")
    pipeline = row["pipeline"]
    if pipeline["algorithm"] != "pysindy" or pipeline["mode"] != "default":
        raise ValueError("offline replay accepts only pysindy/default (no model calls)")
    if set(pipeline) != {"algorithm", "mode", "config", "input", "conditioning"}:
        raise ValueError("pipeline has missing or unknown fields")
    conditioning = pipeline["conditioning"]
    if set(conditioning) != {"library_rule"}:
        raise ValueError("replay conditioning must use library_rule, not named terms")
    check_conditioning(conditioning)
    if "terms" in pipeline["config"] or "library" in pipeline["config"]:
        raise ValueError("config must not carry terms or library")
    _check_preparation(row["preparation"], pipeline["input"])
    selection = row["selection"]
    if set(selection) != {"subsets_up_to", "ratio"}:
        raise ValueError("selection requires subsets_up_to and ratio")
    if type(selection["subsets_up_to"]) is not int or selection["subsets_up_to"] < 1:
        raise ValueError("subsets_up_to must be a positive integer")
    ratio = selection["ratio"]
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or ratio < 1
    ):
        raise ValueError("ratio must be finite and at least 1")
    if set(row["frame"]) != {"method", "trim"}:
        raise ValueError("frame requires method and trim")
    if row["frame"]["method"] not in {"finite_diff", "fd2"}:
        raise ValueError("replay frame method must be finite_diff or fd2")


def _check_preparation(preparation: dict[str, Any] | None, kind: str) -> None:
    if preparation is None:
        if kind != "raw":
            raise ValueError("raw pipeline requires no preparation")
        return
    method = preparation["method"]
    if method == "savgol":
        allowed = {"method", "savgol_window"}
    elif method == "network":
        allowed = {"method", "network_recipe", "seed"}
    else:
        raise ValueError("preparation method must be savgol or network")
    if kind != method or set(preparation) != allowed:
        raise ValueError("preparation fields must match the pipeline input kind")


def _frame(recipe: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    frame: dict[str, Any] = copy.deepcopy(recipe["frame"])
    trim = frame["trim"]
    if isinstance(trim, dict) and set(trim) == {"spatial", "time"}:
        frame["trim"] = {
            axis["name"]: trim[
                "time" if axis["name"] == report["lhs_axis"] else "spatial"
            ]
            for axis in report["axes"]
        }
    return frame


def pipeline_recipe(recipe: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    _check_recipe(recipe)
    return _recorded_pipeline(recipe, _frame(recipe, report))


def _config(recipe: dict[str, Any]) -> dict[str, Any]:
    import kd

    schema = next(
        row for row in kd.instrument_schemas() if row["algorithm"] == "pysindy"
    )
    defaults = {
        field["name"]: field["default"]
        for field in schema["fields"]
        if field["settable_from"] == "json" and field["name"] != "terms"
    }
    defaults.update(
        {
            param["name"]: param["default"]
            for param in schema["facade_params"]
            if param["name"] != "seed"
        }
    )
    return {**defaults, **recipe["pipeline"]["config"]}


def _recorded_pipeline(recipe: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    pipeline: dict[str, Any] = copy.deepcopy(recipe["pipeline"])
    pipeline["config"] = _config(recipe)
    preparation = recipe["preparation"]
    if preparation is not None:
        pipeline["input_steps"] = [
            {"tool": "smooth_input", "dataset_id": "the pinned input", **preparation}
        ]
    pipeline["refinement"] = [
        {
            "tool": "select_support",
            "run_id": "the search run",
            **recipe["selection"],
            **frame,
        },
        {
            "tool": "refine_candidate",
            "run_id": "the search run",
            "terms": "the chosen terms, as written",
            **frame,
        },
        {
            "tool": "submit_answer",
            "run_id": "the search run",
            "law": "the refined law",
            "refinement_id": "the refinement receipt",
        },
    ]
    return pipeline


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class _Refused(Exception):
    pass


class _Trial:
    def __init__(self, workspace: Path, recipe: dict[str, Any]) -> None:
        self.workspace = workspace
        self.started = time.monotonic()
        self.result: dict[str, Any] = {
            "status": "running",
            "law": None,
            "report": None,
            "pipeline": _recorded_pipeline(recipe, recipe["frame"]),
            "calls": [],
            "evaluations": None,
            "selection_evaluations": 0,
            "search_wall_seconds": None,
            "wall_seconds": 0.0,
        }

    def save(self) -> None:
        self.result["wall_seconds"] = time.monotonic() - self.started
        _write(self.workspace / "replay.json", self.result)

    def call(
        self, tools: dict[str, Callable[..., Any]], name: str, **arguments: Any
    ) -> dict[str, Any]:
        from langgraph.types import Command

        returned = tools[name](**arguments)
        if isinstance(returned, Command):
            (message,) = cast(dict[str, Any], returned.update)["messages"]
            returned = json.loads(str(message.content))
        self.result["calls"].append(
            {"tool": name, "arguments": arguments, "result": returned}
        )
        if name == "run_discovery":
            self.result["evaluations"] = returned["diagnostics"].get("evaluations")
            self.result["search_wall_seconds"] = returned["provenance"].get(
                "wallclock_seconds"
            )
        self.save()
        if returned["status"] != "success":
            self.result["status"] = "failure"
            self.result["failures"] = returned["failures"]
            raise _Refused(name)
        return cast(dict[str, Any], returned)


def _discover(
    trial: _Trial,
    tools: dict[str, Callable[..., Any]],
    recipe: dict[str, Any],
    dataset: str,
    terms: list[str],
) -> str:
    result = trial.call(
        tools,
        "run_discovery",
        dataset_id=dataset,
        tool_call_id="replay-search",
        algorithm="pysindy",
        params={**trial.result["pipeline"]["config"], "terms": terms},
        seed=0,
        checkpoint_every=None,
        checkpoint_keep_last=None,
    )
    return str(result["provenance"]["run_id"])


def _select(
    trial: _Trial,
    tools: dict[str, Callable[..., Any]],
    recipe: dict[str, Any],
    run_id: str,
    terms: list[str],
    frame: dict[str, Any],
) -> dict[str, Any]:
    from kdagent.adapter import _nmse_order, _support_selection

    calls = _subset_calls(
        {"tool": "select_support", "run_id": run_id, **recipe["selection"], **frame},
        terms,
    )
    if len(calls) == 1:
        return trial.call(
            tools,
            calls[0]["tool"],
            **{k: v for k, v in calls[0].items() if k != "tool"},
        )
    if frame.get("trim") == "by_order":
        _resolve_batch_trim(trial, tools, calls)
    entries = []
    for call in calls:
        result = trial.call(
            tools,
            call["tool"],
            **{
                key: value
                for key, value in call.items()
                if key not in {"tool", "choose"}
            },
        )
        entries.extend(result["primary"]["entries"])

    entries = sorted(
        ({**entry, "size": len(entry["terms"])} for entry in entries), key=_nmse_order
    )
    selected = _support_selection(entries, recipe["selection"]["ratio"])
    if selected["chosen"] is None:
        trial.result.update(status="failure", failures=[{"code": "no_scored_sets"}])
        trial.save()
        raise _Refused("select_support")
    return {"primary": {"entries": entries, **selected}}


def _resolve_batch_trim(
    trial: _Trial,
    tools: dict[str, Callable[..., Any]],
    calls: list[dict[str, Any]],
) -> None:
    preflight = calls.pop(0)
    resolved = trial.call(
        tools,
        preflight["tool"],
        **{k: v for k, v in preflight.items() if k not in {"tool", "choose"}},
    )
    for call in calls:
        call["trim"] = resolved["diagnostics"]["frame"]["trim"]


def _run_steps(
    trial: _Trial,
    tools: dict[str, Callable[..., Any]],
    recipe: dict[str, Any],
    description: dict[str, Any],
) -> None:
    report = description["summary"]["report"]
    trial.result["report"] = report
    frame = _frame(recipe, report)
    trial.result["pipeline"] = _recorded_pipeline(recipe, frame)
    trial.save()
    terms = library_terms(recipe["pipeline"]["conditioning"]["library_rule"], report)
    dataset = description["ref"]
    if recipe["preparation"] is not None:
        smoothed = trial.call(
            tools, "smooth_input", dataset_id=dataset, **recipe["preparation"]
        )
        dataset = smoothed["primary"]["dataset_id"]
    run_id = _discover(trial, tools, recipe, dataset, terms)
    selected = _select(trial, tools, recipe, run_id, terms, frame)
    trial.result["selection_evaluations"] = len(selected["primary"]["entries"])
    refined = trial.call(
        tools,
        "refine_candidate",
        run_id=run_id,
        terms=selected["primary"]["chosen"]["terms"],
        **frame,
    )
    law = refined["primary"]["law"]
    trial.call(
        tools,
        "submit_answer",
        law=law,
        run_id=run_id,
        tool_call_id="replay-submit",
        refinement_id=refined["primary"]["refinement_id"],
    )
    trial.result.update(
        status="success",
        law=law,
        selected_nmse=selected["primary"]["chosen"]["fit"]["nmse"],
    )


def execute_pipeline(
    *,
    data_path: Path,
    load_options: dict[str, Any],
    task_kind: str,
    pipeline: dict[str, Any],
    workspace: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    from kdagent.adapter import make_tools
    from kdagent.data_source import save_input
    from kdagent.episode_worker import prepare_file
    from kdagent.session import Session

    _check_recipe(pipeline)
    if task_kind != "pde-evolution":
        raise ValueError("replay menu supports pde-evolution only")
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        raise ValueError("replay workspace must be empty")
    trial = _Trial(workspace, pipeline)
    trial.save()
    try:
        description = prepare_file(
            data_path, load_options, task_kind=None, time_cap_seconds=None
        )
        save_input(workspace, description)
        session = Session(workspace, dataset_id=description["ref"], device=device)
        tools = {tool.__name__: tool for tool in make_tools(session)}
        _run_steps(trial, tools, pipeline, description)
    except _Refused:
        pass
    except Exception as error:
        trial.result.update(
            status="error", error={"type": type(error).__name__, "message": str(error)}
        )
    trial.save()
    return trial.result


def evaluate_pipeline(
    *,
    data_path: Path,
    load_options: dict[str, Any],
    task_kind: str,
    pipeline: dict[str, Any],
    workspace: Path,
    seconds: float,
    device: str = "cpu",
) -> dict[str, Any]:
    _check_recipe(pipeline)
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        raise ValueError("replay workspace must be empty")
    request = {
        "data_path": str(data_path.resolve()),
        "load_options": load_options,
        "task_kind": task_kind,
        "pipeline": pipeline,
        "workspace": str((workspace / "work").resolve()),
        "device": device,
    }
    _write(workspace / "request.json", request)
    return _supervised_trial(workspace, seconds, pipeline)


def _supervised_trial(
    workspace: Path, seconds: float, pipeline: dict[str, Any]
) -> dict[str, Any]:
    from kd.harness import run_supervised

    with (workspace / "worker.log").open("w", encoding="utf-8") as log:
        process = run_supervised(
            [
                sys.executable,
                "-m",
                __name__,
                str((workspace / "request.json").resolve()),
            ],
            env=dict(os.environ),
            hard_seconds=seconds,
            grace_seconds=2,
            stdout=log,
            stderr=log,
            poll_seconds=0.05,
        )
    result_path = workspace / "work" / "replay.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else _Trial(workspace, pipeline).result
    )
    if process.stop_reason == "timeout":
        result.update(status="timeout", law=None)
    elif process.returncode != 0:
        result.update(
            status="error",
            law=None,
            error={
                "returncode": process.returncode,
                "log": str(workspace / "worker.log"),
            },
        )
    result["wall_seconds"] = process.wallclock_seconds
    result["process"] = {
        "returncode": process.returncode,
        "group_emptied": process.group_emptied,
        "stop_reason": process.stop_reason,
    }
    _write(workspace / "replay.json", result)
    return result


def _worker(request_path: Path) -> None:
    from kdagent.episode_worker import stop_live_workers

    def interrupted(_signum: int, _frame: Any) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop_live_workers(0.5)
        raise SystemExit(124)

    signal.signal(signal.SIGTERM, interrupted)
    request = json.loads(request_path.read_text())
    execute_pipeline(
        **{
            **request,
            "data_path": Path(request["data_path"]),
            "workspace": Path(request["workspace"]),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    _worker(parser.parse_args().request)
