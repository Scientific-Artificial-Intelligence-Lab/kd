
from __future__ import annotations

import hashlib
import json
import posixpath
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, cast

import kd
from kdagent.data_source import InputIntegrityError
from kdagent.experience import CALLER_ERROR_TYPES
from kdagent.surrogates import SurrogateEntry

Observation = dict[str, Any]
EXPORT_FORMAT = "kdb-observations-v2"
GRADED_FORMAT = "kd-agent-graded-searches-v1"
CONTROLLER_ERRORS = {"SearchTimeout", "SearchWorkerDied", "SearchWorkerError"}


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def strict_json(value: str | bytes) -> Any:
    payload = json.loads(value, parse_constant=_nonfinite)
    json.dumps(payload, allow_nan=False)
    return payload


def read_export(
    path: Path,
) -> tuple[list[Observation], list[str], dict[str, str], dict[str, Any]]:
    raw = path.read_bytes()
    header_line, *row_lines = raw.decode("utf-8").splitlines()
    header = strict_json(header_line)
    if header["format"] != EXPORT_FORMAT:
        raise ValueError(
            f"{path} declares format {header['format']!r}; this reader reads "
            f"{EXPORT_FORMAT!r}"
        )
    if header["instrument_schema_sha256"] != kd.instrument_schema_digest():
        raise ValueError(
            f"{path} was exported under another kd declaration set, and the "
            "exporter's kd fills each config's `generations`; export the "
            "records again under the installed kd"
        )
    rows = [strict_json(line) for line in row_lines]
    withheld = sorted(name for name, side in header["groups"].items() if side == "eval")
    observations = [_from_export(row, header["rules_version"]) for row in rows]
    source = {"kind": "kdb-observations", "digest": hashlib.sha256(raw).hexdigest()}
    read = {
        "rules_version": header["rules_version"],
        "produced_under": header["instrument_schema_sha256"],
        "rows": len(rows),
        "examinees": dict(sorted(Counter(row["examinee_id"] for row in rows).items())),
        "status": dict(sorted(Counter(row["status"] for row in rows).items())),
        "refusals": [],
    }
    return observations, withheld, source, read


def _from_export(row: dict[str, Any], rules_version: str) -> Observation:
    verdict = row["verdict"]
    cost = row["cost"]
    return {
        "problem_key": row["problem_key"],
        "group": row["group"],
        "report": row["report"],
        "task_kind": row["task_kind"],
        "pipeline": row["pipeline"],
        "conditions": row["conditions"],
        "instrument_schema_sha256": row["instrument_schema_sha256"],
        "examinee_version": row["examinee_version"],
        "rules_version": rules_version,
        "structure": verdict["lhs_matches"] is True and verdict["jaccard"] == 1.0,
        "max_relative_error": verdict["max_relative_error"],
        "heldout_nrmse": verdict["heldout"]["nrmse"],
        "evaluations": None if cost is None else cost.get("boundary_results"),
        "wall_seconds": row["resources"]["wallclock_seconds"],
    }


class SourceRefusedError(ValueError):

    def __init__(self, refused: list[dict[str, Any]], read: dict[str, Any]) -> None:
        self.refused, self.read = refused, read
        super().__init__("graded searches refused: " + json.dumps(refused))


def _read(path: Path, dependencies: list[dict[str, Any]]) -> bytes:
    raw = path.read_bytes()
    dependencies.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return raw


def _groups(header: dict[str, Any]) -> list[str]:
    if not all(
        header.get(name)
        for name in ("rules_version", "examinee_version", "instrument_schema_sha256")
    ):
        raise ValueError(
            "rules_version, examinee_version and instrument_schema_sha256 "
            "must be recorded"
        )
    if not header["groups"] or any(
        side not in ("train", "eval") for side in header["groups"].values()
    ):
        raise ValueError("groups must declare train or eval")
    excluded = sorted(name for name, side in header["groups"].items() if side == "eval")
    if excluded:
        mapping = header.get("group_problems")
        if not isinstance(mapping, dict) or set(mapping) != set(header["groups"]):
            raise ValueError(
                "evaluation requires singleton group_problems; re-export per problem"
            )
        if any(
            not isinstance(keys, list) or len(keys) != 1 for keys in mapping.values()
        ):
            raise ValueError(
                "each evaluation group must name exactly one independent problem"
            )
        keys = [values[0] for values in mapping.values()]
        if len(set(keys)) != len(keys):
            raise ValueError("problem appears in multiple fold groups")
    return excluded


class IntegrityError(ValueError):
    pass


@contextmanager
def _row_refusal(errors: list[str]) -> Iterator[None]:
    try:
        yield
    except (IntegrityError, InputIntegrityError):
        raise
    except (ValueError, KeyError, TypeError, OSError) as error:
        errors.append(str(error))


def _fold_row(row: dict[str, Any], header: dict[str, Any]) -> None:
    if header["groups"].get(row["group"]) != "train":
        raise IntegrityError(
            "group is undeclared or eval; eval rows must be withheld at source"
        )
    if "group_problems" in header:
        if row["problem_key"] not in header["group_problems"][row["group"]]:
            raise IntegrityError("problem_key differs from declared group problem")


def read_graded_runs(
    path: Path,
) -> tuple[list[Observation], list[str], dict[str, str], dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    read: dict[str, Any] = {
        "dependencies": dependencies,
        "runs": [],
        "refused": [],
        "refusals": [],
    }
    try:
        lines = _read(path.resolve(), dependencies).splitlines()
        header = strict_json(lines[0])
        if header["format"] != GRADED_FORMAT:
            raise ValueError(f"format must be {GRADED_FORMAT}")
        excluded = _groups(header)
        read["rules_version"] = header["rules_version"]
        read["produced_under"] = header["instrument_schema_sha256"]
        rows = []
        for index, line in enumerate(lines[1:], 1):
            try:
                rows.append(strict_json(line))
            except (ValueError, TypeError) as error:
                raise SourceRefusedError(
                    [{"index": index, "reason": str(error)}], read
                ) from error
        items = _source_items(path, rows, header, dependencies, read)
        failures = Counter(
            (row["failure"]["code"], row["failure"].get("search_started"))
            for row in rows
            if row.get("failure") is not None
        )
        read["failures"] = [
            {"code": code, "search_started": started, "count": count}
            for (code, started), count in sorted(
                failures.items(), key=lambda pair: (pair[0][0], str(pair[0][1]))
            )
        ]
        observations = _admit_items(items, header, read)
        run_fills = {
            (str(item["workspace"]), item["row"]["run_id"]): item["detail"]["filled"]
            for item in items
            if "filled" in item["detail"]
        }
        read["filled"] = dict(
            Counter(name for filled in run_fills.values() for name in filled)
        )
    except (
        IntegrityError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        OSError,
    ) as error:
        if isinstance(error, SourceRefusedError):
            raise
        raise SourceRefusedError([{"index": 0, "reason": str(error)}], read) from error
    refused = [item for item in read["refused"] if item["scope"] != "duplicate"]
    read["admission"] = {
        "source_rows": len(rows),
        "logical_observations": len(observations)
        + sum(item["scope"] == "unassigned" for item in refused),
        "admitted": sum(not obs.get("refused", False) for obs in observations),
        "refused": len(refused),
        "unassigned": sum(item["scope"] == "unassigned" for item in refused),
    }
    digest = hashlib.sha256(
        json.dumps(dependencies, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        observations,
        excluded,
        {"kind": "kd-agent-graded-searches", "digest": digest},
        read,
    )


def _source_items(
    path: Path,
    rows: list[dict[str, Any]],
    header: dict[str, Any],
    dependencies: list[dict[str, Any]],
    read: dict[str, Any],
) -> list[dict[str, Any]]:
    items, seen = [], set()
    problems: dict[str, str] = {}
    for index, row in enumerate(rows, 1):
        detail: dict[str, Any] = {
            "index": index,
            "problem_key": row.get("problem_key"),
            "run_id": row.get("run_id"),
        }
        item: dict[str, Any] = {"row": row, "detail": detail, "errors": []}
        try:
            _source_evidence(item, path, dependencies)
            with _row_refusal(item["errors"]):
                _fold_row(row, header)
                if (
                    row["problem_key"] in problems
                    and problems[row["problem_key"]] != row["group"]
                ):
                    raise IntegrityError("problem_key appears in different groups")
                problems[row["problem_key"]] = row["group"]
        except (IntegrityError, InputIntegrityError) as error:
            raise SourceRefusedError(
                [{**detail, "reason": str(error)}], read
            ) from error
        missing = {
            "workspace",
            "producing_workspace",
            "run_id",
            "producer",
        } - row.keys()
        if missing:
            item["errors"].append(f"missing source metadata: {sorted(missing)}")
        if "workspace" not in item or "run_id" not in row:
            read["runs"].append(detail)
            read["refused"].append(
                {**detail, "reason": "; ".join(item["errors"]), "scope": "unassigned"}
            )
            continue
        key = (
            str(item["workspace"]),
            row["run_id"],
            row.get("refinement_id"),
        )
        if key in seen:
            read["refused"].append(
                {**detail, "reason": "duplicate source identity", "scope": "duplicate"}
            )
            continue
        seen.add(key)
        items.append(item)
    return items


def _source_evidence(
    item: dict[str, Any], path: Path, dependencies: list[dict[str, Any]]
) -> None:
    from kdagent.lineage import SegmentLedger

    row, errors = item["row"], item["errors"]
    with _row_refusal(errors):
        item["workspace"] = (path.parent / row["workspace"]).resolve()
    if "workspace" not in item:
        return
    workspace = item["workspace"]
    with _row_refusal(errors):
        _read(workspace / "tree.jsonl", dependencies)
        ledger = SegmentLedger(workspace)
        item.update(ledger=ledger, entry=ledger.entry(row["run_id"]))
        item["chain"] = ledger.chain(row["run_id"])

    if row.get("producer") == "refinement":
        with _row_refusal(errors):
            _receipt(item, workspace, dependencies)
        if "receipt" in item:
            with _row_refusal(errors):
                item["refinement_recipe"] = _refinement_recipe(
                    item["receipt"], workspace, dependencies, item["detail"]
                )
    if "entry" not in item:
        return
    entry = item["entry"]
    with _row_refusal(errors):
        item["input_recipe"] = _input_recipe(
            entry.selection_forward["input"],
            row.get("root_input"),
            entry.dataset,
            workspace,
            dependencies,
            item["detail"],
        )
    _search_evidence(item, dependencies)


def _search_evidence(item: dict[str, Any], dependencies: list[dict[str, Any]]) -> None:
    row, entry, errors = item["row"], item["entry"], item["errors"]
    workspace = item["workspace"]
    with _row_refusal(errors):
        item["network"] = (
            None
            if entry.surrogate is None
            else _surrogate_receipt(
                workspace, entry.surrogate, dependencies, item["detail"]
            )
        )
    with _row_refusal(errors):
        _sealed(item, workspace, dependencies)
    if item.get("raised"):
        item["detail"].update(
            instrument=entry.instrument,
            requested=entry.params,
            failure=row.get("failure"),
        )
        with _row_refusal(errors):
            if item.get("sealed"):
                _forward_binding(entry, None)
            if entry.surrogate is not None:
                raise ValueError(
                    "injected surrogate on an unsealed search has no artifact identity"
                )
            item["artifact_key"] = None
        return
    if "record" not in item:
        return
    record = item["record"]
    with _row_refusal(errors):
        if record.record_hash != row["record_hash"]:
            raise IntegrityError("grade record_hash differs from sealed record")
    with _row_refusal(errors):
        _forward_binding(entry, record)
    if "network" in item:
        with _row_refusal(errors):
            item["artifact_key"] = _injected(
                entry, record, item["network"], item["detail"]
            )


def _lineage_prefixes(
    searches: dict[tuple[str, str], dict[str, Any]], read: dict[str, Any]
) -> set[tuple[str, str]]:
    prefixes: set[tuple[str, str]] = set()
    visited: set[tuple[str, str]] = set()
    for (workspace, run_id), item in searches.items():
        if "entry" not in item:
            continue
        ledger = item["ledger"]
        pending = [run_id]
        while pending:
            current = pending.pop()
            key = (workspace, current)
            if key in visited:
                continue
            visited.add(key)
            children = ledger.children(current)
            if children:
                prefixes.add(key)
                pending.extend(child.run_id for child in children)
            elif key not in searches:
                detail = {
                    "index": None,
                    "problem_key": item["detail"]["problem_key"],
                    "workspace": workspace,
                    "run_id": current,
                    "parent_run_id": ledger.entry(current).parent_run_id,
                }
                read["runs"].append(detail)
                read["refused"].append(
                    {
                        **detail,
                        "scope": "unassigned",
                        "reason": "terminal lineage segment absent from source manifest",
                    }
                )
    return prefixes


def _admit_items(
    items: list[dict[str, Any]],
    header: dict[str, Any],
    read: dict[str, Any],
) -> list[Observation]:


    searches = {
        (str(item["workspace"]), item["row"]["run_id"]): item
        for item in items
        if item["row"].get("refinement_id") is None
    }
    prefixes = _lineage_prefixes(searches, read)
    observations = []
    for item in items:
        row, detail = item["row"], item["detail"]
        read["runs"].append(detail)
        if (
            row.get("refinement_id") is None
            and (str(item["workspace"]), row["run_id"]) in prefixes
        ):
            continue
        try:
            observation, errors = _logical_row(item, searches, header)
        except (IntegrityError, InputIntegrityError) as error:
            raise SourceRefusedError(
                [{**detail, "reason": str(error)}], read
            ) from error
        except PreSearchRefusal as error:
            if (
                item.get("raised")
                and row["failure"]["code"] in CALLER_ERROR_TYPES
                and not item["errors"]
            ):
                read["refusals"].append(_refusal_observation(item))
            read["refused"].append(
                {
                    **detail,
                    "reason": "; ".join([*item["errors"], str(error)]),
                    "scope": "unassigned",
                }
            )
            continue
        except (ValueError, KeyError, TypeError, OSError) as error:
            read["refused"].append(
                {
                    **detail,
                    "reason": "; ".join([*item["errors"], str(error)]),
                    "scope": "unassigned",
                }
            )
            continue
        try:
            if errors:
                raise ValueError("; ".join(errors))
            _grade(row["grade"])
            observation.update(row["grade"])
        except (ValueError, KeyError, TypeError) as error:
            read["refused"].append(
                {**detail, "reason": str(error), "scope": "pipeline"}
            )
            observation.update(
                structure=False,
                max_relative_error=None,
                heldout_nrmse=None,
                refused=True,
            )
        observations.append(observation)
    return observations


def _logical_row(
    item: dict[str, Any],
    searches: dict[tuple[str, str], dict[str, Any]],
    header: dict[str, Any],
) -> tuple[Observation, list[str]]:
    row, workspace, detail = item["row"], item["workspace"], item["detail"]
    if row["producer"] not in {"search", "refinement"}:
        raise ValueError("producer must be search or refinement")
    segments = []
    errors: list[str] = []
    observations: list[Observation] = []
    for entry in item["chain"]:
        segment = (
            item
            if entry.run_id == row["run_id"]
            else searches[(str(workspace), entry.run_id)]
        )
        errors.extend(segment["errors"])
        if segment.get("raised"):
            _raised_failure(segment)
            errors.append(
                f"search failed after it started: {segment['row']['failure']['code']}"
            )
        elif not segment.get("sealed", False):
            raise ValueError("segment lacks supported sealed provenance")
        segment_row, record = segment["row"], segment.get("record")
        if entry.run_id != row["run_id"] and segment_row["producer"] != "search":
            raise ValueError("continuation ancestor must identify a search producer")
        obs = _graded_row(
            segment_row,
            header,
            workspace,
            segment["detail"],
            entry,
            record,
            segment["artifact_key"],
            segment["input_recipe"],
            errors,
        )
        if observations and (
            obs["problem_key"] != observations[0]["problem_key"]
            or obs["pipeline"].get("conditioning")
            != observations[0]["pipeline"].get("conditioning")
        ):
            raise ValueError("continuation problem or conditioning changed")
        observations.append(obs)
        segments.append(
            {
                "run_id": entry.run_id,
                "record_hash": None if record is None else record.record_hash,
                "parent_run_id": entry.parent_run_id,
                "resume_from": entry.resume_from,
                "resume_iteration": None
                if entry.resume_from is None
                else entry.resume_from["iteration"],
                "reseed": entry.reseed,
                "config": obs["pipeline"]["config"],
                "conditions": obs["conditions"],
                "evaluations": obs["evaluations"],
                "wall_seconds": obs["wall_seconds"],
            }
        )
    observation = observations[-1]
    detail["segments"] = segments
    if len(segments) > 1:
        observation["pipeline"]["continuation"] = [
            {
                key: segment[key]
                for key in ("config", "conditions", "resume_iteration", "reseed")
            }
            for segment in segments
        ]
        evaluations = [obs["evaluations"] for obs in observations]
        observation["evaluations"] = None if None in evaluations else sum(evaluations)
        walls = [obs["wall_seconds"] for obs in observations]
        observation["wall_seconds"] = None if None in walls else sum(walls)
    if row["producer"] == "refinement":
        receipt = item["receipt"]
        if receipt["search_input_record"] != detail["input"]:
            raise ValueError("refinement search input differs from source search")
        observation["pipeline"]["refinement"] = item["refinement_recipe"]
        detail["refinement"] = receipt
    return observation, errors


class PreSearchRefusal(ValueError):
    pass


def _refusal_observation(item: dict[str, Any]) -> Observation:
    row = item["row"]
    root = row["root_input"]
    return {
        "problem_key": row["problem_key"],
        "group": row["group"],
        "task_kind": row["task_kind"],
        "report": root["report"],
        "dataset_id": root["dataset_id"],
        "instrument": item["entry"].instrument,
        "requested": item["entry"].params or {},
        "defaults_applied": row.get("defaults_applied") or {},
        "code": row["failure"]["code"],
        "search_started": row["failure"]["search_started"],
        "detail": row["failure"]["detail"],
    }


def _raised_failure(item: dict[str, Any]) -> None:
    row = item["row"]
    failure = row["failure"]
    code = failure["code"]
    if code in CALLER_ERROR_TYPES:
        raise PreSearchRefusal(
            f"refused before any search started: {code}: {failure['detail']}"
        )
    if not item.get("sealed") and not item.get("verified_failure"):
        raise ValueError(f"unverified failure: {code}")
    if failure.get("search_started") is not True:
        raise ValueError(f"raised before the search started: {code}")
    if item["entry"].surrogate is not None:
        raise ValueError(
            "injected surrogate on an unsealed search has no artifact identity"
        )
    if "conditions" not in row:
        raise ValueError("cap not recorded")


def _grade(grade: dict[str, Any]) -> None:
    if type(grade["structure"]) is not bool:
        raise ValueError("grade.structure must be a boolean")
    for name in ("max_relative_error", "heldout_nrmse"):
        value = grade[name]
        if value is not None and (type(value) not in (int, float) or value < 0):
            raise ValueError(f"grade.{name} must be nonnegative or null")


def _receipt(
    item: dict[str, Any], workspace: Path, dependencies: list[dict[str, Any]]
) -> None:
    row = item["row"]
    receipts = [
        strict_json(line)
        for line in _read(workspace / "refinements.jsonl", dependencies).splitlines()
    ]
    matches = [
        receipt
        for receipt in receipts
        if receipt["refinement_id"] == row["refinement_id"]
    ]
    if len(matches) != 1:
        raise ValueError("refinement_id requires exactly one recorded receipt")
    receipt: dict[str, Any] = matches[0]

    item["receipt"] = receipt
    digest = hashlib.sha256(
        json.dumps(receipt["law"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    errors: list[str] = []
    for source in (receipt, row):
        with _row_refusal(errors):
            if digest != source["law_sha256"]:
                raise IntegrityError("refinement law_sha256 digest mismatch")
    if errors:
        raise ValueError("; ".join(errors))
    if receipt["source_run_id"] != row["run_id"] or receipt["parent_ref"] != {
        "kind": "search",
        "run_id": row["run_id"],
    }:
        raise ValueError("refinement source run differs from graded search")
    if receipt["format"] != "kdagent-refinement-receipt-v1":
        raise ValueError("unsupported refinement receipt format")


def _refinement_recipe(
    receipt: dict[str, Any],
    workspace: Path,
    dependencies: list[dict[str, Any]],
    detail: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    with _row_refusal(errors):
        output_steps = _refinement_output(receipt, workspace, dependencies, detail)


    parameters = deepcopy(receipt["parameters"])
    with _row_refusal(errors):
        _refinement_evaluation(parameters, receipt, workspace, dependencies, detail)
    with _row_refusal(errors):
        surrogate_id = parameters.pop("surrogate_id", None)
        if surrogate_id is not None:
            network = _surrogate_receipt(workspace, surrogate_id, dependencies, detail)
            if receipt["artifact_sha256"] != network.file_sha256:
                raise IntegrityError("refinement surrogate artifact digest mismatch")
            parameters["surrogate_recipe"] = {
                key: value
                for key, value in network.recipe.items()
                if key != "dataset_id"
            }
    if errors:
        raise ValueError("; ".join(errors))
    if output_steps is not None:
        parameters["output_input_steps"] = output_steps
    return {
        "method": receipt["method"],
        "parameters": parameters,
        "support_origin": receipt["support_origin"],
        "support": receipt["law"]["support"],
    }


def _refinement_evaluation(
    parameters: dict[str, Any],
    receipt: dict[str, Any],
    workspace: Path,
    dependencies: list[dict[str, Any]],
    detail: dict[str, Any],
) -> None:
    evaluation_id = parameters.pop("evaluation_input_id", None)
    if evaluation_id is not None:
        evaluation = receipt["evaluation_input_record"]
        if evaluation["dataset_id"] != evaluation_id:
            raise ValueError("refinement evaluation input identity differs")
        if evaluation != receipt["search_input_record"]:
            search = receipt["search_input_record"]
            root = search["chain"][0] if search["chain"] else search
            _, steps = _input_recipe(
                evaluation, root, evaluation_id, workspace, dependencies, detail
            )
            parameters["evaluation_input_steps"] = steps
        else:
            parameters["evaluation_input"] = "search"


def _refinement_output(
    receipt: dict[str, Any],
    workspace: Path,
    dependencies: list[dict[str, Any]],
    detail: dict[str, Any],
) -> list[dict[str, Any]] | None:
    output = receipt["output_input_record"]
    if output is None:
        return None
    errors: list[str] = []
    with _row_refusal(errors):
        if receipt["artifact_sha256"] != output["file_sha256"]:
            raise IntegrityError("refinement output artifact digest mismatch")
    with _row_refusal(errors):
        search = receipt["search_input_record"]
        root = search["chain"][0] if search["chain"] else search
        _, steps = _input_recipe(
            output, root, output["dataset_id"], workspace, dependencies, detail
        )
    if errors:
        raise ValueError("; ".join(errors))
    return steps


def _sealed(
    item: dict[str, Any], workspace: Path, dependencies: list[dict[str, Any]]
) -> None:
    from kd.search import RunRecord

    entry = item["entry"]
    directory = workspace / entry.run_dir
    manifest_path = directory / "manifest.json"
    if (
        not manifest_path.exists()
        and item["row"].get("failure", {}).get("code") in CONTROLLER_ERRORS
    ):
        item.update(raised=True, sealed=False)
        item["verified_failure"] = _unsealed_binding(item, workspace, dependencies)
        return
    manifest = strict_json(_read(manifest_path, dependencies))
    if manifest.get("status") == "raised":
        for key, value in (("run_id", entry.run_id), ("instrument", entry.instrument)):
            if manifest[key] != value:
                raise IntegrityError(f"manifest/ledger identity mismatch: {key}")
        if (manifest["scheme"], manifest["schema_version"]) != ("kd-rundir-v1", 1):
            raise ValueError("unsupported run manifest")
        _continuation_binding(
            entry,
            manifest,
            workspace,
            dependencies,
            raised=True,
            producing_workspace=item["row"]["producing_workspace"],
        )
        item.update(raised=True, sealed=True)
        return
    raw = _read(directory / "record.json", dependencies)
    try:
        record = RunRecord.from_dict(strict_json(raw))
    except (ValueError, TypeError, KeyError) as error:
        raise IntegrityError(str(error)) from error


    item["record"] = record
    if (
        entry.dataset != record.evidence.dataset_name
        or entry.seed != record.evidence.seed
        or entry.instrument != record.evidence.instrument
    ):
        raise IntegrityError(
            "ledger dataset/seed/instrument differs from sealed record"
        )
    _manifest_binding(
        entry,
        record,
        manifest,
        hashlib.sha256(raw).hexdigest(),
        workspace,
        dependencies,
        producing_workspace=item["row"]["producing_workspace"],
    )
    item["sealed"] = True


def _no_law_binding(entry: Any) -> bool:
    forward = entry.selection_forward
    return forward is not None and all(
        key in forward and forward.get(key) == value
        for key, value in (
            ("run_id", entry.run_id),
            ("record_hash", None),
            ("law", None),
            ("status", "inapplicable"),
            ("reason", "no_law"),
        )
    )


def _unsealed_binding(
    item: dict[str, Any], workspace: Path, dependencies: list[dict[str, Any]]
) -> bool:
    entry, failure = item["entry"], item["row"]["failure"]
    if not _no_law_binding(entry) or failure.get("search_started") is not True:
        return False
    path = workspace / "trace.jsonl"
    if not path.exists():
        return False
    for line in _read(path, dependencies).splitlines():
        trace = strict_json(line)
        if trace["run_type"] != "tool" or trace["name"] != "run_discovery":
            continue
        output = (trace.get("outputs") or {}).get("output") or {}
        if "content" in output:
            content = output["content"]
        else:
            messages = (output.get("update") or {}).get("messages") or []
            content = messages[0].get("content", "") if messages else ""
        if not content.startswith("{"):
            continue
        reply = strict_json(content)
        provenance = reply["provenance"]
        if (
            provenance.get("run_id") == entry.run_id
            and provenance.get("instrument") == entry.instrument
            and provenance.get("search_started") is True
        ):
            return True
    return False


def _manifest_binding(
    entry: Any,
    record: Any,
    manifest: dict[str, Any],
    record_digest: str,
    workspace: Path,
    dependencies: list[dict[str, Any]],
    *,
    producing_workspace: str,
) -> None:
    errors: list[str] = []
    with _row_refusal(errors):
        if manifest["artifacts"]["record.json"]["sha256"] != record_digest:
            raise IntegrityError("manifest record.json digest mismatch")
    expected = {
        "run_id": entry.run_id,
        "instrument": entry.instrument,
        "dataset_cache_fingerprint": record.run_spec.dataset_cache_fingerprint,
        "config_hash": record.run_spec_hash,
    }
    for key, value in expected.items():
        with _row_refusal(errors):
            if manifest[key] != value:
                raise IntegrityError(f"manifest/ledger/record identity mismatch: {key}")
    with _row_refusal(errors):
        if (manifest["scheme"], manifest["schema_version"]) != ("kd-rundir-v1", 1):
            raise ValueError("unsupported run manifest")
    with _row_refusal(errors):
        _continuation_binding(
            entry,
            manifest,
            workspace,
            dependencies,
            raised=False,
            producing_workspace=producing_workspace,
        )
    if errors:
        raise ValueError("; ".join(errors))


def _forward_binding(entry: Any, record: Any) -> None:
    forward = entry.selection_forward
    if record is None:
        if not _no_law_binding(entry):
            raise IntegrityError(
                "raised selection_forward requires a bound no_law record"
            )
        return
    if forward is None:
        raise ValueError("selection_forward missing")
    if forward["record_hash"] != record.record_hash:
        no_law = (
            forward["record_hash"] is None
            and forward["law"] is None
            and forward["status"] == "inapplicable"
            and forward["reason"] == "no_law"
        )
        if not no_law:
            raise IntegrityError("selection_forward record_hash mismatch")
    if forward["run_id"] != entry.run_id:
        raise ValueError("selection_forward run_id mismatch")


def _continuation_binding(
    entry: Any,
    manifest: dict[str, Any],
    workspace: Path,
    dependencies: list[dict[str, Any]],
    *,
    raised: bool,
    producing_workspace: str,
) -> None:
    from kdagent.lineage import SegmentLedger

    lineage = manifest["lineage"]
    if entry.parent_run_id is None:
        if lineage is not None or entry.resume_from is not None:
            raise ValueError("continuation manifest/ledger disagree")
        return
    if lineage is None:
        raise ValueError("continuation requires manifest and ledger lineage")




    attempted = raised and all(
        lineage[key] is None
        for key in ("source_config_hash", "source_final_status", "source_iteration")
    )
    errors: list[str] = []
    for key, value in (
        ("source_run_id", entry.parent_run_id),
        ("reseed", entry.reseed),
    ):
        with _row_refusal(errors):
            if lineage[key] != value:
                raise IntegrityError(
                    f"continuation manifest/ledger identity mismatch: {key}"
                )
    if not attempted:
        with _row_refusal(errors):
            parent = SegmentLedger(workspace).entry(entry.parent_run_id)
            parent_manifest = strict_json(
                _read(workspace / parent.run_dir / "manifest.json", dependencies)
            )
            if lineage["source_config_hash"] != parent_manifest["config_hash"]:
                raise IntegrityError(
                    "continuation manifest/ledger identity mismatch: source_config_hash"
                )
    with _row_refusal(errors):
        if entry.resume_from is None:
            raise ValueError("continuation requires manifest and ledger lineage")
        if (
            not attempted
            and lineage["source_iteration"] != entry.resume_from["iteration"]
        ):
            raise IntegrityError(
                "continuation manifest/ledger identity mismatch: source_iteration"
            )
    if entry.resume_from is not None:





        ledger, recorded = cast(str, entry.resume_from["path"]), lineage["resume_from"]
        with _row_refusal(errors):
            local = (workspace / recorded).resolve() == (workspace / ledger).resolve()
            fetched = PurePosixPath(recorded).is_absolute() and posixpath.normpath(
                PurePosixPath(producing_workspace) / ledger
            ) == posixpath.normpath(recorded)
            if not (local or fetched):
                raise IntegrityError(
                    "continuation manifest/ledger identity mismatch: resume_from"
                )
    if errors:
        raise ValueError("; ".join(errors))


def _input_label(launch: dict[str, Any], root: dict[str, Any], dataset: str) -> str:
    own = {key: launch[key] for key in ("dataset_id", "file_sha256")}
    if own["dataset_id"] != dataset:
        raise ValueError("launch input differs from ledger dataset")
    chain, derived = launch["chain"], launch["derived"]
    if chain and chain[-1] != own:
        raise ValueError("input chain does not end at launch input")
    first = chain[0] if chain else own
    if first != {key: root[key] for key in ("dataset_id", "file_sha256")}:
        raise ValueError("root_input differs from launch chain root")
    if derived is None:
        if len(chain) > 1 or (not chain and own["file_sha256"] is not None):
            raise ValueError("raw input requires a root-only chain")
        return "raw"
    if len(chain) != 2:
        raise ValueError("only one derived input step is supported")
    if (derived["from"], derived["parent_sha256"]) != (
        first["dataset_id"],
        first["file_sha256"],
    ):
        raise ValueError("derived parent differs from chain root")
    if derived.get("network") is not None:
        raise ValueError("network-sampled fields are outside the input labels")
    if derived.get("pinn") is not None:
        if not derived["pinn"]:
            raise ValueError("pinn input requires recorded settings")
        return "pinn"
    if derived.get("savgol") is not None:
        return "savgol"
    if "lowpass_modes" not in derived or "time_window" not in derived:
        raise ValueError("lowpass input requires recorded parameters")
    return "lowpass"


def _surrogate_receipt(
    workspace: Path,
    surrogate_id: str,
    dependencies: list[dict[str, Any]],
    detail: dict[str, Any],
) -> SurrogateEntry:
    from kdagent.surrogates import SurrogateRegistry, recipe_key

    raw = _read(workspace / "surrogates" / "registry.jsonl", dependencies)
    receipts = [strict_json(line) for line in raw.splitlines() if line.strip()]
    receipts = [row for row in receipts if row["surrogate_id"] == surrogate_id]
    detail["surrogate_registrations"] = receipts
    network = SurrogateRegistry(workspace).entry(surrogate_id)
    if network.recipe_key != recipe_key(network.recipe):
        raise IntegrityError("surrogate recipe digest mismatch")
    raw = _read(workspace / network.file, dependencies)
    if hashlib.sha256(raw).hexdigest() != network.file_sha256:
        raise IntegrityError("surrogate file digest mismatch")


    if len({row["file_sha256"] for row in receipts}) > 1:
        raise ValueError(
            "ambiguous surrogate launch binding: the same ID was re-registered "
            "with different file bytes"
        )
    return network


def _injected(
    entry: Any,
    record: Any,
    network: SurrogateEntry | None,
    detail: dict[str, Any],
) -> str | None:
    if network is None:
        return None
    if (
        network.recipe["dataset_id"] != entry.dataset
        or network.recipe["algorithm"] != entry.instrument
    ):
        raise ValueError("surrogate recipe dataset/instrument mismatch")
    if (
        network.identity is not None
        and network.identity.get("dataset") != record.run_spec.dataset_cache_fingerprint
    ):
        raise ValueError("surrogate input fingerprint mismatch")
    schema = next(
        s for s in kd.instrument_schemas() if s["algorithm"] == entry.instrument
    )
    key = network.key
    if key not in schema["config_artifact_keys"] or not record.run_spec.config.get(key):
        raise ValueError("surrogate injection is not represented in sealed config")
    if not record.run_spec.artifacts or key not in record.run_spec.artifacts:
        raise ValueError("surrogate has no sealed artifact identity")
    detail["surrogate"] = network.to_dict()
    detail["sealed_artifact"] = record.run_spec.artifacts[key]
    return key


def _declared_fill(
    schema: dict[str, Any],
    effective: dict[str, Any],
    *,
    produced_under: str,
    filled: dict[str, Any],
    launch: bool,
    prefix: str = "",
) -> dict[str, Any]:
    result = deepcopy(effective)
    same_declarations = produced_under == kd.instrument_schema_digest()
    for field in schema["fields"]:
        name = field["name"]
        path = prefix + name
        if name not in result:
            if field["required"]:
                where = "launch record" if launch else "sealed configuration"
                raise ValueError(f"{path}: required parameter absent from the {where}")
            if same_declarations:
                value = field["default"]
            elif field["legacy"] is not None:
                value = field["legacy"]["value"]
            elif launch:
                raise ValueError(
                    f"{path}: absent from the launch record, which holds only "
                    "values the agent typed and the controller filled; the "
                    "search ran under other declarations and kd declares no "
                    "legacy value"
                )
            else:
                raise ValueError(
                    f"{path}: added after this search ran and kd declares no legacy value"
                )
            result[name] = deepcopy(value)
            filled[path] = deepcopy(value)
        if field["object_fields"] is not None and isinstance(result[name], dict):
            result[name] = _declared_fill(
                {"fields": field["object_fields"]},
                result[name],
                produced_under=produced_under,
                filled=filled,
                launch=launch,
                prefix=path + ".",
            )
    return result


def _launch_config(
    schema: dict[str, Any],
    requested: dict[str, Any],
    sketch: Any,
    defaults_applied: dict[str, Any] | None = None,
    *,
    seed: int,
    produced_under: str,
    filled: dict[str, Any],
) -> dict[str, Any]:
    requested = {**requested, **(defaults_applied or {})}
    effective = deepcopy(requested)
    for field in schema["fields"]:
        name, key = field["name"], field["facade_param"] or field["name"]
        if key in requested:
            value = requested[key]
        elif name in requested:
            value = requested[name]
        else:
            continue
        if name == "use_autograd" and key == "derivatives" and name not in requested:
            value = value == "autograd"
        effective[name] = deepcopy(value)
        if key != name:
            effective.pop(key, None)
    for param in schema["facade_params"]:
        if param["config_field"] is None:
            effective.pop(param["name"], None)
    if effective.get("terms") is not None:
        effective["terms"] = [kd.lower_term(term) for term in effective["terms"]]
    effective["sketch"] = deepcopy(sketch)
    effective["seed"] = seed
    return _declared_fill(
        schema, effective, produced_under=produced_under, filled=filled, launch=True
    )


def _recipe(
    entry: Any,
    record: Any,
    row: dict[str, Any],
    key: str | None,
    produced_under: str,
    detail: dict[str, Any],
) -> dict[str, Any]:
    from kdagent.action_memory import normalize_pipeline

    schema = next(
        s for s in kd.instrument_schemas() if s["algorithm"] == entry.instrument
    )
    filled: dict[str, Any] = {}
    detail["filled"] = filled





    sealed = record.run_spec.config if record is not None else row.get("resolved")
    effective = (
        _declared_fill(
            schema,
            sealed,
            produced_under=produced_under,
            filled=filled,
            launch=False,
        )
        if sealed is not None
        else _launch_config(
            schema,
            entry.params or {},
            entry.sketch,
            row["defaults_applied"],
            seed=entry.seed,
            produced_under=produced_under,
            filled=filled,
        )
    )
    facade = _declared_fill(
        {
            "fields": [
                {**param, "required": False, "object_fields": None}
                for param in schema["facade_params"]
                if param["config_field"] is None
            ]
        },
        row["facade_params"],
        produced_under=produced_under,
        filled=filled,
        launch=False,
        prefix="facade:",
    )
    if key is not None:


        if any(field["name"] == key for field in schema["fields"]):
            effective[key] = None
        else:
            del effective[key]
    if effective.get("sketch") != entry.sketch:
        raise ValueError("sealed config and ledger sketch differ")


    pipeline = normalize_pipeline(
        entry.instrument,
        effective_config=effective,
        facade_params=facade,
        report=row["search_report"],
        retain_conditioning=True,
    )
    if entry.params is not None:
        for name in ("terms", "library"):
            if name in entry.params and effective.get(name) is not None:
                pipeline.setdefault("conditioning", {})[name] = deepcopy(
                    effective[name]
                )
    generation = next(p for p in schema["facade_params"] if p["name"] == "generations")
    if generation["effect"] != "unused":
        field = generation["config_field"]
        pipeline["config"]["generations"] = (
            effective[field] if field is not None else facade["generations"]
        )
    return pipeline


def _graded_row(
    row: dict[str, Any],
    header: dict[str, Any],
    workspace: Path,
    detail: dict[str, Any],
    entry: Any,
    record: Any,
    artifact_key: str | None,
    input_recipe: tuple[str, list[dict[str, Any]]],
    errors: list[str],
) -> Observation:
    if not isinstance(row["problem_key"], str) or not row["problem_key"]:
        raise ValueError("problem_key must identify the independent problem")
    root = row["root_input"]
    if "name" in root["report"] or "name" in row["search_report"]:
        raise ValueError("root/search reports must omit name")
    launch = entry.selection_forward["input"]
    label, steps = input_recipe
    pipeline = _recipe(
        entry, record, row, artifact_key, header["instrument_schema_sha256"], detail
    )
    pipeline["input"] = label + ("+surrogate" if artifact_key is not None else "")
    if steps:
        pipeline["input_steps"] = steps
    _conditions(row)
    detail.update(
        workspace=str(workspace),
        record_hash=None if record is None else record.record_hash,
        forward_record_hash=entry.selection_forward.get("record_hash"),
        input=launch,
        original_pipeline=None if record is None else record.run_spec.config,
        pipeline=pipeline,
        grade=row.get("grade"),
        search_law=None if record is None else entry.law,
        search_evidence=None if record is None else record.evidence.to_dict(),
    )
    wall_seconds = None
    with _row_refusal(errors):
        value = row["wall_seconds"]
        if value is not None and (type(value) not in (int, float) or value < 0):
            raise ValueError("wall_seconds must be nonnegative or null")
        wall_seconds = value
    return {
        "problem_key": row["problem_key"],
        "group": row["group"],
        "report": root["report"],
        "search_report": row["search_report"],
        "task_kind": row["task_kind"],
        "pipeline": pipeline,
        "conditions": row["conditions"],
        "rules_version": header["rules_version"],
        "examinee_version": header["examinee_version"],
        "instrument_schema_sha256": kd.instrument_schema_digest(),
        "produced_under": header["instrument_schema_sha256"],
        "evaluations": None if record is None else record.cost.boundary_results,
        "wall_seconds": wall_seconds,
    }


def _input_recipe(
    launch: dict[str, Any],
    root: dict[str, Any] | None,
    dataset: str,
    workspace: Path,
    dependencies: list[dict[str, Any]],
    detail: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    derived = launch["derived"]
    if len(launch["chain"]) <= 2 and (
        derived is None or derived.get("network") is None
    ):
        if root is None:
            raise ValueError("missing root_input metadata")
        label = _input_label(launch, root, dataset)


        return label, [] if derived is None else [
            _transform(derived, workspace, dependencies, detail)
        ]
    from kdagent.data_source import derivation_chain, input_record

    descriptions = derivation_chain(
        workspace,
        dataset,
        written={item["dataset_id"]: item["file_sha256"] for item in launch["chain"]},
    )
    if input_record(dataset, descriptions) != launch:
        raise ValueError("input descriptors differ from launch snapshot")
    if root is None:
        raise ValueError("missing root_input metadata")
    if launch["chain"][0] != {key: root[key] for key in ("dataset_id", "file_sha256")}:
        raise ValueError("root_input differs from launch chain root")
    for description in descriptions:
        _read(workspace / "inputs" / (description["ref"] + ".json"), dependencies)
        dependencies.append(
            {"path": description["path"], "sha256": description["source"]["sha256"]}
        )
    detail["input_descriptors"] = descriptions
    steps = [
        _transform(description["derived"], workspace, dependencies, detail)
        for description in descriptions[1:]
    ]
    return (steps[0]["kind"] if len(steps) == 1 else "composite"), steps


def _transform(
    derived: dict[str, Any],
    workspace: Path,
    dependencies: list[dict[str, Any]],
    detail: dict[str, Any],
) -> dict[str, Any]:
    if derived.get("network") is not None:
        network = derived["network"]
        receipt = _surrogate_receipt(
            workspace, network["surrogate_id"], dependencies, detail
        )
        if receipt.file_sha256 != network["artifact_sha256"]:
            raise IntegrityError("network artifact digest differs from receipt")
        parameters = {
            key: value for key, value in receipt.recipe.items() if key != "dataset_id"
        }
        return {"kind": "network", "parameters": parameters}
    if derived.get("pinn") is not None:
        return {"kind": "pinn", "parameters": deepcopy(derived["pinn"])}
    if derived.get("savgol") is not None:
        return {"kind": "savgol", "parameters": deepcopy(derived["savgol"])}
    return {
        "kind": "lowpass",
        "parameters": {key: derived[key] for key in ("lowpass_modes", "time_window")},
    }


def _conditions(row: dict[str, Any]) -> None:
    conditions = row["conditions"]
    if set(conditions) != {
        "device",
        "soft_seconds",
        "budget_tier",
        "blas_threads",
        "python_hash_seed",
    }:
        raise ValueError(
            "conditions must record device, soft_seconds, budget_tier, blas_threads, python_hash_seed"
        )
    if conditions["budget_tier"] not in ("full", "short", None):
        raise ValueError("conditions.budget_tier must be full, short or null")
    if conditions["device"] not in ("cpu", "cuda"):
        raise ValueError("conditions.device must be cpu or cuda")
    for name, minimum in (("blas_threads", 1), ("python_hash_seed", 0)):
        value = conditions[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"conditions.{name} must be an integer >= {minimum}")
    value = conditions["soft_seconds"]
    if value is not None and (type(value) not in (int, float) or value < 0):
        raise ValueError("soft_seconds must be nonnegative or null")
