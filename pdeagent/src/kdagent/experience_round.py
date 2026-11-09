
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kdagent.experience_sources import strict_json

_TERMINAL = {
    "completed",
    "raised",
    "timeout_soft",
    "timeout_term",
    "killed",
    "worker_crash",
    "not_applicable",
    "missing_data",
    "env_unavailable",
    "no_public_entry",
}


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _lhs(value: str | dict[str, Any] | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    from kd import LhsSpec, render_lhs_label

    return render_lhs_label(LhsSpec(**value))


def grade(law: dict[str, Any] | None, reference: dict[str, Any]) -> dict[str, Any]:
    from kd import judge_law

    if law is None:
        return {
            "graded": True,
            "overall": False,
            "structure_ok": False,
            "coefficients_ok": False,
            "lhs_ok": False,
            "max_relative_error": None,
            "reason": "no submitted law",
        }
    try:
        verdict = judge_law(
            law["support"],
            law["coefficients"],
            reference["terms"],
            reference["coefficients"],
            axes=reference["axes"],
            fields=reference["fields"],
            small_term_rtol=0.0,
            coeff_rtol=0.05,
        ).to_dict()
    except ValueError as error:
        return {
            "graded": False,
            "overall": False,
            "structure_ok": False,
            "coefficients_ok": False,
            "lhs_ok": _lhs(law["lhs"]) == reference["lhs"],
            "max_relative_error": None,
            "reason": str(error),
        }
    lhs_ok = _lhs(law["lhs"]) == reference["lhs"]
    verdict.update(graded=True, lhs_ok=lhs_ok, overall=verdict["overall"] and lhs_ok)
    return verdict


def _bound(binding: dict[str, Any], base: Path) -> tuple[Path, bytes]:
    path = (base / binding["path"]).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError(f"source sha256 mismatch: {path}")
    return path, raw


def _json_lines(raw: bytes) -> list[dict[str, Any]]:
    return [strict_json(line) for line in raw.splitlines() if line.strip()]


def _bind_run(run: dict[str, Any], base: Path) -> dict[str, Any]:
    bound = {
        key: _bound(run[key], base)
        for key in ("data", "input", "result", "trace", "verdict")
    }
    records = [_bound(binding, base) for binding in run["records"]]
    if run["refinements"] is not None:
        bound["refinements"] = _bound(run["refinements"], base)
    manifests = []
    for path, _ in records:
        manifest_path = path.parent / "manifest.json"
        manifests.append((manifest_path, manifest_path.read_bytes()))
    return {"wire": run, "bound": bound, "records": records, "manifests": manifests}


def _sealed_records(item: dict[str, Any], root: str) -> dict[str, Any]:
    from kd.search import RunRecord

    records = {}
    for (path, raw), (_, manifest_raw) in zip(
        item["records"], item["manifests"], strict=True
    ):
        record = RunRecord.from_dict(strict_json(raw))
        manifest = strict_json(manifest_raw)
        if (manifest["scheme"], manifest["schema_version"]) != ("kd-rundir-v1", 1):
            raise ValueError(f"unsupported source run manifest: {path}")
        expected = {
            "instrument": record.evidence.instrument,
            "dataset_cache_fingerprint": record.run_spec.dataset_cache_fingerprint,
            "config_hash": record.run_spec_hash,
        }
        if any(manifest[key] != value for key, value in expected.items()):
            raise ValueError(f"source manifest identity mismatch: {path}")
        if manifest.get("status") not in {"completed", "interrupted"}:
            raise ValueError(f"source run manifest is not sealed: {path}")
        artifact = manifest["artifacts"]["record.json"]
        if artifact["sha256"] != hashlib.sha256(raw).hexdigest() or artifact[
            "bytes"
        ] != len(raw):
            raise ValueError(f"source manifest record hash mismatch: {path}")
        if record.evidence.dataset_name.split("+", 1)[0] != root:
            raise ValueError(f"search belongs to another input: {path}")
        run_id = manifest["run_id"]
        if run_id in records:
            raise ValueError(f"duplicate source search: {run_id}")
        records[run_id] = record
    return records


def _reference(
    run: dict[str, Any], verdict: dict[str, Any], report: dict[str, Any]
) -> None:
    from kd import judge_law

    reference, answer = run["reference"], verdict["diagnostics"]["answer"]
    if reference["axes"] != [a["name"] for a in report["axes"]] or reference[
        "fields"
    ] != [f["name"] for f in report["fields"]]:
        raise ValueError("reference vocabulary differs from bound input report")
    match = judge_law(
        reference["terms"],
        reference["coefficients"],
        [t for t, _ in answer["terms"]],
        [c for _, c in answer["terms"]],
        axes=reference["axes"],
        fields=reference["fields"],
        small_term_rtol=0.0,
        coeff_rtol=0.0,
    )
    if reference["lhs"] != answer["lhs"] or not match.overall:
        raise ValueError("reference differs from selected verdict answer")


def _receipt_input(record: dict[str, Any], descriptor: dict[str, Any]) -> None:
    root, digest = descriptor["ref"], descriptor["source"]["sha256"]
    chain = record["chain"]
    if not chain or (chain[0]["dataset_id"], chain[0]["file_sha256"]) != (root, digest):
        raise ValueError("refinement input root sha256 differs from round")
    if record["dataset_id"].split("+", 1)[0] != root:
        raise ValueError("refinement input belongs to another root")
    if (chain[-1]["dataset_id"], chain[-1]["file_sha256"]) != (
        record["dataset_id"],
        record["file_sha256"],
    ):
        raise ValueError("refinement input chain endpoint differs from record")


def _validate_receipts(
    receipts: list[dict[str, Any]], records: dict[str, Any], descriptor: dict[str, Any]
) -> None:
    for receipt in receipts:
        if receipt["source_run_id"] not in records:
            raise ValueError("refinement has no bound sealed search")
        if receipt["format"] != "kdagent-refinement-receipt-v1":
            raise ValueError("unsupported refinement receipt format")
        if _digest(receipt["law"]) != receipt["law_sha256"]:
            raise ValueError("refinement law_sha256 mismatch")
        for key in (
            "search_input_record",
            "evaluation_input_record",
            "output_input_record",
        ):
            if receipt.get(key) is not None:
                _receipt_input(receipt[key], descriptor)


def _answer_producer(
    output: dict[str, Any],
    records: dict[str, Any],
    receipts: list[dict[str, Any]],
    root: str,
) -> None:
    provenance = output.get("answer_provenance")
    if provenance is None:
        return
    if provenance["dataset"].split("+", 1)[0] != root:
        raise ValueError("answer provenance belongs to another input root")
    if provenance["run_id"] not in records:
        raise ValueError("answer provenance has no bound sealed search")
    producer = provenance.get("producer_ref")
    if producer is None:
        return
    if producer["kind"] == "search":
        if producer["run_id"] != provenance["run_id"]:
            raise ValueError("answer producer differs from cited search")
    elif producer["kind"] == "refinement":
        matching = [
            r for r in receipts if r["refinement_id"] == producer["refinement_id"]
        ]
        if len(matching) != 1 or matching[0]["source_run_id"] != provenance["run_id"]:
            raise ValueError("answer producer has no bound refinement")
        if (
            output["answer"] is not None
            and _digest(output["answer"]) != matching[0]["law_sha256"]
        ):
            raise ValueError("answer law differs from bound refinement")
    else:
        raise ValueError("unsupported answer producer kind")


def _result_input(
    result: dict[str, Any],
    descriptor: dict[str, Any],
    records: dict[str, Any],
    receipts: list[dict[str, Any]],
) -> None:
    raw = result["raw"]
    if raw is None:
        raise ValueError("unattributable result: no worker input provenance")
    digest = raw.get("provenance", {}).get("data_file_sha256")
    if digest != descriptor["source"]["sha256"]:
        raise ValueError("result input sha256 differs from round data")
    output = raw["output"]
    if output is None:
        return
    answer_input = output.get("answer_input")
    root = descriptor["ref"]
    if answer_input is not None:
        if answer_input["verified"] is not True:
            raise ValueError("answer input provenance is unverified")
        actual_root = answer_input.get("root") or answer_input["dataset_id"]
        if actual_root != root or answer_input["dataset_id"].split("+", 1)[0] != root:
            raise ValueError("answer input root differs from round")
    _answer_producer(output, records, receipts, root)


def _output_task(result: dict[str, Any], run: dict[str, Any]) -> None:
    raw = result["raw"]
    if raw is not None and raw["output"] is not None:
        output = raw["output"]
        if (
            output["task_kind"] != run["task_kind"]
            or output["target"] != result["row"]["target"]
        ):
            raise ValueError("result output task/target differs from round")


def _validate_identity(item: dict[str, Any]) -> dict[str, Any]:
    run, bound = item["wire"], item["bound"]
    descriptor, result = (strict_json(bound[key][1]) for key in ("input", "result"))
    verdicts = [
        v
        for v in _json_lines(bound["verdict"][1])
        if (v["row_hash"], v["attempt"])
        == (run["verdict"]["row_hash"], run["verdict"]["attempt"])
    ]
    if len(verdicts) != 1:
        raise ValueError("round needs exactly one matching verdict")
    verdict = verdicts[0]
    if result["status"] not in _TERMINAL or result["status"] != verdict["status"]:
        raise ValueError("result is not terminal or differs from verdict status")
    if result["row"] != verdict["row"] or result["attempt"] != verdict["attempt"]:
        raise ValueError("result row/attempt differs from verdict")
    if result["row"]["problem_id"] != run["problem_key"]:
        raise ValueError("result problem differs from round")
    _output_task(result, run)
    if descriptor["source"]["sha256"] != run["data"]["sha256"] or descriptor.get(
        "derived"
    ):
        raise ValueError("round data must be the bound root input")
    summary = descriptor["summary"]
    if (
        descriptor["task_kind"] != run["task_kind"]
        or summary["task_kind"] != run["task_kind"]
    ):
        raise ValueError("round task kind differs from input")
    if (
        summary["target"] != result["row"]["target"]
        or summary["report"]["lhs_label"] != summary["target"]
    ):
        raise ValueError("round target differs from input report")
    records = _sealed_records(item, descriptor["ref"])
    receipts = _json_lines(bound["refinements"][1]) if "refinements" in bound else []
    _validate_receipts(receipts, records, descriptor)
    _result_input(result, descriptor, records, receipts)
    return {
        **item,
        "descriptor": descriptor,
        "result": result,
        "verdict": verdict,
        "decoded_records": records,
        "receipts": receipts,
    }


def _candidate_counts(trace: list[dict[str, Any]]) -> dict[str, int]:
    llm_times = [r["start_time"] for r in trace if r.get("run_type") == "llm"]
    counts = {}
    for event in trace:
        if event.get("run_type") != "tool" or event["name"] not in {
            "run_discovery",
            "refine_candidate",
        }:
            continue
        output = event["outputs"]["output"]
        if isinstance(output, str):
            content = output
        elif "content" in output:
            content = output["content"]
        else:
            content = output["update"]["messages"][0]["content"]
        payload = strict_json(content)
        primary = payload.get("primary", {})
        producer = primary.get("refinement_id") or payload.get("provenance", {}).get(
            "run_id"
        )
        if producer is not None:
            counts[producer] = sum(t < event["start_time"] for t in llm_times)
    return counts


def _candidates(
    item: dict[str, Any], reference: dict[str, Any]
) -> list[dict[str, Any]]:
    counts = _candidate_counts(_json_lines(item["bound"]["trace"][1]))
    candidates = []
    for run_id, record in item["decoded_records"].items():
        evidence = record.evidence
        law = (
            {
                "support": evidence.support,
                "coefficients": evidence.coefficients,
                "lhs": evidence.catalog_fit["lhs_spec"]
                if evidence.catalog_fit
                else None,
            }
            if evidence.is_valid
            and evidence.support is not None
            and evidence.coefficients is not None
            else None
        )
        candidates.append(
            {
                "producer": "search",
                "producer_id": run_id,
                "grade": grade(law, reference),
                "model_calls": counts.get(run_id),
            }
        )
    for receipt in item["receipts"]:
        rid = receipt["refinement_id"]
        candidates.append(
            {
                "producer": "refinement",
                "producer_id": rid,
                "grade": grade(receipt["law"], reference),
                "model_calls": counts.get(rid),
            }
        )
    return candidates


def _diagnose(item: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    raw = item["result"]["raw"]
    output = {} if raw is None else raw["output"] or {}
    final = grade(output.get("answer"), reference)
    candidates = _candidates(item, reference)
    successful = [
        candidate for candidate in candidates if candidate["grade"]["overall"]
    ]
    if final["overall"]:
        category = "solved"
    elif successful:
        category = "found-but-not-selected"
    elif final["structure_ok"] and final["lhs_ok"]:
        category = "coefficient-error"
    elif (
        final.get("missing")
        and not final.get("extra")
        and any(
            set(final["missing"]).issubset(candidate["grade"].get("law", {}))
            for candidate in candidates
        )
    ):
        category = "dropped-term"
    else:
        category = "never-found"
    return {
        "category": category,
        "final": final,
        "candidates": candidates,
        "successful_candidates": successful,
    }


def _project(item: dict[str, Any]) -> dict[str, Any]:
    import kd

    run, descriptor, bound = item["wire"], item["descriptor"], item["bound"]
    data = kd.load(bound["data"][0], **descriptor["load_options"])
    report = kd.preview_report(data).to_dict()
    original = descriptor["summary"]["report"]
    for key in ("lhs_field", "lhs_axis", "lhs_order", "lhs_label", "topology"):
        if report[key] != original[key]:
            raise ValueError(f"bound input report differs from data: {key}")
    if [(f["name"], f["shape"]) for f in report["fields"]] != [
        (f["name"], f["shape"]) for f in original["fields"]
    ]:
        raise ValueError("bound input report field shape differs from data")
    if [(a["name"], a["n"], a["periodic"]) for a in report["axes"]] != [
        (a["name"], a["n"], a["periodic"]) for a in original["axes"]
    ]:
        raise ValueError("bound input report axes differ from data")
    _reference(run, item["verdict"], report)
    raw = item["result"]["raw"]
    model_calls = 0 if raw is None else raw["cost"]["model_calls"]
    paths = [p for p, _ in bound.values()] + [
        p for p, _ in item["records"] + item["manifests"]
    ]
    identity = {k: run[k] for k in ("run_id", "problem_key", "task_kind", "reference")}
    identity["artifacts"] = {
        key: hashlib.sha256(value).hexdigest() for key, (_, value) in bound.items()
    }
    identity["searches"] = sorted(
        hashlib.sha256(value).hexdigest()
        for _, value in item["records"] + item["manifests"]
    )
    return {
        "run_id": run["run_id"],
        "problem_key": run["problem_key"],
        "task_kind": run["task_kind"],
        "data_path": bound["data"][0],
        "source_workspace": bound["input"][0].parent.parent,
        "data_sha256": run["data"]["sha256"],
        "load_options": descriptor["load_options"],
        "report": report,
        "reference": run["reference"],
        "diagnosis": _diagnose(item, run["reference"]),
        "source_digest": _digest(identity),
        "model_calls": model_calls,
        "artifact_paths": paths,
    }


def read_round(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    document = strict_json(path.read_bytes())
    if document["format"] != "kd-experience-round-v1":
        raise ValueError("unsupported experience round format")
    runs = document["runs"]
    if not runs or len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError("round needs nonempty, unique run_id values")
    bound = [_bind_run(run, path.parent) for run in runs]
    admitted = [_validate_identity(item) for item in bound]
    projected = [_project(item) for item in admitted]
    for run in projected:
        run["artifact_paths"].append(path.resolve())
    return projected
