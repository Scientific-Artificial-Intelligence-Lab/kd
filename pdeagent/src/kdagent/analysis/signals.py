
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import kd

from kdagent.analysis.sources import (
    COST_WORDS,
    NAMED_PARAM,
    call_params,
)


def signal_s1(calls: list[dict[str, Any]]) -> dict[str, Any]:
    laws = [
        json.dumps(c["inputs"].get("law"), sort_keys=True)
        for c in calls
        if c["name"] == "submit_answer"
    ]
    distinct = len(set(laws))
    verdict = "no_trigger" if not laws else "hit" if len(laws) == distinct else "miss"
    return {
        "submits": len(laws),
        "distinct": distinct,
        "repeats": len(laws) - distinct,
        "verdict": verdict,
    }


def signal_s2(calls: list[dict[str, Any]]) -> dict[str, Any]:
    events = []
    searches = [c for c in calls if c["name"] == "run_discovery"]
    for index, call in enumerate(searches):
        env = call["envelope"] or {}





        if env.get("status") == "success":
            continue
        text = json.dumps(env.get("failures") or [])
        named = {m.group(1) for m in NAMED_PARAM.finditer(text)}
        if not named:
            continue
        algorithm = call["inputs"].get("algorithm", "sga")
        followed = [
            later
            for later in searches[index + 1:]
            if later["inputs"].get("algorithm", "sga") == algorithm
            and named & set(call_params(later))
        ]
        events.append(
            {
                "algorithm": algorithm,
                "named": sorted(named),
                "message": text[:300],
                "followed": bool(followed),
            }
        )
    if not events:
        return {"events": [], "verdict": "no_trigger"}
    return {
        "events": events,
        "verdict": "hit" if any(e["followed"] for e in events) else "miss",
    }


def _budget_before(
    key: str,
    params: dict[str, Any],
    env: dict[str, Any],
    algorithm: str,
    defaults: dict[str, dict[str, Any]],
) -> Any:
    if key in params:
        return params[key]
    provenance = (env.get("provenance") or {}).get("params") or {}
    applied = provenance.get("defaults_applied") or {}
    if key in applied:
        return applied[key]
    return defaults.get(algorithm, {}).get(key)


def signal_s3(
    calls: list[dict[str, Any]],
    budget: set[str],
    defaults: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    requested, escalations = [], []
    searches = [c for c in calls if c["name"] == "run_discovery"]
    for index, call in enumerate(searches):
        params = call_params(call)
        requested.append(params)
        env = call["envelope"] or {}
        nmse = ((env.get("diagnostics") or {}).get("nmse")) if env else None
        if nmse is None or nmse <= 0.01:
            continue
        algorithm = call["inputs"].get("algorithm", "sga")
        for later in searches[index + 1:]:
            if later["inputs"].get("algorithm", "sga") != algorithm:
                continue
            after = call_params(later)




            changes = [
                {
                    "param": key,
                    "before": _budget_before(key, params, env, algorithm, defaults),
                    "after": value,
                }
                for key, value in after.items()
                if key in budget and isinstance(value, int | float)
            ]
            grew = any(
                c["before"] is not None and c["after"] > c["before"] for c in changes
            )
            escalations.append(
                {
                    "nmse": nmse,
                    "algorithm": algorithm,
                    "raised": grew,
                    "changes": changes,
                }
            )
            break
    with_budget = [p for p in requested if set(p) & budget]
    return {
        "searches": len(searches),
        "requested": requested,
        "with_budget": len(with_budget),
        "escalations": escalations,
        "verdict": "hit" if with_budget else "miss",
    }


def signal_s4(
    calls: list[dict[str, Any]], texts: list[str], classes: dict[str, str]
) -> dict[str, Any]:
    used = [
        c["inputs"].get("algorithm", "sga")
        for c in calls
        if c["name"] == "run_discovery"
    ]
    used_classes = sorted({classes.get(a, "unknown") for a in used})
    mentions = []
    for text in texts:
        for match in COST_WORDS.finditer(text):
            start = max(0, match.start() - 60)
            mentions.append(text[start: match.end() + 60].replace("\n", " "))
    avoided_heavy = bool(used) and "heavy" not in used_classes
    if mentions:
        verdict = "hit_mentioned"
    elif avoided_heavy:



        verdict = "hit_avoided_only"
    else:
        verdict = "miss"
    return {
        "algorithms": sorted(set(used)),
        "cost_classes": used_classes,
        "avoided_heavy": avoided_heavy,
        "mentions": mentions[:6],
        "verdict": verdict,
    }


CARD_PATH = re.compile(r"/?skills/([a-z0-9_]+)/SKILL\.md", re.IGNORECASE)



API_GUESS_FAIL = re.compile(
    r"AttributeError|TypeError|KeyError|IndexError|NameError|Traceback|"
    r"is not defined|not subscriptable|has no attribute",
)


















CEILING = {
    "allen-cahn": 0.00011837,
    "burgers": 1.118e-05,
    "burgers-2d": 0.00010152,
    "chafee-infante": 0.00025261,
    "convection-diffusion": 0.00831944,
    "eq-6-2-12": 0.04785,
    "kdv": 0.00123165,
    "klein-gordon": 1.99e-06,
    "llm4ed-fisher": 5.549e-05,
    "llm4ed-fisher-nonlinear": 1.407e-05,
    "llm4ed-heat": 1.929e-05,
    "pde-compound": 1.37801634,
    "pde-divide": 0.00206,
    "wave": 0.07653936,
}









CAPABLE = {
    "allen-cahn": {"pysindy"},
    "burgers": {"pysindy"},
    "burgers-2d": {"pysindy"},
    "chafee-infante": {"pysindy"},
    "convection-diffusion": {"pysindy"},




    "eq-6-2-12": {"pysindy", "dlga"},
    "kdv": {"pysindy"},
    "llm4ed-fisher": {"pysindy"},
    "llm4ed-fisher-nonlinear": {"pysindy"},
    "llm4ed-heat": {"pysindy"},
    "pde-compound": {"pysindy", "dlga"},
    "pde-divide": {"pysindy", "sga"},
}














CAPABLE_COMPLETE = {"pde-divide", "eq-6-2-12", "pde-compound"}

SHELL_TOOLS = {"execute", "shell", "bash"}


def _failed(call: dict[str, Any]) -> bool:
    env = call["envelope"] or {}
    return call["status"] != "success" or env.get("status") in {"failure", "partial"}


def signal_s5(calls: list[dict[str, Any]]) -> dict[str, Any]:
    reads = [
        (index, match.group(1))
        for index, call in enumerate(calls)
        if call["name"] == "read_file"
        and (match := CARD_PATH.search(str(call["inputs"].get("file_path", ""))))
    ]
    failures = [index for index, call in enumerate(calls) if _failed(call)]
    if not reads:
        return {"reads": [], "verdict": "miss"}
    first_read = reads[0][0]
    before = not failures or first_read < failures[0]
    return {
        "reads": [name for _, name in reads],
        "first_read_at": first_read,
        "first_failure_at": failures[0] if failures else None,
        "verdict": "hit" if before else "hit_after_failure",
    }


def signal_s6(dataset: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    written: list[str] = []
    for call in calls:
        if call["name"] == "run_discovery":
            params = call_params(call)
            for key in ("terms", "library"):
                written += [str(t) for t in params.get(key) or []]
        if call["name"] == "submit_answer":
            law = call["inputs"].get("law") or {}
            written += [str(t) for t in law.get("support") or law.get("terms") or []]
    if not written:
        return {"terms": [], "bad": [], "verdict": "no_trigger"}
    report = kd.validate_terms(
        kd.get_dataset(dataset).loader(), sorted(set(written)), max_order=3
    )
    bad = sorted({rejection.term for rejection in report.rejected})
    return {"terms": written, "bad": bad, "verdict": "miss" if bad else "hit"}


def _evidence(workspace: Path, row: dict[str, Any]) -> dict[str, Any]:
    run_dir = row.get("run_dir")
    if not run_dir:
        return {}


    base = workspace if row.get("format") == "kd-segtree-v1" else workspace / "runs"
    record = base / str(run_dir) / "record.json"
    if not record.is_file():
        return {}
    evidence: dict[str, Any] = json.loads(record.read_text(encoding="utf-8"))[
        "evidence"
    ]
    return evidence


def _support_key(support: Any, coefficients: Any) -> tuple[Any, ...] | None:
    if not isinstance(support, list) or not isinstance(coefficients, list):
        return None
    if len(support) != len(coefficients):
        return None
    return tuple(
        sorted(
            (str(t).replace(" ", ""), round(float(c), 12))
            for t, c in zip(support, coefficients, strict=True)
        )
    )


def parsed_answer(console: dict[str, str]) -> dict[str, Any] | None:
    try:
        answer = ast.literal_eval(console.get("answer", ""))
    except (ValueError, SyntaxError):
        return None
    return answer if isinstance(answer, dict) else None


def answer_row(
    console: dict[str, str], rows: list[dict[str, Any]], workspace: Path
) -> dict[str, Any] | None:
    named = parsed_answer({"answer": console.get("answer_provenance", "")})
    if named is not None and named.get("run_id"):




        return next((row for row in rows if row.get("run_id") == named["run_id"]), None)


    answer = parsed_answer(console)
    if answer is None:
        return None
    want = _support_key(answer.get("support"), answer.get("coefficients"))
    if want is None:
        return None
    matched = []
    for row in rows:
        evidence = _evidence(workspace, row)
        if _support_key(evidence.get("support"), evidence.get("coefficients")) == want:
            matched.append(row)
    return matched[-1] if matched else None


def signal_s7(
    dataset: str,
    console: dict[str, str],
    rows: list[dict[str, Any]],
    workspace: Path,
) -> dict[str, Any]:
    row = answer_row(console, rows, workspace)
    ceiling = CEILING.get(dataset)
    capable = CAPABLE.get(dataset)
    nmse = row["nmse"] if row else None
    return {
        "answer_from": None if row is None else row["instrument"],
        "answer_ledger": (
            None
            if row is None
            else "tree"
            if row.get("format") == "kd-segtree-v1"
            else "catalog"
        ),
        "answer_nmse": nmse,
        "final_vs_ceiling": None if (nmse is None or not ceiling) else nmse / ceiling,
        "instruments": sorted({r["instrument"] for r in rows}),
        "verdict": (
            "no_table"
            if capable is None
            else "no_trigger"
            if parsed_answer(console) is None
            else "unattributable"
            if row is None
            else "hit"
            if row["instrument"] in capable


            else "miss"
            if dataset in CAPABLE_COMPLETE
            else "unmeasured"
        ),
    }


def signal_s8(calls: list[dict[str, Any]]) -> dict[str, Any]:
    shells = [c for c in calls if c["name"] in SHELL_TOOLS]
    failed = [c for c in shells if API_GUESS_FAIL.search(c["content"])]
    hunting = [
        c
        for c in shells
        if re.search(r"\b(find|locate|glob)\b", str(c["inputs"].get("command", "")))
        and "import kd" not in str(c["inputs"].get("command", ""))
    ]
    return {
        "shell_calls": len(shells),
        "api_guess_failures": len(failed),
        "file_hunting": len(hunting),
        "verdict": "hit" if not failed else "miss",
    }


def free_observations(calls: list[dict[str, Any]]) -> dict[str, Any]:
    path_errors = sum(1 for c in calls if "path_not_found" in c["content"])
    skill_reads = sum(
        1
        for c in calls
        if c["name"] == "read_file"
        and "skills" in str(c["inputs"].get("file_path", ""))
    )
    return {
        "path_not_found": path_errors,
        "skill_reads": skill_reads,
        "tool_calls": len(calls),
    }
