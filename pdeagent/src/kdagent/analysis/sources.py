
from __future__ import annotations

import json
import re
from datetime import UTC
from pathlib import Path
from typing import Any

import kd



NAMED_PARAM = re.compile(r"\b([a-z][a-z0-9_]{2,})\s*=\s*([0-9][0-9.eE+-]*)")



COST_WORDS = re.compile(
    r"\b(heavy|medium|light|cost|costly|expensive|cheap|minutes?|hours?|"
    r"wall[- ]?clock|slow(?:er|est)?|fast(?:er|est)?)\b",
    re.IGNORECASE,
)


def cost_classes() -> dict[str, str]:
    return {s["algorithm"]: s["cost_class"] for s in kd.instrument_schemas()}


def facade_defaults() -> dict[str, dict[str, Any]]:
    return {
        schema["algorithm"]: {
            row["name"]: row["default"] for row in schema["facade_params"]
        }
        for schema in kd.instrument_schemas()
    }


def budget_params() -> set[str]:
    names = {"population", "num", "width", "pop_size"}
    for schema in kd.instrument_schemas():
        for param in schema["facade_params"]:
            if param.get("effect") == "max_iterations":
                names.add(param["name"])
    return names


def load_trace(workspace: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in workspace.joinpath("trace.jsonl").open()]
    rows.sort(key=lambda r: r["start_time"])
    return rows


def tool_payload(row: dict[str, Any]) -> tuple[str, str]:
    payload = (row.get("outputs") or {}).get("output") or {}
    if "content" in payload:
        return payload["content"], payload.get("status") or "success"
    messages = (payload.get("update") or {}).get("messages") or []
    if messages:
        return messages[0].get("content", ""), messages[0].get("status") or "success"
    return "", "no_output"


def envelope(content: str) -> dict[str, Any] | None:
    if not content.startswith("{"):
        return None
    payload: dict[str, Any] = json.loads(content)
    return payload


def call_params(call: dict[str, Any]) -> dict[str, Any]:
    params = call["inputs"].get("params")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return {}
    return params if isinstance(params, dict) else {}


def tool_calls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for row in rows:
        if row["run_type"] != "tool":
            continue
        content, status = tool_payload(row)
        calls.append(
            {
                "name": row["name"],
                "inputs": row.get("inputs") or {},
                "content": content,
                "status": status,
                "envelope": envelope(content),
            }
        )
    return calls


def llm_facts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls, prompt, completion, texts = 0, 0, 0, []
    for row in rows:
        if row["run_type"] != "llm":
            continue
        calls += 1
        for batch in (row.get("outputs") or {}).get("generations") or []:
            for generation in batch:
                message = (generation.get("message") or {}).get("kwargs") or {}
                usage = message.get("usage_metadata") or {}
                prompt += usage.get("input_tokens", 0)
                completion += usage.get("output_tokens", 0)
                if isinstance(message.get("content"), str):
                    texts.append(message["content"])
    return {
        "model_calls": calls,
        "input_tokens": prompt,
        "output_tokens": completion,
        "texts": texts,
    }


def truncated_replies(rows: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    truncated = 0
    for row in rows:
        if row["run_type"] != "llm" or row["id"] in seen:
            continue
        seen.add(row["id"])
        reasons = []
        for batch in (row.get("outputs") or {}).get("generations") or []:
            for generation in batch:
                message = (generation.get("message") or {}).get("kwargs") or {}
                metadata = message.get("response_metadata") or {}
                reasons.append(metadata.get("finish_reason"))
        if "length" in reasons:
            truncated += 1
    return truncated


def console_summary(log: Path) -> dict[str, str]:
    if not log.exists():
        return {}
    summary = {}
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("=== "):
            key, _, value = line[4:].partition(": ")
            summary[key] = value
    return summary


def wallclock_seconds(rows: list[dict[str, Any]]) -> float:
    starts = [r["start_time"] for r in rows]
    ends = [r["end_time"] for r in rows if r.get("end_time")]
    if not ends:
        return 0.0
    from datetime import datetime

    first = datetime.fromisoformat(min(starts))
    last = datetime.fromisoformat(max(ends))
    return (last - first).total_seconds()


def search_seconds(row: dict[str, Any]) -> float | None:
    from datetime import datetime

    parts = str(row.get("run_dir", "")).split("-")
    if len(parts) < 4:
        return None
    started = datetime.strptime(f"{parts[-3]}{parts[-2]}", "%Y%m%d%H%M%S").replace(
        tzinfo=UTC
    )
    return (datetime.fromisoformat(row["created_at"]) - started).total_seconds()


def catalog_rows(workspace: Path) -> list[dict[str, Any]]:
    catalog = workspace / "runs" / "catalog.jsonl"
    if not catalog.exists():
        return []
    return [json.loads(line) for line in catalog.open()]


def tree_rows(workspace: Path) -> list[dict[str, Any]]:


    tree = workspace / "tree.jsonl"
    if not tree.exists():
        return []
    rows = [json.loads(line) for line in tree.open() if line.strip()]
    return [row for row in rows if row["event"] == "segment"]


def search_rows(workspace: Path) -> list[dict[str, Any]]:
    rows_catalog = catalog_rows(workspace)
    on_catalog = {row["run_id"] for row in rows_catalog}
    return rows_catalog + [
        row for row in tree_rows(workspace) if row["run_id"] not in on_catalog
    ]
