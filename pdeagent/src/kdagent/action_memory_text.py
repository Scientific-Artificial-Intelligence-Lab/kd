
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def render(hits: Sequence[Mapping[str, Any]]) -> str:
    if not hits:
        return ""
    lines = [
        "Historical action evidence. Choose among legal actions using current feedback.",
        "The metric is OLS-refitted support error on the input grid; it is not "
        "held-out validation or equation recovery. Observational improvements "
        "do not establish an action's causal benefit. Missing measurements are null.",
    ]
    if "effects" in hits[0]["row"]:
        lines = [
            "Historical offline supervised action effects under a fixed selector.",
            "Selection outcomes use a fixed structure, native-coefficient and reference-relative residual protocol. "
            "They do not establish current-task correctness or live-Agent benefit. Proxy error improvements remain diagnostics. "
            "Candidate gains are distinct from selected-answer gains; candidate unknown counts conservatively cover either unknown axis. "
            "Costs, failures, timeouts and unknown outcomes remain in their denominators.",
        ]
    for hit in hits:
        row = hit["row"]
        payload = {"evidence_index": hit["row_index"], **dict(row)}
        lines.append(json.dumps(payload, sort_keys=True, allow_nan=False))
    return "\n".join(lines)


def diagnostic(
    status: str, digest: str, hits: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    text = render(hits)
    if status == "no_match":
        text = "No matching action experience."
    elif status == "unavailable":
        text = "Action experience unavailable: the current search state is incomplete."
    return {
        "action_memory": {
            "status": status,
            "memory_sha256": digest,
            "row_indices": [hit["row_index"] for hit in hits],
            "text": text,
        }
    }
