
from __future__ import annotations

from typing import Any, Mapping


def is_truncated(response_metadata: Mapping[str, Any]) -> bool:
    return (
        response_metadata.get("finish_reason") == "length"
        or response_metadata.get("stop_reason") == "max_tokens"
        or (
            response_metadata.get("status") == "incomplete"
            and (response_metadata.get("incomplete_details") or {}).get("reason")
            == "max_output_tokens"
        )
    )


def is_refusal(response_metadata: Mapping[str, Any]) -> bool:
    return response_metadata.get("stop_reason") == "refusal"
