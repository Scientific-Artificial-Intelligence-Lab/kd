
from __future__ import annotations

import json
import logging
import platform as _platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kd.search.discover.golden.constants import PROJECT_ROOT, SCHEMA_VERSION
from kd.search.discover.golden.summarise import GoldenRunResult

logger = logging.getLogger(__name__)


def fixture_id(pde: str, mode: str, seed: int) -> str:
    return f"{pde}_{mode}_seed{seed}"


def write_fixture(
    *,
    output_path: Path,
    pde: str,
    mode: str,
    seed: int,
    config: dict[str, Any],
    result: GoldenRunResult,
    regenerate: bool,
) -> None:
    if output_path.exists() and not regenerate:
        raise FileExistsError(
            f"Fixture exists at {output_path}; pass regenerate=True to "
            "overwrite.",
        )
    payload = _build_payload(
        pde=pde, mode=mode, seed=seed, config=config, result=result,
    )
    _write_json_atomic(output_path, payload)
    logger.info("Wrote fixture %s -> %s", payload["fixture_id"], output_path)


def load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Fixture {path} is not a JSON object.")
    schema = payload.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Fixture {path} has schema_version={schema!r}; "
            f"expected {SCHEMA_VERSION}",
        )
    return payload


def _build_payload(
    *,
    pde: str,
    mode: str,
    seed: int,
    config: dict[str, Any],
    result: GoldenRunResult,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture_id(pde, mode, seed),
        "commit": _git_commit(),
        "timestamp": _now_iso(),
        "config": config,
        "result": {
            "expression_canonical": result.expression_canonical,
            "expression_raw": result.expression_raw,
            "term_set_sorted": list(result.term_set_sorted),
            "coefs_by_term": dict(result.coefs_by_term),
            "reward": result.reward,
            "mse": result.mse,
            "nmse": result.nmse,
            "n_iterations_to_best": result.n_iterations_to_best,
            "wall_time_seconds": result.wall_time_seconds,
        },
        "platform": _platform_payload(),
    }


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _platform_payload() -> dict[str, str]:
    return {
        "device": "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": _platform.python_version(),
    }


def _write_json_atomic(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


__all__ = [
    "SCHEMA_VERSION",
    "fixture_id",
    "load_fixture",
    "write_fixture",
]
