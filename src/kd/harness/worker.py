
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kd.api import Model
from kd.harness.dispatch import (
    DispatchManifestError,
    read_dispatch_manifest,
    resolve_dataset,
)
from kd.harness.plan import ExperimentPlan
from kd.harness.runner import run_plan

logger = logging.getLogger(__name__)


def run_worker(
    dispatch_path: Path,
    shard_id: str,
    *,
    model_factory: Callable[..., Any] = Model,
) -> Path:
    dispatch_path = Path(dispatch_path)
    manifest = read_dispatch_manifest(dispatch_path)
    shard = next((s for s in manifest.shards if s.shard_id == shard_id), None)
    if shard is None:
        raise DispatchManifestError(
            f"unknown shard_id {shard_id!r}; manifest shards are "
            f"{[s.shard_id for s in manifest.shards]!r}"
        )

    shard_root = dispatch_path.parent / "shards" / shard_id
    sub_entries = tuple(manifest.plan.entries[i] for i in shard.entry_indices)
    sub_plan = ExperimentPlan(
        name=f"{manifest.plan.name}::{shard_id}", entries=sub_entries
    )

    ordered_refs: list[str] = []
    seen: set[str] = set()
    for index in shard.entry_indices:
        ref = manifest.plan.entries[index].dataset_ref
        if ref not in seen:
            seen.add(ref)
            ordered_refs.append(ref)
    datasets = {ref: resolve_dataset(manifest.datasets[ref]) for ref in ordered_refs}

    device_kwargs: dict[str, Any] = {}
    if shard.device is not None:
        device_kwargs["device"] = shard.device





    recording_kwargs: dict[str, Any] = {}
    if manifest.recording is not None:
        recording = manifest.recording
        batch_root = dispatch_path.parent
        resume_map: dict[int, Path] | None = None
        if recording.resume_from is not None:
            resume_map = {
                index: (
                    Path(raw) if Path(raw).is_absolute() else batch_root / raw
                )
                for index, raw in recording.resume_from.items()
            }
        recording_kwargs = {
            "recording": recording.options,
            "catalog_path": (
                batch_root / recording.catalog
                if recording.catalog is not None
                else None
            ),
            "entry_indices": shard.entry_indices,
            "resume_from": resume_map,
            "plan_hash": manifest.plan_hash,
        }

    logger.info(
        "worker: running shard %s (%d entries) into %s",
        shard_id,
        len(sub_entries),
        shard_root,
    )
    run_plan(
        sub_plan,
        datasets=datasets,
        store_root=shard_root,
        model_factory=model_factory,
        **device_kwargs,
        **recording_kwargs,
    )
    return shard_root


def main(argv: list[str] | None = None) -> None:


    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="kd.harness.worker")
    parser.add_argument("--dispatch", required=True, help="path to dispatch.json")
    parser.add_argument("--shard", required=True, help="shard id, e.g. shard-00")
    args = parser.parse_args(argv)
    run_worker(Path(args.dispatch), args.shard)


if __name__ == "__main__":
    main()
