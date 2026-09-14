
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kd.search.checkpoint_manifest import (
    FINAL_STATUS_COMPLETED,
    KIND_FINAL,
    KIND_PERIODIC,
    CheckpointManifestEntry,
    load_checkpoint_manifest,
)


class CheckpointSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedCheckpoint:

    path: Path
    iteration: int


def _eligible_entries(checkpoint_dir: Path) -> tuple[CheckpointManifestEntry, ...]:
    if not checkpoint_dir.exists():
        return ()
    return tuple(
        entry
        for entry in load_checkpoint_manifest(checkpoint_dir)
        if entry.kind == KIND_PERIODIC
        or (entry.kind == KIND_FINAL and entry.final_status == FINAL_STATUS_COMPLETED)
    )


def eligible_checkpoint_iterations(checkpoint_dir: Path | str) -> tuple[int, ...]:
    return tuple(
        sorted({entry.iteration for entry in _eligible_entries(Path(checkpoint_dir))})
    )


def resolve_checkpoint(
    checkpoint_dir: Path | str, iteration: int | None = None
) -> ResolvedCheckpoint:
    checkpoint_dir = Path(checkpoint_dir)
    eligible = _eligible_entries(checkpoint_dir)
    available = sorted({entry.iteration for entry in eligible})
    if not available:
        cause = (
            f"no checkpoint directory at {checkpoint_dir}"
            if not checkpoint_dir.exists()
            else "the archive holds no periodic checkpoint and no completed final"
        )
        if iteration is None:


            raise CheckpointSelectionError(f"no resumable checkpoint: {cause}")
        raise CheckpointSelectionError(
            f"no resumable iteration {iteration}; the available iterations are "
            f"[] ({cause})"
        )
    target = max(available) if iteration is None else iteration
    matches = [entry for entry in eligible if entry.iteration == target]
    if not matches:
        raise CheckpointSelectionError(
            f"no resumable iteration {target}; the available iterations are {available}"
        )
    selected = matches[-1]
    return ResolvedCheckpoint(
        path=(checkpoint_dir / selected.filename).resolve(),
        iteration=selected.iteration,
    )
