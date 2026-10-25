
from __future__ import annotations

import json
import logging
import signal
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from kd.api import Model
from kd.core.interrupt import SearchInterrupted
from kd.harness.episode import EpisodeOutcome, run_episode
from kd.harness.plan import PlanEntry
from kd.harness.recording import RecordingOptions
from kd.harness.runner import stage_outcome

if TYPE_CHECKING:
    from types import FrameType


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise SearchInterrupted("SIGTERM received: the parent asked this entry to stop")


def _stage(job_dir: Path, outcome: EpisodeOutcome) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    stage_outcome(job_dir, outcome)


def main(argv: list[str]) -> None:
    signal.signal(signal.SIGTERM, _interrupt)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    (job_arg,) = argv
    job_path = Path(job_arg)
    job = json.loads(job_path.read_text(encoding="utf-8"))




    dataset = torch.load(job["dataset_file"], map_location="cpu", weights_only=False)
    recording = job["recording"]
    run_episode(
        entry=PlanEntry.from_dict(job["entry"]),
        entry_index=job["entry_index"],
        dataset=dataset,
        model_factory=Model,
        device=job["device"],
        run_dir=None if job["run_dir"] is None else Path(job["run_dir"]),
        recording=None if recording is None else RecordingOptions(**recording),
        record_ref=job["record_ref"],
        persist_outcome=partial(_stage, job_path.parent),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
