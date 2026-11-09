
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3


@dataclass(frozen=True)
class Workspace:

    root: Path

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    @property
    def history(self) -> Path:
        return self.root / "history"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def capabilities_file(self) -> Path:
        return self.cache / "capabilities.json"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs / job_id

    def run_dir(self, job_id: str) -> Path:
        return self.runs / job_id

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets / dataset_id

    def upload_dir(self, upload_id: str) -> Path:
        return self.uploads / upload_id

    def reference_marker(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "reference_shown"

    def create(self) -> None:
        for directory in (
            self.jobs,
            self.runs,
            self.datasets,
            self.uploads,
            self.history,
            self.cache,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def of_job_dir(cls, job_dir: Path) -> Workspace:
        return cls(job_dir.resolve().parent.parent)


@dataclass(frozen=True)
class Settings:

    workspace: Workspace
    token: str
    host: str = "127.0.0.1"
    port: int = 8765

    fit_slots: int = 1
    light_slots: int = 1

    reserve_min_bytes: int = 4 * GIB
    reserve_fraction: float = 0.25
    supervisor_allowance_bytes: int = GIB
    fit_cap_fraction: float = 0.5
    fit_cap_floor_bytes: int = 2 * GIB
    light_cap_bytes: int = 4 * GIB

    hard_slack_seconds: float = 120.0
    light_hard_seconds: float = 1800.0
    kill_grace_seconds: float = 45.0

    checkpoint_every: int = 10
    checkpoint_keep_last: int = 2

    max_upload_bytes: int = 2 * GIB
    preview_max_points: int = 5000
    preview_max_cells: int = 100_000
    page_max_bytes: int = 256 * 1024
    scheduler_interval_seconds: float = 1.0
    kdproc_timeout_seconds: float = 120.0
