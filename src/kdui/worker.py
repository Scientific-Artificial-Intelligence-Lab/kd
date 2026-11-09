
from __future__ import annotations

import importlib
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kdui import jobstore
from kdui.jobstore import JobContext
from kdui.models import JobError, JobKind, JobOutcome
from kdui.settings import Workspace

logger = logging.getLogger(__name__)

_INTERRUPT_POLL_SECONDS = 0.5

_OPERATIONS: dict[JobKind, tuple[str, str]] = {
    JobKind.FIT: ("kdui.ops.fit", "run_fit"),
    JobKind.VERIFY: ("kdui.ops.verify", "run_verify"),
    JobKind.COMPARE: ("kdui.ops.compare", "run_compare"),
}


def work(job_dir: Path) -> int:
    job = jobstore.read_request(job_dir)
    context = JobContext(Workspace.of_job_dir(job_dir), job.job_id)
    module, function = _OPERATIONS[job.kind]
    try:
        operation: Callable[..., Any] = getattr(
            importlib.import_module(module), function
        )
        result = operation(job.request, context)
    except Exception as error:


        logger.exception("job %s (%s) raised", job.job_id, job.kind.value)
        if not (job_dir / jobstore.OUTCOME_FILE).exists():
            jobstore.write_outcome(
                job_dir,
                JobOutcome(
                    at=datetime.now(UTC),
                    status="raised",
                    error=JobError(error_type=type(error).__name__, message=str(error)),
                ),
            )
        return 1
    if not (job_dir / jobstore.OUTCOME_FILE).exists():
        jobstore.write_outcome(
            job_dir, JobOutcome(at=datetime.now(UTC), status="completed")
        )
    jobstore.write_result(job_dir, result)
    return 0


def _raise_sigterm_when_told(job_dir: Path) -> None:
    while not jobstore.interrupt_requested(job_dir):
        time.sleep(_INTERRUPT_POLL_SECONDS)
    logger.info("the supervisor asks this operation to end")
    signal.raise_signal(signal.SIGTERM)


def main(argv: Sequence[str] | None = None) -> int:
    (job_dir,) = sys.argv[1:] if argv is None else argv
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if sys.platform == "win32":
        threading.Thread(
            target=_raise_sigterm_when_told,
            args=(Path(job_dir),),
            name="kdui-interrupt",
            daemon=True,
        ).start()
    return work(Path(job_dir))


if __name__ == "__main__":
    sys.exit(main())
