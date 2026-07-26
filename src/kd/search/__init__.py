"""Search algorithm interfaces for kd."""

from __future__ import annotations

from kd.search.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    RunnerCallback,
    VizDataCollector,
)
from kd.search.checkpoint_manifest import (
    CKPTMAN_SCHEMA_VERSION,
    CKPTMAN_SCHEME,
    CheckpointManifestEntry,
    CheckpointManifestError,
    load_checkpoint_manifest,
)
from kd.search.discover import DiscoverConfig, DISCOVERPlugin
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.iteration_events import (
    ITEREVENT_SCHEMA_VERSION,
    ITEREVENT_SCHEME,
    IterationEvent,
    IterationEventEmitter,
    IterationEventSinkError,
)
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
    SearchAlgorithm,
    TerminatingSearchAlgorithm,
)
from kd.search.recorder import VizRecorder
from kd.search.result import (
    ExperimentResult,
    RunResult,
    default_final_result,
)
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin

__all__ = [
    "CKPTMAN_SCHEMA_VERSION",
    "CKPTMAN_SCHEME",
    "CheckpointCallback",
    "CheckpointManifestEntry",
    "CheckpointManifestError",
    "DISCOVERPlugin",
    "DLGAConfig",
    "DLGAPlugin",
    "DiscoverConfig",
    "EarlyStoppingCallback",
    "ExperimentResult",
    "ExperimentRunner",
    "ITEREVENT_SCHEMA_VERSION",
    "ITEREVENT_SCHEME",
    "IterationEvent",
    "IterationEventEmitter",
    "IterationEventSinkError",
    "IterativeSearchAlgorithm",
    "LoggingCallback",
    "PlatformComponents",
    "RunResult",
    "RunnerCallback",
    "SGAConfig",
    "SGAPlugin",
    "SearchAlgorithm",
    "TerminatingSearchAlgorithm",
    "VizDataCollector",
    "VizRecorder",
    "default_final_result",
    "load_checkpoint_manifest",
]
