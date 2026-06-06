"""Search algorithm interfaces for kd."""

from __future__ import annotations

from kd.search.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    RunnerCallback,
    VizDataCollector,
)
from kd.search.discover import DiscoverConfig, DISCOVERPlugin
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
    SearchAlgorithm,
)
from kd.search.recorder import VizRecorder
from kd.search.result import (
    ExperimentResult,
    ResultBuilder,
    ResultTargetProvider,
    RunResult,
)
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin

__all__ = [
    "CheckpointCallback",
    "DISCOVERPlugin",
    "DLGAConfig",
    "DLGAPlugin",
    "DiscoverConfig",
    "EarlyStoppingCallback",
    "ExperimentResult",
    "ExperimentRunner",
    "IterativeSearchAlgorithm",
    "LoggingCallback",
    "PlatformComponents",
    "ResultBuilder",
    "ResultTargetProvider",
    "RunResult",
    "RunnerCallback",
    "SGAConfig",
    "SGAPlugin",
    "SearchAlgorithm",
    "VizDataCollector",
    "VizRecorder",
]
