"""Search algorithm interfaces for kd.

Two-tier surface. This facade is the ENGINE level: the framework surface
(protocols, callbacks, the runner, checkpoint/iteration-event schemas,
``ExperimentResult`` / ``RunResult``) plus the ``Config`` + ``Plugin`` pair of
all seven registered engines (``sga`` / ``dlga`` / ``discover`` / ``pysr`` /
``eqgpt`` / ``llm4ed`` / ``pysindy``). The per-engine package facades
(``kd.search.sga``, ``kd.search.eqgpt``, ...) are the PLUGIN level: the same
pair plus each package's own deeper names (backend seams, IR converters,
vocabularies, solver helpers), which stay there and are not re-exported here.

Carrying all seven pairs costs no extra import: ``import kd.search`` runs
``kd/__init__`` first, which already imports every engine package eagerly, so
the four pairs this facade used to withhold (on eager-solver-import grounds)
were loaded anyway and the omission only made the engine surface non-uniform.
That debt is still open, but the eager import it has to remove is
``kd/__init__``'s, not this one's.

Sanctioned direct-import module paths beyond this facade:
``kd.search.records`` / ``kd.search.mini_table`` / ``kd.search.descriptor``
-- their key types (``RunRecord`` / ``build_mini_table`` /
``InstrumentDescriptor`` / ``InstrumentMode`` / ``Knob`` / ``tool_schema``)
are re-exported here (045 rows 4a-4c), but each module carries more than
what is promoted (e.g. ``mini_table``'s ``escape_cell``) and stays
direct-import-sanctioned at its own module path.
"""






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
from kd.search.descriptor import InstrumentDescriptor, InstrumentMode, Knob, tool_schema
from kd.search.discover import DiscoverConfig, DISCOVERPlugin
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.eqgpt import EqGPTConfig, EqGPTPlugin
from kd.search.iteration_events import (
    ITEREVENT_SCHEMA_VERSION,
    ITEREVENT_SCHEME,
    IterationEvent,
    IterationEventEmitter,
    IterationEventSinkError,
)
from kd.search.llm4ed import Llm4edConfig, Llm4edPlugin
from kd.search.mini_table import build_mini_table
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
    SearchAlgorithm,
    TerminatingSearchAlgorithm,
)
from kd.search.pysindy import PySINDyConfig, PySINDyPlugin
from kd.search.pysr import PySRConfig, PySRPlugin
from kd.search.recorder import BEST_SCORE_KEY, VizRecorder
from kd.search.records import RunRecord
from kd.search.result import (
    ExperimentResult,
    RunResult,
    default_final_result,
)
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin

__all__ = [
    "BEST_SCORE_KEY",
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
    "EqGPTConfig",
    "EqGPTPlugin",
    "ExperimentResult",
    "ExperimentRunner",
    "ITEREVENT_SCHEMA_VERSION",
    "ITEREVENT_SCHEME",
    "InstrumentDescriptor",
    "InstrumentMode",
    "IterationEvent",
    "IterationEventEmitter",
    "IterationEventSinkError",
    "IterativeSearchAlgorithm",
    "Knob",
    "Llm4edConfig",
    "Llm4edPlugin",
    "LoggingCallback",
    "PlatformComponents",
    "PySINDyConfig",
    "PySINDyPlugin",
    "PySRConfig",
    "PySRPlugin",
    "RunRecord",
    "RunResult",
    "RunnerCallback",
    "SGAConfig",
    "SGAPlugin",
    "SearchAlgorithm",
    "TerminatingSearchAlgorithm",
    "VizDataCollector",
    "VizRecorder",
    "build_mini_table",
    "default_final_result",
    "load_checkpoint_manifest",
    "tool_schema",
]
