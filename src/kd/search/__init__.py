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
``InstrumentDescriptor`` / ``InstrumentMode`` / ``Knob`` / ``Segmentation`` /
``tool_schema``)
are re-exported here (045 rows 4a-4c), but each module carries more than
what is promoted (e.g. ``mini_table``'s ``escape_cell``) and stays
direct-import-sanctioned at its own module path.
"""






from __future__ import annotations

from kd.core.platform.sketch_compile import SketchClauseLevels
from kd.search.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    RunnerCallback,
    VizDataCollector,
    WallClockBudgetCallback,
)
from kd.search.checkpoint_manifest import (
    CKPTMAN_SCHEMA_VERSION,
    CKPTMAN_SCHEME,
    CheckpointManifestEntry,
    CheckpointManifestError,
    load_checkpoint_manifest,
)
from kd.search.descriptor import (
    InstrumentDescriptor,
    InstrumentMode,
    Knob,
    Segmentation,
    tool_schema,
)
from kd.search.discover import DiscoverConfig, DISCOVERPlugin
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.eqgpt import EqGPTConfig, EqGPTPlugin
from kd.search.iteration_events import (
    ITEREVENT_DIAGNOSTICS_KEYS,
    ITEREVENT_SCHEMA_VERSION,
    ITEREVENT_SCHEME,
    PHASE_SCHEMA_VERSION,
    PHASE_SCHEME,
    PHASE_VOCABULARY,
    IterationEvent,
    IterationEventEmitter,
    IterationEventSinkError,
    PhaseEvent,
    PhaseWriter,
)
from kd.search.llm4ed import Llm4edConfig, Llm4edPlugin
from kd.search.mini_table import build_mini_table
from kd.search.protocol import (
    DiscoveryTask,
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
    ParetoEntry,
    RunResult,
    default_final_result,
)
from kd.search.run_catalog import (
    CATALOG_FILENAME,
    DEFAULT_RUNS_ROOT,
    RUNCAT_SCHEME,
    append_catalog_row,
    catalog_row_from_record,
    catalog_row_from_result,
)
from kd.search.run_dir import (
    RUNDIR_SCHEME,
    RunDirPaths,
    create_run_dir,
    finalize_run_dir,
    new_run_id,
    run_id_of_run_dir,
)
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin
from kd.search.sketch_outcome import SketchOutcome, write_sketch_artifact

__all__ = [
    "BEST_SCORE_KEY",
    "CATALOG_FILENAME",
    "CKPTMAN_SCHEMA_VERSION",
    "CKPTMAN_SCHEME",
    "CheckpointCallback",
    "CheckpointManifestEntry",
    "CheckpointManifestError",
    "DEFAULT_RUNS_ROOT",
    "DISCOVERPlugin",
    "DLGAConfig",
    "DLGAPlugin",
    "DiscoverConfig",
    "DiscoveryTask",
    "EarlyStoppingCallback",
    "EqGPTConfig",
    "EqGPTPlugin",
    "ExperimentResult",
    "ExperimentRunner",
    "ITEREVENT_DIAGNOSTICS_KEYS",
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
    "PHASE_SCHEMA_VERSION",
    "PHASE_SCHEME",
    "PHASE_VOCABULARY",
    "ParetoEntry",
    "PhaseEvent",
    "PhaseWriter",
    "PlatformComponents",
    "PySINDyConfig",
    "PySINDyPlugin",
    "PySRConfig",
    "PySRPlugin",
    "RUNCAT_SCHEME",
    "RUNDIR_SCHEME",
    "RunDirPaths",
    "RunRecord",
    "RunResult",
    "RunnerCallback",
    "SGAConfig",
    "SGAPlugin",
    "SearchAlgorithm",
    "Segmentation",
    "SketchClauseLevels",
    "SketchOutcome",
    "TerminatingSearchAlgorithm",
    "VizDataCollector",
    "VizRecorder",
    "WallClockBudgetCallback",
    "append_catalog_row",
    "build_mini_table",



    "catalog_row_from_record",
    "catalog_row_from_result",
    "create_run_dir",
    "default_final_result",
    "finalize_run_dir",
    "load_checkpoint_manifest",
    "new_run_id",
    "run_id_of_run_dir",
    "tool_schema",
    "write_sketch_artifact",
]
