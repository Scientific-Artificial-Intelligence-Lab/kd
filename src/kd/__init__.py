"""kd - Symbolic regression platform for PDE discovery."""

from kd.api import Model, instrument_schemas
from kd.core.equation import LhsSpec, Sketch, law_signature
from kd.core.equation.rendering import render_lhs_label
from kd.core.evaluator import EvaluationResult
from kd.core.expr.sympy_bridge import FormattedEquation, format_pde
from kd.core.interrupt import SearchInterrupted
from kd.core.rates import RateSummary, paired_exact_test, rate_summary, wilson_interval
from kd.core.recovery import (
    RecoveryVerdict,
    judge_recovery,
    load_bearing_recall,
    span_floor,
    term_set_jaccard,
)
from kd.core.verify import (
    SKETCH_EXIT_VERIFY,
    VerificationReport,
    VerifyPolicy,
    verify_equation,
)
from kd.data import (
    DATASET_CATALOG,
    AxisInfo,
    DatasetSource,
    DatasetSpec,
    DataTopology,
    FieldData,
    PDEDataset,
    TabularDataset,
    TaskType,
    add_noise,
    generate_advection_data,
    generate_burgers_data,
    generate_diffusion_data,
    get_dataset,
    list_datasets,
    list_datasets_answer_blind,
    list_remote_datasets,
    load,
    load_allen_cahn,
    load_burgers,
    load_burgers_2d,
    load_chafee_infante,
    load_convection_diffusion,
    load_eq_6_2_12,
    load_from_hub,
    load_kdv,
    load_klein_gordon,
    load_llm4ed_fisher,
    load_llm4ed_fisher_nonlinear,
    load_llm4ed_heat,
    load_pde_compound,
    load_pde_divide,
    load_tlc_cc,
    load_wave,
    load_wave_breaking,
)
from kd.evaluate import (
    EvaluationFailedError,
    InvalidTermsError,
    TermRejection,
    TermValidationReport,
    evaluate_terms,
    validate_terms,
)
from kd.inspect import (
    ArrayReport,
    AxisReport,
    DatasetReport,
    FieldReport,
    SourceReport,
    inspect_file,
    preview,
    preview_report,
)
from kd.search.checkpoint_manifest import (
    FINAL_STATUS_COMPLETED,
    KIND_FINAL,
    CheckpointManifestEntry,
    CheckpointManifestError,
    load_checkpoint_manifest,
)
from kd.search.checkpoint_select import (
    CheckpointSelectionError,
    ResolvedCheckpoint,
    eligible_checkpoint_iterations,
    resolve_checkpoint,
)
from kd.search.discover import DiscoverConfig
from kd.search.dlga import DLGAConfig
from kd.search.eqgpt import EqGPTConfig
from kd.search.llm4ed import Llm4edConfig
from kd.search.pysindy import PySINDyConfig
from kd.search.pysr import PySRConfig
from kd.search.result import ExperimentResult
from kd.search.sga import SGAConfig
from kd.viz.engine import VizEngine

__version__ = "0.8.1"







__all__ = [
    "ArrayReport",
    "AxisInfo",
    "AxisReport",
    "CheckpointManifestEntry",
    "CheckpointManifestError",
    "CheckpointSelectionError",
    "DATASET_CATALOG",
    "DLGAConfig",
    "DataTopology",
    "DatasetReport",
    "DatasetSource",
    "DatasetSpec",
    "DiscoverConfig",
    "EqGPTConfig",
    "EvaluationFailedError",
    "EvaluationResult",
    "ExperimentResult",
    "FINAL_STATUS_COMPLETED",
    "FieldData",
    "FieldReport",
    "FormattedEquation",
    "InvalidTermsError",
    "KIND_FINAL",
    "LhsSpec",
    "Llm4edConfig",
    "Model",
    "PDEDataset",
    "PySINDyConfig",
    "PySRConfig",
    "RateSummary",
    "RecoveryVerdict",
    "ResolvedCheckpoint",
    "SGAConfig",
    "SKETCH_EXIT_VERIFY",
    "SearchInterrupted",
    "Sketch",
    "SourceReport",
    "TabularDataset",
    "TaskType",
    "TermRejection",
    "TermValidationReport",
    "VerificationReport",
    "VerifyPolicy",
    "VizEngine",
    "__version__",
    "add_noise",
    "eligible_checkpoint_iterations",
    "evaluate_terms",
    "format_pde",
    "generate_advection_data",
    "generate_burgers_data",
    "generate_diffusion_data",
    "get_dataset",
    "inspect_file",
    "instrument_schemas",
    "judge_recovery",
    "law_signature",
    "list_datasets",
    "list_datasets_answer_blind",
    "list_remote_datasets",
    "load",
    "load_allen_cahn",
    "load_bearing_recall",
    "load_burgers",
    "load_burgers_2d",
    "load_chafee_infante",
    "load_checkpoint_manifest",
    "load_convection_diffusion",
    "load_eq_6_2_12",
    "load_from_hub",
    "load_kdv",
    "load_klein_gordon",
    "load_llm4ed_fisher",
    "load_llm4ed_fisher_nonlinear",
    "load_llm4ed_heat",
    "load_pde_compound",
    "load_pde_divide",
    "load_tlc_cc",
    "load_wave",
    "load_wave_breaking",
    "paired_exact_test",
    "preview",
    "preview_report",
    "rate_summary",
    "render_lhs_label",
    "resolve_checkpoint",
    "span_floor",
    "term_set_jaccard",
    "validate_terms",
    "verify_equation",
    "wilson_interval",
]
