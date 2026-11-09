"""kd - Symbolic regression platform for PDE discovery."""

from typing import TYPE_CHECKING

from kd import _registry
from kd._refit_methods import (
    FrameFit,
    RefitMethod,
    ResolvedFrame,
    refit_method,
    refit_methods,
    register_refit_method,
)
from kd._registry import extension_dependencies, extension_origin, extension_source
from kd.api import (
    Model,
    format_record_equation,
    instrument_schema_digest,
    instrument_schemas,
)
from kd.core.equation import (
    LhsSpec,
    law_signature,
    sketch_from_dict,
    sketch_to_dict,
)
from kd.core.equation._sketch_terms import lower_term
from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.sketch_builder import sketch_from_anchors
from kd.core.equation.sketch_input import (
    Hole,
    Sketch,
    SketchSpec,
    load_sketch,
    save_sketch,
)
from kd.core.evaluator import EvaluationResult
from kd.core.expr.jet import jet_polynomial
from kd.core.expr.monomial_keys import term_monomial_keys, undefined_terms
from kd.core.expr.sympy_bridge import (
    FormattedEquation,
    format_pde,
    from_sympy,
    to_sympy,
)
from kd.core.interrupt import SearchInterrupted
from kd.core.law_judge import ParsedLaw, TermVerdict, judge_law, law_from_equation
from kd.core.linear_solve import register_sparse_solver, sparse_solvers
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
    NetworkSmoothing,
    NetworkSmoothingQuality,
    PDEDataset,
    TabularDataset,
    TaskType,
    Trajectory,
    TrajectorySet,
    add_noise,
    compute_dataset_fingerprint,
    derivative_field,
    derivative_target,
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
    load_wave_breaking_datasets,
    lowpass_smooth,
    lowpass_spectrum,
    network_smooth,
    select_lowpass_modes,
    surrogate_field,
    trajectories_to_tabular,
)
from kd.data.derivatives import (
    derivative_provider,
    derivative_providers,
    register_derivative_provider,
)
from kd.data.tabular_file import load_tabular
from kd.evaluate import (
    AblationResult,
    AblationRow,
    ConstantRefit,
    EvaluationFailedError,
    ForwardError,
    InvalidTermsError,
    RefitFrame,
    RefitResult,
    TermRejection,
    TermSetEntry,
    TermSetScores,
    TermValidationReport,
    TrajectoryForwardError,
    WeakFormDiagnostics,
    WeakFormOptions,
    ablate_terms,
    block_dispersion,
    evaluate_terms,
    forward_error,
    forward_error_trajectories,
    refit_constants,
    refit_terms,
    score_term_sets,
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
from kd.pinn_refine import PinnRefinement, refine_pinn
from kd.prediction import predict_terms
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
from kd.search.descriptor import ModeVerdict, mode_accepts
from kd.search.discover import DiscoverConfig
from kd.search.dlga import DLGAConfig
from kd.search.eqgpt import EqGPTConfig
from kd.search.llm4ed import Llm4edConfig
from kd.search.pysindy import PySINDyConfig
from kd.search.pysr import PySRConfig
from kd.search.registry import (
    execution_dependencies,
    execution_mode_for_request,
    register_algorithm,
)
from kd.search.result import ExperimentResult
from kd.search.sga import SGAConfig
from kd.term_votes import TermVoteResult, TermVoteRow, vote_terms

__version__ = "0.10.0"








if TYPE_CHECKING:
    from kd.viz.engine import VizEngine


def __getattr__(name: str) -> object:

    if name == "VizEngine":
        from kd.viz.engine import VizEngine

        globals()[name] = VizEngine
        return VizEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"VizEngine"})


__all__ = [
    "AblationResult",
    "AblationRow",
    "ArrayReport",
    "AxisInfo",
    "AxisReport",
    "CheckpointManifestEntry",
    "CheckpointManifestError",
    "CheckpointSelectionError",
    "ConstantRefit",
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
    "ForwardError",
    "FrameFit",
    "Hole",
    "InvalidTermsError",
    "KIND_FINAL",
    "LhsSpec",
    "Llm4edConfig",
    "ModeVerdict",
    "Model",
    "NetworkSmoothing",
    "NetworkSmoothingQuality",
    "PDEDataset",
    "ParsedLaw",
    "PinnRefinement",
    "PySINDyConfig",
    "PySRConfig",
    "RateSummary",
    "RecoveryVerdict",
    "RefitFrame",
    "RefitMethod",
    "RefitResult",
    "ResolvedCheckpoint",
    "ResolvedFrame",
    "SGAConfig",
    "SKETCH_EXIT_VERIFY",
    "SearchInterrupted",
    "Sketch",
    "SketchSpec",
    "SourceReport",
    "TabularDataset",
    "TaskType",
    "TermRejection",
    "TermSetEntry",
    "TermSetScores",
    "TermValidationReport",
    "TermVerdict",
    "TermVoteResult",
    "TermVoteRow",
    "Trajectory",
    "TrajectoryForwardError",
    "TrajectorySet",
    "VerificationReport",
    "VerifyPolicy",
    "VizEngine",
    "WeakFormDiagnostics",
    "WeakFormOptions",
    "__version__",
    "ablate_terms",
    "add_noise",
    "block_dispersion",
    "compute_dataset_fingerprint",
    "derivative_field",
    "derivative_provider",
    "derivative_providers",
    "derivative_target",
    "eligible_checkpoint_iterations",
    "evaluate_terms",
    "execution_dependencies",
    "execution_mode_for_request",
    "extension_dependencies",
    "extension_origin",
    "extension_source",
    "format_pde",
    "format_record_equation",
    "forward_error",
    "forward_error_trajectories",
    "from_sympy",
    "generate_advection_data",
    "generate_burgers_data",
    "generate_diffusion_data",
    "get_dataset",
    "inspect_file",
    "instrument_schema_digest",
    "instrument_schemas",
    "jet_polynomial",
    "judge_law",
    "judge_recovery",
    "law_from_equation",
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
    "load_sketch",
    "load_tabular",
    "load_tlc_cc",
    "load_wave",
    "load_wave_breaking",
    "load_wave_breaking_datasets",
    "lower_term",
    "lowpass_smooth",
    "lowpass_spectrum",
    "mode_accepts",
    "network_smooth",
    "paired_exact_test",
    "predict_terms",
    "preview",
    "preview_report",
    "rate_summary",
    "refine_pinn",
    "refit_constants",
    "refit_method",
    "refit_methods",
    "refit_terms",
    "register_algorithm",
    "register_derivative_provider",
    "register_refit_method",
    "register_sparse_solver",
    "render_lhs_label",
    "resolve_checkpoint",
    "save_sketch",
    "score_term_sets",
    "select_lowpass_modes",
    "sketch_from_anchors",
    "sketch_from_dict",
    "sketch_to_dict",
    "span_floor",
    "sparse_solvers",
    "surrogate_field",
    "term_monomial_keys",
    "term_set_jaccard",
    "to_sympy",
    "trajectories_to_tabular",
    "undefined_terms",
    "validate_terms",
    "verify_equation",
    "vote_terms",
    "wilson_interval",
]



_registry.load_extensions()
