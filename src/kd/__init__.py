"""kd - Symbolic regression platform for PDE discovery."""

from kd.api import Model, instrument_schemas
from kd.core.equation import Sketch, law_signature
from kd.core.evaluator import EvaluationResult
from kd.core.verify import VerificationReport, VerifyPolicy, verify_equation
from kd.data import (
    DATASET_CATALOG,
    AxisInfo,
    DatasetSpec,
    DataTopology,
    FieldData,
    PDEDataset,
    TabularDataset,
    TaskType,
    generate_advection_data,
    generate_burgers_data,
    generate_diffusion_data,
    get_dataset,
    list_datasets,
    list_remote_datasets,
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
    AxisReport,
    DatasetReport,
    FieldReport,
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
from kd.search.discover import DiscoverConfig
from kd.search.dlga import DLGAConfig
from kd.search.eqgpt import EqGPTConfig
from kd.search.llm4ed import Llm4edConfig
from kd.search.pysindy import PySINDyConfig
from kd.search.pysr import PySRConfig
from kd.search.result import ExperimentResult
from kd.search.sga import SGAConfig
from kd.viz.engine import VizEngine

__version__ = "0.7.1"





__all__ = [
    "AxisInfo",
    "AxisReport",
    "CheckpointManifestEntry",
    "CheckpointManifestError",
    "DATASET_CATALOG",
    "DLGAConfig",
    "DataTopology",
    "DatasetReport",
    "DatasetSpec",
    "DiscoverConfig",
    "EqGPTConfig",
    "EvaluationFailedError",
    "EvaluationResult",
    "ExperimentResult",
    "FINAL_STATUS_COMPLETED",
    "FieldData",
    "FieldReport",
    "InvalidTermsError",
    "KIND_FINAL",
    "Llm4edConfig",
    "Model",
    "PDEDataset",
    "PySINDyConfig",
    "PySRConfig",
    "SGAConfig",
    "Sketch",
    "TabularDataset",
    "TaskType",
    "TermRejection",
    "TermValidationReport",
    "VerificationReport",
    "VerifyPolicy",
    "VizEngine",
    "__version__",
    "evaluate_terms",
    "generate_advection_data",
    "generate_burgers_data",
    "generate_diffusion_data",
    "get_dataset",
    "instrument_schemas",
    "law_signature",
    "list_datasets",
    "list_remote_datasets",
    "load_allen_cahn",
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
    "preview",
    "preview_report",
    "validate_terms",
    "verify_equation",
]
