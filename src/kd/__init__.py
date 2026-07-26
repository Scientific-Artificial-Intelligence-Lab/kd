"""kd - Symbolic regression platform for PDE discovery."""

from kd.api import Model, instrument_schemas
from kd.core.equation import law_signature
from kd.core.evaluator import EvaluationResult
from kd.core.verify import VerifyPolicy, verify_equation
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
from kd.inspect import preview
from kd.search.checkpoint_manifest import (
    CheckpointManifestEntry,
    CheckpointManifestError,
    load_checkpoint_manifest,
)
from kd.search.discover import DiscoverConfig
from kd.search.dlga import DLGAConfig
from kd.search.result import ExperimentResult
from kd.search.sga import SGAConfig
from kd.viz.engine import VizEngine

__version__ = "0.5.0"

__all__ = [
    "AxisInfo",
    "CheckpointManifestEntry",
    "CheckpointManifestError",
    "DATASET_CATALOG",
    "DLGAConfig",
    "DataTopology",
    "DatasetSpec",
    "DiscoverConfig",
    "EvaluationFailedError",
    "EvaluationResult",
    "ExperimentResult",
    "FieldData",
    "InvalidTermsError",
    "Model",
    "PDEDataset",
    "SGAConfig",
    "TabularDataset",
    "TaskType",
    "TermRejection",
    "TermValidationReport",
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
    "validate_terms",
    "verify_equation",
]
