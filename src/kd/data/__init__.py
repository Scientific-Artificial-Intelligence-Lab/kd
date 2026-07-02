"""kd data module: PDE dataset schema and synthetic data generation."""

from kd.data.catalog import (
    DATASET_CATALOG,
    DatasetSpec,
    get_dataset,
    list_datasets,
)
from kd.data.noise import (
    NoiseScale,
    discover_unnormalized,
    xu2020_relative,
)
from kd.data.regression import (
    TabularDataset,
    load_tlc_cc,
    load_wave_breaking,
)
from kd.data.remote import (
    list_remote_datasets,
    load_from_hub,
    load_llm4ed_fisher,
    load_llm4ed_fisher_nonlinear,
    load_llm4ed_heat,
)
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
    compute_dataset_fingerprint,
)
from kd.data.synthetic import (
    generate_advection_data,
    generate_burgers_data,
    generate_diffusion_data,
    load_allen_cahn,
    load_burgers,
    load_burgers_2d,
    load_chafee_infante,
    load_convection_diffusion,
    load_eq_6_2_12,
    load_kdv,
    load_klein_gordon,
    load_pde_compound,
    load_pde_divide,
    load_wave,
)

__all__ = [
    "AxisInfo",
    "DATASET_CATALOG",
    "DataTopology",
    "DatasetSpec",
    "FieldData",
    "NoiseScale",
    "PDEDataset",
    "TabularDataset",
    "TaskType",
    "compute_dataset_fingerprint",
    "discover_unnormalized",
    "generate_advection_data",
    "generate_burgers_data",
    "generate_diffusion_data",
    "get_dataset",
    "list_datasets",
    "list_remote_datasets",
    "load_allen_cahn",
    "load_burgers",
    "load_burgers_2d",
    "load_chafee_infante",
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
    "xu2020_relative",
]
