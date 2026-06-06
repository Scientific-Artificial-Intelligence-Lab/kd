"""kd data module: PDE dataset schema and synthetic data generation."""

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
    load_burgers,
    load_chafee_infante,
    load_kdv,
    load_pde_compound,
    load_pde_divide,
)

__all__ = [
    "AxisInfo",
    "DataTopology",
    "FieldData",
    "PDEDataset",
    "TaskType",
    "compute_dataset_fingerprint",
    "generate_advection_data",
    "generate_burgers_data",
    "generate_diffusion_data",
    "load_burgers",
    "load_chafee_infante",
    "load_kdv",
    "load_pde_compound",
    "load_pde_divide",
]
