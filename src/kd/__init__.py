"""kd - Symbolic regression platform for PDE discovery."""

from kd.api import Model
from kd.data import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
    generate_advection_data,
    generate_burgers_data,
    generate_diffusion_data,
    load_burgers,
    load_chafee_infante,
    load_kdv,
    load_pde_compound,
    load_pde_divide,
)
from kd.inspect import preview
from kd.search.discover import DiscoverConfig
from kd.search.dlga import DLGAConfig
from kd.search.result import ExperimentResult
from kd.search.sga import SGAConfig
from kd.viz.engine import VizEngine

__version__ = "0.1.0"

__all__ = [
    "AxisInfo",
    "DLGAConfig",
    "DataTopology",
    "DiscoverConfig",
    "ExperimentResult",
    "FieldData",
    "Model",
    "PDEDataset",
    "SGAConfig",
    "TaskType",
    "VizEngine",
    "__version__",
    "generate_advection_data",
    "generate_burgers_data",
    "generate_diffusion_data",
    "load_burgers",
    "load_chafee_infante",
    "load_kdv",
    "load_pde_compound",
    "load_pde_divide",
    "preview",
]
