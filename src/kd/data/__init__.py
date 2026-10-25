"""kd data module: PDE dataset schema and synthetic data generation.

Sanctioned direct-import module path beyond this facade: ``kd.data.lowpass``
for the result and refit types of ``select_lowpass_modes``
(``LowpassSelection``, ``LowpassStep``, ``LowpassRefit``, ``RefitOutcome``),
which are not re-exported here.
"""

from kd.data._load import load
from kd.data.catalog import (
    DATASET_CATALOG,
    DatasetSpec,
    get_dataset,
    list_datasets,
    list_datasets_answer_blind,
)
from kd.data.loaders import load_wave_breaking_datasets
from kd.data.lowpass import lowpass_smooth, lowpass_spectrum, select_lowpass_modes
from kd.data.network_smoothing import (
    NetworkSmoothing,
    NetworkSmoothingQuality,
    network_smooth,
    surrogate_field,
)
from kd.data.noise import (
    NoiseScale,
    add_noise,
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
from kd.data.resample import resample_to_grid
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
    compute_dataset_fingerprint,
)
from kd.data.selection import interface_mask
from kd.data.smoothing import savgol_smooth
from kd.data.source import DatasetSource
from kd.data.subsample import ratio_subsample, stride_subsample
from kd.data.synthetic import (
    ODE_CORE_TRACK,
    ODE_SYSTEMS,
    OdeSystem,
    generate_advection_data,
    generate_burgers_data,
    generate_diffusion_data,
    generate_ode_trajectories,
    load_allen_cahn,
    load_burgers,
    load_burgers_2d,
    load_chafee_infante,
    load_convection_diffusion,
    load_eq_6_2_12,
    load_kdv,
    load_klein_gordon,
    load_laplacian_eitech,
    load_laplacian_smile,
    load_pde_compound,
    load_pde_divide,
    load_poisson_disk,
    load_wave,
)
from kd.data.tabular_file import load_tabular
from kd.data.trajectory import Trajectory, TrajectorySet, derivative_target
from kd.data.trajectory_regression import (
    TrajectoryNNConfig,
    TrajectoryRegression,
    prepare_trajectory_regression,
)
from kd.data.trajectory_regression_io import (
    load_trajectory_regression,
    save_trajectory_regression,
)
from kd.data.trajectory_table import trajectories_to_tabular

__all__ = [
    "AxisInfo",
    "DATASET_CATALOG",
    "DataTopology",
    "DatasetSource",
    "DatasetSpec",
    "FieldData",
    "NoiseScale",
    "ODE_CORE_TRACK",
    "ODE_SYSTEMS",
    "OdeSystem",
    "PDEDataset",
    "TabularDataset",
    "TaskType",
    "Trajectory",
    "TrajectoryNNConfig",
    "TrajectoryRegression",
    "TrajectorySet",
    "compute_dataset_fingerprint",
    "derivative_target",
    "discover_unnormalized",
    "generate_advection_data",
    "generate_burgers_data",
    "generate_diffusion_data",
    "generate_ode_trajectories",
    "get_dataset",
    "interface_mask",
    "list_datasets",
    "list_datasets_answer_blind",
    "list_remote_datasets",
    "load",
    "load_allen_cahn",
    "load_burgers",
    "load_burgers_2d",
    "load_chafee_infante",
    "load_convection_diffusion",
    "load_eq_6_2_12",
    "load_from_hub",
    "load_kdv",
    "load_klein_gordon",
    "load_laplacian_eitech",
    "load_laplacian_smile",
    "load_llm4ed_fisher",
    "load_llm4ed_fisher_nonlinear",
    "load_llm4ed_heat",
    "load_pde_compound",
    "load_pde_divide",
    "load_poisson_disk",
    "load_tabular",
    "load_tlc_cc",
    "load_trajectory_regression",
    "load_wave",
    "load_wave_breaking",
    "load_wave_breaking_datasets",
    "lowpass_smooth",
    "lowpass_spectrum",
    "select_lowpass_modes",
    "NetworkSmoothing",
    "NetworkSmoothingQuality",
    "network_smooth",
    "surrogate_field",
    "add_noise",
    "ratio_subsample",
    "prepare_trajectory_regression",
    "resample_to_grid",
    "savgol_smooth",
    "save_trajectory_regression",
    "stride_subsample",
    "trajectories_to_tabular",
    "xu2020_relative",
]
