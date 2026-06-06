
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kd.data.schema import PDEDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
DATA_DIR = (
    PROJECT_ROOT
    / "refs" / "discover" / "dso" / "dso" / "task" / "pde" / "data_new"
)








_FAST_TEMPLATE: dict[str, float | int] = {
    "pretrain_epoch": 10_000, "pinn_epoch": 200, "n_iterations": 20,
    "n_cycles": 2, "batch_size": 500, "n_collocation": 10_000,
    "epsilon": 0.02, "entropy_weight": 0.03, "entropy_gamma": 0.7,
    "lr": 0.001, "early_stop_patience": 200,
}
_MEDIUM_TEMPLATE: dict[str, float | int] = {
    "pretrain_epoch": 50_000, "pinn_epoch": 500, "n_iterations": 100,
    "n_cycles": 2, "batch_size": 500, "n_collocation": 20_000,
    "epsilon": 0.02, "entropy_weight": 0.03, "entropy_gamma": 0.7,
    "lr": 0.001, "early_stop_patience": 300,
}
_FULL_TEMPLATE: dict[str, float | int] = {
    "pretrain_epoch": 200_000, "pinn_epoch": 1_000, "n_iterations": 200,
    "n_cycles": 3, "batch_size": 500, "n_collocation": 50_000,
    "epsilon": 0.02, "entropy_weight": 0.03, "entropy_gamma": 0.7,
    "lr": 0.001, "early_stop_patience": 500,
}







@dataclass(frozen=True, slots=True)
class TierSettings:

    pretrain_epoch: int
    pinn_epoch: int
    n_iterations: int
    n_cycles: int
    batch_size: int
    n_collocation: int
    epsilon: float
    entropy_weight: float
    entropy_gamma: float
    lr: float
    early_stop_patience: int
    data_path: Path
    operators: tuple[str, ...]
    max_length: int
    attention: bool
    attn_length: int
    stability_selection: int
    controller_learning_rate: float
    soft_length_loc: float
    soft_length_scale: float
    coef_pde: float
    cycle_n_iterations: int | None
    collocation_cut_ratio: float


@dataclass(frozen=True, slots=True)
class PDESpec:

    pde_name: str
    ground_truth: str
    state_vars: tuple[str, ...]
    coord_vars: tuple[str, ...]
    operators_default: tuple[str, ...]
    operators_aligned: tuple[str, ...]
    presets: dict[str, TierSettings] = field(repr=False)
    default_noise_level: float = 0.5



    output_prefix: str = ""

    def __post_init__(self) -> None:
        if not self.output_prefix:
            object.__setattr__(self, "output_prefix", self.pde_name)

    def data_path_for_tier(self, tier: str) -> Path:
        if tier not in self.presets:
            raise KeyError(
                f"Unknown tier {tier!r} for PDE {self.pde_name!r}"
            )
        return self.presets[tier].data_path

    def load_data(self, tier: str) -> PDEDataset:
        from kd.search.discover.data.loader import (
            load_burgers_mat,
            load_chafee_infante_npy,
            load_fisher_linear_mat,
            load_fisher_nonlinear_mat,
            load_kdv_mat,
            load_pde_compound_npy,
            load_pde_divide_npy,
        )

        path = self.data_path_for_tier(tier)
        if self.pde_name == "burgers":
            return load_burgers_mat(path)
        if self.pde_name == "chafee":

            return load_chafee_infante_npy(path.parent)
        if self.pde_name == "fisher_linear":
            return load_fisher_linear_mat(path)
        if self.pde_name == "fisher_nonlinear":
            return load_fisher_nonlinear_mat(path)
        if self.pde_name == "kdv":
            return load_kdv_mat(path)
        if self.pde_name == "pde_compound":
            return load_pde_compound_npy(path)
        if self.pde_name == "pde_divide":
            return load_pde_divide_npy(path)
        raise ValueError(
            f"No data loader registered for PDE {self.pde_name!r}"
        )
