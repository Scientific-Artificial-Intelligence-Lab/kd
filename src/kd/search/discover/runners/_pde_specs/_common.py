
from __future__ import annotations

from typing import Any


def materialize_common_tier_fields(
    preset: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "pretrain_epoch": int(preset["pretrain_epoch"]),
        "pinn_epoch": int(preset["pinn_epoch"]),
        "n_iterations": int(preset["n_iterations"]),
        "n_cycles": int(preset["n_cycles"]),
        "batch_size": int(preset["batch_size"]),
        "n_collocation": int(preset["n_collocation"]),
        "epsilon": float(preset["epsilon"]),
        "entropy_weight": float(preset["entropy_weight"]),
        "entropy_gamma": float(preset["entropy_gamma"]),
        "lr": float(preset["lr"]),
        "early_stop_patience": int(preset["early_stop_patience"]),
    }
