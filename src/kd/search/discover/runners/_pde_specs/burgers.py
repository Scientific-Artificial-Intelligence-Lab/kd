
from __future__ import annotations

from kd.search.discover.runners._pde_specs._base import (
    _FAST_TEMPLATE,
    _FULL_TEMPLATE,
    _MEDIUM_TEMPLATE,
    DATA_DIR,
    PDESpec,
    TierSettings,
)
from kd.search.discover.runners._pde_specs._common import (
    materialize_common_tier_fields,
)





_BURGERS_DEFAULT_OPERATORS = (
    "add", "mul", "sub", "div", "diff_x", "diff2_x", "n2", "n3",
)
_BURGERS_ALIGNED_OPERATORS = (*_BURGERS_DEFAULT_OPERATORS, "diff3_x")
_BURGERS_DEFAULT_DATA = DATA_DIR / "burgers.mat"
_BURGERS_ALIGNED_DATA = DATA_DIR / "burgers2.mat"
_BURGERS_DEFAULT_MAX_LENGTH = 30
_BURGERS_ALIGNED_MAX_LENGTH = 256
_BURGERS_DEFAULT_ATTN_LENGTH = 10
_BURGERS_ALIGNED_ATTN_LENGTH = 20
_BURGERS_DEFAULT_CONTROLLER_LR = 0.001
_BURGERS_ALIGNED_CONTROLLER_LR = 0.0025
_BURGERS_ALIGNED_STABILITY_SELECTION = 3

_BURGERS_ALIGNED_CYCLE_N_ITERATIONS = 20
_BURGERS_DEFAULT_COLL_CUT_RATIO = 0.05
_BURGERS_ALIGNED_COLL_CUT_RATIO = 0.0
_BURGERS_SOFT_LENGTH_LOC = 12.0
_BURGERS_SOFT_LENGTH_SCALE = 5.0








BURGERS_COEF_PDE = 1.0
_BURGERS_NOISE_LEVEL = 0.5






_BURGERS_FAST_PRESET: dict[str, float | int] = dict(_FAST_TEMPLATE)
_BURGERS_MEDIUM_PRESET: dict[str, float | int] = dict(_MEDIUM_TEMPLATE)
_BURGERS_FULL_PRESET: dict[str, float | int] = dict(_FULL_TEMPLATE)
_BURGERS_ALIGNED_PRESET: dict[str, float | int] = {
    "pretrain_epoch": 200_000, "pinn_epoch": 1_000,
    "n_iterations": 20,
    "n_cycles": 3, "batch_size": 1000, "n_collocation": 50_000,
    "epsilon": 0.01,
    "entropy_weight": 0.03, "entropy_gamma": 0.7,
    "lr": 0.001, "early_stop_patience": 500,
}

BURGERS_RAW_PRESETS: dict[str, dict[str, float | int]] = {
    "fast": dict(_BURGERS_FAST_PRESET),
    "medium": dict(_BURGERS_MEDIUM_PRESET),
    "full": dict(_BURGERS_FULL_PRESET),
    "aligned": dict(_BURGERS_ALIGNED_PRESET),
}







def _build_burgers_tier(
    preset: dict[str, float | int], tier: str,
) -> TierSettings:
    is_aligned = tier == "aligned"
    return TierSettings(
        **materialize_common_tier_fields(preset),
        data_path=(
            _BURGERS_ALIGNED_DATA if is_aligned else _BURGERS_DEFAULT_DATA
        ),
        operators=(
            _BURGERS_ALIGNED_OPERATORS if is_aligned
            else _BURGERS_DEFAULT_OPERATORS
        ),
        max_length=(
            _BURGERS_ALIGNED_MAX_LENGTH if is_aligned
            else _BURGERS_DEFAULT_MAX_LENGTH
        ),
        attention=is_aligned,
        attn_length=(
            _BURGERS_ALIGNED_ATTN_LENGTH if is_aligned
            else _BURGERS_DEFAULT_ATTN_LENGTH
        ),
        stability_selection=(
            _BURGERS_ALIGNED_STABILITY_SELECTION if is_aligned else 0
        ),
        controller_learning_rate=(
            _BURGERS_ALIGNED_CONTROLLER_LR if is_aligned
            else _BURGERS_DEFAULT_CONTROLLER_LR
        ),
        soft_length_loc=_BURGERS_SOFT_LENGTH_LOC,
        soft_length_scale=_BURGERS_SOFT_LENGTH_SCALE,
        coef_pde=BURGERS_COEF_PDE,
        cycle_n_iterations=(
            _BURGERS_ALIGNED_CYCLE_N_ITERATIONS if is_aligned else None
        ),
        collocation_cut_ratio=(
            _BURGERS_ALIGNED_COLL_CUT_RATIO if is_aligned
            else _BURGERS_DEFAULT_COLL_CUT_RATIO
        ),
    )


def _build_burgers_spec() -> PDESpec:
    presets = {
        name: _build_burgers_tier(preset, name)
        for name, preset in (
            ("fast", _BURGERS_FAST_PRESET),
            ("medium", _BURGERS_MEDIUM_PRESET),
            ("full", _BURGERS_FULL_PRESET),
            ("aligned", _BURGERS_ALIGNED_PRESET),
        )
    }
    return PDESpec(
        pde_name="burgers",
        ground_truth="u_t = -u * u_x + 0.1 * u_xx",
        state_vars=("u",),
        coord_vars=("x", "t"),
        operators_default=_BURGERS_DEFAULT_OPERATORS,
        operators_aligned=_BURGERS_ALIGNED_OPERATORS,
        presets=presets,
        default_noise_level=_BURGERS_NOISE_LEVEL,
        output_prefix="burgers",
    )


BURGERS_SPEC: PDESpec = _build_burgers_spec()
