
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





_CHAFEE_DEFAULT_OPERATORS = (
    "add",
    "mul",
    "sub",
    "div",
    "n2",
    "n3",
    "diff_x",
    "diff2_x",
)
_CHAFEE_ALIGNED_OPERATORS = (*_CHAFEE_DEFAULT_OPERATORS, "diff3_x")
_CHAFEE_DEFAULT_MAX_LENGTH = 30
_CHAFEE_ALIGNED_MAX_LENGTH = 256
_CHAFEE_DEFAULT_ATTN_LENGTH = 10
_CHAFEE_ALIGNED_ATTN_LENGTH = 20
_CHAFEE_DEFAULT_CONTROLLER_LR = 0.001
_CHAFEE_ALIGNED_CONTROLLER_LR = 0.0025
_CHAFEE_ALIGNED_STABILITY_SELECTION = 3





_CHAFEE_ALIGNED_CYCLE_N_ITERATIONS = 20
_CHAFEE_DEFAULT_SOFT_LENGTH_LOC = 12.0
_CHAFEE_ALIGNED_SOFT_LENGTH_LOC = 10.0
_CHAFEE_SOFT_LENGTH_SCALE = 5.0












CHAFEE_COEF_PDE = 1.0
_CHAFEE_NOISE_LEVEL = 0.5
_CHAFEE_DATA = DATA_DIR / "chafee_infante_CI.npy"






_CHAFEE_FAST_PRESET: dict[str, float | int] = dict(_FAST_TEMPLATE)
_CHAFEE_MEDIUM_PRESET: dict[str, float | int] = dict(_MEDIUM_TEMPLATE)
_CHAFEE_FULL_PRESET: dict[str, float | int] = dict(_FULL_TEMPLATE)
_CHAFEE_ALIGNED_PRESET: dict[str, float | int] = {
    "pretrain_epoch": 200_000,
    "pinn_epoch": 1_000,
    "n_iterations": 100,
    "n_cycles": 2,
    "batch_size": 500,
    "n_collocation": 50_000,
    "epsilon": 0.02,
    "entropy_weight": 0.03,
    "entropy_gamma": 0.7,
    "lr": 0.001,
    "early_stop_patience": 500,
}

CHAFEE_RAW_PRESETS: dict[str, dict[str, float | int]] = {
    "fast": dict(_CHAFEE_FAST_PRESET),
    "medium": dict(_CHAFEE_MEDIUM_PRESET),
    "full": dict(_CHAFEE_FULL_PRESET),
    "aligned": dict(_CHAFEE_ALIGNED_PRESET),
}







def _build_chafee_tier(
    preset: dict[str, float | int],
    tier: str,
) -> TierSettings:
    is_aligned = tier == "aligned"
    return TierSettings(
        **materialize_common_tier_fields(preset),
        data_path=_CHAFEE_DATA,
        operators=(
            _CHAFEE_ALIGNED_OPERATORS if is_aligned else _CHAFEE_DEFAULT_OPERATORS
        ),
        max_length=(
            _CHAFEE_ALIGNED_MAX_LENGTH if is_aligned else _CHAFEE_DEFAULT_MAX_LENGTH
        ),
        attention=is_aligned,
        attn_length=(
            _CHAFEE_ALIGNED_ATTN_LENGTH if is_aligned else _CHAFEE_DEFAULT_ATTN_LENGTH
        ),
        stability_selection=(_CHAFEE_ALIGNED_STABILITY_SELECTION if is_aligned else 0),
        controller_learning_rate=(
            _CHAFEE_ALIGNED_CONTROLLER_LR
            if is_aligned
            else _CHAFEE_DEFAULT_CONTROLLER_LR
        ),
        soft_length_loc=(
            _CHAFEE_ALIGNED_SOFT_LENGTH_LOC
            if is_aligned
            else _CHAFEE_DEFAULT_SOFT_LENGTH_LOC
        ),
        soft_length_scale=_CHAFEE_SOFT_LENGTH_SCALE,
        coef_pde=CHAFEE_COEF_PDE,
        cycle_n_iterations=(_CHAFEE_ALIGNED_CYCLE_N_ITERATIONS if is_aligned else None),
        collocation_cut_ratio=0.0,
    )


def _build_chafee_spec() -> PDESpec:
    presets = {
        name: _build_chafee_tier(preset, name)
        for name, preset in (
            ("fast", _CHAFEE_FAST_PRESET),
            ("medium", _CHAFEE_MEDIUM_PRESET),
            ("full", _CHAFEE_FULL_PRESET),
            ("aligned", _CHAFEE_ALIGNED_PRESET),
        )
    }
    return PDESpec(
        pde_name="chafee",
        ground_truth="u_t = u_xx + u^3 - u",
        state_vars=("u",),
        coord_vars=("x", "t"),
        operators_default=_CHAFEE_DEFAULT_OPERATORS,
        operators_aligned=_CHAFEE_ALIGNED_OPERATORS,
        presets=presets,
        default_noise_level=_CHAFEE_NOISE_LEVEL,
        output_prefix="chafee",
    )


CHAFEE_SPEC: PDESpec = _build_chafee_spec()
