
from __future__ import annotations

from kd.search.discover.runners._pde_specs._base import (
    _FAST_TEMPLATE,
    DATA_DIR,
    PDESpec,
    TierSettings,
)
from kd.search.discover.runners._pde_specs._common import (
    materialize_common_tier_fields,
)








_KDV_DEFAULT_OPERATORS = (
    "add", "mul", "sub", "div", "n2", "n3",
    "diff_x", "diff2_x", "diff3_x",
)
_KDV_DATA = DATA_DIR / "Kdv.mat"
_KDV_DEFAULT_MAX_LENGTH = 30



_KDV_DEFAULT_ATTN_LENGTH = 10
_KDV_DEFAULT_CONTROLLER_LR = 0.001
_KDV_SOFT_LENGTH_LOC = 10.0
_KDV_SOFT_LENGTH_SCALE = 5.0


KDV_COEF_PDE = 1.0
_KDV_NOISE_LEVEL = 0.5


_KDV_FAST_PRESET: dict[str, float | int] = dict(_FAST_TEMPLATE)

KDV_RAW_PRESETS: dict[str, dict[str, float | int]] = {
    "fast": dict(_KDV_FAST_PRESET),
}







def _build_kdv_tier(
    preset: dict[str, float | int], tier: str,
) -> TierSettings:
    del tier
    return TierSettings(
        **materialize_common_tier_fields(preset),
        data_path=_KDV_DATA,
        operators=_KDV_DEFAULT_OPERATORS,
        max_length=_KDV_DEFAULT_MAX_LENGTH,
        attention=False,
        attn_length=_KDV_DEFAULT_ATTN_LENGTH,
        stability_selection=0,
        controller_learning_rate=_KDV_DEFAULT_CONTROLLER_LR,
        soft_length_loc=_KDV_SOFT_LENGTH_LOC,
        soft_length_scale=_KDV_SOFT_LENGTH_SCALE,
        coef_pde=KDV_COEF_PDE,
        cycle_n_iterations=None,
        collocation_cut_ratio=0.0,
    )


def _build_kdv_spec() -> PDESpec:
    presets = {
        "fast": _build_kdv_tier(_KDV_FAST_PRESET, "fast"),
    }
    return PDESpec(
        pde_name="kdv",
        ground_truth="u_t = -u * u_x - 0.0025 * u_xxx",
        state_vars=("u",),
        coord_vars=("x", "t"),
        operators_default=_KDV_DEFAULT_OPERATORS,

        operators_aligned=_KDV_DEFAULT_OPERATORS,
        presets=presets,
        default_noise_level=_KDV_NOISE_LEVEL,
        output_prefix="kdv",
    )


KDV_SPEC: PDESpec = _build_kdv_spec()
