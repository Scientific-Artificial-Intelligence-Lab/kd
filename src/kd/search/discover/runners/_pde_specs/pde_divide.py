
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








_PDE_DIVIDE_DEFAULT_OPERATORS = (
    "add", "mul", "sub", "div", "n2", "n3", "diff_x", "diff2_x",
)
_PDE_DIVIDE_DATA = DATA_DIR / "PDE_divide.npy"
_PDE_DIVIDE_DEFAULT_MAX_LENGTH = 30




_PDE_DIVIDE_DEFAULT_ATTN_LENGTH = 10
_PDE_DIVIDE_DEFAULT_CONTROLLER_LR = 0.001
_PDE_DIVIDE_SOFT_LENGTH_LOC = 10.0
_PDE_DIVIDE_SOFT_LENGTH_SCALE = 5.0


PDE_DIVIDE_COEF_PDE = 1.0
_PDE_DIVIDE_NOISE_LEVEL = 0.5


_PDE_DIVIDE_FAST_PRESET: dict[str, float | int] = dict(_FAST_TEMPLATE)

PDE_DIVIDE_RAW_PRESETS: dict[str, dict[str, float | int]] = {
    "fast": dict(_PDE_DIVIDE_FAST_PRESET),
}







def _build_pde_divide_tier(
    preset: dict[str, float | int], tier: str,
) -> TierSettings:
    del tier
    return TierSettings(
        **materialize_common_tier_fields(preset),
        data_path=_PDE_DIVIDE_DATA,
        operators=_PDE_DIVIDE_DEFAULT_OPERATORS,
        max_length=_PDE_DIVIDE_DEFAULT_MAX_LENGTH,
        attention=False,
        attn_length=_PDE_DIVIDE_DEFAULT_ATTN_LENGTH,
        stability_selection=0,
        controller_learning_rate=_PDE_DIVIDE_DEFAULT_CONTROLLER_LR,
        soft_length_loc=_PDE_DIVIDE_SOFT_LENGTH_LOC,
        soft_length_scale=_PDE_DIVIDE_SOFT_LENGTH_SCALE,
        coef_pde=PDE_DIVIDE_COEF_PDE,
        cycle_n_iterations=None,
        collocation_cut_ratio=0.0,
    )


def _build_pde_divide_spec() -> PDESpec:
    presets = {
        "fast": _build_pde_divide_tier(_PDE_DIVIDE_FAST_PRESET, "fast"),
    }
    return PDESpec(
        pde_name="pde_divide",
        ground_truth="u_t = -u_x / x + 0.25 * u_xx",
        state_vars=("u",),
        coord_vars=("x", "t"),
        operators_default=_PDE_DIVIDE_DEFAULT_OPERATORS,

        operators_aligned=_PDE_DIVIDE_DEFAULT_OPERATORS,
        presets=presets,
        default_noise_level=_PDE_DIVIDE_NOISE_LEVEL,
        output_prefix="pde_divide",
    )


PDE_DIVIDE_SPEC: PDESpec = _build_pde_divide_spec()
