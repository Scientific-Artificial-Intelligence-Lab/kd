
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








_PDE_COMPOUND_DEFAULT_OPERATORS = (
    "add", "mul", "sub", "div", "n2", "n3", "diff_x", "diff2_x",
)
_PDE_COMPOUND_DATA = DATA_DIR / "PDE_compound.npy"
_PDE_COMPOUND_DEFAULT_MAX_LENGTH = 30
_PDE_COMPOUND_DEFAULT_ATTN_LENGTH = 10
_PDE_COMPOUND_DEFAULT_CONTROLLER_LR = 0.001
_PDE_COMPOUND_SOFT_LENGTH_LOC = 12.0
_PDE_COMPOUND_SOFT_LENGTH_SCALE = 5.0


PDE_COMPOUND_COEF_PDE = 1.0
_PDE_COMPOUND_NOISE_LEVEL = 0.5


_PDE_COMPOUND_FAST_PRESET: dict[str, float | int] = dict(_FAST_TEMPLATE)

PDE_COMPOUND_RAW_PRESETS: dict[str, dict[str, float | int]] = {
    "fast": dict(_PDE_COMPOUND_FAST_PRESET),
}







def _build_pde_compound_tier(
    preset: dict[str, float | int], tier: str,
) -> TierSettings:
    del tier
    return TierSettings(
        **materialize_common_tier_fields(preset),
        data_path=_PDE_COMPOUND_DATA,
        operators=_PDE_COMPOUND_DEFAULT_OPERATORS,
        max_length=_PDE_COMPOUND_DEFAULT_MAX_LENGTH,
        attention=False,
        attn_length=_PDE_COMPOUND_DEFAULT_ATTN_LENGTH,
        stability_selection=0,
        controller_learning_rate=_PDE_COMPOUND_DEFAULT_CONTROLLER_LR,
        soft_length_loc=_PDE_COMPOUND_SOFT_LENGTH_LOC,
        soft_length_scale=_PDE_COMPOUND_SOFT_LENGTH_SCALE,
        coef_pde=PDE_COMPOUND_COEF_PDE,
        cycle_n_iterations=None,
        collocation_cut_ratio=0.0,
    )


def _build_pde_compound_spec() -> PDESpec:
    presets = {
        "fast": _build_pde_compound_tier(_PDE_COMPOUND_FAST_PRESET, "fast"),
    }
    return PDESpec(
        pde_name="pde_compound",
        ground_truth="u_t = u * u_xx + (u_x)^2",
        state_vars=("u",),
        coord_vars=("x", "t"),
        operators_default=_PDE_COMPOUND_DEFAULT_OPERATORS,

        operators_aligned=_PDE_COMPOUND_DEFAULT_OPERATORS,
        presets=presets,
        default_noise_level=_PDE_COMPOUND_NOISE_LEVEL,
        output_prefix="pde_compound",
    )


PDE_COMPOUND_SPEC: PDESpec = _build_pde_compound_spec()
