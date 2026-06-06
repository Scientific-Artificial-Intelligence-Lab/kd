
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





_FISHER_NONLINEAR_DEFAULT_OPERATORS = (
    "add", "mul", "sub", "div", "n2", "n3", "diff_x", "diff2_x",
)
_FISHER_NONLINEAR_DATA = DATA_DIR / "fisher_nonlin_groundtruth.mat"
_FISHER_NONLINEAR_DEFAULT_MAX_LENGTH = 30
_FISHER_NONLINEAR_DEFAULT_ATTN_LENGTH = 10
_FISHER_NONLINEAR_DEFAULT_CONTROLLER_LR = 0.001
_FISHER_NONLINEAR_SOFT_LENGTH_LOC = 12.0
_FISHER_NONLINEAR_SOFT_LENGTH_SCALE = 5.0


FISHER_NONLINEAR_COEF_PDE = 1.0
_FISHER_NONLINEAR_NOISE_LEVEL = 0.5


_FISHER_NONLINEAR_FAST_PRESET: dict[str, float | int] = dict(_FAST_TEMPLATE)

FISHER_NONLINEAR_RAW_PRESETS: dict[str, dict[str, float | int]] = {
    "fast": dict(_FISHER_NONLINEAR_FAST_PRESET),
}







def _build_fisher_nonlinear_tier(
    preset: dict[str, float | int], tier: str,
) -> TierSettings:
    del tier
    return TierSettings(
        **materialize_common_tier_fields(preset),
        data_path=_FISHER_NONLINEAR_DATA,
        operators=_FISHER_NONLINEAR_DEFAULT_OPERATORS,
        max_length=_FISHER_NONLINEAR_DEFAULT_MAX_LENGTH,
        attention=False,
        attn_length=_FISHER_NONLINEAR_DEFAULT_ATTN_LENGTH,
        stability_selection=0,
        controller_learning_rate=_FISHER_NONLINEAR_DEFAULT_CONTROLLER_LR,
        soft_length_loc=_FISHER_NONLINEAR_SOFT_LENGTH_LOC,
        soft_length_scale=_FISHER_NONLINEAR_SOFT_LENGTH_SCALE,
        coef_pde=FISHER_NONLINEAR_COEF_PDE,
        cycle_n_iterations=None,
        collocation_cut_ratio=0.0,
    )


def _build_fisher_nonlinear_spec() -> PDESpec:
    presets = {
        "fast": _build_fisher_nonlinear_tier(
            _FISHER_NONLINEAR_FAST_PRESET, "fast",
        ),
    }
    return PDESpec(
        pde_name="fisher_nonlinear",
        ground_truth="u_t = u * u_xx + (u_x)^2 + u + u^2",
        state_vars=("u",),
        coord_vars=("x", "t"),
        operators_default=_FISHER_NONLINEAR_DEFAULT_OPERATORS,

        operators_aligned=_FISHER_NONLINEAR_DEFAULT_OPERATORS,
        presets=presets,
        default_noise_level=_FISHER_NONLINEAR_NOISE_LEVEL,
        output_prefix="fisher_nonlinear",
    )


FISHER_NONLINEAR_SPEC: PDESpec = _build_fisher_nonlinear_spec()
