
from __future__ import annotations

from typing import Any


from kd.search.discover.runners._pde_specs._base import (
    DATA_DIR,
    PROJECT_ROOT,
    PDESpec,
    TierSettings,
)
from kd.search.discover.runners._pde_specs.burgers import (
    BURGERS_COEF_PDE,
    BURGERS_RAW_PRESETS,
    BURGERS_SPEC,
)
from kd.search.discover.runners._pde_specs.chafee import (
    CHAFEE_COEF_PDE,
    CHAFEE_RAW_PRESETS,
    CHAFEE_SPEC,
)
from kd.search.discover.runners._pde_specs.fisher_linear import (
    FISHER_LINEAR_COEF_PDE,
    FISHER_LINEAR_RAW_PRESETS,
    FISHER_LINEAR_SPEC,
)
from kd.search.discover.runners._pde_specs.fisher_nonlinear import (
    FISHER_NONLINEAR_COEF_PDE,
    FISHER_NONLINEAR_RAW_PRESETS,
    FISHER_NONLINEAR_SPEC,
)
from kd.search.discover.runners._pde_specs.kdv import (
    KDV_COEF_PDE,
    KDV_RAW_PRESETS,
    KDV_SPEC,
)
from kd.search.discover.runners._pde_specs.pde_compound import (
    PDE_COMPOUND_COEF_PDE,
    PDE_COMPOUND_RAW_PRESETS,
    PDE_COMPOUND_SPEC,
)
from kd.search.discover.runners._pde_specs.pde_divide import (
    PDE_DIVIDE_COEF_PDE,
    PDE_DIVIDE_RAW_PRESETS,
    PDE_DIVIDE_SPEC,
)










_BURGERS_COEF_PDE: float = BURGERS_COEF_PDE
_CHAFEE_COEF_PDE: float = CHAFEE_COEF_PDE
_FISHER_LINEAR_COEF_PDE: float = FISHER_LINEAR_COEF_PDE
_FISHER_NONLINEAR_COEF_PDE: float = FISHER_NONLINEAR_COEF_PDE
_KDV_COEF_PDE: float = KDV_COEF_PDE
_PDE_COMPOUND_COEF_PDE: float = PDE_COMPOUND_COEF_PDE
_PDE_DIVIDE_COEF_PDE: float = PDE_DIVIDE_COEF_PDE







PDE_REGISTRY: dict[str, PDESpec] = {
    "burgers": BURGERS_SPEC,
    "chafee": CHAFEE_SPEC,
    "fisher_linear": FISHER_LINEAR_SPEC,
    "fisher_nonlinear": FISHER_NONLINEAR_SPEC,
    "kdv": KDV_SPEC,
    "pde_compound": PDE_COMPOUND_SPEC,
    "pde_divide": PDE_DIVIDE_SPEC,
}













_RAW_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "burgers": dict(BURGERS_RAW_PRESETS),
    "chafee": dict(CHAFEE_RAW_PRESETS),
    "fisher_linear": dict(FISHER_LINEAR_RAW_PRESETS),
    "fisher_nonlinear": dict(FISHER_NONLINEAR_RAW_PRESETS),
    "kdv": dict(KDV_RAW_PRESETS),
    "pde_compound": dict(PDE_COMPOUND_RAW_PRESETS),
    "pde_divide": dict(PDE_DIVIDE_RAW_PRESETS),
}


def _assert_raw_presets_consistent() -> None:
    for pde, tier_dict in _RAW_PRESETS.items():
        spec = PDE_REGISTRY[pde]
        for tier, raw in tier_dict.items():
            settings = spec.presets[tier]
            for key, raw_value in raw.items():
                if not hasattr(settings, key):
                    continue
                materialized = getattr(settings, key)
                assert materialized == raw_value, (
                    f"_RAW_PRESETS[{pde!r}][{tier!r}][{key!r}]={raw_value!r}"
                    f" != PDE_REGISTRY[{pde!r}].presets[{tier!r}].{key}="
                    f"{materialized!r}"
                )


_assert_raw_presets_consistent()


__all__ = [
    "BURGERS_SPEC",
    "CHAFEE_SPEC",
    "DATA_DIR",
    "FISHER_LINEAR_SPEC",
    "FISHER_NONLINEAR_SPEC",
    "KDV_SPEC",
    "PDE_COMPOUND_SPEC",
    "PDE_DIVIDE_SPEC",
    "PDE_REGISTRY",
    "PDESpec",
    "PROJECT_ROOT",
    "TierSettings",
]
