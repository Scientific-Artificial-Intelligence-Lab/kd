"""KD visualization package: façade, helpers, and legacy modules."""

from __future__ import annotations

from . import core as _core
from .core import (
    ResidualPlotData,
    VizContext,
    VizRequest,
    VizResult,
    configure,
    list_capabilities,
    render,
)
from .registry import (
    clear_registry,
    iter_registered,
    register_adapter,
    unregister_adapter,
)
from .api import (
    plot_residuals,
    plot_search_evolution,
    plot_training_curve,
    plot_validation_curve,
    render_equation,
)

# Attempt to register default adapters; degrade gracefully if optional deps missing.
try:  # pragma: no cover - defensive import
    from .adapters import (
        DLGAVizAdapter,
        DSCVVizAdapter,
        register_default_adapters,
    )
except Exception:  # pragma: no cover - adapter import failure shouldn't break base API
    DLGAVizAdapter = None  # type: ignore
    DSCVVizAdapter = None  # type: ignore
    register_default_adapters = None  # type: ignore
else:
    register_default_adapters()

# Re-export legacy visualization modules for backward compatibility.
from . import dlga_viz  # noqa: E402,F401
from . import dscv_viz  # noqa: E402,F401
from . import equation_renderer  # noqa: E402,F401
from . import dlga_eq2latex  # noqa: E402,F401
from . import discover_eq2latex  # noqa: E402,F401

__all__ = [
    'ResidualPlotData',
    'VizContext',
    'VizRequest',
    'VizResult',
    'configure',
    'render',
    'list_capabilities',
    'register_adapter',
    'unregister_adapter',
    'clear_registry',
    'iter_registered',
    'plot_training_curve',
    'plot_validation_curve',
    'plot_search_evolution',
    'plot_residuals',
    'render_equation',
    'DLGAVizAdapter',
    'DSCVVizAdapter',
    'register_default_adapters',
    'dlga_viz',
    'dscv_viz',
    'equation_renderer',
    'dlga_eq2latex',
    'discover_eq2latex',
]
