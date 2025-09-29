"""Unified visualization façade core primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import registry
from ._contracts import (
    FieldComparisonData,
    OptimizationHistoryData,
    ParityPlotData,
    ResidualPlotData,
    RewardEvolutionData,
    TermRelationshipData,
    TimeSliceComparisonData,
)
from ._helpers import resolve_output_path
from ._style import VizConfig, configure as configure_style, get_config, style_context


@dataclass
class VizRequest:
    kind: str
    target: Any
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VizResult:
    intent: str
    figure: Any = None
    paths: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        return bool(self.figure) or bool(self.paths)


@dataclass
class VizContext:
    config: VizConfig
    backend: str
    options: Dict[str, Any]

    def save_path(self, filename: str) -> Path:
        return resolve_output_path(filename, self.config.save_dir)

    @property
    def style(self) -> Dict[str, Any]:
        return self.config.style


def configure(
    *,
    style: Optional[Dict[str, Any]] = None,
    save_dir: Optional[Path] = None,
    backend: Optional[str] = None,
) -> VizConfig:
    return configure_style(style=style, save_dir=save_dir, backend=backend)


def render(request: VizRequest, *, backend: str = 'matplotlib') -> VizResult:
    adapter = registry.get_adapter(request.target)
    requested_backend = None if backend == 'matplotlib' else backend

    config = get_config()
    if requested_backend and requested_backend != config.backend:
        configure_style(backend=requested_backend)
        config = get_config()

    active_backend = requested_backend or config.backend or 'matplotlib'
    ctx = VizContext(config=config, backend=active_backend, options=request.options)

    if adapter is None:
        return VizResult(
            intent=request.kind,
            warnings=[
                f"No visualization adapter registered for {type(request.target).__name__}."
            ],
        )

    capabilities: Iterable[str] = getattr(adapter, 'capabilities', [])
    if request.kind not in capabilities:
        return VizResult(
            intent=request.kind,
            warnings=[
                f"Adapter {adapter.__class__.__name__} does not support intent '{request.kind}'."
            ],
            metadata={'capabilities': sorted(capabilities)},
        )

    with style_context(request.options.get('style')):
        return adapter.render(request, ctx)


def list_capabilities(target: Any) -> Iterable[str]:
    adapter = registry.get_adapter(target)
    if adapter is None:
        return []
    return tuple(adapter.capabilities)


__all__ = [
    'ResidualPlotData',
    'OptimizationHistoryData',
    'FieldComparisonData',
    'TimeSliceComparisonData',
    'TermRelationshipData',
    'ParityPlotData',
    'RewardEvolutionData',
    'VizRequest',
    'VizResult',
    'VizContext',
    'configure',
    'render',
    'list_capabilities',
]
