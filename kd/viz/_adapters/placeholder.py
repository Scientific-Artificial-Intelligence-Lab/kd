"""Temporary placeholder adapters for models awaiting unified viz integration."""

from __future__ import annotations

from typing import Iterable

from ..core import VizResult


class DSCVVizAdapter:
    capabilities: Iterable[str] = ()

    def render(self, request, ctx):  # type: ignore[override]
        return VizResult(
            intent=request.kind,
            warnings=['KD_DSCV visualization not integrated yet.'],
        )
