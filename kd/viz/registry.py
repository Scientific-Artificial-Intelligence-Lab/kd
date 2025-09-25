"""Adapter registry for KD visualization façade."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, Type


class VizAdapter(Protocol):
    capabilities: Iterable[str]

    def render(self, request, ctx):  # pragma: no cover - protocol stub
        ...


_ADAPTERS: Dict[Type[Any], VizAdapter] = {}


def register_adapter(model_cls: Type[Any], adapter: VizAdapter) -> None:
    _ADAPTERS[model_cls] = adapter


def unregister_adapter(model_cls: Type[Any]) -> None:
    _ADAPTERS.pop(model_cls, None)


def clear_registry() -> None:
    _ADAPTERS.clear()


def get_adapter(target: Any) -> Optional[VizAdapter]:
    if target is None:
        return None
    model_cls = target if isinstance(target, type) else type(target)

    adapter = _ADAPTERS.get(model_cls)
    if adapter is not None:
        return adapter

    for registered_cls, registered_adapter in _ADAPTERS.items():
        try:
            if issubclass(model_cls, registered_cls):
                return registered_adapter
        except TypeError:
            continue
    return None


def iter_registered() -> Iterable[tuple[Type[Any], VizAdapter]]:
    return tuple(_ADAPTERS.items())
