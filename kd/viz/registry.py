"""Adapter registry for KD visualization façade."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, Type


class VizAdapter(Protocol):
    capabilities: Iterable[str]

    def render(self, request, ctx):  # pragma: no cover - protocol stub
        ...


_ADAPTERS: Dict[Type[Any], VizAdapter] = {}

# Lazy adapters: resolved on first lookup to avoid heavy imports at load time.
# Maps a class name (str) to a callable that returns (model_cls, adapter) or (None, None).
_LAZY_ADAPTERS: Dict[str, Any] = {}


def register_adapter(model_cls: Type[Any], adapter: VizAdapter) -> None:
    _ADAPTERS[model_cls] = adapter


def register_lazy_adapter(class_name: str, resolver: Any) -> None:
    """Register an adapter that will be resolved on first lookup."""
    _LAZY_ADAPTERS[class_name] = resolver


def unregister_adapter(model_cls: Type[Any]) -> None:
    _ADAPTERS.pop(model_cls, None)


def clear_registry() -> None:
    _ADAPTERS.clear()
    _LAZY_ADAPTERS.clear()


def _resolve_lazy(class_name: str) -> None:
    """Resolve a lazy adapter and promote it to the eager registry."""
    resolver = _LAZY_ADAPTERS.pop(class_name, None)
    if resolver is None:
        return
    model_cls, adapter = resolver()
    if model_cls is not None and adapter is not None:
        _ADAPTERS[model_cls] = adapter


def get_adapter(target: Any) -> Optional[VizAdapter]:
    if target is None:
        return None
    model_cls = target if isinstance(target, type) else type(target)

    adapter = _ADAPTERS.get(model_cls)
    if adapter is not None:
        return adapter

    # Check lazy adapters by class name before subclass walk.
    class_name = model_cls.__name__
    if class_name in _LAZY_ADAPTERS:
        _resolve_lazy(class_name)
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
