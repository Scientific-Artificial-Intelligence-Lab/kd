
from __future__ import annotations

import contextlib
import dataclasses
import re
from collections.abc import Iterator
from importlib.metadata import Distribution, entry_points
from typing import Any, Final, Generic, Literal, TypeVar, cast


ENTRY_POINT_GROUP: Final = "kd.extensions"

Source = Literal["builtin", "entry_point", "process"]

_T = TypeVar("_T")
_NAME_RULE: Final = re.compile(r"[a-z][a-z0-9_]*")


@dataclasses.dataclass(frozen=True)
class Origin:

    source: Source
    distribution: str | None = None
    version: str | None = None
    hook: str | None = None


_BUILTIN: Final = Origin("builtin")
_PROCESS: Final = Origin("process")



_current: Origin = _PROCESS


_REGISTRIES: dict[str, Registry[Any]] = {}


class Registry(Generic[_T]):

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.entries: dict[str, _T] = {}
        self.origins: dict[str, Origin] = {}
        _REGISTRIES[kind] = self

    def check_available(self, name: str) -> None:
        if _NAME_RULE.fullmatch(name) is None:
            raise ValueError(
                f"{self.kind} name {name!r} does not match [a-z][a-z0-9_]*"
            )
        if name in self.entries:
            raise ValueError(
                f"{self.kind} name {name!r} is already registered "
                f"({self.origins[name].source}); a name registers once per process"
            )

    def __contains__(self, name: object) -> bool:
        return name in self.entries

    def register(self, name: str, entry: _T) -> None:
        self.check_available(name)
        self.entries[name] = entry
        self.origins[name] = _current

    def get(self, name: str) -> _T:
        self._require(name)
        return self.entries[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def origin(self, name: str) -> Origin:
        self._require(name)
        return self.origins[name]

    def _require(self, name: str) -> None:
        if name not in self.entries:
            raise KeyError(
                f"no {self.kind} is registered as {name!r}; registered: "
                f"{list(self.entries)}"
            )


def extension_source(kind: str, name: str) -> Source:
    return _registry_of(kind).origin(name).source


def extension_origin(kind: str, name: str) -> dict[str, str | None]:
    registry = _registry_of(kind)
    origin = registry.origin(name)
    entry = registry.entries[name]
    return {
        "source": origin.source,
        "distribution": origin.distribution,
        "version": origin.version,
        "hook": origin.hook,
        "implementation": f"{entry.__module__}:{entry.__qualname__}",
    }


def _registry_of(kind: str) -> Registry[Any]:
    if kind not in _REGISTRIES:
        raise KeyError(f"unknown extension kind {kind!r}; kinds: {list(_REGISTRIES)}")
    return _REGISTRIES[kind]


@contextlib.contextmanager
def _registering_as(origin: Origin) -> Iterator[None]:
    global _current
    previous = _current
    _current = origin
    try:
        yield
    finally:
        _current = previous


def registering_builtins() -> contextlib.AbstractContextManager[None]:
    return _registering_as(_BUILTIN)


def load_extensions() -> None:
    for hook in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name):

        dist = cast(Distribution, hook.dist)
        origin = Origin(
            "entry_point",
            distribution=dist.name,
            version=dist.version,
            hook=f"{hook.name} = {hook.value}",
        )
        try:
            with _registering_as(origin):
                hook.load()()
        except Exception as exc:
            raise ImportError(
                f"{ENTRY_POINT_GROUP} hook {origin.hook!r} of distribution "
                f"{dist.name} {dist.version} failed: {exc}"
            ) from exc


__all__ = [
    "ENTRY_POINT_GROUP",
    "Origin",
    "Registry",
    "Source",
    "extension_origin",
    "extension_source",
    "load_extensions",
    "registering_builtins",
]
