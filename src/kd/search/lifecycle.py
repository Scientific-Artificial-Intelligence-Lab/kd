
from __future__ import annotations

from enum import Enum
from typing import Any


class LifecycleState(Enum):

    CREATED = "created"
    PREPARED = "prepared"
    RUNNING = "running"
    DONE = "done"


class LifecycleError(RuntimeError):
    pass


class SearchLifecycle:

    def __init__(self) -> None:
        self._state: LifecycleState = LifecycleState.CREATED
        self._restored: bool = False

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def restored(self) -> bool:
        return self._restored

    def restore(self, payload: dict[str, Any]) -> None:
        if self._state is not LifecycleState.CREATED:
            raise self._illegal("restore")
        self._restored = bool(payload)

    def prepare(self) -> None:
        if self._state is not LifecycleState.CREATED:
            raise self._illegal("prepare")
        self._state = LifecycleState.PREPARED

    def iterate(self) -> None:
        if self._state not in (LifecycleState.PREPARED, LifecycleState.RUNNING):
            raise self._illegal("iterate")
        self._state = LifecycleState.RUNNING

    def finish(self) -> None:
        if self._state not in (LifecycleState.PREPARED, LifecycleState.RUNNING):
            raise self._illegal("finish")
        self._state = LifecycleState.DONE

    def _illegal(self, transition: str) -> LifecycleError:
        return LifecycleError(
            f"illegal lifecycle transition '{transition}' from state "
            f"{self._state.name}"
        )
