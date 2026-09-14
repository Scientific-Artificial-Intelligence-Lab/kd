
from __future__ import annotations

__all__ = ["Session", "run"]


def __getattr__(name: str) -> object:
    if name == "run":
        from kdagent.run import run

        return run
    if name == "Session":
        from kdagent.session import Session

        return Session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
