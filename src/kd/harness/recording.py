
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RecordingOptions"]


@dataclass(frozen=True, kw_only=True)
class RecordingOptions:

    events_every_n: int = 1
    checkpoint_every: int | None = None
    checkpoint_keep_last: int | None = None
    phases: bool = True

    def __post_init__(self) -> None:
        if type(self.events_every_n) is not int or self.events_every_n < 1:
            raise ValueError(
                f"events_every_n must be an int >= 1; got {self.events_every_n!r}"
            )
        if self.checkpoint_every is not None and (
            type(self.checkpoint_every) is not int or self.checkpoint_every < 1
        ):
            raise ValueError(
                "checkpoint_every must be an int >= 1 or None; "
                f"got {self.checkpoint_every!r}"
            )
        if self.checkpoint_keep_last is not None:
            if self.checkpoint_every is None:
                raise ValueError(
                    "checkpoint_keep_last requires checkpoint_every"
                )
            if (
                type(self.checkpoint_keep_last) is not int
                or self.checkpoint_keep_last < 1
            ):
                raise ValueError(
                    "checkpoint_keep_last must be an int >= 1 or None; "
                    f"got {self.checkpoint_keep_last!r}"
                )
        if not isinstance(self.phases, bool):
            raise ValueError(f"phases must be a bool; got {self.phases!r}")
