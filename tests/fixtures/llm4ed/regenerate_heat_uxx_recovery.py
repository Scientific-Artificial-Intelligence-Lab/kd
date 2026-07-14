
from __future__ import annotations

import sys
from pathlib import Path

from kd.api import Model
from kd.llm import TapeRecordingProvider



_TESTS = Path(__file__).resolve().parents[2] / "unit" / "search" / "llm4ed"
sys.path.insert(0, str(_TESTS))
from _plugin_helpers import (
    GOOD,
    FakeProvider,
    heat_dataset,
    make_config,
)

_FIXTURE = Path(__file__).resolve().parent / "heat_uxx_recovery.jsonl"


def main() -> None:
    if _FIXTURE.exists():
        _FIXTURE.unlink()
    provider = TapeRecordingProvider(FakeProvider(GOOD), path=_FIXTURE)
    model = Model(
        algorithm="llm4ed",
        generations=2,
        config=make_config(),
        provider=provider,
    )
    model.fit(heat_dataset())
    n_lines = len(_FIXTURE.read_text(encoding="utf-8").splitlines())
    print(
        f"Recorded {_FIXTURE} ({n_lines} entries); "
        f"recovered {model.best_expr_!r} at reward {model.best_score_:.4f}."
    )


if __name__ == "__main__":
    main()
