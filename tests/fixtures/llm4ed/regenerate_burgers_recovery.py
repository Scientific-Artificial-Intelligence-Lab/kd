
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

from kd.api import Model
from kd.llm import TapeRecordingProvider



_VALIDATION = Path(__file__).resolve().parents[2] / "validation" / "llm4ed"
_UNIT = Path(__file__).resolve().parents[2] / "unit" / "search" / "llm4ed"
sys.path.insert(0, str(_VALIDATION))
sys.path.insert(0, str(_UNIT))

from _plugin_helpers import FakeProvider
from _recovery_helpers import (
    GENERATIONS,
    RESPONSES,
    TAPE_FILENAME,
    burgers_recovery_config,
    burgers_recovery_dataset,
)

_TAPE = Path(__file__).resolve().parent / TAPE_FILENAME


def main() -> None:
    if _TAPE.exists():
        _TAPE.unlink()
    provider = TapeRecordingProvider(FakeProvider(list(RESPONSES)), path=_TAPE)
    model = Model(
        algorithm="llm4ed",
        generations=GENERATIONS,
        config=burgers_recovery_config(),
        provider=provider,
    )
    start = time.perf_counter()
    model.fit(burgers_recovery_dataset())
    elapsed = time.perf_counter() - start

    n_lines = len(_TAPE.read_text(encoding="utf-8").splitlines())
    final = model.result_.final_eval
    print(f"Recorded {_TAPE} ({n_lines} entries)")
    print(f"best_expr_ = {model.best_expr_!r}")
    print(f"best_score_ = {model.best_score_:.6f}")
    print(f"iterations = {model.result_.iterations}")
    print(f"final.is_valid = {final.is_valid}")
    print(f"final.r2 = {final.r2:.6f}")
    print(f"final.terms = {final.terms}")
    coeffs = final.coefficients
    if final.terms is not None and coeffs is not None:
        for term, coeff in zip(final.terms, coeffs, strict=True):
            print(f" {term!r}: {float(coeff):+.6f}")
    print(f"fit() elapsed = {elapsed:.3f} s")
    digest = hashlib.sha256(_TAPE.read_bytes()).hexdigest()
    print(f"sha256(tape) = {digest}")



    print(
        " -> paste into BURGERS_TAPE_SHA256 in the llm4ed validation conftest "
        "(same commit as the tape)"
    )


if __name__ == "__main__":
    main()
