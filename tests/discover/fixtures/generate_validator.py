from __future__ import annotations

from pathlib import Path

import numpy as np


NAMES = [
    "x", "t", "u",
    "add", "mul",
    "sin", "cos", "diff_x", "diff_t", "n2", "neg",
    "div",
]
ARITIES = np.array([0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32)
MAX_LENGTH = 10



def _natural_length(tokens: list[int]) -> int:
    if not tokens:
        return 0
    dangling = 1
    for i, t in enumerate(tokens):
        dangling += int(ARITIES[t]) - 1
        if dangling == 0:
            return i + 1
    return -1


def _is_valid(tokens: list[int]) -> bool:
    nat = _natural_length(tokens)
    if nat <= 0:
        return False
    if nat > MAX_LENGTH:
        return False

    return nat > 1




CASES: list[tuple[str, list[int]]] = [

    ("valid_burgers", [3, 4, 2, 7, 2, 8, 2]),
    ("valid_simple", [4, 2, 2]),
    ("valid_unary", [7, 2]),
    ("valid_nested", [5, 7, 2]),
    ("valid_at_max", [3, 4, 5, 2, 7, 2, 4, 2, 8, 2]),

    ("invalid_incomplete", [3, 4, 2]),
    ("invalid_incomplete_unary", [7]),
    ("invalid_too_long", [3, 4, 5, 2, 7, 2, 4, 2, 8, 9, 2]),
    ("invalid_trivial", [2]),
    ("invalid_trivial_coord", [0]),
]


def main() -> None:
    data: dict[str, np.ndarray] = {
        "names": np.array(NAMES),
        "arities": ARITIES,
        "max_length": np.int32(MAX_LENGTH),
    }


    for name, tokens in CASES:
        data[f"case_{name}_tokens"] = np.array(tokens, dtype=np.int32)
        data[f"case_{name}_expected"] = np.bool_(_is_valid(tokens))


    empty_action = len(NAMES)
    max_len = max(len(t) for _, t in CASES)
    batch = np.full((len(CASES), max_len), empty_action, dtype=np.int32)
    mask = np.zeros(len(CASES), dtype=np.bool_)
    for i, (_, tokens) in enumerate(CASES):
        batch[i,: len(tokens)] = tokens
        mask[i] = _is_valid(tokens)

    data["batch_tokens"] = batch
    data["batch_mask"] = mask

    out = Path(__file__).parent / "validator.npz"
    np.savez(out, **data)

    for name, tokens in CASES:
        nat = _natural_length(tokens)
        valid = _is_valid(tokens)
        print(f" {name}: len={len(tokens)}, nat={nat}, valid={valid}")
    print(f"Batch: {mask.sum()}/{len(CASES)} valid")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
