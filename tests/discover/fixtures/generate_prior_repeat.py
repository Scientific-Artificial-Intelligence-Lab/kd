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
L = len(NAMES)
EMPTY_ACTION = L

ADD_IDX = NAMES.index("add")
MUL_IDX = NAMES.index("mul")
SIN_IDX = NAMES.index("sin")
U_IDX = NAMES.index("u")
NEG_IDX = NAMES.index("neg")




def _ref_repeat(
    actions: np.ndarray,
    target_tokens: np.ndarray,
    max_: int,
    n_choices: int,
) -> np.ndarray:
    batch_size = actions.shape[0]
    prior = np.zeros((batch_size, n_choices), dtype=np.float32)

    if actions.shape[1] == 0:
        return prior

    counts = np.sum(np.isin(actions, target_tokens), axis=1)
    mask = counts >= max_
    if np.any(mask):
        prior[np.ix_(mask, target_tokens)] = -np.inf

    return prior




def _generate() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    target_add = np.array([ADD_IDX], dtype=np.int32)



    actions_below = np.array(
        [[ADD_IDX, U_IDX, ADD_IDX, U_IDX, ADD_IDX, U_IDX, U_IDX]],
        dtype=np.int32,
    )
    arrays["below_max_actions"] = actions_below
    arrays["below_max_targets"] = target_add
    arrays["below_max_max"] = np.int32(5)
    arrays["below_max_expected"] = _ref_repeat(actions_below, target_add, 5, L)



    actions_at = np.array(
        [[ADD_IDX, ADD_IDX, ADD_IDX, ADD_IDX, ADD_IDX, U_IDX, U_IDX]],
        dtype=np.int32,
    )
    arrays["at_max_actions"] = actions_at
    arrays["at_max_targets"] = target_add
    arrays["at_max_max"] = np.int32(5)
    arrays["at_max_expected"] = _ref_repeat(actions_at, target_add, 5, L)



    actions_above = np.array(
        [[ADD_IDX, ADD_IDX, ADD_IDX, ADD_IDX, ADD_IDX, ADD_IDX, ADD_IDX]],
        dtype=np.int32,
    )
    arrays["above_max_actions"] = actions_above
    arrays["above_max_targets"] = target_add
    arrays["above_max_max"] = np.int32(5)
    arrays["above_max_expected"] = _ref_repeat(actions_above, target_add, 5, L)



    actions_mixed = np.array(
        [
            [ADD_IDX, U_IDX, ADD_IDX, U_IDX, U_IDX, U_IDX, U_IDX],
            [ADD_IDX, ADD_IDX, ADD_IDX, ADD_IDX, ADD_IDX, U_IDX, U_IDX],
        ],
        dtype=np.int32,
    )
    arrays["mixed_batch_actions"] = actions_mixed
    arrays["mixed_batch_targets"] = target_add
    arrays["mixed_batch_max"] = np.int32(5)
    arrays["mixed_batch_expected"] = _ref_repeat(
        actions_mixed, target_add, 5, L
    )




    actions_padded = np.array(
        [[ADD_IDX, ADD_IDX, ADD_IDX, EMPTY_ACTION, EMPTY_ACTION,
          EMPTY_ACTION, EMPTY_ACTION]],
        dtype=np.int32,
    )
    arrays["padding_safe_actions"] = actions_padded
    arrays["padding_safe_targets"] = target_add
    arrays["padding_safe_max"] = np.int32(5)
    arrays["padding_safe_expected"] = _ref_repeat(
        actions_padded, target_add, 5, L
    )



    target_add_mul = np.array([ADD_IDX, MUL_IDX], dtype=np.int32)
    actions_multi = np.array(
        [[ADD_IDX, MUL_IDX, ADD_IDX, MUL_IDX, ADD_IDX, U_IDX, U_IDX]],
        dtype=np.int32,
    )
    arrays["multi_target_actions"] = actions_multi
    arrays["multi_target_targets"] = target_add_mul
    arrays["multi_target_max"] = np.int32(5)
    arrays["multi_target_expected"] = _ref_repeat(
        actions_multi, target_add_mul, 5, L
    )




    actions_any = np.array(
        [[U_IDX, SIN_IDX, U_IDX]],
        dtype=np.int32,
    )
    arrays["max_zero_actions"] = actions_any
    arrays["max_zero_targets"] = target_add
    arrays["max_zero_max"] = np.int32(0)
    arrays["max_zero_expected"] = _ref_repeat(
        actions_any, target_add, 0, L
    )

    return arrays


def main() -> None:
    fixtures_dir = Path(__file__).parent.parent
    arrays = _generate()
    out_path = fixtures_dir / "prior_repeat.npz"
    np.savez(out_path, **arrays)
    print(f"Saved {len(arrays)} arrays to {out_path}")


    data = dict(np.load(out_path, allow_pickle=True))


    assert np.all(data["below_max_expected"] == 0.0), "below_max should be zeros"


    assert data["at_max_expected"][0, ADD_IDX] == -np.inf, "at_max: add should be -inf"
    assert data["at_max_expected"][0, U_IDX] == 0.0, "at_max: u should be 0"


    assert data["above_max_expected"][0, ADD_IDX] == -np.inf, (
        "above_max: add should be -inf"
    )


    assert np.all(data["mixed_batch_expected"][0] == 0.0), "mixed row 0 should be zeros"
    assert data["mixed_batch_expected"][1, ADD_IDX] == -np.inf, (
        "mixed row 1 add should be -inf"
    )


    assert np.all(data["padding_safe_expected"] == 0.0), (
        "padding should not inflate count"
    )


    assert data["multi_target_expected"][0, ADD_IDX] == -np.inf
    assert data["multi_target_expected"][0, MUL_IDX] == -np.inf
    assert data["multi_target_expected"][0, U_IDX] == 0.0


    assert data["max_zero_expected"][0, ADD_IDX] == -np.inf, (
        "max_zero: add should be forbidden"
    )
    assert data["max_zero_expected"][0, U_IDX] == 0.0

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
