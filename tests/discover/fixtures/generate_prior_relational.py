from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


NAMES = [
    "x", "t", "u",
    "add", "mul",
    "sin", "cos", "diff_x", "diff_t", "n2", "neg",
    "div",
]
ARITIES = np.array([0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32)
L = len(NAMES)
EMPTY_ACTION = L


EXTENDED_ARITIES = np.append(ARITIES, np.int32(0))


X, T, U = 0, 1, 2
ADD, MUL = 3, 4
SIN, COS, DIFF_X, DIFF_T, N2, NEG = 5, 6, 7, 8, 9, 10
DIV = 11


TERMINAL = np.where(ARITIES == 0)[0].astype(np.int32)
DIFF = np.array([DIFF_X, DIFF_T], dtype=np.int32)
TRIG = np.array([SIN, COS], dtype=np.int32)

TRIG_DIFF = np.concatenate([TRIG, DIFF]).astype(np.int32)


PA = np.full(L, -1, dtype=np.int32)
_c = 0
for _i in range(L):
    if ARITIES[_i] > 0:
        PA[_i] = _c
        _c += 1




def _ancestors(
    actions: np.ndarray,
    arities: np.ndarray,
    ancestor_tokens: np.ndarray,
) -> np.ndarray:
    batch_size, seq_len = actions.shape
    ancestor_set = set(int(t) for t in ancestor_tokens)
    mask = np.zeros(batch_size, dtype=np.bool_)
    for r in range(batch_size):
        dangling = 0
        threshold = None
        for c in range(seq_len):
            arity = int(arities[actions[r, c]])
            dangling += arity - 1
            if threshold is None:
                if int(actions[r, c]) in ancestor_set:
                    threshold = dangling - arity
            else:
                if dangling == threshold:
                    threshold = None
        if threshold is not None:
            mask[r] = True
    return mask


def _ref_relational_child(
    parent: np.ndarray,
    effectors: np.ndarray,
    targets: np.ndarray,
    n_choices: int,
) -> np.ndarray:
    batch_size = parent.shape[0]
    prior = np.zeros((batch_size, n_choices), dtype=np.float32)
    adj_parents = PA[effectors]
    mask = np.isin(parent, adj_parents)
    if np.any(mask):
        prior[np.ix_(mask, targets)] = -np.inf
    return prior


def _ref_relational_descendant(
    actions: np.ndarray,
    effectors: np.ndarray,
    targets: np.ndarray,
    n_choices: int,
) -> np.ndarray:
    batch_size = actions.shape[0]
    prior = np.zeros((batch_size, n_choices), dtype=np.float32)
    mask = _ancestors(actions, EXTENDED_ARITIES, effectors)
    if np.any(mask):
        prior[np.ix_(mask, targets)] = -np.inf
    return prior




def _generate() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}








    actions_1 = np.array([[SIN]], dtype=np.int32)
    arrays["anc_1_actions"] = actions_1
    arrays["anc_1_ancestor_tokens"] = TRIG
    arrays["anc_1_expected"] = _ancestors(actions_1, EXTENDED_ARITIES, TRIG)




    actions_2 = np.array([[SIN, U]], dtype=np.int32)
    arrays["anc_2_actions"] = actions_2
    arrays["anc_2_ancestor_tokens"] = TRIG
    arrays["anc_2_expected"] = _ancestors(actions_2, EXTENDED_ARITIES, TRIG)




    actions_3 = np.array([[ADD, SIN]], dtype=np.int32)
    arrays["anc_3_actions"] = actions_3
    arrays["anc_3_ancestor_tokens"] = TRIG
    arrays["anc_3_expected"] = _ancestors(actions_3, EXTENDED_ARITIES, TRIG)



    actions_4 = np.array([[ADD, SIN, U]], dtype=np.int32)
    arrays["anc_4_actions"] = actions_4
    arrays["anc_4_ancestor_tokens"] = TRIG
    arrays["anc_4_expected"] = _ancestors(actions_4, EXTENDED_ARITIES, TRIG)


    actions_5 = np.array([[DIFF_X]], dtype=np.int32)
    arrays["anc_5_actions"] = actions_5
    arrays["anc_5_ancestor_tokens"] = DIFF
    arrays["anc_5_expected"] = _ancestors(actions_5, EXTENDED_ARITIES, DIFF)





    actions_6 = np.array([[DIFF_X, DIFF_X]], dtype=np.int32)
    arrays["anc_6_actions"] = actions_6
    arrays["anc_6_ancestor_tokens"] = DIFF
    arrays["anc_6_expected"] = _ancestors(actions_6, EXTENDED_ARITIES, DIFF)


    actions_7 = np.array([[]], dtype=np.int32).reshape(1, 0)
    arrays["anc_7_actions"] = actions_7
    arrays["anc_7_ancestor_tokens"] = TRIG
    arrays["anc_7_expected"] = _ancestors(actions_7, EXTENDED_ARITIES, TRIG)





    actions_8 = np.array([
        [ADD, SIN, U],
        [ADD, U, SIN],
        [ADD, U, U],
    ], dtype=np.int32)
    arrays["anc_8_actions"] = actions_8
    arrays["anc_8_ancestor_tokens"] = TRIG
    arrays["anc_8_expected"] = _ancestors(actions_8, EXTENDED_ARITIES, TRIG)







    actions_9 = np.array([[ADD, U]], dtype=np.int32)
    ancestor_add = np.array([ADD], dtype=np.int32)
    arrays["anc_9_actions"] = actions_9
    arrays["anc_9_ancestor_tokens"] = ancestor_add
    arrays["anc_9_expected"] = _ancestors(
        actions_9, EXTENDED_ARITIES, ancestor_add
    )



    actions_10 = np.array([[MUL, U, SIN]], dtype=np.int32)
    arrays["anc_10_actions"] = actions_10
    arrays["anc_10_ancestor_tokens"] = TRIG
    arrays["anc_10_expected"] = _ancestors(actions_10, EXTENDED_ARITIES, TRIG)






    parent_1 = np.array([PA[SIN]], dtype=np.int32)
    arrays["child_1_parent"] = parent_1
    arrays["child_1_effectors"] = TRIG
    arrays["child_1_targets"] = TRIG
    arrays["child_1_expected"] = _ref_relational_child(
        parent_1, TRIG, TRIG, L
    )


    parent_2 = np.array([PA[ADD]], dtype=np.int32)
    arrays["child_2_parent"] = parent_2
    arrays["child_2_effectors"] = TRIG
    arrays["child_2_targets"] = TRIG
    arrays["child_2_expected"] = _ref_relational_child(
        parent_2, TRIG, TRIG, L
    )


    parent_3 = np.array([PA[DIFF_X], PA[ADD]], dtype=np.int32)
    arrays["child_3_parent"] = parent_3
    arrays["child_3_effectors"] = DIFF
    arrays["child_3_targets"] = np.array([ADD], dtype=np.int32)
    arrays["child_3_expected"] = _ref_relational_child(
        parent_3, DIFF, np.array([ADD], dtype=np.int32), L
    )






    actions_trig1 = np.array([[SIN, COS]], dtype=np.int32)
    arrays["trig_1_actions"] = actions_trig1
    arrays["trig_1_expected"] = _ref_relational_descendant(
        actions_trig1, TRIG_DIFF, TRIG_DIFF, L
    )


    actions_trig2 = np.array([[DIFF_X, DIFF_X]], dtype=np.int32)
    arrays["trig_2_actions"] = actions_trig2
    arrays["trig_2_expected"] = _ref_relational_descendant(
        actions_trig2, TRIG_DIFF, TRIG_DIFF, L
    )


    actions_trig3 = np.array([[SIN, U]], dtype=np.int32)
    arrays["trig_3_actions"] = actions_trig3
    arrays["trig_3_expected"] = _ref_relational_descendant(
        actions_trig3, TRIG_DIFF, TRIG_DIFF, L
    )


    actions_trig4 = np.array([[ADD, U]], dtype=np.int32)
    arrays["trig_4_actions"] = actions_trig4
    arrays["trig_4_expected"] = _ref_relational_descendant(
        actions_trig4, TRIG_DIFF, TRIG_DIFF, L
    )



    actions_trig5 = np.array([[SIN]], dtype=np.int32)
    arrays["trig_5_actions"] = actions_trig5
    arrays["trig_5_expected"] = _ref_relational_descendant(
        actions_trig5, TRIG, TRIG, L
    )







    DIFF_DESC_TARGETS = np.array([ADD], dtype=np.int32)


    actions_dd1 = np.array([[DIFF_X]], dtype=np.int32)
    arrays["diffdes_1_actions"] = actions_dd1
    arrays["diffdes_1_expected"] = _ref_relational_descendant(
        actions_dd1, DIFF, DIFF_DESC_TARGETS, L
    )


    actions_dd2 = np.array([[ADD, U]], dtype=np.int32)
    arrays["diffdes_2_actions"] = actions_dd2
    arrays["diffdes_2_expected"] = _ref_relational_descendant(
        actions_dd2, DIFF, DIFF_DESC_TARGETS, L
    )



    actions_dd3 = np.array([[DIFF_X, MUL]], dtype=np.int32)
    arrays["diffdes_3_actions"] = actions_dd3
    arrays["diffdes_3_expected"] = _ref_relational_descendant(
        actions_dd3, DIFF, DIFF_DESC_TARGETS, L
    )

    return arrays


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    fixtures_dir = Path(__file__).parent.parent
    arrays = _generate()
    out_path = fixtures_dir / "prior_relational.npz"
    np.savez(out_path, **arrays)
    logger.info("Saved %d arrays to %s", len(arrays), out_path)


    data = dict(np.load(out_path, allow_pickle=True))


    assert data["anc_1_expected"][0], "sin(?) → inside"
    assert not data["anc_2_expected"][0], "sin(u) → outside"
    assert data["anc_3_expected"][0], "add(sin(?)) → inside"
    assert not data["anc_4_expected"][0], "add(sin(u),?) → outside"
    assert data["anc_5_expected"][0], "diff_x(?) → inside"
    assert data["anc_6_expected"][0], "diff_x(diff_x(?)) → inside"
    assert not data["anc_7_expected"][0], "empty → outside"
    assert data["anc_9_expected"][0], (
        "binary ancestor arity-fix: add(u,?) → inside"
    )
    assert data["anc_10_expected"][0], "mul(u, sin(?)) → inside"


    np.testing.assert_array_equal(
        data["anc_8_expected"], [False, True, False]
    )


    assert data["trig_1_expected"][0, SIN] == -np.inf, (
        "sin forbidden inside cos subtree"
    )
    assert data["trig_2_expected"][0, DIFF_X] == -np.inf, (
        "diff_x forbidden inside diff_x subtree"
    )
    assert np.all(data["trig_3_expected"] == 0.0), "sin(u) → zeros"
    assert np.all(data["trig_4_expected"] == 0.0), "add(u,?) → zeros"


    assert data["diffdes_1_expected"][0, ADD] == -np.inf, (
        "add forbidden inside diff subtree"
    )
    assert data["diffdes_1_expected"][0, MUL] == 0.0, (
        "mul allowed inside diff subtree"
    )
    assert np.all(data["diffdes_2_expected"] == 0.0), (
        "not inside diff → zeros"
    )

    logger.info("All sanity checks passed.")


if __name__ == "__main__":
    main()
