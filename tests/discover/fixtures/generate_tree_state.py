
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent.parent





def parents_siblings_at_once(
    tokens: np.ndarray,
    arities: np.ndarray,
    parent_adjust: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    N, L = tokens.shape
    empty_parent = int(np.max(parent_adjust) + 1)
    empty_sibling = len(arities)

    adj_parents = np.full((N, L), empty_parent, dtype=np.int32)
    siblings = np.full((N, L), empty_sibling, dtype=np.int32)

    for b in range(1, L):
        for r in range(N):
            arity = arities[tokens[r, b - 1]]
            if arity > 0:
                adj_parents[r, b] = parent_adjust[tokens[r, b - 1]]
                continue


            dangling = 0
            for c in range(b):
                arity = arities[tokens[r, b - c - 1]]
                dangling += arity - 1
                if dangling == 0:
                    adj_parents[r, b] = parent_adjust[tokens[r, b - c - 1]]
                    siblings[r, b] = tokens[r, b - c]
                    break

    return adj_parents, siblings


def parents_siblings_incremental(
    tokens: np.ndarray,
    arities: np.ndarray,
    parent_adjust: np.ndarray,
    empty_parent: int,
    empty_sibling: int,
) -> tuple[np.ndarray, np.ndarray]:
    N, L = tokens.shape
    adj_parents = np.full(N, empty_parent, dtype=np.int32)
    siblings_out = np.full(N, empty_sibling, dtype=np.int32)

    for r in range(N):
        arity = arities[tokens[r, -1]]
        if arity > 0:
            adj_parents[r] = parent_adjust[tokens[r, -1]]
            continue
        dangling = 0
        for c in range(L):
            arity = arities[tokens[r, L - c - 1]]
            dangling += arity - 1
            if dangling == 0:
                adj_parents[r] = parent_adjust[tokens[r, L - c - 1]]
                siblings_out[r] = tokens[r, L - c]
                break

    return adj_parents, siblings_out


def compute_dangling(
    tokens: np.ndarray,
    arities: np.ndarray,
) -> np.ndarray:
    N, L = tokens.shape
    dangling = np.ones((N, L), dtype=np.int32)
    for t in range(1, L):
        dangling[:, t] = dangling[:, t - 1] + arities[tokens[:, t - 1]] - 1
    return dangling


def compute_batch_obs(
    tokens: np.ndarray,
    arities: np.ndarray,
    parent_adjust: np.ndarray,
) -> np.ndarray:
    N, L = tokens.shape
    empty_action = len(arities)

    adj_parents, siblings = parents_siblings_at_once(tokens, arities, parent_adjust)
    dangling = compute_dangling(tokens, arities)

    prev_action = np.full((N, L), empty_action, dtype=np.int32)
    if L > 1:
        prev_action[:, 1:] = tokens[:, :-1]

    obs = np.stack([prev_action, adj_parents, siblings, dangling], axis=1)
    return obs.astype(np.float32)




NAMES = ["x1", "u1", "add", "mul", "n2", "n3", "sub"]
ARITIES = np.array([0, 0, 2, 2, 1, 1, 2], dtype=np.int32)
PARENT_ADJUST = np.array([-1, -1, 0, 1, 2, 3, 4], dtype=np.int32)

X1, U1, ADD, MUL, N2, N3, SUB = range(7)

N_TOKENS = len(ARITIES)
EMPTY_ACTION = N_TOKENS
EMPTY_PARENT = int(np.max(PARENT_ADJUST) + 1)
EMPTY_SIBLING = N_TOKENS




NAMES_DIFF = ["x1", "u1", "add", "mul", "diff", "diff2"]
ARITIES_DIFF = np.array([0, 0, 2, 2, 1, 1], dtype=np.int32)
PARENT_ADJUST_DIFF = np.array([-1, -1, 0, 1, 2, 3], dtype=np.int32)




CASES = {
    "single_terminal": np.array([[X1]], dtype=np.int32),
    "binary_add": np.array([[ADD, X1, U1]], dtype=np.int32),
    "unary_n2": np.array([[N2, X1]], dtype=np.int32),
    "nested": np.array([[ADD, MUL, U1, N2, U1, N2, U1]], dtype=np.int32),
    "deep_unary": np.array([[N2, N3, N2, X1]], dtype=np.int32),
    "with_sub": np.array([[SUB, ADD, X1, U1, MUL, U1, X1]], dtype=np.int32),
}


BATCH_TOKENS = np.array([
    [ADD, X1, U1],
    [N2, N2, X1],
    [MUL, X1, U1],
], dtype=np.int32)


DIFF_TOKENS = np.array([[2, 3, 1, 4, 1, 5, 1]], dtype=np.int32)


def main() -> None:
    arrays: dict[str, np.ndarray] = {}


    arrays["arities"] = ARITIES
    arrays["parent_adjust"] = PARENT_ADJUST
    arrays["names"] = np.array(NAMES)
    arrays["empty_action"] = np.array(EMPTY_ACTION)
    arrays["empty_parent"] = np.array(EMPTY_PARENT)
    arrays["empty_sibling"] = np.array(EMPTY_SIBLING)


    for name, tokens in CASES.items():
        obs = compute_batch_obs(tokens, ARITIES, PARENT_ADJUST)
        arrays[f"{name}_tokens"] = tokens
        arrays[f"{name}_obs"] = obs


    batch_obs = compute_batch_obs(BATCH_TOKENS, ARITIES, PARENT_ADJUST)
    arrays["batch_tokens"] = BATCH_TOKENS
    arrays["batch_obs"] = batch_obs


    diff_obs = compute_batch_obs(DIFF_TOKENS, ARITIES_DIFF, PARENT_ADJUST_DIFF)
    arrays["diff_tokens"] = DIFF_TOKENS
    arrays["diff_obs"] = diff_obs
    arrays["arities_diff"] = ARITIES_DIFF
    arrays["parent_adjust_diff"] = PARENT_ADJUST_DIFF
    arrays["names_diff"] = np.array(NAMES_DIFF)


    for name, tokens in CASES.items():
        obs = arrays[f"{name}_obs"]
        _N, L = tokens.shape

        for t in range(1, L):
            inc_p, inc_s = parents_siblings_incremental(
                tokens[:, :t], ARITIES, PARENT_ADJUST,
                EMPTY_PARENT, EMPTY_SIBLING,
            )
            batch_p = int(obs[0, 1, t])
            batch_s = int(obs[0, 2, t])
            assert inc_p[0] == batch_p, (
                f"{name} t={t}: inc parent {inc_p[0]} != batch {batch_p}"
            )
            assert inc_s[0] == batch_s, (
                f"{name} t={t}: inc sibling {inc_s[0]} != batch {batch_s}"
            )


    output_path = FIXTURES_DIR / "tree_state.npz"
    np.savez(output_path, **arrays)


    data = np.load(output_path, allow_pickle=True)
    print(f"Generated {len(data.files)} arrays:")
    for key in sorted(data.files):
        arr = data[key]
        print(f" {key}: shape={arr.shape}, dtype={arr.dtype}")

    print("\nIncremental == batch sanity checks passed!")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
