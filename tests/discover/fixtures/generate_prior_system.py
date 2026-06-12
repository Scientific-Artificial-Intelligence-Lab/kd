from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


NAMES = [
    "x", "t", "u",
    "add", "mul",
    "sin", "cos", "diff_x", "diff_t", "n2", "neg",
    "div",
]
ARITIES = np.array([0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32)
L = len(NAMES)

TERMINAL = np.where(ARITIES == 0)[0].astype(np.int32)
UNARY = np.where(ARITIES == 1)[0].astype(np.int32)
BINARY = np.where(ARITIES == 2)[0].astype(np.int32)
NONTERMINAL = np.where(ARITIES > 0)[0].astype(np.int32)
DIFF = np.array([NAMES.index("diff_x"), NAMES.index("diff_t")], dtype=np.int32)
STATE = np.array([NAMES.index("u")], dtype=np.int32)


PA = np.full(L, -1, dtype=np.int32)
_c = 0
for _i in range(L):
    if ARITIES[_i] > 0:
        PA[_i] = _c
        _c += 1

N_NT = int((ARITIES > 0).sum())
EMPTY_PARENT = N_NT
EMPTY_ACTION = L
EMPTY_SIBLING = L

MIN_LEN = 3
MAX_LEN = 10



SOFT_LOC = 6.0
SOFT_SCALE = 2.0


SOFT_EARLY_CUTOFF = 3
SOFT_EARLY_SCALE = 10.0
_ADD_NAMES: frozenset[str] = frozenset({"add", "add_t", "sub", "sub_t"})
NONADD = np.array(
    [i for i, name in enumerate(NAMES) if name not in _ADD_NAMES],
    dtype=np.int32,
)




def _ref_length(step_idx: int, dangling: np.ndarray) -> np.ndarray:
    B = dangling.shape[0]
    prior = np.zeros((B, L), dtype=np.float32)

    if step_idx == 0:

        prior[:, TERMINAL] = -np.inf
        return prior

    i = step_idx - 1


    if MAX_LEN is not None and (i + 2) >= MAX_LEN // 2:
        remaining = MAX_LEN - (i + 1)
        for t in BINARY:
            prior[dangling >= remaining - 1, t] = -np.inf
        for t in UNARY:
            prior[dangling == remaining, t] = -np.inf


    if MIN_LEN is not None and (i + 2) < MIN_LEN:
        for t in TERMINAL:
            prior[dangling == 1, t] = -np.inf

    return prior


def _ref_diff_child(parent: np.ndarray) -> np.ndarray:
    B = parent.shape[0]
    prior = np.zeros((B, L), dtype=np.float32)

    adj_diff = PA[DIFF]
    mask = np.isin(parent.astype(np.int32), adj_diff)

    allowed = set(STATE.tolist() + DIFF.tolist())
    targets = np.array([i for i in range(L) if i not in allowed], dtype=np.int32)
    for t in targets:
        prior[mask, t] = -np.inf

    return prior


def _ref_soft_length(step_idx: int, batch_size: int) -> np.ndarray:
    prior = np.zeros((batch_size, L), dtype=np.float32)
    if step_idx == 0:

        return prior
    t = float(step_idx)
    if t < SOFT_LOC:
        if t < SOFT_EARLY_CUTOFF:
            early_penalty = -((t - SOFT_EARLY_CUTOFF) ** 2) / SOFT_EARLY_SCALE
            prior[:, NONADD] += np.float32(early_penalty)

    elif t > SOFT_LOC:
        late_penalty = -((t - SOFT_LOC) ** 2) / (2.0 * SOFT_SCALE)
        prior[:, NONTERMINAL] += np.float32(late_penalty)

    return prior


def _combine_hard(*priors: np.ndarray) -> np.ndarray:


    combined = sum(priors)
    return combined.astype(np.float32)


def _combine_all(*priors: np.ndarray) -> np.ndarray:

    return sum(priors).astype(np.float32)




def _scan_ps(prefix: np.ndarray) -> tuple[int, int]:
    if len(prefix) == 0:
        return EMPTY_PARENT, EMPTY_SIBLING
    last = int(prefix[-1])
    if ARITIES[last] > 0:
        return int(PA[last]), EMPTY_SIBLING
    d = 0
    n = len(prefix)
    for off in range(n):
        idx = n - off - 1
        d += int(ARITIES[int(prefix[idx])]) - 1
        if d == 0:
            return int(PA[int(prefix[idx])]), int(prefix[idx + 1])
    return EMPTY_PARENT, EMPTY_SIBLING




def _step_cases() -> dict[str, dict[str, Any]]:
    cases = {}

    def _case(name: str, step_idx: int, obs_row: list[float]) -> None:
        obs = np.array([obs_row], dtype=np.float32)
        adjustment = _combine_hard(
            _ref_length(step_idx, obs[:, 3]),
            _ref_diff_child(obs[:, 1]),
        )
        cases[name] = {"step_idx": step_idx, "obs": obs, "mask": adjustment}


    _case("initial", 0, [EMPTY_ACTION, EMPTY_PARENT, EMPTY_SIBLING, 1])


    _case("unconstrained", 5, [3, PA[3], EMPTY_SIBLING, 2])


    _case("near_max_d2", 8, [2, PA[4], 2, 2])


    _case("near_max_d1", 8, [2, PA[4], 2, 1])


    _case("diff_child", 5, [7, PA[7], EMPTY_SIBLING, 1])


    _case("below_min", 1, [5, PA[5], EMPTY_SIBLING, 1])


    _case("diff_below_min", 1, [7, PA[7], EMPTY_SIBLING, 1])

    return cases


def _soft_length_cases() -> dict[str, dict[str, Any]]:
    cases = {}




    obs_initial = np.array(
        [[EMPTY_ACTION, EMPTY_PARENT, EMPTY_SIBLING, 1]], dtype=np.float32,
    )
    adjustment_initial = _ref_soft_length(0, batch_size=1)
    cases["soft_initial"] = {
        "step_idx": 0,
        "obs": obs_initial,
        "mask": adjustment_initial,
    }


    obs_early = np.array(
        [[3, PA[3], EMPTY_SIBLING, 2]], dtype=np.float32,
    )
    adjustment_early = _ref_soft_length(2, batch_size=1)
    cases["soft_early"] = {
        "step_idx": 2,
        "obs": obs_early,
        "mask": adjustment_early,
    }


    obs_loc = np.array(
        [[3, PA[3], EMPTY_SIBLING, 2]], dtype=np.float32,
    )
    adjustment_loc = _ref_soft_length(6, batch_size=1)
    cases["soft_at_loc"] = {
        "step_idx": 6,
        "obs": obs_loc,
        "mask": adjustment_loc,
    }


    obs_late = np.array(
        [[3, PA[3], EMPTY_SIBLING, 2]], dtype=np.float32,
    )
    adjustment_late = _ref_soft_length(9, batch_size=1)
    cases["soft_late"] = {
        "step_idx": 9,
        "obs": obs_late,
        "mask": adjustment_late,
    }

    return cases


def _batch_case() -> dict[str, np.ndarray]:
    actions = np.array([[3, 4, 2, 7, 2, 8, 2]], dtype=np.int32)
    B, T = actions.shape


    prev_action = np.full((B, T), EMPTY_ACTION, dtype=np.int32)
    prev_action[:, 1:] = actions[:, :-1]

    dangling = np.ones((B, T), dtype=np.int32)
    for t in range(1, T):
        dangling[:, t] = dangling[:, t - 1] + ARITIES[actions[:, t - 1]] - 1

    parent = np.full((B, T), EMPTY_PARENT, dtype=np.int32)
    sibling = np.full((B, T), EMPTY_SIBLING, dtype=np.int32)
    for t in range(1, T):
        for b in range(B):
            parent[b, t], sibling[b, t] = _scan_ps(actions[b, :t])

    obs = np.stack([prev_action, parent, sibling, dangling], axis=1).astype(
        np.float32
    )

    adjustments = np.zeros((B, T, L), dtype=np.float32)
    for t in range(T):
        adjustments[:, t,:] = _combine_hard(
            _ref_length(t, dangling[:, t].astype(np.float32)),
            _ref_diff_child(parent[:, t].astype(np.float32)),
        )

    return {"actions": actions, "obs": obs, "masks": adjustments}


def main() -> None:
    step = _step_cases()
    soft = _soft_length_cases()
    batch = _batch_case()



    data: dict[str, np.ndarray | np.generic] = {
        "names": np.array(NAMES),
        "arities": ARITIES,
        "min_len": np.int32(MIN_LEN),
        "max_len": np.int32(MAX_LEN),
        "soft_loc": np.float32(SOFT_LOC),
        "soft_scale": np.float32(SOFT_SCALE),
    }


    for name, case in step.items():
        data[f"step_{name}_obs"] = case["obs"]
        data[f"step_{name}_mask"] = case["mask"]
        data[f"step_{name}_step_idx"] = np.int32(case["step_idx"])


    for name, case in soft.items():
        data[f"step_{name}_obs"] = case["obs"]
        data[f"step_{name}_mask"] = case["mask"]
        data[f"step_{name}_step_idx"] = np.int32(case["step_idx"])


    data["batch_actions"] = batch["actions"]
    data["batch_obs"] = batch["obs"]
    data["batch_masks"] = batch["masks"]

    out = Path(__file__).parent / "prior_system.npz"
    np.savez(out, **data)
    print(f"Saved {len(data)} arrays to {out}")


    print("\n── Hard prior step cases (0.0 = allow, -inf = forbid) ──")
    for name, case in step.items():
        mask = case["mask"]
        allowed = int(np.isfinite(mask).sum())
        print(f" step/{name}: idx={case['step_idx']}, {allowed}/{L} allowed")

    print("\n── Soft prior step cases (continuous logit adjustments) ──")
    for name, case in soft.items():
        mask = case["mask"]
        min_v = float(mask.min())
        max_v = float(mask.max())
        print(
            f" step/{name}: idx={case['step_idx']}, "
            f"range=[{min_v:.4f}, {max_v:.4f}]"
        )

    print("\n── Batch case ──")
    for t in range(batch["masks"].shape[1]):
        mask = batch["masks"][0, t]
        allowed = int(np.isfinite(mask).sum())
        tok = NAMES[batch["actions"][0, t]]
        print(f" batch t={t} ({tok}): {allowed}/{L} allowed")


if __name__ == "__main__":
    main()
