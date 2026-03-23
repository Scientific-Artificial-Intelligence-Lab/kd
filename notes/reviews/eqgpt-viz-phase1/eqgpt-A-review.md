# Review: eqgpt-A migration

## Reviewers
- code-reviewer: 3 High, 2 Medium, 2 Low
- logic-reviewer: 5 High, 4 Medium, 2 Low
- adversarial-reviewer: 4 High, 4 Medium, 3 Low

## Rebuttal Results

### CONFIRMED (fixed)
| Issue | Source | Fix |
|-------|--------|-----|
| posterial_solution_PINN.py indentation breakage | code-H1 | +4 spaces lines 97-242 |
| posterial_solution_FD.py indentation breakage | code-H2 | +4 spaces lines 52-351 |
| process_dataset_wave_breaking.py indentation breakage | code-H3 | +8 spaces lines 30-44 |

### DISMISSED (out of Task A scope)
| Cluster | Issues | Reason |
|---------|--------|--------|
| Global variable deps in functions | logic-H1-H5, adv-H2 | Task A non-goal: "不重写算法逻辑". Step 5 only targets calculate_reward + get_meta in continue_train_GPT_all.py. Track for Task B. |
| Pre-existing ref_lib bugs | adv-H1, adv-H3, adv-L11 | epochs undefined, get_concise_form signature, uyyt index — all from ref_lib, not introduced by migration |
| 145MB .pt in git | adv-M6 | User explicitly accepted: "作为临时 demo 完全足够" |
| Star import pollution | adv-L9, code-L2 | Known risk (Task A Risk #5), Task B uses explicit imports |
| Module-level json.load | adv-H4 | Task explicitly says "轻量, 保留" |

### DOWNGRADED (Medium → Low, tracked)
| Issue | Source | Note |
|-------|--------|------|
| plot_save/ path not _DIR based | logic-M7, adv-M7 | Only in functions, not import-time |
| read_dataset.py write path | logic-M8 | Only in calculate_words(), not import-time |
| set_random_seeds() at import | logic-M6 | Minor: changes random state on import of continue_train_GPT.py |
| _device.py docstring outdated | code-M1 | Cosmetic |
