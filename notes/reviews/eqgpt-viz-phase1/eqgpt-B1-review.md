# Review: eqgpt-B1 KD_EqGPT wrapper + App

## Reviewers
- code-reviewer: 2 High, 4 Medium, 3 Low
- logic-reviewer: 2 High, 3 Medium, 3 Low
- adversarial-reviewer: 3 High, 5 Medium, 3 Low

## CONFIRMED (fixed)
| Issue | Sources | Fix |
|-------|---------|-----|
| _REF_LIB_DIR off by one parent | all 3 | .parent.parent.parent |
| find_min_no_repeat crash with small samples | logic-H2, adv-H1 | pass samples_k parameter |
| Magic numbers in _format_results | code-H2, logic-M3, adv-M7 | use word2id dict |
| pickle file handle leak | code-M1, logic-L1, adv-L8 | with statement |
| Unused imports | code-M3 | removed NN, calculate_terms |
| torch.tensor copy warning | adv-M5 | clone().detach().requires_grad_() |
| Missing surrogate model crash | logic-M2 | directory existence check |

## NOTED (documented, not blocking)
| Issue | Source | Note |
|-------|--------|------|
| Fine-tuning divergence from ref_lib | logic-M1 | Added comment explaining deliberate fix |

## DISMISSED
| Issue | Source | Reason |
|-------|--------|--------|
| Function > 50 lines | code-M2 | fit_pretrained is main flow, splitting = over-engineering |
| Physics magic numbers | code-M4 | From ref_lib, not rewriting |
| 12 models in memory | adv-M6 | Ref_lib design, MPS verified |
| MPS masked_fill_ | adv-M4 | Already working on MPS |
| KD_EqGPT always in dropdown | adv-L10 | By design per task spec |
