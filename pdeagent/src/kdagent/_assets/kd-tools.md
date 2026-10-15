---
name: kd-tools
description: Data tools, search records, continuation, sketches and surrogate reuse.
---

## Tools

`list_datasets` lists a file input as id, task kind, target, data report and
array inventory, and a catalog dataset as id, axes, lhs, fmt, tier and tags
with no report.
`run_discovery` requires `dataset_id` and `algorithm`; JSON parameters go in
`params`. `train_surrogate` trains and registers a reusable derivative model.
`prune_node` abandons a search branch; `select_node` chooses an existing search
for publication. `submit_answer` publishes a structured answer with support
and aligned coefficients. Use the pinned input's id for every data tool.

When omitted, `generations` is {DEFAULT_GENERATIONS} where consumed and a
fresh run uses seed {DEFAULT_SEED}. Supply `seed` separately from `params`.
A resumed run inherits its seed; `reseed` derives a new random stream from
restored search state. A new segment's generation budget is additional work.

## Results and records

Read `failures` and status first; a `partial` result is not a successful
search. `primary.law` contains support, aligned coefficients and lhs;
`primary.equation` is the equation display. Read structured values instead of
extracting coefficients from display text. A null law is not submittable.
`diagnostics.score_kind`, `score_direction`, `nmse`, `mse` and
`segment_report` describe fit and segment facts. Native scores and derivative
domains do not establish comparisons across algorithms.
`platform_nmse` and `platform_coefficients` are separate neutral refit
measurements, with `unavailable_reason` when that evaluation is unsupported.
`provenance.headline_coefficient_source` says whether coefficients are native
or refitted; `provenance.run_id` names the search for selection and resume.
`provenance.run_dir` points to `record.json` (sealed evidence), `recorder.json`
(iteration series) and `phases.jsonl` (phase boundaries). The fitted target is
in `evidence.catalog_fit.lhs_spec`.

## Continuation

Use `resume_from_run_id` for the latest checkpoint; add `resume_iteration`
to roll back and fork. `resume_safe` parameters can change on resume,
`init_only` parameters cannot affect restored state, and `identity_breaking`
parameters require a new lineage. An archive classified `conclusion` cannot
resume; rerun it with other parameters. `decision_parent` records the parent
of a fresh switch or narrowing decision. A segment report contains
observations, not a decision about which branch deserves more budget.
A supplied catalog prior is an annotated historical row per instrument;
it is not a judgment of the current input or result.

## Sketches

Pass a `kd-sketch-v1` JSON payload as `sketch`. Its `lhs_spec` specifies
`field`, `axis`, `order`; `vocabulary` names fields and coordinates;
`pinned` fixes `term_ir` and `value`; `anchored` fits mandatory `term_ir` entries.
Each hole names an `id`, `min_count`, `max_count` and a `constraint` over
`max_deriv_order`, `operators`, `fields` and `axes`. `match_policy` specifies
coefficient tolerances, support threshold and identity/assignment schemes.
An example payload is
```json
{{
  "schema": "kd-sketch-v1",
  "lhs_spec": {{
    "field": "u",
    "axis": "t",
    "order": 1
  }},
  "vocabulary": {{
    "fields": [
      "u"
    ],
    "coordinates": [
      "t",
      "x"
    ]
  }},
  "pinned": [],
  "anchored": [],
  "holes": [
    {{
      "id": "rest",
      "min_count": 0,
      "max_count": 3,
      "constraint": {{
        "max_deriv_order": 2,
        "operators": [
          "mul"
        ],
        "fields": [
          "u"
        ],
        "axes": [
          "x"
        ]
      }}
    }}
  ],
  "match_policy": {{
    "coeff_atol": 1e-08,
    "coeff_rtol": 1e-05,
    "support_threshold": 0.0,
    "term_identity": "kd-lawsig-v2",
    "hole_assignment": "disjoint",
    "derivative_order": "total-effective"
  }}
}}
```
Use the input's names. The per-algorithm table declares each clause's level;
tabular modes do not accept sketches. Operator constraints use IR feature
names, such as `mul` and `recip`, not search-backend spellings.

What the levels mean. `lowered`: the clause is compiled into the search arithmetic itself, as a pinned term deducted from the regression target before the fit. `generation_enforced`: the clause filters which candidates the search may generate, which decides how fast a compliant law is found rather than whether it is compliant. `fit_enforced`: the anchored column is in every fit, never thresholded, and its coefficient is not shrunk by the penalty; the exit still requires |coefficient| >= support_threshold, and at support_threshold=0 an anchored clause on a fit_enforced backend fails only on an exactly zero coefficient. `exit_checked`: the search itself is not constrained by that clause at all, and compliance is judged only at the exit.

There are two exit gates. First, every used clause is judged whatever its level; `diagnostics.sketch.satisfied` reports that clause verdict. Second, the full law must pass the exit residual verification policy on the platform footprint (by default NMSE <= 0.05). `diagnostics.sketch.certified` reports whether kd delivered a solution after both gates. A law can satisfy every clause yet fail the residual gate: `satisfied` true, `certified` false and a null `primary.law`; `diagnostics.sketch.failure` explains the refusal. kd never delivers a law that fails either gate, but the search may find no certified law. A false `certified` and a null `primary.law` are expected outcomes, and clause failure is likelier when more used clauses sit at `exit_checked`.

## Surrogates

Only algorithms declaring an injection channel accept `train_surrogate` and
`run_discovery(surrogate=<id>)`. Training accepts the declared surrogate
fields plus mode order parameters and a separate seed. Equal recipes reuse
the registered model. A resumed lineage inherits its surrogate and sketch;
changing either requires a fresh lineage. Read each algorithm's Surrogate
section for the mode selection and training behavior.
