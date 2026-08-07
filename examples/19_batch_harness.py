"""Example 19 - Run a batch of fits as one plan, then read the consensus.

Examples 01-18 each run ONE fit. Research questions are usually plural:
does this result hold across seeds? across engines? across datasets?
``kd.harness`` answers those by running a declarative matrix and sealing
the evidence, so the aggregate is computed from records rather than from
notes:

    ExperimentPlan an ORDERED matrix of (instrument, dataset_ref, seed,
                       model_kwargs) entries; the plan hash covers the order,
                       so a plan identifies the exact matrix that was run
    run_plan runs every entry serially into a fresh EvidenceStore,
                       stamping an environment fingerprint
    EvidenceStore.load re-opens a store READ-ONLY and re-verifies it: plan
                       hash, every record's own hash, no orphan files
    build_consensus a pure function over a sealed store -> a report tree
                       grouping runs into structure classes

A per-entry failure does not abort the batch: it is recorded as an outcome
with ``status="raised"`` and the remaining entries still run. A failure to
PERSIST evidence does abort, because a batch that cannot record what it did
is not worth continuing.

This example varies the seed only, which is the cheapest useful question
(is the result reproducible?). Widening the matrix is a plan edit, not a
code change - see the comment at the bottom.

Run: python examples/19_batch_harness.py
"""

import shutil
from pathlib import Path

import kd
from kd.harness import (
    EvidenceStore,
    ExperimentPlan,
    PlanEntry,
    build_consensus,
    render_consensus_markdown,
    run_plan,
    write_consensus_artifact,
)

OUT = Path(__file__).parent / "out" / "19_harness"
# ``run_plan`` requires a fresh store root: an evidence store is append-only
# and sealed on completion, so it refuses to reuse a populated directory.
shutil.rmtree(OUT, ignore_errors=True)

# 1. One dataset, held in memory. The plan refers to it by NAME
# (``dataset_ref``); ``run_plan`` resolves the name against the mapping you
# pass, so a plan stays a piece of data with no dataset bytes in it.
dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")

# 2. The plan: the same engine and budget at three seeds.
# ``model_kwargs`` carries only facade knobs the harness does not own.
# ``algorithm`` / ``seed`` / ``verbose`` / ``config`` and friends are
# reserved - the harness sets them from the entry itself and rejects a
# plan that tries to shadow them.
plan = ExperimentPlan(
    name="burgers-seed-stability",
    entries=tuple(
        PlanEntry(
            instrument="sga",
            dataset_ref="burgers",
            seed=seed,
            model_kwargs={"generations": 6, "population": 12},
        )
        for seed in (0, 1, 2)
    ),
)
print(f"Plan: {len(plan.entries)} entries, hash {plan.plan_hash()[:10]}")

# 3. Run it. Serial by design: the harness composes the Model facade and adds
# no scheduling, routing, or budget logic of its own.
result = run_plan(plan, datasets={"burgers": dataset}, store_root=OUT / "evidence")

print("\n--- episodes ---")
for outcome in result.outcomes:
    print(
        f" [{outcome.entry_index}] {outcome.entry.instrument} "
        f"seed={outcome.entry.seed} {outcome.status} "
        f"{outcome.wallclock_seconds:.2f}s"
    )
assert all(o.status == "completed" for o in result.outcomes), "an episode failed"

# 4. Re-open the store the way a later analysis would: from disk, read-only.
# ``load`` re-verifies the whole directory and raises on any mismatch, so
# an aggregate can never be computed over silently edited evidence.
store = EvidenceStore.load(result.store_root)
print(f"\nStore sealed at {store.root} (plan hash {store.plan_hash[:10]})")

# 5. Aggregate. Passing ``datasets`` lets the report add the empirical axis
# (each law re-checked on the data with its own reported coefficients);
# omit it and the structural axes are still computed.
report = build_consensus(store, datasets={"burgers": dataset})

print("\n--- consensus ---")
for dataset_consensus in report.datasets:
    print(f"dataset {dataset_consensus.dataset_ref}:")
    for cls in dataset_consensus.classes:
        members = ", ".join(
            f"{m.instrument}/seed{m.seed}" for m in cls.members
        )
        print(f" class {cls.structure_key[:10]} {{{', '.join(cls.terms)}}}")
        print(f" members: {members}")
    for adjacency in dataset_consensus.adjacency:
        print(
            f" {adjacency.structure_key_a[:10]} vs "
            f"{adjacency.structure_key_b[:10]}: {adjacency.relation} "
            f"(jaccard {adjacency.jaccard:.2f})"
        )
    for stability in dataset_consensus.instrument_stability:
        print(
            f" {stability.instrument}: {stability.n_signable}/"
            f"{stability.n_completed} runs signable, modal class share "
            f"{stability.modal_share:.2f}"
        )

# 6. Two renderings of the same report: Markdown to read, JSON to diff.
# Both are deterministic functions of the report, so re-rendering the same
# evidence gives byte-identical output.
markdown = OUT / "consensus.md"
markdown.write_text(render_consensus_markdown(report), encoding="utf-8")
artifact = write_consensus_artifact(report, path=OUT / "consensus.json")
print(f"\nWrote {markdown} and {artifact.name}")

# 7. Reading the result. Classes group runs by CANONICAL TERM SET, so two
# runs share a class when they selected the same terms - not merely when
# their printed equations look alike. The converse also holds and is worth
# seeing: on this dataset seeds 0 and 1 land in one class, while seed 2
# typically lands in its own with terms like diff_x(n2(u)) and
# diff2_x(sub(u, x)). That is the SAME law in different notation -
# d/dx(u^2) = 2*u*u_x and d2/dx2(u - x) = u_xx - so the split is a
# notation difference, not a disagreement about physics. The adjacency
# line is what tells you the two classes are related; read it before
# concluding that an engine is unstable.
print(
    "\nA class split can be a notation difference rather than a real "
    "disagreement.\nCheck the adjacency relation before drawing conclusions."
)

# Widening the matrix is a plan edit. To compare engines, or to add a
# dataset, extend the entry list and pass every referenced dataset:
#
# entries=tuple(
# PlanEntry(instrument=name, dataset_ref=ref, seed=seed, model_kwargs={})
# for name in ("sga", "discover")
# for ref in ("burgers", "advection")
# for seed in (0, 1, 2)
# )
# run_plan(plan, datasets={"burgers": ds1, "advection": ds2}, store_root=...)
#
# Give each engine a budget it can actually work with: a cross-engine table
# built from budgets that suit one engine is a statement about the budgets,
# not about the engines.
