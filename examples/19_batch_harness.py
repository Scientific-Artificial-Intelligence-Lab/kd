
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


shutil.rmtree(OUT, ignore_errors=True)




dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")






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



result = run_plan(plan, datasets={"burgers": dataset}, store_root=OUT / "evidence")

print("\n--- episodes ---")
for outcome in result.outcomes:
    print(
        f" [{outcome.entry_index}] {outcome.entry.instrument} "
        f"seed={outcome.entry.seed} {outcome.status} "
        f"{outcome.wallclock_seconds:.2f}s"
    )
assert all(o.status == "completed" for o in result.outcomes), "an episode failed"




store = EvidenceStore.load(result.store_root)
print(f"\nStore sealed at {store.root} (plan hash {store.plan_hash[:10]})")




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




markdown = OUT / "consensus.md"
markdown.write_text(render_consensus_markdown(report), encoding="utf-8")
artifact = write_consensus_artifact(report, path=OUT / "consensus.json")
print(f"\nWrote {markdown} and {artifact.name}")











print(
    "\nA class split can be a notation difference rather than a real "
    "disagreement.\nCheck the adjacency relation before drawing conclusions."
)















