"""Example 10 - Checkpoint a long run, then resume it.

``ExperimentResult.save`` (example 05) persists a FINISHED result. But what
if the process dies mid-search - power cut, OOM kill, Ctrl-C? With
``checkpoint_dir`` set, the model writes its full search state to disk as
it runs:

    checkpoint_000000.pt ... every ``checkpoint_every`` iterations
    checkpoint_final.pt at the end of fit - also on a Python-level
                                failure (exception / Ctrl-C) via the runner's
                                finally-block. A HARD kill (power cut, OOM
                                SIGKILL) leaves no final file: resume from
                                the last periodic checkpoint instead.
    manifest.json the ledger: one entry per checkpoint above,
                                recording its kind, iteration, best score and,
                                for the final one, whether the run COMPLETED
                                or CRASHED.

Pick the resume point from that ledger, never from the filenames. The
finally-block writes ``checkpoint_final.pt`` for a crashed run too, so the
name alone cannot tell a finished run from a dead one; only the manifest's
``final_status`` can. ``kd.load_checkpoint_manifest`` reads and verifies the
ledger (read-only and fail-loud; call it on a directory whose run has
terminated) and hands back the recorded facts to select on.

``fit(dataset, resume_from=<file>)`` then restores the SEARCH STATE
(population / controller weights / best-so-far) into a fresh Model and keeps
searching. Config is NOT restored - it comes from the new Model - so
"resume with a larger generations budget" is the natural way to extend a
run. Checkpoints are written atomically (tmp + rename): a process killed
mid-write can never leave a torn checkpoint behind.

``checkpoint_dir`` works for all seven facade algorithms, in two flavours.
Six of them resume by restoring the search state and continuing to search:
the five iterative engines (sga / dlga / discover / eqgpt / llm4ed) plus pysr,
whose resumed segment is one warm ``PySRRegressor.fit`` from the archived
populations, with ``generations`` as that segment's PySR iteration increment.
Only pysindy is recover-without-rerun: the fitted result is reloaded (no
second PySINDy pass) rather than the search being extended.

Run: python examples/10_checkpoint_resume.py
"""

import shutil
from pathlib import Path

import kd

CKPT_DIR = Path(__file__).parent / "out" / "10_checkpoints"
CKPT_DIR_PHASE2 = Path(__file__).parent / "out" / "10_checkpoints_phase2"
# Fresh dirs so the listings are honest AND so each run gets an unused
# checkpoint_dir: a manifest-managed directory refuses reuse (one run per dir).
shutil.rmtree(CKPT_DIR, ignore_errors=True)
shutil.rmtree(CKPT_DIR_PHASE2, ignore_errors=True)

# 1. Generate synthetic data with a known ground truth.
dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")

# 2. an internal milestone - a budget-limited first run that checkpoints as it goes.
# (Stand-in for the run that gets interrupted: if the process died after
# iteration 5, checkpoint_000004.pt would survive and be resumable.)
model = kd.Model(
    algorithm="sga",
    generations=8,
    population=12,
    seed=0,
    verbose=False,
    checkpoint_dir=CKPT_DIR,
    checkpoint_every=4,
)
model.fit(dataset)

print("\n--- an internal milestone: 8 generations, checkpointed ---")
print(f"Discovered: {model.best_expr_}")
print(f"Best AIC: {model.best_score_:.4f}")

# 3. Read the ledger to see what the run left behind. load_checkpoint_manifest
# verifies the whole directory (files the ledger does not name, torn writes)
# and returns the entries in write order, or raises
# kd.CheckpointManifestError rather than handing back a half-truth.
entries = kd.load_checkpoint_manifest(CKPT_DIR)
for entry in entries:
    print(
        f" wrote {entry.filename:<21} kind={entry.kind:<9} "
        f"status={entry.final_status or '-':<10} iter={entry.iteration}"
    )

# 4. Select the resume point from those recorded facts. Prefer the final
# checkpoint of a COMPLETED run; if the ledger has none (it crashed, or the
# process was hard-killed before the finally-block ran), fall back to the
# newest periodic checkpoint. A filename glob cannot make this distinction.
resume_entry: kd.CheckpointManifestEntry | None = next(
    (
        entry
        for entry in entries
        if entry.kind == kd.KIND_FINAL
        and entry.final_status == kd.FINAL_STATUS_COMPLETED
    ),
    None,
)
if resume_entry is None:
    periodic = [entry for entry in entries if entry.kind != kd.KIND_FINAL]
    if not periodic:
        raise SystemExit("no resumable checkpoint recorded in the manifest")
    resume_entry = periodic[-1]
print(f"Resume point: {resume_entry.filename} (iteration {resume_entry.iteration})")

# 5. an internal milestone - resume from the selected checkpoint with a FRESH Model (think:
# a new process after the crash). The algorithm must match the checkpoint;
# the budget is the new Model's, so this run extends the search.
resumed = kd.Model(
    algorithm="sga",
    # The NEW budget. Iteration numbering restarts at 0 on resume, so this
    # runs 15 MORE generations on top of the restored search state.
    generations=15,
    population=12,
    seed=0,
    verbose=False,
    # Checkpoint the resumed run too - a second crash is just as fatal. A
    # checkpoint_dir is manifest-managed and refuses reuse, so the resumed run
    # writes to its OWN fresh directory (one run per dir); resume_from still
    # points at the phase-1 checkpoint the manifest selected.
    checkpoint_dir=CKPT_DIR_PHASE2,
    checkpoint_every=4,
)
resumed.fit(dataset, resume_from=CKPT_DIR / resume_entry.filename)

print("\n--- an internal milestone: resumed, +15 generations ---")
print(f"Discovered: {resumed.best_expr_}")
print(f"Best AIC: {resumed.best_score_:.4f}")

# 6. The restored best-so-far is the resumed run's starting ratchet, so
# phase 2 can only match or improve phase 1's AIC - never regress.
# (Synthetic Burgers converges within a few generations, so the two
# scores match here; on harder problems the extra budget keeps digging.)
assert resumed.best_score_ <= model.best_score_ + 1e-9, "Resume lost progress!"
print("\nResumed best <= phase-1 best: resume kept the search progress.")
