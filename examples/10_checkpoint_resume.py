
import shutil
from pathlib import Path

import kd

CKPT_DIR = Path(__file__).parent / "out" / "10_checkpoints"
shutil.rmtree(CKPT_DIR, ignore_errors=True)


dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")




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
for path in sorted(CKPT_DIR.glob("checkpoint_*.pt")):
    print(f" wrote {path.name}")




resumed = kd.Model(
    algorithm="sga",


    generations=15,
    population=12,
    seed=0,
    verbose=False,


    checkpoint_dir=CKPT_DIR,
    checkpoint_every=4,
)
resumed.fit(dataset, resume_from=CKPT_DIR / "checkpoint_final.pt")

print("\n--- an internal milestone: resumed, +15 generations ---")
print(f"Discovered: {resumed.best_expr_}")
print(f"Best AIC: {resumed.best_score_:.4f}")





assert resumed.best_score_ <= model.best_score_ + 1e-9, "Resume lost progress!"
print("\nResumed best <= phase-1 best: resume kept the search progress.")
