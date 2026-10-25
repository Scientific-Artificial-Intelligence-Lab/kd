"""Gallery: two KdV solitons, discovered by SGA and integrated forward.

SGA searches the bundled KdV field (256 x 201, periodic x on [-1, 1),
t in [0, 1]) with the documentation site's recipe: 200 generations, seed 42,
finite-difference derivatives. The discovered equation is then integrated
from the first time slice over the whole grid and drawn against the data.

Outputs in ``examples/gallery/out/kdv/``:

    kdv-cover.png profiles at eight times, stacked (observed vs integrated)
    kdv.mp4 the profile over time, 60 frames at 12 fps
    report/ the run's HTML report
    kdv-run.json the run record: settings, law, verdict, forward error

Run: python examples/gallery/kdv.py (a few minutes on a CPU; needs ffmpeg)
"""

import json
import logging
import sys
import time
from pathlib import Path

import kd
from kd.data import compute_dataset_fingerprint
from kd.harness import environment_fingerprint
from kd.viz import build_integration_result
from kd.viz.plots import plot_profile_waterfall, save_profile_animation
from kd.viz.style import SAVE_DPI

OUT = Path(__file__).parent / "out" / "kdv"
GENERATIONS, SEED = 200, 42
FPS, MAX_FRAMES = 12, 60
# judge_law thresholds: the u_xxx coefficient is 0.0025 of the u*u_x one, so
# the trimming threshold sits below that ratio; coefficients within 5 %.
SMALL_TERM_RTOL, COEFF_RTOL = 1e-4, 0.05

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("gallery.kdv")
OUT.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
environment = environment_fingerprint()

# 1. Data: the bundled KdV benchmark, noise-free.
dataset = kd.load_kdv()

# 2. Discovery: SGA with the site's recipe.
model = kd.Model(
    algorithm="sga", generations=GENERATIONS, seed=SEED, verbose=False
).fit(dataset)
result = model.result_
terms = [term for term, _ in result.equation.terms]
coefficients = [float(scalar.value) for _, scalar in result.equation.terms]
log.info("Discovered: %s", result.equation)

# 3. Forward: integrate the discovered equation from the first time slice.
integration, integration_notes = build_integration_result(result, dataset)

# 4. Cover: profiles at eight evenly spaced times, stacked.
cover, cover_notes = plot_profile_waterfall(dataset, integration)
cover.savefig(OUT / "kdv-cover.png", dpi=SAVE_DPI)

# 5. Movie: the observed profile and the integrated law over time.
movie_notes = save_profile_animation(
    dataset, integration, OUT / "kdv.mp4", fps=FPS, max_frames=MAX_FRAMES
)

# 6. Report: every figure of the run in one HTML page.
report = model.report(OUT / "report")

# 7. Measurements: forward error over the grid, and the term-by-term verdict.
forward = kd.forward_error(dataset, terms, coefficients)
truth = kd.law_from_equation(dataset.ground_truth, axes=dataset.axis_order)
verdict = kd.judge_law(
    terms,
    coefficients,
    truth.terms,
    truth.coefficients,
    axes=dataset.axis_order,
    small_term_rtol=SMALL_TERM_RTOL,
    coeff_rtol=COEFF_RTOL,
)
log.info(
    "Verdict: structure %s, coefficients %s",
    verdict.structure_ok,
    verdict.coefficients_ok,
)
log.info("Forward error: %s over t in %s", forward.value, forward.interval)

# 8. Run record.
assets = [OUT / "kdv-cover.png", OUT / "kdv.mp4", report.report]
record = {
    "piece": "kdv",
    "environment": environment,
    "command": "python examples/gallery/kdv.py",
    "data": {
        "loader": "kd.load_kdv()",
        "shape": list(dataset.get_field(dataset.lhs_field).shape),
        "fingerprint": compute_dataset_fingerprint(dataset),
        "noise": None,
    },
    "seed": SEED,
    "config": result.config,
    "device": str(dataset.get_field(dataset.lhs_field).device),
    "budget": {"generations": GENERATIONS},
    "iterations": result.iterations,
    "early_stopped": result.early_stopped,
    "cost": result.run_record.to_dict()["cost"],
    "discovered": {
        "equation": str(result.equation),
        "terms": terms,
        "coefficients": coefficients,
        "score_kind": result.score_kind,
        "score": model.best_score_,
        "nmse": result.final_eval.nmse,
    },
    "reference": dataset.ground_truth,
    "verdict": verdict.to_dict(),
    "forward_error": {
        "value": forward.value,
        "reason": forward.reason,
        "interval": forward.interval,
        "boundary": forward.boundary,
        "definition": "||pred - u||_2 / ||u||_2 over the full grid",
    },
    "notes": {
        "integration": integration_notes,
        "cover": cover_notes,
        "movie": movie_notes,
        "report": report.warnings,
    },
    "movie": {"fps": FPS, "max_frames": MAX_FRAMES},
    "assets": {str(p.relative_to(OUT)): p.stat().st_size for p in assets},
    "wall_seconds": time.perf_counter() - started,
    "argv": sys.argv,
}
(OUT / "kdv-run.json").write_text(json.dumps(record, indent=2) + "\n")
log.info("Wrote %s", OUT)
