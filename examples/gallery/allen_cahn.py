"""Gallery: Allen-Cahn fronts, discovered by SGA and integrated forward.

SGA searches the bundled Allen-Cahn field (256 x 201, periodic x on [-1, 1),
t in [0, 10]) with the settings of kd's recovery test for the same equation
family (Chafee-Infante, u_t = u_xx - u + u^3): 150 generations, population 20,
STRidge maxit 15, str_iters 15, d_tol 0.3, finite-difference derivatives.
That test runs five search seeds (42, 123, 256, 314, 777); pass one of them on
the command line. The discovered equation is integrated from the first time
slice over the whole grid and drawn against the data.

Outputs in ``examples/gallery/out/allen_cahn/``:

    allen_cahn-cover.png space-time view: observed, integrated, difference
    allen_cahn.mp4 the profile over time, 60 frames at 12 fps
    report/ the run's HTML report
    allen_cahn-run.json the run record: settings, law, verdict, forward error

Run: python examples/gallery/allen_cahn.py 42 (minutes on a CPU; needs ffmpeg)
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
from kd.viz.plots import plot_field_comparison, save_profile_animation
from kd.viz.style import SAVE_DPI

OUT = Path(__file__).parent / "out" / "allen_cahn"
SEED = int(sys.argv[1])
SEARCH = {"generations": 150, "population": 20, "maxit": 15, "str_iters": 15}
D_TOL = 0.3
FPS, MAX_FRAMES = 12, 60
# judge_law thresholds: the u_xx coefficient is 0.003 of the u and u^3 ones,
# so the trimming threshold sits below that ratio; coefficients within 5 %.
SMALL_TERM_RTOL, COEFF_RTOL = 1e-4, 0.05

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("gallery.allen_cahn")
OUT.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
environment = environment_fingerprint()

# 1. Data: the bundled Allen-Cahn benchmark, noise-free.
dataset = kd.load_allen_cahn()

# 2. Discovery: SGA with the recovery test's settings.
model = kd.Model(algorithm="sga", seed=SEED, d_tol=D_TOL, verbose=False, **SEARCH).fit(
    dataset
)
result = model.result_
terms = [term for term, _ in result.equation.terms]
coefficients = [float(scalar.value) for _, scalar in result.equation.terms]
log.info("Discovered: %s", result.equation)

# 3. Forward: integrate the discovered equation from the first time slice.
integration, integration_notes = build_integration_result(result, dataset)

# 4. Cover: the space-time field, observed and integrated, on a diverging map
# (the two phases are u = -1 and u = +1, so zero is the interface).
cover, cover_notes = plot_field_comparison(dataset, integration, field_cmap="diverging")
cover.savefig(OUT / "allen_cahn-cover.png", dpi=SAVE_DPI)

# 5. Movie: the observed profile and the integrated law over time.
movie_notes = save_profile_animation(
    dataset, integration, OUT / "allen_cahn.mp4", fps=FPS, max_frames=MAX_FRAMES
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
assets = [OUT / "allen_cahn-cover.png", OUT / "allen_cahn.mp4", report.report]
record = {
    "piece": "allen_cahn",
    "environment": environment,
    "command": f"python examples/gallery/allen_cahn.py {SEED}",
    "data": {
        "loader": "kd.load_allen_cahn()",
        "shape": list(dataset.get_field(dataset.lhs_field).shape),
        "fingerprint": compute_dataset_fingerprint(dataset),
        "noise": None,
    },
    "seed": SEED,
    "config": result.config,
    "device": str(dataset.get_field(dataset.lhs_field).device),
    "budget": {"generations": SEARCH["generations"]},
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
(OUT / "allen_cahn-run.json").write_text(json.dumps(record, indent=2) + "\n")
log.info("Wrote %s", OUT)
