"""Gallery: the Lorenz attractor, discovered from two noisy trajectories.

kd's Lorenz system (``generate_ode_trajectories("chaotic")``: sigma 10,
rho 28, beta 8/3, t in [0, 8], here 961 samples) gives two trajectories from
initial states drawn with seed 0. Each state of each trajectory receives
Gaussian noise with a standard deviation of exactly 1 % of that state's
standard deviation (``kd.data.xu2020_relative``). Central differences of
noisy samples this close together are mostly noise, so the derivatives come
from one neural network per trajectory, fitted to the noisy states and
differentiated by autograd (``prepare_trajectory_regression``). SGA then fits
one law per state on that state's regression table, with the settings of kd's
ODE recovery test (200 generations, population 40, depth 4, width 5) and the
complexity penalty ``aic_ratio`` halved to 0.5; the search seed, shared by
the three fits, is the command-line argument. The three laws are integrated
together from each trajectory's first noisy sample and compared with the
observations.

Nearby trajectories of a chaotic system separate exponentially, so pointwise
agreement can only hold for a while; the state panels are cut at ``WINDOW``
and the record holds the error inside that window and over the full interval,
next to the same measurement for the reference law integrated from the same
noisy samples.

Outputs in ``examples/gallery/out/lorenz/``:

    lorenz-cover.png both trajectories in 3D, observed and integrated
    lorenz-states.png x, y, z of one trajectory against time, t <= WINDOW
    lorenz.mp4 the 3D curves growing over time, 300 frames at 24 fps
    report-d?_dt/ one HTML report per state law
    lorenz-run.json the run record: settings, laws, verdicts, errors

Run: python examples/gallery/lorenz.py 1 (about ten minutes on a CPU; needs ffmpeg)
"""

import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import kd
from kd.data import (
    ODE_SYSTEMS,
    TrajectoryNNConfig,
    compute_dataset_fingerprint,
    derivative_target,
    generate_ode_trajectories,
    prepare_trajectory_regression,
    xu2020_relative,
)
from kd.harness import environment_fingerprint
from kd.viz.plots import (
    plot_state_panels,
    plot_trajectory_overlay,
    save_trajectory_animation,
)
from kd.viz.style import SAVE_DPI

OUT = Path(__file__).parent / "out" / "lorenz"
SYSTEM, DATA_SEED, N_TRAJECTORIES, N_POINTS = "chaotic", 0, 2, 961
NOISE, NOISE_SEED = 0.01, 1000
NN = TrajectoryNNConfig(seed=0, hidden_sizes=(128, 128, 128), max_epochs=20000)
SEARCH = {"generations": 200, "population": 40, "depth": 4, "width": 5}
AIC_RATIO, SEARCH_SEED = 0.5, int(sys.argv[1])
TRIM = 2
# The state panels show one trajectory (one colour pair per figure), cut where
# pointwise agreement still holds.
SHOWN, WINDOW = "chaotic_traj_01", 2.5
FPS, MAX_FRAMES = 24, 300
# Coefficients within 10 %: the bar kd's ODE recovery test sets for neural
# derivatives of noisy trajectories.
SMALL_TERM_RTOL, COEFF_RTOL = 1e-4, 0.10

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("gallery.lorenz")
OUT.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
environment = environment_fingerprint()
system = ODE_SYSTEMS[SYSTEM]

# 1. Data: two trajectories of kd's Lorenz system.
clean = generate_ode_trajectories(
    SYSTEM, seed=DATA_SEED, n_trajectories=N_TRAJECTORIES, n_points=N_POINTS
)

# 2. Noise: 1 % of each state's standard deviation, one seed per state series.
observed = dataclasses.replace(
    clean,
    source=f"{clean.source}; {NOISE:.0%} noise per state, seeds from {NOISE_SEED}",
    trajectories=tuple(
        dataclasses.replace(
            trajectory,
            state=np.stack(
                [
                    xu2020_relative(
                        torch.from_numpy(row.copy()),
                        NOISE,
                        seed=NOISE_SEED + len(system.state_vars) * k + i,
                    ).numpy()
                    for i, row in enumerate(trajectory.state)
                ]
            ),
        )
        for k, trajectory in enumerate(clean.trajectories)
    ),
)

# 3. Derivatives: one network per trajectory, differentiated by autograd.
prepared = prepare_trajectory_regression(observed, derivatives="autograd", nn_config=NN)

# 4. Discovery: one SGA fit per state, on that state's regression table.
laws, models = {}, {}
for state in system.state_vars:
    table = prepared.table(state, state_source="observed", trim=TRIM)
    model = kd.Model(
        algorithm="sga",
        seed=SEARCH_SEED,
        aic_ratio=AIC_RATIO,
        verbose=False,
        **SEARCH,
    ).fit(table)
    equation = model.result_.equation
    laws[table.target_name] = (
        [term for term, _ in equation.terms],
        [float(scalar.value) for _, scalar in equation.terms],
    )
    models[table.target_name] = model
    log.info("Discovered: %s", model.best_expr_)

# 5. Forward: integrate the three laws together along both trajectories.
forward = kd.forward_error_trajectories(observed, laws)
log.info("Forward error, full interval: %s %s", forward.value, forward.reason)

# 6. Reference: the published laws, judged term by term and integrated the same way.
truths = {
    derivative_target(state): kd.law_from_equation(
        answer, axes=[observed.time_var], fields=system.state_vars
    )
    for state, answer in zip(system.state_vars, system.answers, strict=True)
}
verdicts = {
    target: kd.judge_law(
        *laws[target],
        truth.terms,
        truth.coefficients,
        axes=[observed.time_var],
        fields=system.state_vars,
        small_term_rtol=SMALL_TERM_RTOL,
        coeff_rtol=COEFF_RTOL,
    )
    for target, truth in truths.items()
}
reference_laws = {t: (law.terms, law.coefficients) for t, law in truths.items()}
reference_forward = kd.forward_error_trajectories(observed, reference_laws)

# 7. The shown trajectory, cut to the window: its own forward error there.
shown = dataclasses.replace(
    observed, trajectories=tuple(t for t in observed.trajectories if t.id == SHOWN)
)
windowed = dataclasses.replace(
    shown,
    trajectories=tuple(
        dataclasses.replace(t, t=t.t[t.t <= WINDOW], state=t.state[:, t.t <= WINDOW])
        for t in shown.trajectories
    ),
)
window_forward = kd.forward_error_trajectories(windowed, laws)
log.info("Forward error, %s for t <= %s: %s", SHOWN, WINDOW, window_forward.value)

# 8. Cover: both trajectories in 3D, observed and integrated.
cover_notes = plot_trajectory_overlay(observed, forward.predicted)
plt.gcf().savefig(OUT / "lorenz-cover.png", dpi=SAVE_DPI)

# 9. State panels: one trajectory against time inside the window.
panels, panel_notes = plot_state_panels(shown, forward.predicted, t_max=WINDOW)
panels.savefig(OUT / "lorenz-states.png", dpi=SAVE_DPI)

# 10. Movie: the 3D curves growing over time.
movie_notes = save_trajectory_animation(
    observed, forward.predicted, OUT / "lorenz.mp4", fps=FPS, max_frames=MAX_FRAMES
)

# 11. Reports: one per state law.
reports = {
    target: model.report(OUT / f"report-{target}") for target, model in models.items()
}


# 12. Run record.
def _forward_record(result: kd.TrajectoryForwardError) -> dict:
    return {
        "value": result.value,
        "reason": result.reason,
        "per_trajectory": result.per_trajectory,
        "per_target": result.per_target,
        "intervals": result.intervals,
        "solver": result.solver,
    }


assets = [
    OUT / "lorenz-cover.png",
    OUT / "lorenz-states.png",
    OUT / "lorenz.mp4",
    *(report.report for report in reports.values()),
]
record = {
    "piece": "lorenz",
    "environment": environment,
    "command": f"python examples/gallery/lorenz.py {SEARCH_SEED}",
    "data": {
        "generator": f"kd.data.generate_ode_trajectories({SYSTEM!r}, "
        f"seed={DATA_SEED}, n_trajectories={N_TRAJECTORIES}, n_points={N_POINTS})",
        "trajectories": [
            {"id": t.id, "samples": int(t.t.size), "t": [float(t.t[0]), float(t.t[-1])]}
            for t in observed.trajectories
        ],
        "fingerprint_clean": compute_dataset_fingerprint(clean),
        "fingerprint_observed": compute_dataset_fingerprint(observed),
        "noise": {
            "recipe": "kd.data.xu2020_relative per state of each trajectory",
            "level": NOISE,
            "seeds": f"{NOISE_SEED} + 3 * trajectory index + state index",
        },
        "derivatives": prepared.metadata["derivatives"],
        "nn_config": prepared.metadata["nn_config"],
        "nn_seeds": prepared.metadata["derived_seeds"],
        "nn_training": {
            k: {
                "epochs": r.epochs_run,
                "final_loss": r.final_loss,
                "seconds": r.elapsed_seconds,
            }
            for k, r in prepared.training_reports.items()
        },
        "trim": TRIM,
    },
    "seed": SEARCH_SEED,
    "config": {t: m.result_.config for t, m in models.items()},
    "device": NN.device,
    "budget": {"generations": SEARCH["generations"]},
    "iterations": {t: m.result_.iterations for t, m in models.items()},
    "early_stopped": {t: m.result_.early_stopped for t, m in models.items()},
    "cost": {t: m.result_.run_record.to_dict()["cost"] for t, m in models.items()},
    "discovered": {
        t: {
            "equation": m.best_expr_,
            "terms": laws[t][0],
            "coefficients": laws[t][1],
            "score_kind": m.result_.score_kind,
            "score": m.best_score_,
            "nmse": m.result_.final_eval.nmse,
        }
        for t, m in models.items()
    },
    "reference": list(system.answers),
    "verdict": {t: v.to_dict() for t, v in verdicts.items()},
    "forward_error": {
        "definition": "||pred - obs||_2 / ||obs||_2 over trajectories, states, times",
        "start": "each trajectory's first noisy sample",
        "full_interval": _forward_record(forward),
        "window": {
            "trajectory": SHOWN,
            "t_max": WINDOW,
            **_forward_record(window_forward),
        },
        "reference_law_full_interval": _forward_record(reference_forward),
    },
    "notes": {
        "cover": cover_notes,
        "panels": panel_notes,
        "movie": movie_notes,
        "reports": {t: r.warnings for t, r in reports.items()},
    },
    "movie": {"fps": FPS, "max_frames": MAX_FRAMES},
    "assets": {str(p.relative_to(OUT)): p.stat().st_size for p in assets},
    "wall_seconds": time.perf_counter() - started,
    "argv": sys.argv,
}
(OUT / "lorenz-run.json").write_text(json.dumps(record, indent=2) + "\n")
log.info("Wrote %s", OUT)
