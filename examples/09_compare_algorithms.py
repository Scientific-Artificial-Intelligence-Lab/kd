
import importlib.util
import sys
from pathlib import Path



if importlib.util.find_spec("pysr") is None:
    sys.exit(
        "This example runs all four engines and needs the optional PySR "
        "backend.\nInstall it first: uv sync --extra pysr "
        '(or: pip install "kd[pysr]")'
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kd
from kd.core.expr.sympy_bridge import format_pde
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.search.dlga import DLGAConfig
from kd.search.pysr.config import PySRConfig

OUT_DIR = Path(__file__).parent / "out" / "09_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)



dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")


MODELS = {
    "SGA": kd.Model(algorithm="sga", generations=30, population=15, seed=0),
    "DLGA": kd.Model(
        algorithm="dlga",
        generations=20,
        config=DLGAConfig.burgers_preset(
            pop_size=80,
            seed=0,
            surrogate_max_epochs=2000,
        ),
    ),
    "DISCOVER": kd.Model(algorithm="discover", generations=20, seed=0),
    "PySR": kd.Model(
        algorithm="pysr",
        config=PySRConfig(
            terms=("u", "u_x", "u_xx"),
            niterations=40,
            seed=0,

            extra_pysr_kwargs={"deterministic": True, "parallelism": "serial"},
        ),
    ),
}



for name, model in MODELS.items():
    print(f"\n=== {name} ===")
    model.fit(dataset)
    kind = model.result_.score_kind
    print(f"{name} best: {model.best_expr_} (native {kind}: {model.best_score_:.4g})")
    model.result_.save(OUT_DIR / f"{name.lower()}.json")

results = {name: m.result_ for name, m in MODELS.items()}























def discovered_structure(result: kd.ExperimentResult) -> list[str]:
    ev = result.final_eval
    if ev.terms is None:
        return []
    if ev.selected_indices is not None:
        return [ev.terms[i] for i in ev.selected_indices]
    return list(ev.terms)




components = PlatformBuilder(dataset, DerivativeReqs(max_atomic_order=3)).build()
UNIFIED_LHS = "u_t"

structures = {name: discovered_structure(m.result_) for name, m in MODELS.items()}
lhs_names = {
    name: (m.result_.final_eval.lhs_name or UNIFIED_LHS) for name, m in MODELS.items()
}
unified = {
    name: (
        components.evaluator.evaluate_terms(structures[name])
        if lhs_names[name] == UNIFIED_LHS
        else None
    )
    for name in MODELS
}

print("\n=== Unified platform evaluation (same Evaluator, same NMSE) ===")
print(f"{'Algorithm':<10} {'unified NMSE':>14} {'R^2':>9} discovered structure")
for name, ev in unified.items():
    structure = " + ".join(structures[name])
    if ev is None:
        print(
            f"{name:<10} {'(excluded)':>14} {'-':>9} "
            f"LHS={lhs_names[name]} != {UNIFIED_LHS} -- not comparable"
        )
    elif ev.is_valid:
        print(f"{name:<10} {ev.nmse:>14.4g} {ev.r2:>9.4f} {structure}")
    else:
        print(f"{name:<10} {'INVALID':>14} {'-':>9} {ev.error_message}")



warnings: list[str] = []


def equation_cell(name: str) -> str:
    ev = unified[name]
    if ev is None:
        return f"LHS={lhs_names[name]} (excluded from the {UNIFIED_LHS} ruler)"
    if not ev.is_valid:
        return "invalid under the unified ruler"
    eq = format_pde(structures[name], ev.coefficients, lhs=UNIFIED_LHS, sig_figs=3)
    return f"${eq.latex}$"


fig = plt.figure(figsize=(13, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[0.70, 1.05], hspace=0.10, wspace=0.30)




ax_t = fig.add_subplot(gs[0,:])
ax_t.axis("off")
col_labels = [
    "Engine",
    "Discovered equation (unified re-fit)",
    "NMSE",
    "$R^2$",
    "Iters",
]
rows = []
for name, result in results.items():
    ev = unified[name]
    ok = ev is not None and ev.is_valid
    rows.append(
        [
            name,
            equation_cell(name),
            f"{ev.nmse:.2e}" if ok else "--",
            f"{ev.r2:.4f}" if ok else "--",
            "1 (one-shot)" if name == "PySR" else str(result.iterations),
        ]
    )
table = ax_t.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
    colWidths=[0.09, 0.55, 0.12, 0.10, 0.14],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)
for (row, _col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#eef2f7")
ax_t.set_title(
    "One dataset, one ruler: every structure re-fit by the same platform "
    "Evaluator (ground truth: $u_t = -1.0\\,u u_x + 0.1\\,u_{xx}$)",
    fontsize=12,
    pad=14,
)




ax_p = fig.add_subplot(gs[1, 0])
for name, result in results.items():
    series = [float(s) for s in result.recorder.get("_best_score")]
    if not series:
        warnings.append(f"{name}: no _best_score series, progress curve skipped")
        continue
    if len(series) == 1 or series[-1] == series[0]:
        prog = [1.0] * len(series)
    else:
        span = series[-1] - series[0]
        prog = [(s - series[0]) / span for s in series]
    ax_p.plot(range(len(prog)), prog, marker=".", markersize=4, label=name)
ax_p.set_xlabel("Iteration")
ax_p.set_ylabel("Normalized best-score progress")
ax_p.set_title("Search progress (per-engine normalized to [0, 1])")
ax_p.grid(alpha=0.25)
ax_p.legend(fontsize=9, loc="lower right")



ax_b = fig.add_subplot(gs[1, 1])
shown: dict[str, float] = {}
for n, e in unified.items():
    if e is None:
        warnings.append(f"{n}: LHS={lhs_names[n]} != {UNIFIED_LHS}, off the ruler")
    elif not e.is_valid:
        warnings.append(f"{n}: unified evaluation invalid -- {e.error_message}")
    else:
        shown[n] = e.nmse
bars = ax_b.bar(
    list(shown.keys()), list(shown.values()), color="darkorange", alpha=0.85
)
if shown:
    ax_b.set_ylim(0, max(shown.values()) * 1.30)
ax_b.set_ylabel("NMSE (unified Evaluator)")
ax_b.set_title("One ruler: platform re-fit NMSE (lower is better)")
ax_b.grid(axis="y", alpha=0.25)
for bar, value in zip(bars, shown.values(), strict=True):
    ax_b.annotate(
        f"{value:.2e}",
        xy=(bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

fig.suptitle(
    "kd: four engines, one Burgers dataset (nx=64, nt=32, nu=0.1, seed=0)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "comparison.svg")
plt.close(fig)
print(f"\nComparison sheet: {OUT_DIR / 'comparison.svg'}")
for w in warnings:
    print(f"[viz warning] {w}")



report = kd.VizEngine(output_dir=OUT_DIR / "pysr_report").render_all(
    MODELS["PySR"].result_, algorithm=MODELS["PySR"].algorithm_, dataset=dataset
)
print(f"PySR report: {report.report}")
