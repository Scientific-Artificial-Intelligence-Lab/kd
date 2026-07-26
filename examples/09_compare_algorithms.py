
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kd
from kd.core.expr.sympy_bridge import format_pde
from kd.llm import LLMRequest, LLMResponse
from kd.search.dlga import DLGAConfig
from kd.search.eqgpt.backend import resolve_asset_path
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.llm4ed.config import Llm4edConfig

OUT_DIR = Path(__file__).parent / "out" / "09_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class OfflineProvider:

    def prepare(self) -> None:
        return None

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="<res>u*u_x+u_xx</res>", model="offline", usage=None)




dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")





MODELS: dict[str, kd.Model] = {
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
}
skipped: dict[str, str] = {}


if importlib.util.find_spec("pysr") is not None:
    from kd.search.pysr.config import PySRConfig

    MODELS["PySR"] = kd.Model(
        algorithm="pysr",
        config=PySRConfig(
            terms=("u", "u_x", "u_xx"),
            niterations=40,
            seed=0,

            extra_pysr_kwargs={"deterministic": True, "parallelism": "serial"},
        ),
    )
else:
    skipped["PySR"] = "needs the `pysr` extra (uv sync --extra pysr); see examples/12"




if importlib.util.find_spec("pysindy") is not None:
    from kd.search.pysindy.config import PySINDyConfig

    MODELS["PySINDy"] = kd.Model(
        algorithm="pysindy",
        config=PySINDyConfig(
            terms=("u", "u_x", "u_xx", "mul(u, u_x)"),
            threshold=0.05,
            seed=0,
        ),
    )
else:
    skipped["PySINDy"] = "needs the `pysindy` extra (uv sync --extra pysindy)"





try:
    resolve_asset_path()
except FileNotFoundError:
    skipped["EqGPT"] = "needs pretrained weights (.pt); see examples/16_eqgpt.py"
else:
    MODELS["EqGPT"] = kd.Model(
        algorithm="eqgpt", generations=3, config=EqGPTConfig.burgers_preset()
    )




MODELS["llm4ed"] = kd.Model(
    algorithm="llm4ed",
    generations=3,
    config=Llm4edConfig(samples_per_epoch=4, max_llm_calls_per_propose=4),
    provider=OfflineProvider(),
)
PRESET_ENGINES = {"llm4ed"}


def display_name(name: str) -> str:
    return f"{name} (preset)" if name in PRESET_ENGINES else name


if skipped:
    print("\nSkipped (optional asset missing):")
    for name, why in skipped.items():
        print(f" - {name}: {why}")



for name, model in MODELS.items():
    print(f"\n=== {display_name(name)} ===")
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


def canonical_terms(terms: list[str]) -> list[str]:
    return terms


UNIFIED_LHS = "u_t"

structures = {
    name: canonical_terms(discovered_structure(m.result_))
    for name, m in MODELS.items()
}
lhs_names = {
    name: (m.result_.final_eval.lhs_name or UNIFIED_LHS) for name, m in MODELS.items()
}



unified: dict[str, kd.EvaluationResult | None] = {}
exclusions: dict[str, str] = {}
for name in MODELS:
    if lhs_names[name] != UNIFIED_LHS:
        unified[name] = None
        exclusions[name] = f"LHS={lhs_names[name]} != {UNIFIED_LHS} -- not comparable"
        continue
    if not structures[name]:



        unified[name] = None
        exclusions[name] = "engine reported no valid structure"
        continue
    try:


        unified[name] = kd.evaluate_terms(dataset, structures[name], max_order=3)
    except (kd.InvalidTermsError, kd.EvaluationFailedError) as err:
        unified[name] = None
        exclusions[name] = f"unified re-fit refused: {err}"

print("\n=== Unified platform evaluation (kd.evaluate_terms, same NMSE) ===")
print(f"{'Algorithm':<16} {'unified NMSE':>14} {'R^2':>9} discovered structure")
for name, ev in unified.items():
    if ev is None:
        line = f"{display_name(name):<16} {'(excluded)':>14} {'-':>9} "
        print(line + exclusions[name])
    else:
        structure = " + ".join(structures[name])
        print(f"{display_name(name):<16} {ev.nmse:>14.4g} {ev.r2:>9.4f} {structure}")



warnings: list[str] = []


def equation_cell(name: str) -> str:
    ev = unified[name]
    if ev is None:
        reason = exclusions[name]
        return reason if len(reason) <= 90 else reason[:87] + "..."
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
    ok = ev is not None
    rows.append(
        [
            display_name(name),
            equation_cell(name),
            f"{ev.nmse:.2e}" if ok else "--",
            f"{ev.r2:.4f}" if ok else "--",
            "1 (one-shot)" if name in ("PySR", "PySINDy") else str(result.iterations),
        ]
    )
table = ax_t.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
    colWidths=[0.15, 0.49, 0.12, 0.10, 0.14],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)
for (row, _col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#eef2f7")
ax_t.set_title(
    "One dataset, one ruler: every structure re-fit via kd.evaluate_terms "
    "(ground truth: $u_t = -1.0\\,u u_x + 0.1\\,u_{xx}$)",
    fontsize=12,
    pad=14,
)
if PRESET_ENGINES & set(MODELS):

    ax_t.text(
        0.5,
        -0.06,
        "(preset) = canned offline provider proposes a known-good structure; "
        "it demonstrates the unified re-fit, not search quality (see examples/17)",
        transform=ax_t.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
        color="#555555",
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
    ax_p.plot(
        range(len(prog)), prog, marker=".", markersize=4, label=display_name(name)
    )
ax_p.set_xlabel("Iteration")
ax_p.set_ylabel("Normalized best-score progress")
ax_p.set_title("Search progress (per-engine normalized to [0, 1])")
ax_p.grid(alpha=0.25)
ax_p.legend(fontsize=8, loc="lower right")



ax_b = fig.add_subplot(gs[1, 1])
shown: dict[str, float] = {}
for n, e in unified.items():
    if e is None:
        warnings.append(f"{n}: off the ruler -- {exclusions[n]}")
    else:
        shown[n] = e.nmse
bar_colors = ["#c0392b" if n in PRESET_ENGINES else "darkorange" for n in shown]
bars = ax_b.bar(
    [display_name(n) for n in shown], list(shown.values()), color=bar_colors, alpha=0.85
)
if shown:
    ax_b.set_ylim(0, max(shown.values()) * 1.30)
ax_b.set_ylabel("NMSE (unified Evaluator)")
ax_b.set_title("One ruler: platform re-fit NMSE (lower is better)")
ax_b.grid(axis="y", alpha=0.25)
ax_b.tick_params(axis="x", labelrotation=20, labelsize=8)
for bar, value in zip(bars, shown.values(), strict=True):
    ax_b.annotate(
        f"{value:.2e}",
        xy=(bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

engine_list = ", ".join(display_name(n) for n in MODELS)
fig.suptitle(
    f"kd engines on one Burgers dataset (nx=64, nt=32, nu=0.1, seed=0)\n{engine_list}",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "comparison.svg")
plt.close(fig)
print(f"\nComparison sheet: {OUT_DIR / 'comparison.svg'}")
for w in warnings:
    print(f"[viz warning] {w}")




report_engine = next((n for n in ("PySR", "EqGPT") if n in MODELS), None)
if report_engine is not None:
    engine = kd.VizEngine(output_dir=OUT_DIR / f"{report_engine.lower()}_report")
    report = engine.render_all(
        MODELS[report_engine].result_,
        algorithm=MODELS[report_engine].algorithm_,
        dataset=dataset,
    )
    print(f"{report_engine} report: {report.report}")
