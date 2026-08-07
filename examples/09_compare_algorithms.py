"""Example 09 - Many engines, one dataset, one ruler.

kd ships seven discovery engines behind the one-line ``kd.Model`` facade. This
example runs whichever are available on the SAME synthetic Burgers dataset
(identical generator call, identical seed), then renders a single comparison
sheet. The roster:

  - SGA, DLGA, DISCOVER -- always available (no optional asset).
  - PySR -- external baseline, needs the ``pysr`` extra.
  - PySINDy -- external baseline (native STLSQ over the kd term
                            library), needs the ``pysindy`` extra.
  - EqGPT -- pretrained GPT proposer, needs the ``.pt`` weights.
  - llm4ed -- LLM proposer, run here OFFLINE via a canned
                            provider (see the honesty note below).

Engines whose optional asset is missing SKIP gracefully (reported, never
silently dropped), so a bare install still produces a sheet from the four
always-on engines. The sheet has:

  - equations table: every discovered structure re-fit through
    ``kd.evaluate_terms`` (the stateless public entry: one shared
    finite-difference feature set + least squares for everyone) and
    rendered via the SymPy bridge -- so DISCOVER's algebraically nested
    form simplifies to its u*u_x equivalent and a spurious term shows up
    with its ~1e-16 re-fit coefficient in plain sight
  - per-engine normalized search progress (each engine's native score
    mapped to [0, 1] -- AIC vs GA fitness vs RL reward vs NMSE are
    different rulers, so raw values are deliberately never overlaid on
    one axis; ``kd.viz.plots.render_overlaid_convergence`` is
    the right tool when runs DO share a ruler, e.g. multi-seed batches
    of one algorithm)
  - unified NMSE bars: the apples-to-apples ranking on the one ruler.

Each engine reports its own score kind natively, so the only honest
cross-algorithm ranking is to re-fit every discovered structure on the
same derivative features with the same solver and the same metric. That
is exactly what the platform Evaluator provides.

HONESTY NOTE on llm4ed: its real search operator is a live LLM (network +
API key, non-deterministic). To keep this example offline and deterministic
we inject a CANNED provider that always proposes the Burgers structure --
so llm4ed's row is a PRESET answer, NOT a live search. It is tagged
"(preset)" wherever it appears and demonstrates only the unified re-fit
mechanics on a known-good structure, not search quality. For the real
llm4ed search story (offline replay tape or live backend) see
``examples/17_llm4ed.py``.

Burgers: u_t + u * u_x - 0.1 * u_xx = 0 (i.e., u_t = -u*u_x + 0.1*u_xx)

Runtime: ~6-12 min on CPU, engine mix dependent (DLGA trains its NN_1
surrogate; PySR boots a Julia runtime on first use; EqGPT loads a GPT and
proposes for a few generations; llm4ed's canned path is ~10 s). Budgets
below are demo-sized -- raise them for a serious recovery attempt (see each
algorithm's own example).

Reproducibility note: SGA/DLGA are seed-deterministic here. PySR's seed
alone is only WEAKLY reproducible (statistically similar, not
bit-identical runs -- two demo runs found u*u_x + u_xx vs n3(u)*u_x + u),
so this example pins it with
``extra_pysr_kwargs={"deterministic": True, "parallelism": "serial"}``
(slower; drop it for speed if you don't need a stable story). DISCOVER's
multithreaded torch path may still drift in the last digits. EqGPT, PySINDy
(native STLSQ has no RNG), and the canned llm4ed path are deterministic here.

Run: python examples/09_compare_algorithms.py
      open examples/out/09_compare/comparison.svg
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kd
from kd import DLGAConfig, EqGPTConfig, Llm4edConfig
from kd.core.expr import format_pde
from kd.llm import LLMRequest, LLMResponse
from kd.search import BEST_SCORE_KEY
from kd.search.eqgpt import resolve_asset_path

OUT_DIR = Path(__file__).parent / "out" / "09_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class OfflineProvider:
    """A canned ``kd.llm.LLMProvider`` for the offline llm4ed demo.

    Always proposes the Burgers right-hand-side STRUCTURE (``u*u_x + u_xx``)
    so the example stays network-free and deterministic. The ``<res>...</res>``
    wrapper is the payload format llm4ed's response parser expects. This is a
    PRESET answer, not a search -- see the module docstring's honesty note.
    """

    def prepare(self) -> None:
        return None

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="<res>u*u_x+u_xx</res>", model="offline", usage=None)


# 1. ONE dataset for all engines -- same generator call, same seed as
# examples 01 (SGA), 07 (DISCOVER) and 08 (DLGA).
dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")

# 2. Assemble the roster. In-house engines (SGA/DLGA/DISCOVER) and llm4ed's
# canned offline provider always run; PySR, PySINDy, and EqGPT need an
# optional asset and skip gracefully -- their reason lands in ``skipped``
# (reported below), never a silent drop.
MODELS: dict[str, kd.Model] = {
    "SGA": kd.Model(algorithm="sga", generations=30, population=15, seed=0),
    "DLGA": kd.Model(
        algorithm="dlga",
        generations=20,
        config=DLGAConfig.burgers_preset(
            pop_size=80,
            seed=0,
            surrogate_max_epochs=2000, # paper uses 50000; small = fast demo
        ),
    ),
    "DISCOVER": kd.Model(algorithm="discover", generations=20, seed=0),
}
skipped: dict[str, str] = {}

# PySR: external baseline, needs the optional ``pysr`` extra (Julia backend).
if importlib.util.find_spec("pysr") is not None:
    from kd import PySRConfig

    MODELS["PySR"] = kd.Model(
        algorithm="pysr",
        config=PySRConfig(
            terms=("u", "u_x", "u_xx"),
            niterations=40,
            seed=0,
            # Bit-stable runs (see the reproducibility note above).
            extra_pysr_kwargs={"deterministic": True, "parallelism": "serial"},
        ),
    )
else:
    skipped["PySR"] = "needs the `pysr` extra (uv sync --extra pysr); see examples/12"

# PySINDy: external baseline, native STLSQ over the kd term library, needs the
# optional ``pysindy`` extra. threshold=0.05 keeps the 0.1*u_xx term -- a 0.1
# threshold sits right on that coefficient and would prune it.
if importlib.util.find_spec("pysindy") is not None:
    from kd import PySINDyConfig

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

# EqGPT: pretrained generative GPT proposer, needs the ``.pt`` weights (NOT
# vendored). Probe with the SAME resolver the backend uses so this check
# cannot drift from the real load. ``burgers_preset()`` pins sparsity_alpha
# (D5: no universal default).
try:
    resolve_asset_path()
except FileNotFoundError:
    skipped["EqGPT"] = "needs pretrained weights (.pt); see examples/16_eqgpt.py"
else:
    MODELS["EqGPT"] = kd.Model(
        algorithm="eqgpt", generations=3, config=EqGPTConfig.burgers_preset()
    )

# llm4ed: OFFLINE canned provider -> a PRESET answer, not a live search.
# Included to show the unified re-fit on a known-good structure; tagged
# "(preset)" throughout so it is never mistaken for a searched result.
MODELS["llm4ed"] = kd.Model(
    algorithm="llm4ed",
    generations=3,
    config=Llm4edConfig(samples_per_epoch=4, max_llm_calls_per_propose=4),
    provider=OfflineProvider(),
)
PRESET_ENGINES = {"llm4ed"} # canned, not searched -> tagged in every view


def display_name(name: str) -> str:
    """Engine label for tables/plots: preset engines carry an explicit tag."""
    return f"{name} (preset)" if name in PRESET_ENGINES else name


if skipped:
    print("\nSkipped (optional asset missing):")
    for name, why in skipped.items():
        print(f" - {name}: {why}")

# Fit order is also RNG-safe: each plugin (re)seeds its own generators in
# prepare(), so DISCOVER's torch.manual_seed cannot leak into a later fit.
for name, model in MODELS.items():
    print(f"\n=== {display_name(name)} ===")
    model.fit(dataset)
    kind = model.result_.score_kind # AIC / fitness / reward / NMSE
    print(f"{name} best: {model.best_expr_} (native {kind}: {model.best_score_:.4g})")
    model.result_.save(OUT_DIR / f"{name.lower()}.json")

results = {name: m.result_ for name, m in MODELS.items()}

# 3. Unified re-evaluation -- the "one ruler" step. ``kd.evaluate_terms``
# (the stateless public entry) re-fits every discovered structure on the
# same finite-difference features with the same least-squares solver and
# the same NMSE. Coefficients are re-fit on that common feature set: the
# comparison ranks the discovered STRUCTURE, independent of each engine's
# internal scoring. The entry is fail-loud: a structure it cannot honestly
# fit raises (and is reported below) instead of coming back as a
# penalty-sentinel score.
#
# The structure is taken from ``final_eval.terms`` (platform IR for every
# engine), narrowed to ``selected_indices`` when a sparse solver reported
# a support subset (e.g. SGA's ``terms`` are its pruned candidate's theta
# columns and STRidge picks the nonzero subset; dense solvers report all
# terms with ``selected_indices=None``). ``best_expr_`` is NOT used here --
# it is the human-readable rendering (SGA's includes fitted coefficients
# and an ``u_t =`` prefix, which is display-only).
#
# Two honesty guards: (a) the ruler regresses against ``u_t`` -- an engine
# whose auto-selected LHS differs (DLGA may pick ``u_tt``) is excluded and
# annotated instead of being silently reinterpreted; (b) for DISCOVER the
# unified score roughly equals its native one (it already scores through
# the platform Evaluator) -- the ruler adds nothing new there, it just
# puts all engines on the same axis.


def discovered_structure(result: kd.ExperimentResult) -> list[str]:
    """Return the discovered term structure in platform IR ([] if invalid)."""
    ev = result.final_eval
    if ev.terms is None: # invalid final eval carries no term list
        return []
    if ev.selected_indices is not None:
        return [ev.terms[i] for i in ev.selected_indices]
    return list(ev.terms)


def canonical_terms(terms: list[str]) -> list[str]:
    """Return plugin-emitted platform funcall IR terms unchanged.

    llm4ed parses natural-math proposals inside the plugin and now exposes
    canonical IR (for example, ``mul(u, u_x)``) at ``final_eval.terms``.
    """
    return terms


# kd.evaluate_terms derives its LHS from dataset.lhs_order (not a fixed
# default); UNIFIED_LHS is this script's label for the ruler comparison.
UNIFIED_LHS = "u_t"

structures = {
    name: canonical_terms(discovered_structure(m.result_))
    for name, m in MODELS.items()
}
lhs_names = {
    name: (m.result_.final_eval.lhs_name or UNIFIED_LHS) for name, m in MODELS.items()
}

# Per-engine unified result. An engine that cannot be put on the ruler gets
# an explanation string in ``exclusions`` instead of a silent drop.
unified: dict[str, kd.EvaluationResult | None] = {}
exclusions: dict[str, str] = {}
for name in MODELS:
    if lhs_names[name] != UNIFIED_LHS:
        unified[name] = None
        exclusions[name] = f"LHS={lhs_names[name]} != {UNIFIED_LHS} -- not comparable"
        continue
    if not structures[name]:
        # This guard also keeps evaluate_terms' bare ValueError (raised on an
        # empty term list) out of the except below, which deliberately catches
        # only the two fail-loud contract errors.
        unified[name] = None
        exclusions[name] = "engine reported no valid structure"
        continue
    try:
        # max_order=3: DLGA's default library includes u_xxx -- the shared
        # ruler must be able to serve every term any engine may have selected.
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

# 4. One comparison sheet: equations table (full width) over two panels.
# Every exclusion is reported as a warning -- nothing is dropped silently.
warnings: list[str] = []


def equation_cell(name: str) -> str:
    """Mathtext equation from the unified re-fit (SymPy-simplified)."""
    ev = unified[name]
    if ev is None: # carry the exclusion reason into the table, truncated
        reason = exclusions[name]
        return reason if len(reason) <= 90 else reason[:87] + "..."
    eq = format_pde(structures[name], ev.coefficients, lhs=UNIFIED_LHS, sig_figs=3)
    return f"${eq.latex}$"


fig = plt.figure(figsize=(13, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[0.70, 1.05], hspace=0.10, wspace=0.30)

# Row 1: equations table, full width. Coefficients are the unified re-fit
# values; the SymPy bridge simplifies nested forms (DISCOVER's div-based
# term collapses to u*u_x) so structures are visually comparable.
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
    ok = ev is not None # a returned result is valid (fail-loud contract)
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
    # Loud footnote: the canned engine is a preset structure, not a search.
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

# Row 2 left: per-engine normalized search progress. Native scores live on
# different rulers, so each curve is normalized to its own [first, last]
# span -- shapes are comparable, values deliberately are not.
ax_p = fig.add_subplot(gs[1, 0])
for name, result in results.items():
    series = [float(s) for s in result.recorder.get(BEST_SCORE_KEY)]
    if not series:
        warnings.append(f"{name}: no {BEST_SCORE_KEY} series, progress curve skipped")
        continue
    if len(series) == 1 or series[-1] == series[0]:
        prog = [1.0] * len(series) # converged at (or before) iteration 0
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

# Row 2 right: unified NMSE bars with value labels. Linear axis from zero:
# near-equal bars ARE the story (every engine ties on the one ruler).
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

# 5. Render a standalone HTML report for one engine that would otherwise lack
# a per-engine example report -- PySR when present, else the first available
# optional engine. Skips cleanly if only the always-on engines ran.
report_engine = next((n for n in ("PySR", "EqGPT") if n in MODELS), None)
if report_engine is not None:
    engine = kd.VizEngine(output_dir=OUT_DIR / f"{report_engine.lower()}_report")
    report = engine.render_all(
        MODELS[report_engine].result_,
        algorithm=MODELS[report_engine].algorithm_,
        dataset=dataset,
    )
    print(f"{report_engine} report: {report.report}")
