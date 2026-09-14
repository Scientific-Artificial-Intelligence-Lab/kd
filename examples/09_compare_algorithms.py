"""Example 09 - Many engines, one dataset, one ruler.

kd ships seven discovery engines behind the one-line ``kd.Model`` facade. This
example runs whichever are available on the SAME synthetic Burgers dataset
(identical generator call, identical seed), then renders a single comparison
sheet. The roster:

  - SGA, DLGA, DISCOVER -- always available (no optional asset).
  - PySR -- external baseline, needs the ``pysr`` extra.
  - PySINDy -- native STLSQ over the kd term library, installed
                            with kd.
  - EqGPT -- pretrained GPT proposer, needs the ``.pt`` weights.
  - llm4ed -- LLM proposer, run here OFFLINE via a canned
                            provider (see the honesty note below).

Engines whose optional asset is missing SKIP gracefully (reported, never
silently dropped), so a bare install still produces a sheet from the five
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

import kd
from kd import DLGAConfig, EqGPTConfig, Llm4edConfig, PySINDyConfig
from kd.evaluate import compare_results
from kd.llm import LLMRequest, LLMResponse
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
# canned offline provider and PySINDy always run; PySR and EqGPT need an
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
    skipped["PySR"] = 'needs the `pysr` extra (pip install "sail-kd[pysr]"); see examples/12'

# PySINDy: external baseline, native STLSQ over the kd term library (installed
# with kd). threshold=0.05 keeps the 0.1*u_xx term -- a 0.1 threshold sits
# right on that coefficient and would prune it.
MODELS["PySINDy"] = kd.Model(
    algorithm="pysindy",
    config=PySINDyConfig(
        terms=("u", "u_x", "u_xx", "mul(u, u_x)"),
        threshold=0.05,
        seed=0,
    ),
)

# EqGPT: pretrained generative GPT proposer, needs the ``.pt`` weights. Probe
# with the SAME resolver the backend uses, in offline mode: a comparison sheet
# must not start the 151.7 MB Hub download, so EqGPT joins only when the
# weights are already local (``$KD_EQGPT_ASSET_DIR`` or a cached download).
# ``burgers_preset()`` pins sparsity_alpha (D5: no universal default).
try:
    resolve_asset_path(offline=True)
except FileNotFoundError:
    skipped["EqGPT"] = (
        "pretrained weights not local; run examples/16_eqgpt.py once to "
        "download them, or set KD_EQGPT_ASSET_DIR"
    )
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
    print(f"{name} best: {model.result_.equation} (native {kind}: {model.best_score_:.4g})")
    model.result_.save(OUT_DIR / f"{name.lower()}.json")

results = {name: m.result_ for name, m in MODELS.items()}

# 3. Refit only each discovered law's active terms against the same u_t
# finite-difference features. Native scores and original coefficients stay
# in the saved results. LHS mismatches and refused refits are named in the
# comparison's exclusions, never silently put on the u_t ruler.
comparison = compare_results(
    dataset, {display_name(name): result for name, result in results.items()}
)
print(comparison)

# 4. The library draws the refitted equations, per-run normalized search
# progress, and unified NMSE bars. The preset row retains its label in
# every panel; it demonstrates refitting, not search quality.
sheet = comparison.render(
    OUT_DIR / "comparison.svg",
    note="(preset) = canned known-good proposals; demonstrates the refit, not search quality",
)
print(f"\nComparison sheet: {sheet.figures[0]}")
for warning in sheet.warnings:
    print(f"[viz warning] {warning}")

# 5. Each available optional engine gets the same complete report entry.
for name in ("PySR", "EqGPT"):
    if name in MODELS:
        report = MODELS[name].report(OUT_DIR / f"{name.lower()}_report")
        print(f"{name} report: {report.report}")
