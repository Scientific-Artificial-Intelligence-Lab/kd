"""KD PDE Discovery — Gradio Web Interface.

Designed for deployment on the Bohrium platform.

Usage:
    python app.py
"""

import os
import warnings
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import gradio as gr

from kd.dataset import load_pde
from kd.dataset._registry import PDE_REGISTRY
from kd.viz.core import VizRequest, render, configure


# ── GPU offloading ────────────────────────────────────────────

GPU_MODELS = {"KD_DLGA", "KD_DSCV_SPR", "KD_EqGPT"}

try:
    from kd.job import submit_gpu_job, check_job, download_result, find_active_job
    _HAS_BOHRIUM = True
except ImportError:
    _HAS_BOHRIUM = False


def _is_bohrium_env(request):
    """Return True when running on Bohrium platform with valid auth cookies."""
    if not _HAS_BOHRIUM or request is None:
        return False
    try:
        return "appAccessKey" in (request.headers.get("cookie") or "")
    except Exception:
        return False


# ── Constants ──────────────────────────────────────────────────

MODEL_KEYS = {
    "KD_SGA": "sga",
    "KD_DLGA": "dlga",
    "KD_DSCV": "discover",
    "KD_DSCV_SPR": "discover_spr",
    "KD_EqGPT": "eqgpt",
    "KD_Discover_Regression": "regression",
}

ACTIVE_DATASETS = sorted([
    name for name, info in PDE_REGISTRY.items()
    if info.get("status") != "pending"
])

# Regression datasets (separate from PDE registry)
from kd.dataset import load_regression
_REGRESSION_DATASETS = ["tlc_cc_t1", "tlc_cc_t2"]
ALL_DATASETS = ACTIVE_DATASETS + _REGRESSION_DATASETS

# Operator presets per model
DSCV_BINARY_DEFAULT = "add, mul, diff"
DSCV_UNARY_DEFAULT = "n2"
SPR_BINARY_DEFAULT = "add_t, mul_t, div_t, diff_t, diff2_t"
SPR_UNARY_DEFAULT = "n2_t"
DLGA_OPS_DEFAULT = "u, u_x, u_xx, u_xxx"
REG_BINARY_DEFAULT = "add, sub, mul, div"
REG_UNARY_DEFAULT = "inv"

# Viz intents available for exploration
VIZ_INTENTS = [
    "equation",
    "search_evolution",
    "training_curve",
    "validation_curve",
    "optimization",
    "parity",
    "field_comparison",
    "residual",
    "derivative_relationships",
    "density",
    "tree",
]


# ── Helpers ────────────────────────────────────────────────────

def get_compatible_models(dataset_name):
    """Return model display names compatible with a given dataset."""
    if not dataset_name:
        return []
    # Regression datasets → only regression model
    if dataset_name in _REGRESSION_DATASETS:
        return ["KD_Discover_Regression"]
    info = PDE_REGISTRY.get(dataset_name, {})
    models_map = info.get("models", {})
    return [
        display for display, key in MODEL_KEYS.items()
        if models_map.get(key, False)
    ]


def on_dataset_change(dataset_name):
    """Update model dropdown when dataset changes."""
    models = get_compatible_models(dataset_name)
    default = models[0] if models else None
    return gr.update(choices=models, value=default)


def on_model_change(model_name):
    """Show/hide parameter groups and auto-switch operator presets."""
    sga_vis = model_name == "KD_SGA"
    dlga_vis = model_name == "KD_DLGA"
    dscv_vis = model_name in ("KD_DSCV", "KD_DSCV_SPR")
    spr_vis = model_name == "KD_DSCV_SPR"
    eqgpt_vis = model_name == "KD_EqGPT"
    reg_vis = model_name == "KD_Discover_Regression"

    if model_name == "KD_DSCV_SPR":
        binary_val = SPR_BINARY_DEFAULT
        unary_val = SPR_UNARY_DEFAULT
    else:
        binary_val = DSCV_BINARY_DEFAULT
        unary_val = DSCV_UNARY_DEFAULT

    return (
        gr.update(visible=sga_vis),    # sga_group
        gr.update(visible=dlga_vis),   # dlga_group
        gr.update(visible=dscv_vis),   # dscv_group
        gr.update(visible=spr_vis),    # spr_group
        gr.update(visible=eqgpt_vis),  # eqgpt_group
        gr.update(visible=reg_vis),    # reg_group
        gr.update(value=binary_val),   # dscv_binary
        gr.update(value=unary_val),    # dscv_unary
    )


def _parse_ops(text):
    """Parse comma-separated operator string into list."""
    return [op.strip() for op in text.split(",") if op.strip()]


def _get_equation_text(model, model_name, result=None):
    """Extract equation string from model/result depending on model type."""
    if model_name == "KD_SGA":
        return model.equation_latex()
    elif model_name == "KD_DLGA":
        return getattr(model, "eq_latex", None) or "(equation renderer unavailable)"
    elif model_name == "KD_EqGPT":
        if result is None:
            return "(no equation found)"
        best = result.get("best_equation", "(no equation found)")
        best_r = result.get("best_reward", 0.0)
        lines = [f"Best: {best}  (reward={best_r:.4f})", "", "Top-10:"]
        for i, (eq, rw) in enumerate(
            zip(result.get("equations", []), result.get("rewards", []))
        ):
            lines.append(f"  {i+1:2d}. [reward={rw:.4f}]  {eq}")
        return "\n".join(lines)
    elif model_name == "KD_Discover_Regression":
        if result is not None:
            return result.get("expression_named", result.get("expression", "(no expression found)"))
        return "(no expression found)"
    elif result is not None:
        return result.get("expression", "(no expression found)")
    return "(unknown)"


def _get_model_capabilities(model):
    """Return available viz capabilities for the current model."""
    from kd.viz.core import list_capabilities
    if model is None:
        return []
    caps = list(list_capabilities(model))
    # spr_* intents only work for KD_DSCV_SPR (PINN mode)
    try:
        from kd.model.kd_discover import KD_Discover_SPR as _SPR
        is_spr = isinstance(model, _SPR)
    except ImportError:
        is_spr = False
    if not is_spr:
        caps = [c for c in caps if not c.startswith("spr_")]
    return sorted(caps)


def _render_equation_fig(model, equation_text):
    """Try kd.viz first; fall back to plain matplotlib text rendering."""
    import matplotlib.pyplot as plt

    # Try the adapter-based renderer
    try:
        viz_result = render(VizRequest("equation", model))
        if viz_result.figure is not None and not viz_result.warnings:
            return viz_result.figure
    except Exception:
        pass

    # Fallback: render equation text onto a matplotlib figure
    if not equation_text or equation_text.startswith("("):
        return None
    # Strip existing $ delimiters to avoid double-wrapping ($$...$$)
    eq_clean = equation_text.strip().strip("$").strip()
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    try:
        ax.text(0.5, 0.5, f"${eq_clean}$", fontsize=18,
                ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
    except Exception:
        ax.clear()
        ax.axis("off")
        ax.text(0.5, 0.5, equation_text, fontsize=14,
                ha="center", va="center", transform=ax.transAxes, family="monospace")
        fig.tight_layout()
    return fig


# ── GPU job tracking ──────────────────────────────────────────

_GPU_JOB_FILE = "/tmp/.kd_gpu_job_id"


def _save_job_id(job_id):
    """Persist job_id to file so it survives state loss."""
    try:
        with open(_GPU_JOB_FILE, "w") as f:
            f.write(str(job_id) if job_id else "")
        print(f"[kd] _save_job_id({job_id}) → wrote to {_GPU_JOB_FILE}")
    except Exception as e:
        print(f"[kd] _save_job_id({job_id}) FAILED: {e}")


def _load_job_id():
    """Load persisted job_id from file."""
    try:
        with open(_GPU_JOB_FILE) as f:
            val = f.read().strip()
            result = int(val) if val else None
        print(f"[kd] _load_job_id() → {result}")
        return result
    except Exception as e:
        print(f"[kd] _load_job_id() FAILED: {e}")
        return None


# ── Training ───────────────────────────────────────────────────

# 10-value output tuple: (eq_text, eq_plot, model_state, viz_dd,
#     job_state, job_status, check_btn, gpu_timer, browser_job_id, recover_row)
_NO_GPU = (None, gr.update(visible=False), gr.update(visible=False),
           gr.Timer(active=False), None, gr.update(visible=False))


def run_training(
    dataset_name, model_name,
    # SGA
    sga_run, sga_num, sga_depth, sga_seed,
    # DLGA
    dlga_ops, dlga_epi, dlga_max_iter, dlga_sample,
    # DSCV / SPR shared
    dscv_binary, dscv_unary, dscv_batch, dscv_epochs, dscv_seed,
    # SPR extra
    spr_sample_ratio, spr_epochs,
    # EqGPT
    eqgpt_epochs=5, eqgpt_samples=400, eqgpt_cases="N only (12 cases)",
    eqgpt_seed=0,
    # Regression
    reg_binary="add, sub, mul, div", reg_unary="inv",
    reg_iterations=50, reg_samples=500, reg_seed=1,
    reg_parsimony=0.005,
    # Gradio injects this automatically
    request: gr.Request = None,
):
    """Core training function — routes to local or GPU offline execution."""
    if not dataset_name or not model_name:
        return ("Please select dataset and model first.",
                None, None, gr.update()) + _NO_GPU

    # GPU models on Bohrium → submit as offline task
    if model_name in GPU_MODELS and _is_bohrium_env(request):
        return _submit_gpu_training(
            dataset_name, model_name,
            dlga_ops, dlga_epi, dlga_max_iter, dlga_sample,
            dscv_binary, dscv_unary, dscv_batch, dscv_epochs, dscv_seed,
            spr_sample_ratio, spr_epochs,
            eqgpt_epochs, eqgpt_samples, eqgpt_cases, eqgpt_seed,
            request,
        )

    # All other cases (including local dev) → run in-process
    return _run_local(
        dataset_name, model_name,
        sga_run, sga_num, sga_depth, sga_seed,
        dlga_ops, dlga_epi, dlga_max_iter, dlga_sample,
        dscv_binary, dscv_unary, dscv_batch, dscv_epochs, dscv_seed,
        spr_sample_ratio, spr_epochs,
        eqgpt_epochs, eqgpt_samples, eqgpt_cases, eqgpt_seed,
        reg_binary, reg_unary, reg_iterations, reg_samples, reg_seed,
        reg_parsimony,
    )


def _run_local(
    dataset_name, model_name,
    sga_run, sga_num, sga_depth, sga_seed,
    dlga_ops, dlga_epi, dlga_max_iter, dlga_sample,
    dscv_binary, dscv_unary, dscv_batch, dscv_epochs, dscv_seed,
    spr_sample_ratio, spr_epochs,
    eqgpt_epochs=5, eqgpt_samples=400, eqgpt_cases="N only (12 cases)",
    eqgpt_seed=0,
    reg_binary="add, sub, mul, div", reg_unary="inv",
    reg_iterations=50, reg_samples=500, reg_seed=1,
    reg_parsimony=0.005,
):
    """Run model training in the current process (CPU or local GPU)."""
    configure(save_dir=None)

    try:
        model = None
        result = None

        if model_name == "KD_Discover_Regression":
            from kd.model.kd_discover_regression import KD_Discover_Regression
            X, y, meta = load_regression(dataset_name)
            model = KD_Discover_Regression(
                binary_operators=_parse_ops(reg_binary),
                unary_operators=_parse_ops(reg_unary),
                n_iterations=int(reg_iterations),
                n_samples_per_batch=int(reg_samples),
                seed=int(reg_seed),
                config_out={"task": {"parsimony_coeff": float(reg_parsimony)}},
            )
            result = model.fit(X, y, var_names=meta["var_names"], verbose=True)

            # Compute R² for display
            y_pred = model.predict(X)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            eq_text = _get_equation_text(model, model_name, result)
            eq_text += f"\n\nR² = {r2:.6f}  |  MSE = {result['mse']:.4f}  |  Reward = {result['reward']:.6f}"
            eq_text += f"\nTarget: {meta['target_name']}  |  Variables: {meta['var_names']}"
            fig = _render_equation_fig(model, _get_equation_text(model, model_name, result))

            return (eq_text, fig, model, gr.update(choices=[], value=None)) + _NO_GPU

        elif model_name == "KD_EqGPT":
            from kd.model.kd_eqgpt import KD_EqGPT
            case_filter = "N" if "N only" in str(eqgpt_cases) else "all"
            model = KD_EqGPT(
                optimize_epochs=int(eqgpt_epochs),
                samples_per_epoch=int(eqgpt_samples),
                case_filter=case_filter,
                seed=int(eqgpt_seed),
            )
            result = model.fit_pretrained()
        else:
            dataset = load_pde(dataset_name)

            if model_name == "KD_SGA":
                from kd.model.kd_sga import KD_SGA
                model = KD_SGA(
                    sga_run=int(sga_run),
                    num=int(sga_num),
                    depth=int(sga_depth),
                    seed=int(sga_seed),
                )
                model.fit_dataset(dataset)

            elif model_name == "KD_DLGA":
                from kd.model.kd_dlga import KD_DLGA
                model = KD_DLGA(
                    operators=_parse_ops(dlga_ops),
                    epi=float(dlga_epi),
                    input_dim=2,
                    verbose=True,
                    max_iter=int(dlga_max_iter),
                )
                sample = int(dlga_sample) if dlga_sample else None
                model.fit_dataset(dataset, sample=sample)

            elif model_name == "KD_DSCV":
                from kd.model.kd_discover import KD_Discover as KD_DSCV
                model = KD_DSCV(
                    binary_operators=_parse_ops(dscv_binary),
                    unary_operators=_parse_ops(dscv_unary),
                    n_samples_per_batch=int(dscv_batch),
                    seed=int(dscv_seed),
                )
                np.random.seed(int(dscv_seed))
                result = model.fit_dataset(dataset, n_epochs=int(dscv_epochs))

            elif model_name == "KD_DSCV_SPR":
                from kd.model.kd_discover import KD_Discover_SPR as KD_DSCV_SPR
                model = KD_DSCV_SPR(
                    binary_operators=_parse_ops(dscv_binary),
                    unary_operators=_parse_ops(dscv_unary),
                    n_samples_per_batch=int(dscv_batch),
                    seed=int(dscv_seed),
                )
                np.random.seed(int(dscv_seed))
                result = model.fit_dataset(
                    dataset,
                    n_epochs=int(spr_epochs),
                    sample_ratio=float(spr_sample_ratio),
                )
            else:
                return (f"Unknown model: {model_name}",
                        None, None, gr.update()) + _NO_GPU

        equation = _get_equation_text(model, model_name, result)
        fig = _render_equation_fig(model, equation)
        caps = _get_model_capabilities(model) if model_name != "KD_EqGPT" else []
        viz_update = gr.update(choices=caps, value=caps[0] if caps else None)

        return (equation, fig, model, viz_update) + _NO_GPU

    except Exception as e:
        return (f"Training error:\n{e}\n\n{traceback.format_exc()}",
                None, None, gr.update()) + _NO_GPU


def _submit_gpu_training(
    dataset_name, model_name,
    dlga_ops, dlga_epi, dlga_max_iter, dlga_sample,
    dscv_binary, dscv_unary, dscv_batch, dscv_epochs, dscv_seed,
    spr_sample_ratio, spr_epochs,
    eqgpt_epochs=5, eqgpt_samples=400, eqgpt_cases="N only (12 cases)",
    eqgpt_seed=0,
    request=None,
):
    """Submit training to Bohrium GPU offline task."""
    if model_name == "KD_DLGA":
        runner_model = "dlga"
        params = {
            "operators": dlga_ops,
            "epi": float(dlga_epi),
            "max_iter": int(dlga_max_iter),
            "sample": int(dlga_sample) if dlga_sample else None,
        }
    elif model_name == "KD_EqGPT":
        runner_model = "eqgpt"
        case_filter = "N" if "N only" in str(eqgpt_cases) else "all"
        params = {
            "optimize_epochs": int(eqgpt_epochs),
            "samples_per_epoch": int(eqgpt_samples),
            "case_filter": case_filter,
            "seed": int(eqgpt_seed),
        }
    else:  # KD_DSCV_SPR
        runner_model = "dscv_spr"
        params = {
            "binary_operators": dscv_binary,
            "unary_operators": dscv_unary,
            "n_samples_per_batch": int(dscv_batch),
            "seed": int(dscv_seed),
            "n_epochs": int(spr_epochs),
            "sample_ratio": float(spr_sample_ratio),
        }

    try:
        job_id = submit_gpu_job(runner_model, dataset_name, params, request)
        _save_job_id(job_id)
        return (
            f"GPU task submitted — Job ID: {job_id}\n"
            f"⚠ Please save this ID: {job_id}\n"
            "Auto-polling every 30s. You can also click 'Check GPU Job'.",
            None, None, gr.update(),
            job_id,
            gr.update(visible=True, value=f"Job {job_id}: Submitted"),
            gr.update(visible=True),
            gr.Timer(active=True),
            job_id,                        # → browser_job_id (LocalStorage)
            gr.update(visible=False),      # → hide recover_row
        )
    except Exception as e:
        return (f"GPU submit error:\n{e}\n\n{traceback.format_exc()}",
                None, None, gr.update()) + _NO_GPU


def check_gpu_status(job_id, browser_job_id=None, request: gr.Request = None):
    """Poll GPU job status; download results when finished.

    This function is called by both the manual Check button and the
    auto-polling timer.  It must NEVER raise — any unhandled exception
    would cause Gradio to return a non-JSON response, which the browser
    shows as ``SyntaxError: Unexpected token '<'``.
    """
    try:
        return _check_gpu_status_inner(job_id, browser_job_id, request)
    except Exception as exc:
        # Last-resort safety net — keep polling with existing job_id
        safe_id = job_id or _load_job_id() or _to_int(browser_job_id)
        if safe_id:
            return (
                f"Status check error: {exc}\nWill retry with job {safe_id}...",
                None, None, gr.update(),
                safe_id,
                gr.update(visible=True, value=f"Retrying... ({type(exc).__name__})"),
                gr.update(visible=True),
                gr.Timer(active=True),
                safe_id,                       # keep browser_job_id
                gr.update(visible=False),
            )
        # All recovery failed — show recover input
        return ("GPU job ID lost. Please enter your Job ID below to resume.",
                None, None, gr.update(),
                None,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.Timer(active=False),
                None,
                gr.update(visible=True),       # show recover_row
                )


def _to_int(val):
    """Safely convert to int, return None on failure."""
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


def _check_gpu_status_inner(job_id, browser_job_id, request):
    """Inner implementation of GPU status check."""
    # Recovery chain: Gradio state → file → browser LocalStorage → platform API
    if not job_id:
        job_id = _load_job_id()
        if job_id:
            print(f"[kd] Recovered job_id={job_id} from file")
    if not job_id:
        job_id = _to_int(browser_job_id)
        if job_id:
            _save_job_id(job_id)
            print(f"[kd] Recovered job_id={job_id} from browser LocalStorage")
    if not job_id and _HAS_BOHRIUM and request:
        try:
            job_id = find_active_job(request)
            if job_id:
                _save_job_id(job_id)
                print(f"[kd] Recovered job_id={job_id} from platform API")
        except Exception as e:
            print(f"[kd] find_active_job failed: {e}")
    if not job_id:
        # Show recover input for user to enter job_id manually
        return ("GPU job ID lost. Please enter your Job ID below to resume.",
                None, None, gr.update(),
                None,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.Timer(active=False),
                None,
                gr.update(visible=True),
                )

    try:
        status = check_job(job_id, request)
    except Exception as e:
        # Keep polling on transient errors
        return (
            f"Status check failed: {e}", None, None, gr.update(),
            job_id,
            gr.update(visible=True, value=f"Error: {e}"),
            gr.update(visible=True),
            gr.Timer(active=True),
            job_id, gr.update(visible=False),
        )

    if status["failed"]:
        _save_job_id(None)
        return (
            f"Job {job_id} failed: {status['status_text']}",
            None, None, gr.update(),
            None,
            gr.update(visible=True, value=f"Job {job_id}: {status['status_text']}"),
            gr.update(visible=False),
            gr.Timer(active=False),
            None, gr.update(visible=False),
        )

    if not status["finished"]:
        elapsed = status.get("spend_time", "")
        return (
            f"Job {job_id} still running... ({status['status_text']})",
            None, None, gr.update(),
            job_id,
            gr.update(visible=True,
                      value=f"Job {job_id}: {status['status_text']}  {elapsed}"),
            gr.update(visible=True),
            gr.Timer(active=True),
            job_id, gr.update(visible=False),
        )

    # ── Job finished → download & display ──
    _save_job_id(None)
    try:
        result_data, save_dir = download_result(job_id, request)
        equation = result_data.get("equation", "(unknown)")

        eq_fig = None
        eq_img = os.path.join(save_dir, "equation.png")
        if os.path.exists(eq_img):
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            img = mpimg.imread(eq_img)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            eq_fig = fig

        elapsed_s = result_data.get("elapsed_seconds", "?")
        return (
            equation, eq_fig, None,
            gr.update(choices=[], value=None),
            None,
            gr.update(visible=True, value=f"Job {job_id}: Finished ({elapsed_s}s)"),
            gr.update(visible=False),
            gr.Timer(active=False),
            None, gr.update(visible=False),
        )
    except Exception as e:
        return (f"Download failed: {e}",
                None, None, gr.update()) + _NO_GPU


# ── Visualization ──────────────────────────────────────────────

def run_viz(viz_type, model_state):
    """Generate a visualization for the trained model."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import tempfile

    if model_state is None:
        return None, "Please train a model first."

    # Use a temp dir so adapters that save-to-file still work
    with tempfile.TemporaryDirectory() as tmpdir:
        configure(save_dir=tmpdir)
        try:
            viz_result = render(VizRequest(viz_type, model_state))
            if viz_result.warnings:
                return None, "\n".join(viz_result.warnings)

            # Prefer in-memory figure
            if viz_result.figure is not None:
                return viz_result.figure, f"'{viz_type}' generated."

            # Fallback: read saved image file
            if viz_result.paths:
                img = mpimg.imread(str(viz_result.paths[0]))
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout(pad=0)
                return fig, f"'{viz_type}' generated."

            return None, f"No output for '{viz_type}'."
        except Exception as e:
            return None, f"Visualization error: {e}"
        finally:
            configure(save_dir=None)


# ── Gradio UI ──────────────────────────────────────────────────

def build_app():
    with gr.Blocks(
        title="KD - PDE Discovery",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown(
            "# KD - PDE Discovery Platform\n"
            "Automatically discover symbolic PDE expressions from numerical data."
        )

        model_state = gr.State(None)
        job_state = gr.State(None)

        with gr.Tabs():
            # ── Tab 1: Training ──────────────────────────
            with gr.TabItem("Training"):
                with gr.Row():
                    # Left: controls
                    with gr.Column(scale=1):
                        _init_ds = "burgers" if "burgers" in ALL_DATASETS else ALL_DATASETS[0]
                        _init_models = get_compatible_models(_init_ds)
                        dataset_dd = gr.Dropdown(
                            choices=ALL_DATASETS,
                            value=_init_ds,
                            label="Dataset",
                        )
                        model_dd = gr.Dropdown(
                            choices=_init_models,
                            value=_init_models[0] if _init_models else None,
                            label="Model",
                        )

                        # -- SGA params --
                        _init_model = _init_models[0] if _init_models else None
                        with gr.Group(visible=(_init_model == "KD_SGA")) as sga_group:
                            gr.Markdown("**SGA Parameters**")
                            sga_run = gr.Number(value=3, label="SGA runs", precision=0)
                            sga_num = gr.Number(value=10, label="Population size", precision=0)
                            sga_depth = gr.Number(value=3, label="Tree depth", precision=0)
                            sga_seed = gr.Number(value=0, label="Seed", precision=0)

                        # -- DLGA params --
                        with gr.Group(visible=False) as dlga_group:
                            gr.Markdown("**DLGA Parameters**")
                            dlga_ops = gr.Textbox(
                                value=DLGA_OPS_DEFAULT,
                                label="Operators (comma-separated)",
                                info="Available: u, u_x, u_xx, u_xxx, u_tt",
                            )
                            dlga_epi = gr.Number(value=0.1, label="Sparsity penalty (epi)")
                            dlga_max_iter = gr.Number(value=3000, label="NN max iterations", precision=0)
                            dlga_sample = gr.Number(value=500, label="Sample points", precision=0)

                        # -- DSCV / SPR shared params --
                        with gr.Group(visible=False) as dscv_group:
                            gr.Markdown("**DSCV Parameters**")
                            dscv_binary = gr.Textbox(
                                value=DSCV_BINARY_DEFAULT,
                                label="Binary operators",
                                info="DSCV: add, mul, diff ... | SPR: add_t, mul_t, diff_t ...",
                            )
                            dscv_unary = gr.Textbox(
                                value=DSCV_UNARY_DEFAULT,
                                label="Unary operators",
                                info="DSCV: n2, n3, sin ... | SPR: n2_t ...",
                            )
                            dscv_batch = gr.Number(value=200, label="Samples per batch", precision=0)
                            dscv_epochs = gr.Number(value=5, label="Search epochs", precision=0)
                            dscv_seed = gr.Number(value=0, label="Seed", precision=0)

                        # -- SPR extra params --
                        with gr.Group(visible=False) as spr_group:
                            gr.Markdown("**SPR (PINN) Extra Parameters**")
                            spr_sample_ratio = gr.Slider(
                                0.01, 1.0, value=0.1, step=0.01,
                                label="Observation ratio (sample_ratio)",
                            )
                            spr_epochs = gr.Number(value=5, label="PINN search epochs", precision=0)

                        # -- EqGPT params --
                        with gr.Group(visible=False) as eqgpt_group:
                            gr.Markdown("**EqGPT Parameters**")
                            gr.Markdown(
                                "_Uses pre-trained GPT + surrogate models"
                                " on wave-breaking data._"
                            )
                            eqgpt_epochs = gr.Number(
                                value=5, label="Optimize epochs", precision=0,
                            )
                            eqgpt_samples = gr.Number(
                                value=400, label="Samples per epoch", precision=0,
                            )
                            eqgpt_cases = gr.Dropdown(
                                choices=["N only (12 cases)", "All (23 cases)"],
                                value="N only (12 cases)",
                                label="Case filter",
                            )
                            eqgpt_seed = gr.Number(
                                value=0, label="Seed", precision=0,
                            )

                        # -- Regression params --
                        with gr.Group(visible=False) as reg_group:
                            gr.Markdown("**Symbolic Regression Parameters**")
                            gr.Markdown(
                                "_Discover equations from tabular data (X → y)._"
                            )
                            reg_binary = gr.Textbox(
                                value=REG_BINARY_DEFAULT,
                                label="Binary operators",
                                info="add, sub, mul, div",
                            )
                            reg_unary = gr.Textbox(
                                value=REG_UNARY_DEFAULT,
                                label="Unary operators",
                                info="inv, sin, cos, sqrt, n2, n3, ...",
                            )
                            reg_iterations = gr.Number(
                                value=50, label="Search iterations", precision=0,
                            )
                            reg_samples = gr.Number(
                                value=500, label="Samples per batch", precision=0,
                            )
                            reg_parsimony = gr.Number(
                                value=0.005, label="Parsimony coefficient",
                                info="Higher = simpler equations",
                            )
                            reg_seed = gr.Number(value=1, label="Seed", precision=0)

                        with gr.Row():
                            train_btn = gr.Button("Start Training", variant="primary", size="lg")
                            cancel_btn = gr.Button("Cancel", variant="stop", size="lg")

                    # Right: results
                    with gr.Column(scale=2):
                        eq_text = gr.Textbox(label="Discovered Equation", lines=4, interactive=False)
                        eq_plot = gr.Plot(label="Equation Visualization")
                        job_status = gr.Textbox(
                            label="GPU Job Status", interactive=False, visible=False,
                        )
                        check_btn = gr.Button(
                            "Check GPU Job", visible=False, variant="secondary",
                        )
                        gpu_timer = gr.Timer(30, active=False)
                        browser_job_id = gr.BrowserState(None, storage_key="kd_gpu_job_id")
                        with gr.Row(visible=False) as recover_row:
                            recover_input = gr.Number(
                                label="Enter Job ID to resume",
                                precision=0,
                            )
                            recover_btn = gr.Button("Resume Tracking", variant="primary")

            # ── Tab 2: Visualization ─────────────────────
            with gr.TabItem("Visualization"):
                with gr.Row():
                    viz_dd = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Plot type (train a model first)",
                    )
                    viz_btn = gr.Button("Generate", variant="primary")
                viz_plot = gr.Plot(label="Result")
                viz_msg = gr.Textbox(label="Status", interactive=False)

        # ── Event Wiring ─────────────────────────────────

        # Initialize model list and parameter panel on page load.
        # Two chained .load() calls: first updates the model dropdown,
        # second shows the correct parameter group for the default model.
        _model_change_outputs = [sga_group, dlga_group, dscv_group, spr_group,
                                 eqgpt_group, reg_group, dscv_binary, dscv_unary]
        app.load(on_dataset_change, inputs=[dataset_dd], outputs=[model_dd])
        app.load(on_model_change, inputs=[model_dd], outputs=_model_change_outputs)

        dataset_dd.change(on_dataset_change, inputs=[dataset_dd], outputs=[model_dd])

        model_dd.change(
            on_model_change,
            inputs=[model_dd],
            outputs=_model_change_outputs,
        )

        _train_outputs = [
            eq_text, eq_plot, model_state, viz_dd,
            job_state, job_status, check_btn, gpu_timer,
            browser_job_id, recover_row,
        ]

        train_event = train_btn.click(
            run_training,
            inputs=[
                dataset_dd, model_dd,
                sga_run, sga_num, sga_depth, sga_seed,
                dlga_ops, dlga_epi, dlga_max_iter, dlga_sample,
                dscv_binary, dscv_unary, dscv_batch, dscv_epochs, dscv_seed,
                spr_sample_ratio, spr_epochs,
                eqgpt_epochs, eqgpt_samples, eqgpt_cases, eqgpt_seed,
                reg_binary, reg_unary, reg_iterations, reg_samples, reg_seed,
                reg_parsimony,
            ],
            outputs=_train_outputs,
        )

        cancel_btn.click(
            fn=lambda: ("Training cancelled.", None, None, gr.update(),
                        None, gr.update(visible=False), gr.update(visible=False),
                        gr.Timer(active=False), None, gr.update(visible=False)),
            outputs=_train_outputs,
            cancels=[train_event],
        )

        check_btn.click(
            check_gpu_status,
            inputs=[job_state, browser_job_id],
            outputs=_train_outputs,
        )

        # Auto-poll GPU job status every 30 seconds.
        gpu_timer.tick(
            check_gpu_status,
            inputs=[job_state, browser_job_id],
            outputs=_train_outputs,
        )

        # Recover tracking from user-provided Job ID
        def _recover_job(user_job_id, request: gr.Request = None):
            jid = _to_int(user_job_id)
            if not jid:
                return ("Please enter a valid Job ID.",
                        None, None, gr.update()) + _NO_GPU
            _save_job_id(jid)
            return check_gpu_status(jid, jid, request)

        recover_btn.click(
            _recover_job,
            inputs=[recover_input],
            outputs=_train_outputs,
        )

        viz_btn.click(
            run_viz,
            inputs=[viz_dd, model_state],
            outputs=[viz_plot, viz_msg],
        )

    return app


# ── Entry Point ────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    blocks = build_app()

    # Bohrium serves apps behind an HTTPS proxy that does NOT forward
    # X-Forwarded-Proto.  This ASGI middleware forces the scheme to HTTPS
    # so Gradio generates correct URLs in its frontend config.
    class _ForceHTTPS:
        """Fix Mixed Content when running behind Bohrium's HTTPS proxy.

        1. Injects X-Forwarded-Proto into requests so Gradio generates
           HTTPS URLs in its frontend config (``root``).
        2. Adds Content-Security-Policy: upgrade-insecure-requests to
           responses so the browser auto-upgrades any remaining HTTP
           sub-resource requests (e.g. theme.css loaded by Gradio JS
           via window.location).
        """
        def __init__(self, app):
            self.app = app
        async def __call__(self, scope, receive, send):
            if scope["type"] in ("http", "websocket"):
                headers = list(scope.get("headers", []))
                headers.append((b"x-forwarded-proto", b"https"))
                scope = dict(scope, scheme="https", headers=headers)

                async def send_with_csp(message):
                    if message["type"] == "http.response.start":
                        resp_headers = list(message.get("headers", []))
                        resp_headers.append((
                            b"content-security-policy",
                            b"upgrade-insecure-requests",
                        ))
                        message = dict(message, headers=resp_headers)
                    await send(message)

                await self.app(scope, receive, send_with_csp)
            else:
                await self.app(scope, receive, send)

    import uvicorn
    from fastapi import FastAPI

    fastapi_app = FastAPI()
    inner = gr.mount_gradio_app(fastapi_app, blocks, path="/")
    app = _ForceHTTPS(inner)

    uvicorn.run(app, host="0.0.0.0", port=50000)
