"""KD PDE Discovery — Gradio Web Interface.

Designed for deployment on the Bohrium platform.

Usage:
    python app.py
"""

import json
import os
import shutil
import sys
import threading
import time
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

# KD_FAKE_BOHRIUM=1 enables local debugging of the full GPU path
# without a real Bohrium deployment.  Combine with KD_JOB_BACKEND=mock.
_FAKE_BOHRIUM = os.environ.get("KD_FAKE_BOHRIUM", "") == "1"

try:
    from kd.job import submit_gpu_job, check_job, download_result, find_active_job
    _HAS_BOHRIUM = True
except ImportError:
    _HAS_BOHRIUM = _FAKE_BOHRIUM  # fake mode doesn't need the SDK


def _is_bohrium_env(request):
    """Return True when running on Bohrium platform with valid auth cookies."""
    if _FAKE_BOHRIUM:
        return True
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

# Friendly display names for the model dropdown.
MODEL_DISPLAY = {
    "KD_SGA": "SGA",
    "KD_DLGA": "DLGA",
    "KD_DSCV": "Discover",
    "KD_DSCV_SPR": "Discover_SPR",
    "KD_EqGPT": "EqGPT",
    "KD_Discover_Regression": "Discover_Regression",
}
# Reverse: display name → internal key
_DISPLAY_TO_KEY = {v: k for k, v in MODEL_DISPLAY.items()}

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
    """Return friendly display names compatible with a given dataset."""
    if not dataset_name:
        return []
    # Regression datasets → only regression model
    if dataset_name in _REGRESSION_DATASETS:
        return [MODEL_DISPLAY["KD_Discover_Regression"]]
    info = PDE_REGISTRY.get(dataset_name, {})
    models_map = info.get("models", {})
    return [
        MODEL_DISPLAY[name]
        for name, reg_key in MODEL_KEYS.items()
        if models_map.get(reg_key, False)
    ]


def on_dataset_change(dataset_name):
    """Update model dropdown when dataset changes."""
    models = get_compatible_models(dataset_name)
    default = models[0] if models else None
    return gr.update(choices=models, value=default)


def on_model_change(model_name):
    """Show/hide parameter groups and auto-switch operator presets."""
    model_name = _DISPLAY_TO_KEY.get(model_name, model_name)
    sga_vis = model_name == "KD_SGA"
    dlga_vis = model_name == "KD_DLGA"
    dscv_vis = model_name in ("KD_DSCV", "KD_DSCV_SPR")
    spr_vis = model_name == "KD_DSCV_SPR"
    eqgpt_vis = model_name == "KD_EqGPT"
    reg_vis = model_name == "KD_Discover_Regression"

    gpu_vis = model_name in GPU_MODELS

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
        gr.update(visible=gpu_vis),    # gpu_panel
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
            # Try sympy-based human-readable expression first
            program = getattr(model, "best_program_", None)
            if program is not None:
                try:
                    sympy_expr = program.sympy_expr
                    if sympy_expr and len(sympy_expr) > 0:
                        import sympy
                        expr = sympy_expr[0]
                        if not isinstance(expr, str):
                            names = result.get("var_names", [])
                            for i, name in enumerate(names):
                                expr = expr.subs(
                                    sympy.Symbol(f"x{i + 1}"),
                                    sympy.Symbol(name),
                                )
                            return str(expr)
                except Exception:
                    pass
            return result.get("expression_named", result.get("expression", "(no expression found)"))
        return "(no expression found)"
    elif result is not None:
        return result.get("expression", "(no expression found)")
    return "(unknown)"


def _build_viz_proxy(result_data):
    """Build a proxy model from result.json data for viz adapters.

    The viz registry resolves adapters by ``type(target).__name__``,
    so we create a dynamic class whose name matches the real model class.
    Restores serialized viz attributes so that adapters find the data
    they need (e.g. ``model.searcher.r_train`` for Discover).
    """
    model_name = result_data.get("model", "")
    # Check if any viz data was serialized
    eqgpt_keys = {"equations", "rewards", "best_equation", "reward_history", "parity_data"}
    has_viz = any(k in eqgpt_keys or k.startswith("viz_") for k in result_data)
    if not has_viz:
        return None
    class_map = {
        "eqgpt": "KD_EqGPT",
        "dlga": "KD_DLGA",
        "dscv_spr": "KD_Discover",
    }
    cls_name = class_map.get(model_name, model_name)
    ProxyClass = type(cls_name, (), {})
    proxy = ProxyClass()

    # EqGPT: store full result_data as result_
    proxy.result_ = result_data

    # Discover/DSCV_SPR: restore searcher with r_train / r_history
    if "viz_searcher_r_train" in result_data or "viz_searcher_r_history" in result_data:
        searcher = type("Searcher", (), {})()
        searcher.r_train = result_data.get("viz_searcher_r_train")
        searcher.r_history = result_data.get("viz_searcher_r_history")
        proxy.searcher = searcher

    # DLGA: restore loss/evolution history + equation components
    for attr in ("train_loss_history", "val_loss_history", "evolution_history"):
        key = f"viz_{attr}"
        if key in result_data:
            setattr(proxy, attr, result_data[key])
    for attr in ("Chrom", "coef", "name", "user_operators"):
        key = f"viz_{attr}"
        if key in result_data:
            setattr(proxy, attr, result_data[key])

    return proxy


def _get_model_capabilities(model):
    """Return available viz capabilities for the current model.

    Filters the adapter's static capability set to only include intents
    whose required data actually exists on *model*.  This prevents the
    dropdown from showing options that always fail with a warning.
    """
    from kd.viz.core import list_capabilities
    if model is None:
        return []
    caps = set(list_capabilities(model))

    # spr_* intents only work for KD_DSCV_SPR (PINN mode)
    try:
        from kd.model.kd_discover import KD_Discover_SPR as _SPR
        is_spr = isinstance(model, _SPR)
    except ImportError:
        is_spr = False
    if not is_spr:
        caps -= {c for c in caps if c.startswith("spr_")}

    name = type(model).__name__

    if name == "KD_DLGA":
        # field_comparison / time_slices / residual require ctx.options
        # that run_viz() never supplies → always fail in the UI
        caps -= {"field_comparison", "time_slices", "residual"}
        # parity / derivative_relationships need model.metadata
        if not getattr(model, "metadata", None):
            caps -= {"parity", "derivative_relationships"}
        # optimization / search_evolution need evolution_history
        if not getattr(model, "evolution_history", None):
            caps -= {"optimization", "search_evolution"}
        # validation_curve needs val_loss_history
        if not getattr(model, "val_loss_history", None):
            caps.discard("validation_curve")
        # training_curve needs train_loss_history
        if not getattr(model, "train_loss_history", None):
            caps.discard("training_curve")

    elif name in ("KD_Discover", "KD_DSCV_SPR"):
        searcher = getattr(model, "searcher", None)
        best_p = getattr(model, "best_p", None)
        if best_p is None and searcher is not None:
            best_p = getattr(searcher, "best_p", None)
        # equation / residual / field_comparison / parity need best program
        if not best_p:
            caps -= {"equation", "residual", "field_comparison", "parity",
                      "spr_residual", "spr_field_comparison"}
        # search_evolution needs searcher.r_train
        if not searcher or not getattr(searcher, "r_train", None):
            caps.discard("search_evolution")
        # density needs searcher.r_history
        if not searcher or not getattr(searcher, "r_history", None):
            caps.discard("density")
        # tree needs searcher.plotter with tree_plot method
        if not searcher or not hasattr(getattr(searcher, "plotter", None), "tree_plot"):
            caps.discard("tree")

    elif name == "KD_EqGPT":
        result = getattr(model, "result_", None)
        if not result or not isinstance(result, dict):
            caps.clear()
        else:
            if not result.get("equations") or not result.get("rewards"):
                caps.discard("reward_ranking")
            if not result.get("reward_history"):
                caps.discard("reward_evolution")
            if not result.get("parity_data"):
                caps.discard("parity")

    elif name == "KD_Discover_Regression":
        # equation needs best_program_
        bp = getattr(model, "best_program_", None)
        if bp is None:
            bp = getattr(model, "best_p", None)
            if isinstance(bp, list):
                bp = bp[0] if bp else None
        if not bp:
            caps.discard("equation")
        # parity / residual need data_class with X, y
        dc = getattr(model, "data_class", None)
        if not dc or getattr(dc, "y", None) is None or getattr(dc, "X", None) is None:
            caps -= {"parity", "residual"}
        searcher = getattr(model, "searcher", None)
        if not searcher or not getattr(searcher, "r_train", None):
            caps.discard("search_evolution")
        if not searcher or not getattr(searcher, "r_history", None):
            caps.discard("density")
        if not searcher or not hasattr(getattr(searcher, "plotter", None), "tree_plot"):
            caps.discard("tree")

    elif name == "KD_SGA":
        # field_comparison / time_slices / parity / residual need context_
        if not getattr(model, "context_", None):
            caps -= {"field_comparison", "time_slices", "parity", "residual"}

    elif name == "KD_PySR":
        if not hasattr(model, "X_train_") or not hasattr(model, "y_train_"):
            caps -= {"parity", "residual"}

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


# ── Log capture for real-time training output ─────────────────

class _LogTee:
    """Thread-safe output tee that captures writes while forwarding to the
    original stream.  Multiple instances can share the same buffer so that
    stdout and stderr output is interleaved in capture order."""

    def __init__(self, original, buf=None, lock=None):
        self._orig = original
        self._buf = buf if buf is not None else []
        self._lock = lock if lock is not None else threading.Lock()

    def write(self, s):
        self._orig.write(s)
        with self._lock:
            self._buf.append(s)
        return len(s)

    def flush(self):
        self._orig.flush()

    def isatty(self):
        return self._orig.isatty()

    @property
    def encoding(self):
        return getattr(self._orig, "encoding", "utf-8")

    def fileno(self):
        return self._orig.fileno()

    def get_log(self, max_lines=200):
        """Return captured output as clean multi-line text."""
        with self._lock:
            raw = "".join(self._buf)
        # Convert \r (tqdm carriage returns) into separate lines
        parts = raw.replace("\r\n", "\n").split("\r")
        lines = []
        for p in parts:
            for line in p.split("\n"):
                stripped = line.rstrip()
                if stripped:
                    lines.append(stripped)
        return "\n".join(lines[-max_lines:])


def _log_progress(log_text, elapsed):
    """Build 11-tuple for intermediate log yields during CPU training."""
    if elapsed > 0:
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        status = f"Training in progress... ({time_str})"
    else:
        status = "Initializing..."
    return (
        status,                                    # eq_text
        gr.update(),                               # eq_plot
        gr.update(),                               # model_state
        gr.update(),                               # viz_dd
        gr.update(),                               # job_state
        gr.update(),                               # job_status
        gr.update(),                               # check_btn
        gr.update(),                               # gpu_timer
        gr.update(),                               # browser_job_id
        gr.update(visible=False),                  # gpu_panel
        gr.update(value=log_text, visible=True),   # train_log
    )


# ── GPU job tracking ──────────────────────────────────────────

_GPU_JOB_BASE = "/data/kd_users"


def _get_uid(request=None):
    """Extract user ID from request cookies."""
    if _FAKE_BOHRIUM:
        return "dev_user"
    if request:
        try:
            from http.cookies import SimpleCookie
            sc = SimpleCookie()
            sc.load(request.headers.get("cookie", ""))
            if "userId" in sc:
                return sc["userId"].value
        except Exception:
            pass
    return "default"


def _user_dir(request=None):
    """Return per-user directory under /data/kd_users/{userId}/."""
    uid = _get_uid(request)
    d = os.path.join(_GPU_JOB_BASE, uid)
    os.makedirs(d, exist_ok=True)
    return d


def _save_job_id(job_id, request=None):
    """Persist job_id to per-user file."""
    try:
        path = os.path.join(_user_dir(request), "job_id")
        with open(path, "w") as f:
            f.write(str(job_id) if job_id else "")
        print(f"[kd] _save_job_id({job_id}) → {path}", flush=True)
    except Exception as e:
        print(f"[kd] _save_job_id({job_id}) FAILED: {e}", flush=True)


def _load_job_id(request=None):
    """Load persisted job_id from per-user file."""
    try:
        path = os.path.join(_user_dir(request), "job_id")
        with open(path) as f:
            val = f.read().strip()
            result = int(val) if val else None
        print(f"[kd] _load_job_id() → {result}", flush=True)
        return result
    except Exception as e:
        print(f"[kd] _load_job_id() → None ({e})", flush=True)
        return None


# ── GPU Job History ────────────────────────────────────────────

_HISTORY_COLUMNS = ["Job ID", "Model", "Dataset", "Status", "Equation", "Duration", "Submitted"]


def _history_path(request=None):
    return os.path.join(_user_dir(request), "job_list.txt")


def _append_history(job_id, model_name, dataset_name, request=None):
    """Append a new job entry to per-user history."""
    try:
        path = _history_path(request)
        print(f"[kd] _append_history({job_id}, {model_name}, {dataset_name}) → {path}", flush=True)
        display_name = MODEL_DISPLAY.get(model_name, model_name)
        entry = json.dumps({
            "job_id": str(job_id), "model": display_name,
            "dataset": dataset_name, "status": "Submitted",
            "equation": "", "elapsed": "",
            "submitted_at": time.strftime("%Y-%m-%d %H:%M"),
        }, ensure_ascii=False)
        with open(path, "a") as f:
            f.write(entry + "\n")
        print(f"[kd] _append_history OK, wrote {len(entry)} bytes", flush=True)
    except Exception as e:
        print(f"[kd] _append_history failed: {e}", flush=True)


def _update_history(job_id, status, equation="", elapsed="", request=None):
    """Update status/equation/elapsed for a job in history."""
    try:
        path = _history_path(request)
        print(f"[kd] _update_history({job_id}, {status}, elapsed={elapsed}) path={path} exists={os.path.exists(path)}", flush=True)
        if not os.path.exists(path):
            return
        lines = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("job_id") == str(job_id):
                    entry["status"] = status
                    if equation:
                        entry["equation"] = equation[:200]
                    if elapsed:
                        entry["elapsed"] = str(elapsed)
                lines.append(json.dumps(entry, ensure_ascii=False))
        with open(path, "w") as f:
            for l in lines[-50:]:
                f.write(l + "\n")
    except Exception as e:
        print(f"[kd] _update_history failed: {e}", flush=True)


def _scan_output_dirs():
    """Scan /home/outputs/ for completed job results as history fallback."""
    output_base = "/home/outputs"
    entries = []
    if not os.path.isdir(output_base):
        return entries
    for name in os.listdir(output_base):
        if not name.isdigit():
            continue
        # Search result.json in root or s/ subdirectory
        result_path = None
        for candidate in [
            os.path.join(output_base, name, "result.json"),
            os.path.join(output_base, name, "s", "result.json"),
        ]:
            if os.path.exists(candidate):
                result_path = candidate
                break
        if not result_path:
            continue
        try:
            with open(result_path) as f:
                data = json.loads(f.read())
            # Map runner model name (e.g. "eqgpt") to display name
            runner_model = data.get("model", "")
            _RUNNER_DISPLAY = {"dlga": "DLGA", "eqgpt": "EqGPT",
                               "dscv_spr": "Discover_SPR"}
            display = _RUNNER_DISPLAY.get(runner_model, runner_model)
            elapsed = data.get("elapsed_seconds", "")
            entries.append({
                "job_id": name,
                "model": display,
                "dataset": data.get("dataset", ""),
                "status": "Finished" if data.get("status") == "success" else "Failed",
                "equation": (data.get("equation", "") or "")[:200],
                "elapsed": f"{elapsed}s" if elapsed else "",
                "submitted_at": "",
            })
        except Exception:
            continue
    print(f"[kd] _scan_output_dirs: found {len(entries)} jobs", flush=True)
    return entries


def _load_history(request=None):
    """Load recent GPU job history as list of lists for Dataframe.

    For unfinished jobs, queries the Bohrium API for real-time status
    and updates the local file (following the official example pattern).
    Falls back to scanning /home/outputs/ when job_list.txt is missing.
    """
    path = _history_path(request)
    print(f"[kd] _load_history: path={path} exists={os.path.exists(path)}", flush=True)
    if not os.path.exists(path):
        # Fallback: recover history from downloaded output directories
        fallback = _scan_output_dirs()
        if fallback:
            rows = []
            for e in fallback:
                rows.append([
                    e.get("job_id", ""), e.get("model", ""),
                    e.get("dataset", ""), e.get("status", ""),
                    e.get("equation", "")[:60],
                    e.get("elapsed", ""),
                    e.get("submitted_at", ""),
                ])
            return rows[-10:][::-1]
        return []
    try:
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # For unfinished jobs, query API for real-time status update
        # (mirrors official example: check all non-terminal statuses)
        _TERMINAL = {"Finished", "Failed", "Error"}
        updated = False
        if _HAS_BOHRIUM and request and _is_bohrium_env(request):
            for e in entries:
                status = e.get("status", "")
                if status and status not in _TERMINAL:
                    jid = e.get("job_id")
                    try:
                        from kd.job import check_job, download_result
                        info = check_job(int(jid), request)
                        new_status = info.get("status_text", status)
                        print(f"[kd] _load_history: job {jid} API status={new_status}", flush=True)
                        e["status"] = new_status
                        if info.get("spend_time"):
                            e["elapsed"] = info["spend_time"]
                        updated = True

                        # Auto-download results for newly finished jobs
                        if info.get("finished"):
                            try:
                                result_data, _ = download_result(int(jid), request)
                                eq = result_data.get("equation", "")
                                if eq:
                                    e["equation"] = eq[:200]
                                print(f"[kd] _load_history: auto-downloaded job {jid}", flush=True)
                            except Exception as dl_ex:
                                print(f"[kd] _load_history: auto-download job {jid} failed: {dl_ex}", flush=True)

                    except (Exception, SystemExit) as ex:
                        print(f"[kd] _load_history: API check job {jid} failed: {ex}", flush=True)

        # Write back updated statuses
        if updated:
            try:
                with open(path, "w") as f:
                    for e in entries[-50:]:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                print(f"[kd] _load_history: wrote back {len(entries)} updated entries", flush=True)
            except Exception as ex:
                print(f"[kd] _load_history: write-back failed: {ex}", flush=True)

        rows = []
        for e in entries:
            rows.append([
                e.get("job_id", ""), e.get("model", ""),
                e.get("dataset", ""), e.get("status", ""),
                e.get("equation", "")[:60],
                e.get("elapsed", ""),
                e.get("submitted_at", ""),
            ])
        print(f"[kd] _load_history: {len(rows)} entries loaded", flush=True)
        return rows[-10:][::-1]  # most recent first, max 10
    except Exception as e:
        print(f"[kd] _load_history failed: {e}", flush=True)
        return []


# ── CPU Job History ───────────────────────────────────────────

_CPU_OUTPUT_DIR = os.environ.get("KD_CPU_OUTPUT_DIR", "/data/outputs")


def _cpu_history_path(request=None):
    return os.path.join(_user_dir(request), "cpu_job_list.txt")


def _append_cpu_history(job_id, model_name, dataset_name, request=None):
    """Append a new CPU job entry to per-user history."""
    try:
        path = _cpu_history_path(request)
        display_name = MODEL_DISPLAY.get(model_name, model_name)
        entry = json.dumps({
            "job_id": str(job_id), "model": display_name,
            "dataset": dataset_name, "status": "Submitted",
            "equation": "", "elapsed": "",
            "submitted_at": time.strftime("%Y-%m-%d %H:%M"),
        }, ensure_ascii=False)
        with open(path, "a") as f:
            f.write(entry + "\n")
        print(f"[kd] _append_cpu_history({job_id}) OK", flush=True)
    except Exception as e:
        print(f"[kd] _append_cpu_history failed: {e}", flush=True)


def _update_cpu_history(job_id, status, equation="", elapsed="", request=None):
    """Update status/equation/elapsed for a CPU job in history."""
    try:
        path = _cpu_history_path(request)
        if not os.path.exists(path):
            return
        lines = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("job_id") == str(job_id):
                    entry["status"] = status
                    if equation:
                        entry["equation"] = equation[:200]
                    if elapsed:
                        entry["elapsed"] = str(elapsed)
                lines.append(json.dumps(entry, ensure_ascii=False))
        with open(path, "w") as f:
            for l in lines[-50:]:
                f.write(l + "\n")
    except Exception as e:
        print(f"[kd] _update_cpu_history failed: {e}", flush=True)


def _load_cpu_history(request=None):
    """Load CPU job history as list of lists for Dataframe."""
    path = _cpu_history_path(request)
    if not os.path.exists(path):
        return []
    try:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append([
                    e.get("job_id", ""),
                    e.get("model", ""),
                    e.get("dataset", ""),
                    e.get("status", ""),
                    (e.get("equation", "") or "")[:60],
                    e.get("elapsed", ""),
                    e.get("submitted_at", ""),
                ])
        return rows[-10:][::-1]
    except Exception as e:
        print(f"[kd] _load_cpu_history failed: {e}", flush=True)
        return []


def _clear_cpu_history(request=None):
    """Clear CPU job history and associated result files for current user."""
    path = _cpu_history_path(request)
    if os.path.exists(path):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    jid = e.get("job_id", "")
                    out_dir = os.path.join(_CPU_OUTPUT_DIR, jid)
                    if os.path.isdir(out_dir):
                        shutil.rmtree(out_dir)
                        print(f"[kd] _clear_cpu_history: removed {out_dir}", flush=True)
            open(path, "w").close()
        except Exception as exc:
            print(f"[kd] _clear_cpu_history failed: {exc}", flush=True)
    return gr.update(value=[])


def _clear_gpu_history(request=None):
    """Clear GPU job history and associated result files for current user."""
    path = _history_path(request)
    if os.path.exists(path):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    jid = e.get("job_id", "")
                    out_dir = os.path.join(_CPU_OUTPUT_DIR, jid)
                    if os.path.isdir(out_dir):
                        shutil.rmtree(out_dir)
                        print(f"[kd] _clear_gpu_history: removed {out_dir}", flush=True)
            open(path, "w").close()
        except Exception as exc:
            print(f"[kd] _clear_gpu_history failed: {exc}", flush=True)
    return gr.update(value=[])


def _save_cpu_result(cpu_job_id, model_name, dataset_name, equation, elapsed, model,
                     equation_latex=None):
    """Save CPU training result to /data/outputs/{cpu_job_id}/result.json."""
    try:
        from kd.runner import _serialize_viz_data
        save_dir = os.path.join(_CPU_OUTPUT_DIR, cpu_job_id)
        os.makedirs(save_dir, exist_ok=True)
        result_data = {
            "model": model_name,
            "dataset": dataset_name,
            "equation": equation,
            "elapsed_seconds": round(elapsed, 2),
            "status": "success",
        }
        if equation_latex:
            result_data["equation_latex"] = equation_latex
        _serialize_viz_data(model, result_data)
        with open(os.path.join(save_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False,
                      default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        print(f"[kd] _save_cpu_result({cpu_job_id}) → {save_dir}", flush=True)
    except Exception as e:
        print(f"[kd] _save_cpu_result failed: {e}", flush=True)


# ── Training ───────────────────────────────────────────────────

# 10-value output tuple: (eq_text, eq_plot, model_state, viz_dd,
#     job_state, job_status, check_btn, gpu_timer, browser_job_id, gpu_panel)
# gpu_panel hidden — for local (non-GPU) training paths
_NO_GPU = (None, gr.update(), gr.update(),
           gr.Timer(active=False), None, gr.update(visible=False))
# gpu_panel visible — for GPU paths so user always has manual resume
_NO_GPU_RESUME = (None, gr.update(), gr.update(),
                  gr.Timer(active=False), None, gr.update(visible=True))


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
    reg_iterations=200, reg_samples=1000, reg_seed=42,
    reg_parsimony=0.005,
    # Gradio injects this automatically
    request: gr.Request = None,
):
    """Core training function — generator yielding 11-tuples (10 train + log).

    Routes to GPU offline submission or local CPU training with live log.
    """
    model_name = _DISPLAY_TO_KEY.get(model_name, model_name)
    if not dataset_name or not model_name:
        yield ("Please select dataset and model first.",
                None, None, gr.update()) + _NO_GPU + (gr.update(visible=False),)
        return

    # GPU models on Bohrium → submit as offline task (no log)
    if model_name in GPU_MODELS and _is_bohrium_env(request):
        result = _submit_gpu_training(
            dataset_name, model_name,
            dlga_ops, dlga_epi, dlga_max_iter, dlga_sample,
            dscv_binary, dscv_unary, dscv_batch, dscv_epochs, dscv_seed,
            spr_sample_ratio, spr_epochs,
            eqgpt_epochs, eqgpt_samples, eqgpt_cases, eqgpt_seed,
            request,
        )
        yield result + (gr.update(visible=False),)
        return

    # All other cases (including local dev) → run in-process with live log
    yield from _run_local(
        dataset_name, model_name,
        sga_run, sga_num, sga_depth, sga_seed,
        dlga_ops, dlga_epi, dlga_max_iter, dlga_sample,
        dscv_binary, dscv_unary, dscv_batch, dscv_epochs, dscv_seed,
        spr_sample_ratio, spr_epochs,
        eqgpt_epochs, eqgpt_samples, eqgpt_cases, eqgpt_seed,
        reg_binary, reg_unary, reg_iterations, reg_samples, reg_seed,
        reg_parsimony,
        request=request,
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
    reg_iterations=200, reg_samples=1000, reg_seed=42,
    reg_parsimony=0.005,
    request=None,
):
    """Run model training in a background thread, yielding live log updates.

    Generator that produces 11-tuples (10 train outputs + train_log).
    """
    configure(save_dir=None)
    cpu_job_id = f"cpu_{int(time.time())}"
    t0 = time.time()
    _append_cpu_history(cpu_job_id, model_name, dataset_name, request)

    # Initial yield: show log panel
    yield _log_progress("Initializing model...", 0)

    # Results populated by the training thread
    holder = {}  # 'output': 10-tuple for the final yield

    def _do_train():
        """Actual training — runs in a background thread."""
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

            y_pred = model.predict(X)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            eq_raw = _get_equation_text(model, model_name, result)
            eq_text = eq_raw
            eq_text += f"\n\nR² = {r2:.6f}  |  MSE = {result['mse']:.4f}  |  Reward = {result['reward']:.6f}"
            eq_text += f"\nTarget: {meta['target_name']}  |  Variables: {meta['var_names']}"
            fig = _render_equation_fig(model, eq_raw)

            caps = _get_model_capabilities(model)
            print(f"[kd] _run_local(Regression) caps={caps}", flush=True)
            viz_update = gr.update(choices=caps, value=caps[0] if caps else None)
            elapsed = round(time.time() - t0, 1)
            _save_cpu_result(cpu_job_id, model_name, dataset_name, eq_text, elapsed, model,
                             equation_latex=eq_raw)
            _update_cpu_history(cpu_job_id, "Finished", eq_text, f"{elapsed}s", request)
            holder['output'] = (eq_text, fig, model, viz_update) + _NO_GPU
            return

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
                holder['output'] = (f"Unknown model: {model_name}",
                                    None, None, gr.update()) + _NO_GPU
                return

        equation = _get_equation_text(model, model_name, result)
        fig = _render_equation_fig(model, equation)
        caps = _get_model_capabilities(model)
        print(f"[kd] _run_local({model_name}) caps={caps}", flush=True)
        viz_update = gr.update(choices=caps, value=caps[0] if caps else None)
        elapsed = round(time.time() - t0, 1)
        _save_cpu_result(cpu_job_id, model_name, dataset_name, equation, elapsed, model,
                         equation_latex=equation)
        _update_cpu_history(cpu_job_id, "Finished", equation, f"{elapsed}s", request)
        holder['output'] = (equation, fig, model, viz_update) + _NO_GPU

    # ── Run training in background thread with log capture ──
    shared_buf = []
    shared_lock = threading.Lock()
    out_tee = _LogTee(sys.stdout, shared_buf, shared_lock)
    err_tee = _LogTee(sys.stderr, shared_buf, shared_lock)
    old_out, old_err = sys.stdout, sys.stderr
    error_info = {}
    done = threading.Event()

    def _worker():
        try:
            sys.stdout = out_tee
            sys.stderr = err_tee
            _do_train()
        except Exception as exc:
            error_info['exc'] = exc
            error_info['tb'] = traceback.format_exc()
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # Yield log updates every 2 seconds while training runs
    while not done.wait(timeout=2):
        log = out_tee.get_log()
        yield _log_progress(log or "(waiting for output...)", time.time() - t0)

    # ── Training finished ──
    log = out_tee.get_log()
    elapsed = round(time.time() - t0, 1)

    if error_info:
        exc = error_info['exc']
        tb = error_info['tb']
        _update_cpu_history(cpu_job_id, "Failed", str(exc), "", request)
        final_log = log + f"\n\nTraining failed ({elapsed}s): {exc}"
        yield (f"Training error:\n{exc}\n\n{tb}",
               None, None, gr.update()) + _NO_GPU + (gr.update(value=final_log),)
        return

    final_log = log + f"\n\nTraining complete ({elapsed}s)"
    yield holder['output'] + (gr.update(value=final_log),)


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
        _save_job_id(job_id, request)
        _append_history(job_id, model_name, dataset_name, request)
        return (
            f"GPU task submitted — Job ID: {job_id}\n"
            "Auto-polling every 30s. You can also click 'Check GPU Job'.",
            None, None, gr.update(),
            job_id,
            gr.update(value=f">>> Job ID: {job_id} <<<  Status: Submitted"),
            gr.update(),
            gr.Timer(active=True),
            job_id,                        # → browser_job_id (LocalStorage)
            gr.update(visible=True),       # → show gpu_panel
        )
    except Exception as e:
        return (f"GPU submit error:\n{e}\n\n{traceback.format_exc()}",
                None, None, gr.update()) + _NO_GPU_RESUME


def check_gpu_status(job_id, browser_job_id=None, request: gr.Request = None):
    """Poll GPU job status; download results when finished.

    This function is called by both the manual Check button and the
    auto-polling timer.  It must NEVER raise — any unhandled exception
    would cause Gradio to return a non-JSON response, which the browser
    shows as ``SyntaxError: Unexpected token '<'``.
    """
    try:
        return _check_gpu_status_inner(job_id, browser_job_id, request)
    except (Exception, SystemExit) as exc:
        # Last-resort safety net — keep polling with existing job_id
        safe_id = job_id or _load_job_id(request) or _to_int(browser_job_id)
        if safe_id:
            return (
                f"Status check error: {exc}\nWill retry with job {safe_id}...",
                None, None, gr.update(),
                safe_id,
                gr.update(value=f">>> Job ID: {safe_id} <<<  Retrying..."),
                gr.update(),
                gr.Timer(active=True),
                safe_id,
                gr.update(visible=True),       # → show gpu_panel
            )
        # All recovery failed — show gpu_panel with resume input
        return ("GPU job ID lost. Please enter your Job ID below to resume.",
                None, None, gr.update(),
                None,
                gr.update(),
                gr.update(),
                gr.Timer(active=False),
                None,
                gr.update(visible=True),       # → show gpu_panel
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
        job_id = _load_job_id(request)
        if job_id:
            print(f"[kd] Recovered job_id={job_id} from file", flush=True)
    if not job_id:
        job_id = _to_int(browser_job_id)
        if job_id:
            _save_job_id(job_id, request)
            print(f"[kd] Recovered job_id={job_id} from browser LocalStorage", flush=True)
    if not job_id and _HAS_BOHRIUM and request:
        try:
            job_id = find_active_job(request)
            if job_id:
                _save_job_id(job_id, request)
                print(f"[kd] Recovered job_id={job_id} from platform API", flush=True)
        except (Exception, SystemExit) as e:
            print(f"[kd] find_active_job failed: {e}", flush=True)

    # Ensure recovered job has a history entry (job_list.txt may not exist)
    if job_id:
        hist_path = _history_path(request)
        if not os.path.exists(hist_path) or str(job_id) not in open(hist_path).read():
            print(f"[kd] Adding missing history entry for recovered job {job_id}", flush=True)
            _append_history(job_id, "(recovered)", "", request)
    if not job_id:
        # Show gpu_panel with resume input
        return ("GPU job ID lost. Please enter your Job ID below to resume.",
                None, None, gr.update(),
                None,
                gr.update(),
                gr.update(),
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
            gr.update(value=f">>> Job ID: {job_id} <<<  Error: {e}"),
            gr.update(),
            gr.Timer(active=True),
            job_id, gr.update(visible=True),
        )

    if status["failed"]:
        _save_job_id(None, request)
        _update_history(job_id, "Failed", elapsed=status.get("spend_time", ""), request=request)
        return (
            f"Job {job_id} failed: {status['status_text']}",
            None, None, gr.update(),
            None,
            gr.update(value=f">>> Job ID: {job_id} <<<  {status['status_text']}"),
            gr.update(),
            gr.Timer(active=False),
            None, gr.update(visible=True),
        )

    if not status["finished"]:
        elapsed = status.get("spend_time", "")
        return (
            f"Job {job_id} still running... ({status['status_text']})",
            None, None, gr.update(),
            job_id,
            gr.update(value=f">>> Job ID: {job_id} <<<  {status['status_text']}  {elapsed}"),
            gr.update(),
            gr.Timer(active=True),
            job_id, gr.update(visible=True),
        )

    # ── Job finished → download & display ──
    _save_job_id(None, request)
    try:
        result_data, save_dir = download_result(job_id, request)
        print(f"[kd] GPU result keys: {list(result_data.keys())}", flush=True)
        print(f"[kd] GPU result model={result_data.get('model')}, save_dir={save_dir}", flush=True)
        equation = result_data.get("equation", "(unknown)")
        elapsed_s = result_data.get("elapsed_seconds", "")
        elapsed_str = f"{elapsed_s}s" if elapsed_s else ""
        _update_history(job_id, "Finished", equation, elapsed_str, request)

        # Build a proxy model object from result_data so viz adapters work
        proxy_model = _build_viz_proxy(result_data)

        # Render equation figure on CPU using the same viz adapter as GPU
        eq_fig = _render_equation_fig(proxy_model, equation)
        viz_choices = _get_model_capabilities(proxy_model) if proxy_model else []
        print(f"[kd] proxy_model={proxy_model}, viz_choices={viz_choices}", flush=True)
        viz_update = gr.update(choices=viz_choices, value=viz_choices[0] if viz_choices else None)

        elapsed_s = result_data.get("elapsed_seconds", "?")
        return (
            equation, eq_fig, proxy_model,
            viz_update,
            None,
            gr.update(value=f">>> Job ID: {job_id} <<<  Finished ({elapsed_s}s)"),
            gr.update(),
            gr.Timer(active=False),
            None, gr.update(visible=True),
        )
    except Exception as e:
        return (f"Download failed: {e}\nJob ID: {job_id}",
                None, None, gr.update()) + _NO_GPU_RESUME


# ── Visualization ──────────────────────────────────────────────

def run_viz(viz_type, model_state):
    """Generate a visualization for the trained model."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import tempfile

    if model_state is None:
        return None, "Please train a model first."
    if not viz_type:
        return None, "Please select a plot type."

    print(f"[kd] run_viz(viz_type={viz_type!r}, model={type(model_state).__name__})", flush=True)

    # Use a temp dir so adapters that save-to-file still work
    with tempfile.TemporaryDirectory() as tmpdir:
        configure(save_dir=tmpdir)
        try:
            viz_result = render(VizRequest(viz_type, model_state))
            if viz_result.warnings:
                warning_text = "\n".join(viz_result.warnings)
                if "does not support" in warning_text:
                    return None, f"Visualization '{viz_type}' is not yet implemented for this model."
                return None, warning_text

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
            err_msg = str(e)
            if "not support" in err_msg or "not registered" in err_msg:
                return None, f"Visualization '{viz_type}' is not yet implemented for this model."
            return None, f"Visualization error: {e}"
        finally:
            configure(save_dir=None)


# ── Gradio UI ──────────────────────────────────────────────────

_ERROR_SUPPRESSION_JS = """
() => {
    const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType === 1) {
                    const text = node.textContent || '';
                    if (text.includes('SyntaxError') || text.includes('Unexpected token')) {
                        node.remove();
                    }
                }
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    // Hide ETA estimate from progress display (keep elapsed time only)
    // "processing | 473.4/669.65s" → "processing | 473.4s"
    setInterval(() => {
        document.querySelectorAll('.progress-text, .eta-bar, .timer').forEach(el => {
            const t = el.textContent;
            if (t && t.includes('/')) {
                el.textContent = t.replace(/(\d+\.?\d*)\/\d+\.?\d*s/, '$1s');
            }
        });
        // Hide progress timer on "GPU Job Status" component only
        document.querySelectorAll('label span, .label-wrap span').forEach(lbl => {
            if (lbl.textContent.trim() === 'GPU Job Status') {
                const container = lbl.closest('.block, .gradio-textbox, [class*="textbox"]');
                if (container) {
                    container.querySelectorAll('.progress-text, .eta-bar, .progress-bar, .meta-text, .meta-text-center, [class*="progress"]').forEach(p => {
                        p.style.display = 'none';
                    });
                }
            }
        });
    }, 500);
}
"""


def build_app():
    with gr.Blocks(
        title="KD - PDE Discovery",
        theme=gr.themes.Soft(),
        js=_ERROR_SUPPRESSION_JS,
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

                        # -- Discover / SPR shared params --
                        with gr.Group(visible=False) as dscv_group:
                            gr.Markdown("**Discover Parameters**")
                            dscv_binary = gr.Textbox(
                                value=DSCV_BINARY_DEFAULT,
                                label="Binary operators",
                                info="Discover: add, mul, diff ... | SPR: add_t, mul_t, diff_t ...",
                            )
                            dscv_unary = gr.Textbox(
                                value=DSCV_UNARY_DEFAULT,
                                label="Unary operators",
                                info="Discover: n2, n3, sin ... | SPR: n2_t ...",
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
                                value=200, label="Search iterations", precision=0,
                            )
                            reg_samples = gr.Number(
                                value=1000, label="Samples per batch", precision=0,
                            )
                            reg_parsimony = gr.Number(
                                value=0.005, label="Parsimony coefficient",
                                info="Higher = simpler equations",
                            )
                            reg_seed = gr.Number(value=42, label="Seed", precision=0)

                        with gr.Row():
                            train_btn = gr.Button("Start Training", variant="primary", size="lg")
                            cancel_btn = gr.Button("Cancel", variant="stop", size="lg")

                    # Right: results
                    with gr.Column(scale=2):
                        eq_text = gr.Textbox(label="Discovered Equation", lines=4, interactive=False)
                        eq_plot = gr.Plot(label="Equation Visualization")
                        train_log = gr.Textbox(
                            label="Training Log", lines=10, max_lines=12,
                            interactive=False, visible=False,
                        )
                        with gr.Column(visible=False) as gpu_panel:
                            job_status = gr.Textbox(
                                label="GPU Job Status", interactive=False,
                            )
                            check_btn = gr.Button(
                                "Check GPU Job", variant="secondary",
                            )
                            gpu_timer = gr.Timer(30, active=False)
                            if hasattr(gr, "BrowserState"):
                                browser_job_id = gr.BrowserState(None, storage_key="kd_gpu_job_id", secret="kd_browser_v1")
                            else:
                                browser_job_id = gr.State(None)

                        with gr.Row(equal_height=True):
                            cpu_recover_input = gr.Textbox(
                                label="Enter CPU Job ID to resume",
                                placeholder="cpu_1742000000",
                                scale=3,
                            )
                            cpu_recover_btn = gr.Button(
                                "Resume CPU-Job Tracking", variant="primary",
                                scale=1, min_width=160,
                            )
                        with gr.Row(equal_height=True):
                            recover_input = gr.Number(
                                label="Enter GPU Job ID to resume",
                                precision=0, scale=3,
                            )
                            recover_btn = gr.Button(
                                "Resume GPU-Job Tracking", variant="primary",
                                scale=1, min_width=160,
                            )

                        with gr.Accordion("CPU Job History", open=False):
                            cpu_history_df = gr.Dataframe(
                                headers=_HISTORY_COLUMNS,
                                datatype=["str"] * len(_HISTORY_COLUMNS),
                                column_widths=["160px", "100px", "100px", "80px", "200px", "80px", "130px"],
                                label="Recent CPU Jobs",
                                interactive=False,
                            )
                            with gr.Row():
                                refresh_cpu_history_btn = gr.Button("Refresh", size="sm")
                                clear_cpu_history_btn = gr.Button("Clear History", size="sm", variant="stop")

                        with gr.Accordion("GPU Job History", open=False):
                            history_df = gr.Dataframe(
                                headers=_HISTORY_COLUMNS,
                                datatype=["str"] * len(_HISTORY_COLUMNS),
                                column_widths=["120px", "100px", "100px", "80px", "200px", "80px", "130px"],
                                label="Recent GPU Jobs",
                                interactive=False,
                            )
                            with gr.Row():
                                refresh_history_btn = gr.Button("Refresh", size="sm")
                                clear_gpu_history_btn = gr.Button("Clear History", size="sm", variant="stop")

            # ── Tab 2: Visualization ─────────────────────
            with gr.TabItem("Visualization"):
                with gr.Row():
                    viz_dd = gr.Dropdown(
                        choices=[],
                        value=None,
                        allow_custom_value=True,
                        label="Plot type (train a model first)",
                    )
                    viz_btn = gr.Button("Generate", variant="primary")
                viz_plot = gr.Plot(label="Result")
                viz_msg = gr.Textbox(label="Status", interactive=False)

        # ── Event Wiring ─────────────────────────────────

        _model_change_outputs = [sga_group, dlga_group, dscv_group, spr_group,
                                 eqgpt_group, reg_group, dscv_binary, dscv_unary,
                                 gpu_panel]

        _train_outputs = [
            eq_text, eq_plot, model_state, viz_dd,
            job_state, job_status, check_btn, gpu_timer,
            browser_job_id, gpu_panel,
        ]

        # Initialize model list and parameter panel on page load.
        app.load(on_dataset_change, inputs=[dataset_dd], outputs=[model_dd])
        app.load(on_model_change, inputs=[model_dd], outputs=_model_change_outputs)

        # On page load, check if there's an unfinished GPU job to resume
        def _on_load_recover(browser_jid, request: gr.Request = None):
            jid = _load_job_id(request) or _to_int(browser_jid)
            if not jid and _HAS_BOHRIUM and _is_bohrium_env(request):
                try:
                    jid = find_active_job(request)
                    if jid:
                        print(f"[kd] Page load: recovered job_id={jid} from API", flush=True)
                except (Exception, SystemExit) as e:
                    print(f"[kd] Page load: find_active_job failed: {e}", flush=True)
            if jid:
                print(f"[kd] Page load: recovering job_id={jid}", flush=True)
                return check_gpu_status(jid, browser_jid, request)
            print("[kd] Page load: no job found, showing resume input", flush=True)
            return ("", None, None, gr.update()) + _NO_GPU_RESUME

        _train_outputs = [
            eq_text, eq_plot, model_state, viz_dd,
            job_state, job_status, check_btn, gpu_timer,
            browser_job_id, gpu_panel,
        ]

        app.load(
            _on_load_recover,
            inputs=[browser_job_id],
            outputs=_train_outputs,
            show_progress="hidden",
        )

        dataset_dd.change(on_dataset_change, inputs=[dataset_dd], outputs=[model_dd])

        model_dd.change(
            on_model_change,
            inputs=[model_dd],
            outputs=_model_change_outputs,
        )

        _train_outputs_with_log = _train_outputs + [train_log]

        # Disable button immediately on click → run training → re-enable
        train_event = train_btn.click(
            fn=lambda: gr.update(interactive=False, value="Training..."),
            outputs=[train_btn],
        ).then(
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
            outputs=_train_outputs_with_log,
        ).then(
            fn=lambda: gr.update(interactive=True, value="Start Training"),
            outputs=[train_btn],
        )

        cancel_btn.click(
            fn=lambda: ("Training cancelled.", None, None, gr.update(),
                        None, gr.update(), gr.update(),
                        gr.Timer(active=False), None, gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=True, value="Start Training")),
            outputs=_train_outputs_with_log + [train_btn],
            cancels=[train_event],
        )

        check_btn.click(
            check_gpu_status,
            inputs=[job_state, browser_job_id],
            outputs=_train_outputs,
            show_progress="hidden",
        )

        # Auto-poll GPU job status every 30 seconds.
        gpu_timer.tick(
            check_gpu_status,
            inputs=[job_state, browser_job_id],
            outputs=_train_outputs,
            show_progress="hidden",
        )

        # Recover tracking from user-provided Job ID
        def _recover_job(user_job_id, request: gr.Request = None):
            jid = _to_int(user_job_id)
            if not jid:
                return ("Please enter a valid Job ID.",
                        None, None, gr.update()) + _NO_GPU_RESUME
            _save_job_id(jid, request)
            return check_gpu_status(jid, jid, request)

        recover_btn.click(
            _recover_job,
            inputs=[recover_input],
            outputs=_train_outputs,
            show_progress="hidden",
        )

        # Recover CPU job results from saved result.json
        def _recover_cpu_job(cpu_job_id, request: gr.Request = None):
            cpu_job_id = (cpu_job_id or "").strip()
            if not cpu_job_id:
                return ("Please enter a valid CPU Job ID (e.g. cpu_1742000000).",
                        None, None, gr.update()) + _NO_GPU
            result_path = os.path.join(_CPU_OUTPUT_DIR, cpu_job_id, "result.json")
            if not os.path.exists(result_path):
                return (f"Result not found: {result_path}",
                        None, None, gr.update()) + _NO_GPU
            try:
                with open(result_path) as f:
                    result_data = json.load(f)
                equation = result_data.get("equation", "(unknown)")
                # equation_latex stores only the formula; fall back to first
                # line of equation for old result.json files without it.
                eq_latex = result_data.get("equation_latex") or equation.split("\n")[0]
                proxy_model = _build_viz_proxy(result_data)
                eq_fig = _render_equation_fig(proxy_model, eq_latex)
                viz_choices = _get_model_capabilities(proxy_model) if proxy_model else []
                viz_update = gr.update(choices=viz_choices, value=viz_choices[0] if viz_choices else None)
                elapsed = result_data.get("elapsed_seconds", "?")
                equation += f"\n\n[Restored from {cpu_job_id}, elapsed={elapsed}s]"
                return (equation, eq_fig, proxy_model, viz_update) + _NO_GPU
            except Exception as e:
                return (f"Failed to load CPU job: {e}",
                        None, None, gr.update()) + _NO_GPU

        cpu_recover_btn.click(
            _recover_cpu_job,
            inputs=[cpu_recover_input],
            outputs=_train_outputs,
            show_progress="hidden",
        )

        # CPU Job History
        def _refresh_cpu_history(request: gr.Request = None):
            return _load_cpu_history(request)

        refresh_cpu_history_btn.click(_refresh_cpu_history, outputs=[cpu_history_df])
        clear_cpu_history_btn.click(_clear_cpu_history, outputs=[cpu_history_df])
        app.load(_refresh_cpu_history, outputs=[cpu_history_df])

        # GPU Job History
        def _refresh_history(request: gr.Request = None):
            return _load_history(request)

        refresh_history_btn.click(_refresh_history, outputs=[history_df])
        clear_gpu_history_btn.click(_clear_gpu_history, outputs=[history_df])
        app.load(_refresh_history, outputs=[history_df])

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

        Only activates when the Host header contains ``bohrium`` (i.e. the
        request comes through Bohrium's gateway).  Local development
        (``localhost``) is left untouched to avoid SSL errors.

        1. Injects X-Forwarded-Proto into requests so Gradio generates
           HTTPS URLs in its frontend config (``root``).
        2. Adds Content-Security-Policy: upgrade-insecure-requests to
           responses so the browser auto-upgrades any remaining HTTP
           sub-resource requests (e.g. theme.css loaded by Gradio JS
           via window.location).
        """
        def __init__(self, app):
            self.app = app

        @staticmethod
        def _is_bohrium(headers):
            """Return True if the request comes through Bohrium's proxy."""
            for key, value in headers:
                if key == b"host" and b"bohrium" in value:
                    return True
            return False

        async def __call__(self, scope, receive, send):
            if scope["type"] in ("http", "websocket") and self._is_bohrium(
                scope.get("headers", [])
            ):
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
