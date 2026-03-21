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


# ── Constants ──────────────────────────────────────────────────

MODEL_KEYS = {
    "KD_SGA": "sga",
    "KD_DLGA": "dlga",
    "KD_DSCV": "dscv",
    "KD_DSCV_SPR": "dscv_spr",
}

ACTIVE_DATASETS = sorted([
    name for name, info in PDE_REGISTRY.items()
    if info.get("status") != "pending"
])

# Operator presets per model
DSCV_BINARY_DEFAULT = "add, mul, diff"
DSCV_UNARY_DEFAULT = "n2"
SPR_BINARY_DEFAULT = "add_t, mul_t, div_t, diff_t, diff2_t"
SPR_UNARY_DEFAULT = "n2_t"
DLGA_OPS_DEFAULT = "u, u_x, u_xx, u_xxx"

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

    if model_name == "KD_DSCV_SPR":
        binary_val = SPR_BINARY_DEFAULT
        unary_val = SPR_UNARY_DEFAULT
    else:
        binary_val = DSCV_BINARY_DEFAULT
        unary_val = DSCV_UNARY_DEFAULT

    return (
        gr.update(visible=sga_vis),   # sga_group
        gr.update(visible=dlga_vis),  # dlga_group
        gr.update(visible=dscv_vis),  # dscv_group
        gr.update(visible=spr_vis),   # spr_group
        gr.update(value=binary_val),  # dscv_binary
        gr.update(value=unary_val),   # dscv_unary
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
    from kd.model.kd_dscv import KD_DSCV_SPR
    if not isinstance(model, KD_DSCV_SPR):
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
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    try:
        ax.text(0.5, 0.5, f"${equation_text}$", fontsize=18,
                ha="center", va="center", transform=ax.transAxes)
    except Exception:
        ax.text(0.5, 0.5, equation_text, fontsize=14,
                ha="center", va="center", transform=ax.transAxes, family="monospace")
    fig.tight_layout()
    return fig


# ── Training ───────────────────────────────────────────────────

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
):
    """Core training function called by Gradio."""
    if not dataset_name or not model_name:
        return "Please select dataset and model first.", None, None

    configure(save_dir=None)

    try:
        dataset = load_pde(dataset_name)
        model = None
        result = None

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
            ops = _parse_ops(dlga_ops)
            model = KD_DLGA(
                operators=ops,
                epi=float(dlga_epi),
                input_dim=2,
                verbose=True,
                max_iter=int(dlga_max_iter),
            )
            sample = int(dlga_sample) if dlga_sample else None
            model.fit_dataset(dataset, sample=sample)

        elif model_name == "KD_DSCV":
            from kd.model.kd_dscv import KD_DSCV
            model = KD_DSCV(
                binary_operators=_parse_ops(dscv_binary),
                unary_operators=_parse_ops(dscv_unary),
                n_samples_per_batch=int(dscv_batch),
                seed=int(dscv_seed),
            )
            np.random.seed(int(dscv_seed))
            result = model.fit_dataset(dataset, n_epochs=int(dscv_epochs))

        elif model_name == "KD_DSCV_SPR":
            from kd.model.kd_dscv import KD_DSCV_SPR
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
            return f"Unknown model: {model_name}", None, None

        # Extract equation text
        equation = _get_equation_text(model, model_name, result)

        # Generate equation visualization
        fig = _render_equation_fig(model, equation)

        # Get available viz types for this model
        caps = _get_model_capabilities(model)
        viz_update = gr.update(choices=caps, value=caps[0] if caps else None)

        return equation, fig, model, viz_update

    except Exception as e:
        return f"Training error:\n{e}\n\n{traceback.format_exc()}", None, None, gr.update()


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

        with gr.Tabs():
            # ── Tab 1: Training ──────────────────────────
            with gr.TabItem("Training"):
                with gr.Row():
                    # Left: controls
                    with gr.Column(scale=1):
                        dataset_dd = gr.Dropdown(
                            choices=ACTIVE_DATASETS,
                            value="burgers" if "burgers" in ACTIVE_DATASETS else ACTIVE_DATASETS[0],
                            label="Dataset",
                        )
                        model_dd = gr.Dropdown(choices=[], label="Model")

                        # -- SGA params --
                        with gr.Group(visible=False) as sga_group:
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

                        with gr.Row():
                            train_btn = gr.Button("Start Training", variant="primary", size="lg")
                            cancel_btn = gr.Button("Cancel", variant="stop", size="lg")

                    # Right: results
                    with gr.Column(scale=2):
                        eq_text = gr.Textbox(label="Discovered Equation", lines=4, interactive=False)
                        eq_plot = gr.Plot(label="Equation Visualization")

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

        # Initialize model list for default dataset on page load
        app.load(on_dataset_change, inputs=[dataset_dd], outputs=[model_dd])

        dataset_dd.change(on_dataset_change, inputs=[dataset_dd], outputs=[model_dd])

        model_dd.change(
            on_model_change,
            inputs=[model_dd],
            outputs=[sga_group, dlga_group, dscv_group, spr_group, dscv_binary, dscv_unary],
        )

        train_event = train_btn.click(
            run_training,
            inputs=[
                dataset_dd, model_dd,
                sga_run, sga_num, sga_depth, sga_seed,
                dlga_ops, dlga_epi, dlga_max_iter, dlga_sample,
                dscv_binary, dscv_unary, dscv_batch, dscv_epochs, dscv_seed,
                spr_sample_ratio, spr_epochs,
            ],
            outputs=[eq_text, eq_plot, model_state, viz_dd],
        )

        cancel_btn.click(
            fn=lambda: ("Training cancelled.", None, None, gr.update()),
            outputs=[eq_text, eq_plot, model_state, viz_dd],
            cancels=[train_event],
        )

        viz_btn.click(
            run_viz,
            inputs=[viz_dd, model_state],
            outputs=[viz_plot, viz_msg],
        )

    return app


# ── Entry Point ────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    app.launch(server_name="0.0.0.0", server_port=50001)
