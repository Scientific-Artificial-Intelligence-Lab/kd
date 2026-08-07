
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import kd
import kd.core.equation
import kd.search
import kd.viz.plots
from kd.api import _PLUGIN_CLASS_BY_ALGORITHM





_ENGINE_PAIRS: tuple[tuple[str, str], ...] = (
    ("SGAConfig", "SGAPlugin"),
    ("DLGAConfig", "DLGAPlugin"),
    ("DiscoverConfig", "DISCOVERPlugin"),
    ("PySRConfig", "PySRPlugin"),
    ("EqGPTConfig", "EqGPTPlugin"),
    ("Llm4edConfig", "Llm4edPlugin"),
    ("PySINDyConfig", "PySINDyPlugin"),
)

DECIDED: dict[str, frozenset[str]] = {

    "kd": frozenset(
        {
            "SGAConfig",
            "DLGAConfig",
            "DiscoverConfig",
            "EqGPTConfig",
            "Llm4edConfig",
            "PySRConfig",
            "PySINDyConfig",
            "VerificationReport",
        }
    ),



    "kd.search": frozenset(
        {
            "RunRecord",
            "InstrumentDescriptor",
            "InstrumentMode",
            "Knob",
            "tool_schema",
            "build_mini_table",
            "BEST_SCORE_KEY",
        }
    ),



    "kd.harness": frozenset(
        {
            "MemberVerification",
            "WorkerLogRow",
            "read_dispatch_log",
            "write_dispatch_log",
            "decode_dispatch_log",
        }
    ),

    "kd.core": frozenset(
        {
            "IntegrationResult",
            "integrate_pde",
            "VerificationReport",
            "empirical_agreement",
            "law_agreement",
            "write_verification_artifact",
            "ScorerFn",
            "aic",
            "aic_no_n",
            "aicc",
            "bic",
            "make_aic_scorer",
            "make_bic_scorer",
            "make_sga_scorer",
            "nmse",
        }
    ),

    "kd.core.expr": frozenset(
        {
            "to_sympy",
            "to_latex",
            "to_unicode",
            "format_pde",
            "are_equivalent",
            "from_sympy",
            "symbolic_diff",
            "FormattedEquation",
        }
    ),







    "kd.viz": frozenset(
        {
            "ReportResult",
            "FigureSpec",
            "style_context",
            "EquationDisplay",
            "latex_display",
        }
    ),

    "kd.viz.plots": frozenset(
        {
            "plot_coefficient_bar",
            "plot_convergence",
            "plot_equation",
            "plot_equation_tree",
            "plot_error_heatmap",
            "plot_field_animation",
            "plot_field_comparison",
            "plot_parity",
            "plot_pde_residual_field",
            "plot_residual",
            "plot_score_bar",
            "plot_summary_table",
            "plot_time_slices",
            "render_overlaid_convergence",
        }
    ),

    "kd.data.loaders": frozenset(
        {
            "WaveBreakingCase",
            "WaveBreakingFit",
            "default_wave_pkl_path",
            "evaluate_known_terms",
            "load_wave_breaking_cases",
            "resolve_v1_wave_asset_dir",
            "wave_breaking_case_to_dataset",
            "wave_breaking_star_grid",
            "wave_dataset_name",
            "wave_surrogate_checkpoint_path",
        }
    ),
    "kd.search.llm4ed": frozenset({"Llm4edConfig", "Llm4edPlugin"}),
    "kd.search.pysr": frozenset({"PySRSymbolicRegressor"}),
}


@pytest.mark.parametrize("module_path", sorted(DECIDED))
def test_decided_names_are_exported_and_resolve(module_path: str) -> None:
    module = importlib.import_module(module_path)
    names = DECIDED[module_path]
    assert names <= set(module.__all__)
    for name in names:
        assert getattr(module, name) is not None


def test_engine_configs_uniform_at_top_level() -> None:
    configs = DECIDED["kd"] - {"VerificationReport"}
    for name in configs:
        assert hasattr(kd, name)


def test_viz_plots_facade_carries_decided_names() -> None:
    assert DECIDED["kd.viz.plots"] <= set(kd.viz.plots.__all__)


def test_reserved_coefficient_variants_stay_unexported() -> None:
    for name in ("Field", "Hole", "Posterior", "Coefficient", "Term"):
        assert name not in kd.core.equation.__all__
        assert not hasattr(kd.core.equation, name)


def test_kd_all_full_set_equality() -> None:
    expected = {
        "AxisInfo",
        "AxisReport",
        "CheckpointManifestEntry",
        "CheckpointManifestError",
        "DATASET_CATALOG",
        "DLGAConfig",
        "DataTopology",
        "DatasetReport",
        "DatasetSpec",
        "DiscoverConfig",
        "EqGPTConfig",
        "EvaluationFailedError",
        "EvaluationResult",
        "ExperimentResult",
        "FINAL_STATUS_COMPLETED",
        "FieldData",
        "FieldReport",
        "InvalidTermsError",
        "KIND_FINAL",
        "Llm4edConfig",
        "Model",
        "PDEDataset",
        "PySINDyConfig",
        "PySRConfig",
        "SGAConfig",
        "TabularDataset",
        "TaskType",
        "TermRejection",
        "TermValidationReport",
        "VerificationReport",
        "VerifyPolicy",
        "VizEngine",
        "__version__",
        "evaluate_terms",
        "generate_advection_data",
        "generate_burgers_data",
        "generate_diffusion_data",
        "get_dataset",
        "instrument_schemas",
        "law_signature",
        "list_datasets",
        "list_remote_datasets",
        "load_allen_cahn",
        "load_burgers",
        "load_burgers_2d",
        "load_chafee_infante",
        "load_checkpoint_manifest",
        "load_convection_diffusion",
        "load_eq_6_2_12",
        "load_from_hub",
        "load_kdv",
        "load_klein_gordon",
        "load_llm4ed_fisher",
        "load_llm4ed_fisher_nonlinear",
        "load_llm4ed_heat",
        "load_pde_compound",
        "load_pde_divide",
        "load_tlc_cc",
        "load_wave",
        "load_wave_breaking",
        "preview",
        "preview_report",
        "validate_terms",
        "verify_equation",
    }
    assert set(kd.__all__) == expected


def test_search_facade_carries_all_seven_plugin_pairs() -> None:
    assert len(_ENGINE_PAIRS) == len(_PLUGIN_CLASS_BY_ALGORITHM)
    for config_name, plugin_name in _ENGINE_PAIRS:
        for name in (config_name, plugin_name):
            assert name in kd.search.__all__
            assert getattr(kd.search, name) is not None


def test_import_kd_does_not_load_scipy_integrate() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import kd, sys; "
            "raise SystemExit(1 if 'scipy.integrate' in sys.modules else 0)",
        ],
        check=False,
    )
    assert result.returncode == 0
