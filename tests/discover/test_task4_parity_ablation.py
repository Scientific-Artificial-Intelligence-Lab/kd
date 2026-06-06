
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = (
    _PROJECT_ROOT / "scripts" / "discover" / "research" / "analysis"
    / "task4_parity_ablation.py"
)


def _load_script_module() -> ModuleType:
    scripts_dir = str(_SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "test_task4_parity_ablation_runner",
        _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _load_script_module()


def _make_args(runner: ModuleType, **overrides: object) -> object:
    import argparse

    defaults = {
        "config_name": "A",
        "seed": 0,
        "data_path": Path("dummy"),
        "output_dir": Path("dummy"),
        "heartbeat_iterations": 10,
        "profile_fast": False,
        "pretrain_epoch": None,
        "n_iterations": None,
        "n_cycles": None,
        "stability_selection": None,
        "colloc_chunk_size": None,
        "device": "cpu",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.unit
class TestCollocChunkSizeBudget:

    def test_default_budget_chunk_size_is_none(self, runner: ModuleType) -> None:
        budget = runner.BudgetOverride()
        assert budget.colloc_chunk_size is None

    def test_profile_fast_defaults_chunk_to_8000(self, runner: ModuleType) -> None:
        args = _make_args(runner, profile_fast=True)
        budget = runner._resolve_budget(args)
        assert budget.colloc_chunk_size == 8000

    def test_explicit_chunk_size_threads_through(self, runner: ModuleType) -> None:
        args = _make_args(runner, colloc_chunk_size=1234)
        budget = runner._resolve_budget(args)
        assert budget.colloc_chunk_size == 1234

    def test_explicit_chunk_size_overrides_profile_fast(
        self, runner: ModuleType
    ) -> None:
        args = _make_args(runner, profile_fast=True, colloc_chunk_size=4096)
        budget = runner._resolve_budget(args)
        assert budget.colloc_chunk_size == 4096

    def test_no_flags_keeps_chunk_none(self, runner: ModuleType) -> None:
        args = _make_args(runner)
        budget = runner._resolve_budget(args)
        assert budget.colloc_chunk_size is None


@pytest.mark.unit
class TestPINNConfigChunkSizeWiring:

    def test_chunk_size_propagated_to_pinn_config(self, runner: ModuleType) -> None:
        parity = runner.CONFIGS["A"]
        budget = runner.BudgetOverride(colloc_chunk_size=8000)
        _, pinn_config = runner._build_configs(parity, budget)
        assert pinn_config.colloc_chunk_size == 8000

    def test_default_chunk_size_yields_none_pinn_config(
        self, runner: ModuleType
    ) -> None:
        parity = runner.CONFIGS["A"]
        budget = runner.BudgetOverride()
        _, pinn_config = runner._build_configs(parity, budget)
        assert pinn_config.colloc_chunk_size is None


@pytest.mark.unit
class TestPaperAlignedPConfig:

    def test_p_config_has_paper_arch(self, runner: ModuleType) -> None:
        p = runner.CONFIGS["P"]
        assert p.obs_ratio == 0.80
        assert p.activation == "sin"
        assert p.number_layer == 3
        assert p.n_hidden == 64

    def test_p_config_triggers_paper_controller_overrides(
        self, runner: ModuleType
    ) -> None:
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()
        config, _ = runner._build_configs(parity, budget, config_name="P")
        assert config.batch_size == 2500
        assert config.epsilon == 0.004
        assert config.attn_length == 20

    def test_non_p_config_preserves_legacy_controller_defaults(
        self, runner: ModuleType
    ) -> None:
        for cname in ("A", "C", "D", "F", "G"):
            parity = runner.CONFIGS[cname]
            budget = runner.BudgetOverride()
            config, _ = runner._build_configs(parity, budget, config_name=cname)
            assert config.batch_size == 500, cname
            assert config.epsilon == 0.02, cname
            assert config.attn_length == 10, cname

    def test_p_pinn_config_has_paper_arch(self, runner: ModuleType) -> None:
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()
        _, pinn_config = runner._build_configs(parity, budget, config_name="P")
        assert pinn_config.number_layer == 3
        assert pinn_config.n_hidden == 64
        assert pinn_config.activation == "sin"


@pytest.mark.unit
class TestWave3H19ControllerCLIOverrides:

    def test_cli_epsilon_and_entropy_weight_override_p_config(
        self, runner: ModuleType
    ) -> None:
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()
        config, _ = runner._build_configs(
            parity,
            budget,
            config_name="P",
            cli_epsilon=0.05,
            cli_entropy_weight=0.1,
        )
        assert config.epsilon == 0.05
        assert config.entropy_weight == 0.1

    def test_cli_entropy_gamma_overrides_module_default(
        self, runner: ModuleType
    ) -> None:
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()
        config, _ = runner._build_configs(
            parity, budget, config_name="P", cli_entropy_gamma=0.9
        )
        assert config.entropy_gamma == 0.9

    def test_none_cli_flags_preserve_p_config_overrides(
        self, runner: ModuleType
    ) -> None:
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()
        config, _ = runner._build_configs(parity, budget, config_name="P")
        assert config.epsilon == 0.004
        assert config.entropy_weight == runner.ENTROPY_WEIGHT
        assert config.entropy_gamma == runner.ENTROPY_GAMMA

    def test_p_config_returns_module_defaults_for_entropy(
        self, runner: ModuleType
    ) -> None:
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()
        config, _ = runner._build_configs(
            parity, budget, config_name="P",
            cli_entropy_weight=None, cli_entropy_gamma=None,
        )
        assert config.entropy_weight == runner.ENTROPY_WEIGHT
        assert config.entropy_gamma == runner.ENTROPY_GAMMA

        cli_config, _ = runner._build_configs(
            parity, budget, config_name="P",
            cli_entropy_weight=0.07, cli_entropy_gamma=0.55,
        )
        assert cli_config.entropy_weight == 0.07
        assert cli_config.entropy_gamma == 0.55


@pytest.mark.unit
class TestWave3H19PINNAndPriorCLIOverrides:

    def test_coef_pde_threads_to_pinn_config(self, runner: ModuleType) -> None:
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()

        _, pinn_config = runner._build_configs(
            parity, budget, config_name="P", coef_pde=0.1
        )
        assert pinn_config.coef_pde == 0.1

        _, pinn_config = runner._build_configs(
            parity, budget, config_name="P", coef_pde=0.5
        )
        assert pinn_config.coef_pde == 0.5

        _, pinn_config = runner._build_configs(parity, budget, config_name="P")
        assert pinn_config.coef_pde == 1.0

    def test_disable_diff_child_prior_threads_to_config(
        self, runner: ModuleType
    ) -> None:
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()
        config, _ = runner._build_configs(
            parity, budget, config_name="P", disable_diff_child_prior=True
        )
        assert config.use_diff_child_prior is False

        config, _ = runner._build_configs(parity, budget, config_name="P")
        assert config.use_diff_child_prior is True

    def test_token_bias_args_thread_to_config(self, runner: ModuleType) -> None:
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()
        config, _ = runner._build_configs(
            parity,
            budget,
            config_name="P",
            token_bias_tokens=("u_x", "u_xx"),
            token_bias_weight=2.0,
        )
        assert config.token_bias_tokens == ("u_x", "u_xx")
        assert config.token_bias_weight == 2.0

        config, _ = runner._build_configs(parity, budget, config_name="P")
        assert config.token_bias_tokens == ()
        assert config.token_bias_weight == 0.0


@pytest.mark.unit
class TestAcPaperOverridesAllowedKeys:

    def test_ac_paper_overrides_unknown_key_raises(
        self, runner: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad_overrides = {
            "P": {
                "batch_size": 2500,
                "epsilon": 0.004,
                "attn_length": 20,

                "entropy_weight": 0.07,
            },
        }
        monkeypatch.setattr(
            runner, "_AC_PAPER_CONTROLLER_OVERRIDES", bad_overrides
        )
        parity = runner.CONFIGS["P"]
        budget = runner.BudgetOverride()
        with pytest.raises(ValueError, match="entropy_weight"):
            runner._build_configs(parity, budget, config_name="P")

    def test_ac_paper_overrides_known_keys_pass(
        self, runner: ModuleType
    ) -> None:
        allowed = runner._AC_PAPER_CONTROLLER_OVERRIDE_ALLOWED_KEYS
        for cname, overrides in runner._AC_PAPER_CONTROLLER_OVERRIDES.items():
            extra = set(overrides) - allowed
            assert not extra, (
                f"_AC_PAPER_CONTROLLER_OVERRIDES['{cname}'] has unsupported "
                f"keys {extra}; extend _build_configs precedence chain"
            )









def _make_fake_result_for_gate_payload() -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        pretrain_result=SimpleNamespace(train_loss=1e-4),
        cycle_metrics=[{"physics_loss": 1e-4}],
    )


@pytest.mark.unit
class TestStructuralGateInParityGatesPayload:

    def test_correct_form_structural_ok_true(self, runner: ModuleType) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = runner._gate_payload(
            result,
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            {"max_rel_coef_error": 0.003, "l1_ratio_error": 0.01},
        )
        assert payload["structural_ok"] is True

        assert payload["structure_hit"] is True

    def test_v10_cheating_form_structural_ok_false(
        self, runner: ModuleType
    ) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = runner._gate_payload(
            result,
            "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
            {"max_rel_coef_error": 0.003, "l1_ratio_error": 0.01},
        )
        assert payload["structural_ok"] is False

        assert payload["structure_hit"] is True


@pytest.mark.unit
class TestReleasePassInParityGatesPayload:

    def test_release_pass_field_present(self, runner: ModuleType) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = runner._gate_payload(
            result,
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            {"max_rel_coef_error": 0.003, "l1_ratio_error": 0.01},
        )
        assert "release_pass" in payload

    def test_correct_form_release_pass_true(self, runner: ModuleType) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = runner._gate_payload(
            result,
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            {"max_rel_coef_error": 0.003, "l1_ratio_error": 0.01},
        )
        assert payload["release_pass"] is True

    def test_v10_cheating_release_pass_false(self, runner: ModuleType) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = runner._gate_payload(
            result,
            "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
            {"max_rel_coef_error": 0.003, "l1_ratio_error": 0.01},
        )
        assert payload["release_pass"] is False
        assert payload["structural_ok"] is False
        assert payload["structure_hit"] is True
