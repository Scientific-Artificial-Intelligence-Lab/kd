
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest




import kd.search.discover.tokens.prior

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RUN_DISCOVER_PATH = _PROJECT_ROOT / "scripts" / "discover" / "run_discover.py"
_BURGERS_STUB_PATH = _PROJECT_ROOT / "scripts" / "discover" / "run_mode2_burgers.py"
_CHAFEE_STUB_PATH = _PROJECT_ROOT / "scripts" / "discover" / "run_mode2_chafee.py"
_SUBPROCESS_TIMEOUT_SECONDS = 60
_ENV_VAR_NAME = "DISCOVER_ENABLE_DIAGNOSTICS"





_BURGERS_ALIGNED_CYCLE_N = 20
_CHAFEE_ALIGNED_ATTN_LENGTH = 20
_DEFAULT_SEED = 42





@pytest.fixture
def pde_registry_module() -> ModuleType:
    return importlib.import_module("kd.search.discover.runners.pde_registry")


@pytest.fixture
def mode2_pipeline_module() -> ModuleType:
    return importlib.import_module("kd.search.discover.runners.mode2_pipeline")


@pytest.fixture(scope="module")
def run_discover_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_run_discover_entry", _RUN_DISCOVER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {_RUN_DISCOVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _baseline_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop(_ENV_VAR_NAME, None)
    return env


def _run_subprocess(
    argv: list[str], env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *argv],
        env=env if env is not None else _baseline_env(),
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        cwd=str(_PROJECT_ROOT),
        check=False,
    )





def _resolve_aligned_cycle_n(spec: Any) -> int | None:
    for attr in ("aligned_preset", "ALIGNED_PRESET"):
        preset = getattr(spec, attr, None)
        if isinstance(preset, dict) and "cycle_n_iterations" in preset:
            return int(preset["cycle_n_iterations"])
    presets = getattr(spec, "presets", None)
    if isinstance(presets, dict):
        aligned = presets.get("aligned")
        if isinstance(aligned, dict) and "cycle_n_iterations" in aligned:
            return int(aligned["cycle_n_iterations"])
    direct = getattr(spec, "aligned_cycle_n_iterations", None)
    if isinstance(direct, int):
        return direct
    pipeline = importlib.import_module("kd.search.discover.runners.mode2_pipeline")
    builder = getattr(pipeline, "build_configs", None)
    if callable(builder):
        _, pinn = builder(pde="burgers", tier="aligned", seed=_DEFAULT_SEED)
        return int(pinn.cycle_n_iterations)
    return None


def _resolve_aligned_attn_length(spec: Any) -> int | None:
    for attr in ("aligned_attn_length", "ALIGNED_ATTN_LENGTH"):
        val = getattr(spec, attr, None)
        if isinstance(val, int):
            return val
    presets = getattr(spec, "presets", None)
    if isinstance(presets, dict):
        aligned = presets.get("aligned")
        if isinstance(aligned, dict) and "attn_length" in aligned:
            return int(aligned["attn_length"])
    pipeline = importlib.import_module("kd.search.discover.runners.mode2_pipeline")
    builder = getattr(pipeline, "build_configs", None)
    if callable(builder):
        config, _ = builder(pde="chafee", tier="aligned", seed=_DEFAULT_SEED)
        return int(config.attn_length)
    return None


@pytest.mark.unit
class TestPDERegistry:

    def test_burgers_spec_present(
        self, pde_registry_module: ModuleType,
    ) -> None:
        registry = pde_registry_module.PDE_REGISTRY
        assert "burgers" in registry, f"keys={list(registry)!r}"
        assert getattr(registry["burgers"], "pde_name", None) == "burgers"

    def test_chafee_spec_present(
        self, pde_registry_module: ModuleType,
    ) -> None:
        registry = pde_registry_module.PDE_REGISTRY
        assert "chafee" in registry, f"keys={list(registry)!r}"
        assert getattr(registry["chafee"], "pde_name", None) == "chafee"

    def test_unknown_pde_raises(
        self, pde_registry_module: ModuleType,
    ) -> None:
        registry = pde_registry_module.PDE_REGISTRY
        with pytest.raises(KeyError):
            _ = registry["unknown_pde_definitely_not_real"]

    def test_burgers_data_path_default_uses_burgers_mat(
        self, pde_registry_module: ModuleType,
    ) -> None:
        spec = pde_registry_module.PDE_REGISTRY["burgers"]
        assert hasattr(spec, "data_path_for_tier"), (
            "PDESpec must expose data_path_for_tier(tier) -> Path"
        )
        path = spec.data_path_for_tier("fast")
        assert isinstance(path, Path), f"expected Path; got {type(path)!r}"
        assert path.name == "burgers.mat", f"got {path.name!r}"

    def test_burgers_data_path_aligned_uses_burgers2_mat(
        self, pde_registry_module: ModuleType,
    ) -> None:
        spec = pde_registry_module.PDE_REGISTRY["burgers"]
        path = spec.data_path_for_tier("aligned")
        assert path.name == "burgers2.mat", f"got {path.name!r}"

    def test_chafee_data_path_uses_chafee_npy(
        self, pde_registry_module: ModuleType,
    ) -> None:
        spec = pde_registry_module.PDE_REGISTRY["chafee"]
        for tier in ("fast", "aligned"):
            path = spec.data_path_for_tier(tier)
            ok = path.suffix == ".npy" or "chafee" in path.name.lower()
            assert ok, f"chafee {tier!r} path invalid: {path!r}"

    def test_burgers_aligned_cycle_n_iterations_is_20(
        self, pde_registry_module: ModuleType,
    ) -> None:
        spec = pde_registry_module.PDE_REGISTRY["burgers"]
        cycle_n = _resolve_aligned_cycle_n(spec)
        assert cycle_n == _BURGERS_ALIGNED_CYCLE_N, (
            f"Burgers ALIGNED cycle_n must == {_BURGERS_ALIGNED_CYCLE_N} "
            f"; got {cycle_n!r}"
        )

    def test_chafee_aligned_attn_length_is_20(
        self, pde_registry_module: ModuleType,
    ) -> None:
        spec = pde_registry_module.PDE_REGISTRY["chafee"]
        attn_length = _resolve_aligned_attn_length(spec)
        assert attn_length == _CHAFEE_ALIGNED_ATTN_LENGTH, (
            f"Chafee ALIGNED attn_length must == "
            f"{_CHAFEE_ALIGNED_ATTN_LENGTH} (paper §VI.A); got {attn_length!r}"
        )





@pytest.mark.unit
class TestRunDiscoverArgs:

    def test_pde_flag_required(
        self, run_discover_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["run_discover.py", "--fast"])
        with pytest.raises(SystemExit):
            run_discover_module.parse_args()

    def test_pde_must_be_known_value(
        self, run_discover_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv",
            ["run_discover.py", "--pde", "foobar_definitely_not_real",
             "--fast"],
        )
        with pytest.raises(SystemExit):
            run_discover_module.parse_args()

    def test_tier_mutex_required(
        self, run_discover_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["run_discover.py", "--pde", "burgers"],
        )
        with pytest.raises(SystemExit):
            run_discover_module.parse_args()

    def test_tier_fast_burgers_parses_ok(
        self, run_discover_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv",
            ["run_discover.py", "--pde", "burgers", "--fast", "--seed", "42"],
        )
        args = run_discover_module.parse_args()
        assert getattr(args, "pde", None) == "burgers"
        assert getattr(args, "seed", None) == 42

    def test_seed_default_42(
        self, run_discover_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["run_discover.py", "--pde", "burgers", "--fast"],
        )
        args = run_discover_module.parse_args()
        assert args.seed == _DEFAULT_SEED, f"got {args.seed!r}"

    def test_diagnostic_scaffold_flag_inherits_phase1(
        self, run_discover_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["run_discover.py", "--pde", "burgers", "--fast"],
        )
        args = run_discover_module.parse_args()
        assert hasattr(args, "diagnostic_scaffold")
        assert args.diagnostic_scaffold is False
        for dest in (
            "diagnostic_scaffold_diffusion_tokens",
            "diagnostic_scaffold_reaction_tokens",
            "diagnostic_scaffold_root_tokens",
            "diagnostic_scaffold_neutral_tokens",
        ):
            assert hasattr(args, dest), f"missing {dest!r}"

    def test_noise_scale_default_std(
        self, run_discover_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["run_discover.py", "--pde", "burgers", "--fast"],
        )
        args = run_discover_module.parse_args()
        assert args.noise_scale == "std", f"got {args.noise_scale!r}"

    def test_chafee_specific_noise_level_flag_exists(
        self, run_discover_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv",
            ["run_discover.py", "--pde", "chafee", "--fast",
             "--noise-level", "0.1"],
        )
        args = run_discover_module.parse_args()
        assert hasattr(args, "noise_level"), "--noise-level not registered"
        assert args.noise_level == pytest.approx(0.1)





@pytest.mark.slow
class TestRunDiscoverDispatch:

    def test_main_burgers_fast_dry_run(self) -> None:
        if not _RUN_DISCOVER_PATH.exists():
            pytest.skip("scripts/run_discover.py not yet implemented (RED)")
        help_result = _run_subprocess(
            [str(_RUN_DISCOVER_PATH), "--pde", "burgers", "--help"],
        )
        if "--dry-run" not in help_result.stdout:
            pytest.skip("--dry-run not in --help; dev did not implement")
        result = _run_subprocess(
            [str(_RUN_DISCOVER_PATH), "--pde", "burgers", "--fast",
             "--seed", "42", "--dry-run"],
        )
        assert result.returncode == 0, (
            f"--dry-run dispatch must succeed; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    def test_diagnostic_env_gate_inherited(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not _RUN_DISCOVER_PATH.exists():
            pytest.skip("scripts/run_discover.py not yet implemented (RED)")
        env = _baseline_env()
        result = _run_subprocess(
            [str(_RUN_DISCOVER_PATH), "--pde", "burgers", "--fast",
             "--diagnostic-scaffold",
             "--diagnostic-scaffold-diffusion-tokens", "diff2_x",
             "--diagnostic-scaffold-reaction-tokens", "u"],
            env=env,
        )
        assert result.returncode != 0, (
            f"CLI must fail without {_ENV_VAR_NAME}=1; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert _ENV_VAR_NAME in result.stderr, (
            f"stderr must mention {_ENV_VAR_NAME}; got {result.stderr!r}"
        )

    def test_main_chafee_help_smoke(self) -> None:
        if not _RUN_DISCOVER_PATH.exists():
            pytest.skip("scripts/run_discover.py not yet implemented (RED)")
        result = _run_subprocess(
            [str(_RUN_DISCOVER_PATH), "--pde", "chafee", "--help"],
        )
        assert result.returncode == 0, (
            f"--pde chafee --help must succeed; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )





@pytest.mark.slow
class TestStubBackwardsCompat:

    def _assert_stub_forwards(self, stub_path: Path, pde: str) -> None:
        if not stub_path.exists():
            pytest.skip(f"{stub_path.name} not present (deleted currently?)")
        result = _run_subprocess([str(stub_path), "--help"])
        assert result.returncode == 0, (
            f"{stub_path.name} --help must succeed; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        forwarded_via_help = (
            "--pde" in result.stdout
            or f"--pde {pde}" in result.stdout
        )
        forwarded_via_warn = (
            "run_discover" in result.stderr.lower()
            or "deprecat" in result.stderr.lower()
        )
        assert forwarded_via_help or forwarded_via_warn, (
            f"{stub_path.name} must forward to run_discover.py; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    def test_burgers_stub_forwards_to_run_discover(self) -> None:
        self._assert_stub_forwards(_BURGERS_STUB_PATH, "burgers")

    def test_chafee_stub_forwards_to_run_discover(self) -> None:
        self._assert_stub_forwards(_CHAFEE_STUB_PATH, "chafee")

    def _assert_stub_rejects_explicit_pde(
        self, stub_path: Path, expected_pde: str, foreign_pde: str,
    ) -> None:
        if not stub_path.exists():
            pytest.skip(f"{stub_path.name} not present (deleted currently?)")
        result = _run_subprocess(
            [str(stub_path), "--pde", foreign_pde, "--fast"],
        )
        assert result.returncode == 2, (
            f"{stub_path.name} must exit 2 when --pde flag passed; "
            f"rc={result.returncode} stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        assert "cannot accept an explicit --pde flag" in result.stderr, (
            f"stub stderr must explain --pde rejection contract; "
            f"got {result.stderr!r}"
        )

    def test_burgers_stub_rejects_explicit_pde_flag(self) -> None:
        self._assert_stub_rejects_explicit_pde(
            _BURGERS_STUB_PATH, "burgers", "chafee",
        )

    def test_chafee_stub_rejects_explicit_pde_flag(self) -> None:
        self._assert_stub_rejects_explicit_pde(
            _CHAFEE_STUB_PATH, "chafee", "burgers",
        )





@pytest.mark.unit
class TestMode2Pipeline:

    def test_pipeline_module_exists(
        self, mode2_pipeline_module: ModuleType,
    ) -> None:
        builder = getattr(mode2_pipeline_module, "build_configs", None)
        assert callable(builder), (
            "mode2_pipeline must expose build_configs(pde, tier, seed)"
        )
        obs = getattr(mode2_pipeline_module, "make_observation_data", None)
        assert callable(obs), (
            "mode2_pipeline must expose make_observation_data helper"
        )

    def test_pipeline_build_configs_burgers_aligned(
        self, mode2_pipeline_module: ModuleType,
    ) -> None:
        from kd.search.discover.config import DiscoverConfig, PINNConfig

        result = mode2_pipeline_module.build_configs(
            pde="burgers", tier="aligned", seed=_DEFAULT_SEED,
        )
        assert isinstance(result, tuple) and len(result) == 2, (
            f"expected (DiscoverConfig, PINNConfig); got {type(result)!r}"
        )
        config, pinn = result
        assert isinstance(config, DiscoverConfig)
        assert isinstance(pinn, PINNConfig)
        assert pinn.cycle_n_iterations == _BURGERS_ALIGNED_CYCLE_N

    def test_pipeline_build_configs_chafee_aligned(
        self, mode2_pipeline_module: ModuleType,
    ) -> None:
        from kd.search.discover.config import DiscoverConfig, PINNConfig

        config, pinn = mode2_pipeline_module.build_configs(
            pde="chafee", tier="aligned", seed=_DEFAULT_SEED,
        )
        assert isinstance(config, DiscoverConfig)
        assert isinstance(pinn, PINNConfig)
        assert config.attn_length == _CHAFEE_ALIGNED_ATTN_LENGTH
