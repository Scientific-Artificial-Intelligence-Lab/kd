from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest





import kd.search.discover.tokens.prior
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.runners.diagnostic_cli import (
    build_scaffold_kwargs,
    warn_orphan_token_lists,
)

ENV_VAR_NAME = "DISCOVER_ENABLE_DIAGNOSTICS"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BURGERS_SCRIPT = _PROJECT_ROOT / "scripts" / "discover" / "run_mode2_burgers.py"
_CHAFEE_SCRIPT = _PROJECT_ROOT / "scripts" / "discover" / "run_mode2_chafee.py"
_PARITY_SCRIPT = (
    _PROJECT_ROOT / "scripts" / "discover" / "research" / "analysis" / "task4_parity_ablation.py"
)
_SUBPROCESS_TIMEOUT_SECONDS = 60





@pytest.fixture(autouse=True)
def _ensure_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR_NAME, raising=False)


def _load_script(path: Path, module_name: str) -> ModuleType:
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def burgers_runner() -> ModuleType:
    return _load_script(_BURGERS_SCRIPT, "td012_burgers_runner")


@pytest.fixture(scope="module")
def chafee_runner() -> ModuleType:
    return _load_script(_CHAFEE_SCRIPT, "td012_chafee_runner")


@pytest.fixture(scope="module")
def parity_runner() -> ModuleType:
    return _load_script(_PARITY_SCRIPT, "td012_parity_runner")





@pytest.mark.unit
class TestConfigLayerGate:

    def test_default_config_no_env_required(self) -> None:


        config = DiscoverConfig()
        assert config.diagnostic_scaffold is False

    def test_diagnostic_true_no_env_raises(self) -> None:
        with pytest.raises(RuntimeError):
            DiscoverConfig(
                diagnostic_scaffold=True,
                diagnostic_scaffold_diffusion_tokens=("diff2_x",),
                diagnostic_scaffold_reaction_tokens=("u",),
            )

    def test_diagnostic_true_env_empty_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAME, "")
        with pytest.raises(RuntimeError):
            DiscoverConfig(
                diagnostic_scaffold=True,
                diagnostic_scaffold_diffusion_tokens=("diff2_x",),
                diagnostic_scaffold_reaction_tokens=("u",),
            )

    def test_diagnostic_true_env_zero_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAME, "0")
        with pytest.raises(RuntimeError):
            DiscoverConfig(
                diagnostic_scaffold=True,
                diagnostic_scaffold_diffusion_tokens=("diff2_x",),
                diagnostic_scaffold_reaction_tokens=("u",),
            )

    @pytest.mark.parametrize(
        "env_value",
        ["true", "True", "TRUE", "yes", "on", "Y", "enabled", "2", " 1", "1 "],
    )
    def test_diagnostic_true_env_truthy_strings_raise(
        self, monkeypatch: pytest.MonkeyPatch, env_value: str,
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAME, env_value)
        with pytest.raises(RuntimeError):
            DiscoverConfig(
                diagnostic_scaffold=True,
                diagnostic_scaffold_diffusion_tokens=("diff2_x",),
                diagnostic_scaffold_reaction_tokens=("u",),
            )

    def test_diagnostic_true_env_one_succeeds(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAME, "1")
        config = DiscoverConfig(
            diagnostic_scaffold=True,
            diagnostic_scaffold_diffusion_tokens=("diff2_x",),
            diagnostic_scaffold_reaction_tokens=("u",),
        )
        assert config.diagnostic_scaffold is True

    def test_token_lists_alone_no_env_required(self) -> None:


        config = DiscoverConfig(
            diagnostic_scaffold_diffusion_tokens=("diff2_x",),
            diagnostic_scaffold_reaction_tokens=("u",),
            diagnostic_scaffold_root_tokens=("add", "sub"),
            diagnostic_scaffold_neutral_tokens=("u", "x", "t"),
        )
        assert config.diagnostic_scaffold is False

    def test_error_message_mentions_env_var_name(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            DiscoverConfig(
                diagnostic_scaffold=True,
                diagnostic_scaffold_diffusion_tokens=("diff2_x",),
                diagnostic_scaffold_reaction_tokens=("u",),
            )
        assert ENV_VAR_NAME in str(excinfo.value), (
            "error message must include the env var name verbatim so "
            "users can grep for it; got: " + repr(str(excinfo.value))
        )

    def test_error_message_actionable(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            DiscoverConfig(
                diagnostic_scaffold=True,
                diagnostic_scaffold_diffusion_tokens=("diff2_x",),
                diagnostic_scaffold_reaction_tokens=("u",),
            )
        message = str(excinfo.value)
        assert "=1" in message, (
            "error message must include the activation hint '=1' so the "
            "remediation is obvious from stderr alone; got: " + repr(message)
        )

    def test_error_message_signals_diagnostic_intent(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            DiscoverConfig(
                diagnostic_scaffold=True,
                diagnostic_scaffold_diffusion_tokens=("diff2_x",),
                diagnostic_scaffold_reaction_tokens=("u",),
            )
        message_lower = str(excinfo.value).lower()
        intent_markers = ("diagnostic", "dev", "ground truth", "gt", "leak")
        assert any(marker in message_lower for marker in intent_markers), (
            "error must include at least one diagnostic-intent marker "
            f"({intent_markers}); got: {str(excinfo.value)!r}"
        )

    def test_idempotent_multiple_constructions(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAME, "1")
        for _ in range(5):
            config = DiscoverConfig(
                diagnostic_scaffold=True,
                diagnostic_scaffold_diffusion_tokens=("diff2_x",),
                diagnostic_scaffold_reaction_tokens=("u",),
            )
            assert config.diagnostic_scaffold is True

        assert os.environ.get(ENV_VAR_NAME) == "1"

    def test_default_config_with_env_set_is_still_off(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAME, "1")
        config = DiscoverConfig()
        assert config.diagnostic_scaffold is False





@pytest.mark.unit
class TestCLIInProcessGate:

    def test_burgers_parser_with_env_set_accepts_diagnostic_flag(
        self, burgers_runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAME, "1")
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_mode2_burgers.py",
                "--fast",
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x",
                "--diagnostic-scaffold-reaction-tokens", "u",
            ]
            args = burgers_runner.parse_args()
        finally:
            sys.argv = old_argv
        assert args.diagnostic_scaffold is True

    def test_chafee_parser_with_env_set_accepts_diagnostic_flag(
        self, chafee_runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAME, "1")
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_mode2_chafee.py",
                "--fast",
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x",
                "--diagnostic-scaffold-reaction-tokens", "u",
            ]
            args = chafee_runner.parse_args()
        finally:
            sys.argv = old_argv
        assert args.diagnostic_scaffold is True

    def test_parity_parser_with_env_set_accepts_diagnostic_flag(
        self, parity_runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAME, "1")
        old_argv = sys.argv
        try:
            sys.argv = [
                "task4_parity_ablation.py",
                "--config-name", "A",
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x",
                "--diagnostic-scaffold-reaction-tokens", "u",
            ]
            args = parity_runner.parse_args()
        finally:
            sys.argv = old_argv
        assert args.diagnostic_scaffold is True





@pytest.mark.unit
class TestScaffoldKwargsDefaultOff:

    @staticmethod
    def _default_namespace() -> argparse.Namespace:

        return argparse.Namespace(
            diagnostic_scaffold=False,
            diagnostic_scaffold_diffusion_tokens=None,
            diagnostic_scaffold_reaction_tokens=None,
            diagnostic_scaffold_root_tokens=None,
            diagnostic_scaffold_neutral_tokens=None,
        )

    def test_burgers_scaffold_kwargs_default_returns_empty(
        self, burgers_runner: ModuleType,
    ) -> None:




        _ = burgers_runner
        ns = self._default_namespace()
        kwargs = build_scaffold_kwargs(ns)
        assert kwargs == {}, (
            "default args must yield empty kwargs so DiscoverConfig keeps "
            f"frozen defaults; got: {kwargs!r}"
        )

    def test_chafee_scaffold_kwargs_default_returns_empty(
        self, chafee_runner: ModuleType,
    ) -> None:

        _ = chafee_runner
        ns = self._default_namespace()
        kwargs = build_scaffold_kwargs(ns)
        assert kwargs == {}, (
            "default args must yield empty kwargs so DiscoverConfig keeps "
            f"frozen defaults; got: {kwargs!r}"
        )

    def test_parity_resolve_diagnostic_default_returns_none(
        self, parity_runner: ModuleType,
    ) -> None:
        ns = self._default_namespace()
        result = parity_runner._resolve_diagnostic(ns)
        assert result is None, (
            "default args must yield None so _build_configs leaves "
            f"DiscoverConfig at its frozen defaults; got: {result!r}"
        )

    def test_burgers_scaffold_kwargs_active_propagates_tokens(
        self, burgers_runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import argparse


        _ = burgers_runner
        monkeypatch.setenv(ENV_VAR_NAME, "1")
        ns = argparse.Namespace(
            diagnostic_scaffold=True,
            diagnostic_scaffold_diffusion_tokens=["diff2_x"],
            diagnostic_scaffold_reaction_tokens=["u"],
            diagnostic_scaffold_root_tokens=None,
            diagnostic_scaffold_neutral_tokens=None,
        )
        kwargs = build_scaffold_kwargs(ns)
        assert kwargs.get("diagnostic_scaffold") is True
        assert kwargs.get("diagnostic_scaffold_diffusion_tokens") == (
            "diff2_x",
        )

    def test_parity_resolve_diagnostic_active_returns_dataclass(
        self, parity_runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import argparse

        monkeypatch.setenv(ENV_VAR_NAME, "1")
        ns = argparse.Namespace(
            diagnostic_scaffold=True,
            diagnostic_scaffold_diffusion_tokens=["diff2_x"],
            diagnostic_scaffold_reaction_tokens=["u"],
            diagnostic_scaffold_root_tokens=None,
            diagnostic_scaffold_neutral_tokens=None,
        )
        result = parity_runner._resolve_diagnostic(ns)
        assert result is not None
        assert result.scaffold is True
        assert result.scaffold_diffusion_tokens == ("diff2_x",)


@pytest.mark.unit
class TestOrphanTokenListEmptyListWarn:

    @staticmethod
    def _ns_with_neutral_empty() -> argparse.Namespace:

        return argparse.Namespace(
            diagnostic_scaffold=False,
            diagnostic_scaffold_diffusion_tokens=None,
            diagnostic_scaffold_reaction_tokens=None,
            diagnostic_scaffold_root_tokens=None,
            diagnostic_scaffold_neutral_tokens=[],
        )

    def _assert_warns_for_neutral(
        self,
        runner: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:




        _ = runner
        warn_orphan_token_lists(self._ns_with_neutral_empty())
        captured = capsys.readouterr()
        assert "neutral-tokens" in captured.err, (
            "empty list (nargs='*' with no values) must trigger warn; "
            f"got stderr: {captured.err!r}"
        )

    def test_burgers_warn_on_empty_neutral(
        self,
        burgers_runner: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._assert_warns_for_neutral(burgers_runner, capsys)

    def test_chafee_warn_on_empty_neutral(
        self,
        chafee_runner: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._assert_warns_for_neutral(chafee_runner, capsys)

    def test_parity_warn_on_empty_neutral(
        self,
        parity_runner: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._assert_warns_for_neutral(parity_runner, capsys)





def _run_cli(
    argv: list[str], env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *argv],
        env=env,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        cwd=str(_PROJECT_ROOT),
        check=False,
    )


def _baseline_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    _ = monkeypatch
    env = os.environ.copy()
    env.pop(ENV_VAR_NAME, None)
    return env


@pytest.mark.slow
class TestCLISubprocessGate:

    def test_burgers_cli_diagnostic_without_env_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _baseline_env(monkeypatch)
        result = _run_cli(
            [
                str(_BURGERS_SCRIPT),
                "--fast",
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x",
                "--diagnostic-scaffold-reaction-tokens", "u",
            ],
            env=env,
        )
        assert result.returncode != 0, (
            "CLI must fail without the env opt-in. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert ENV_VAR_NAME in result.stderr, (
            "stderr must mention the env var so users can grep for it. "
            f"got: {result.stderr!r}"
        )

    def test_chafee_cli_diagnostic_without_env_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _baseline_env(monkeypatch)
        result = _run_cli(
            [
                str(_CHAFEE_SCRIPT),
                "--fast",
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x",
                "--diagnostic-scaffold-reaction-tokens", "u",
            ],
            env=env,
        )
        assert result.returncode != 0
        assert ENV_VAR_NAME in result.stderr, (
            f"got: {result.stderr!r}"
        )

    def test_parity_cli_diagnostic_without_env_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _baseline_env(monkeypatch)
        result = _run_cli(
            [
                str(_PARITY_SCRIPT),
                "--config-name", "A",
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x", "diff2_y",
                "--diagnostic-scaffold-reaction-tokens", "u", "n3",
            ],
            env=env,
        )
        assert result.returncode != 0
        assert ENV_VAR_NAME in result.stderr, (
            f"got: {result.stderr!r}"
        )

    def test_burgers_cli_no_diagnostic_flag_unaffected_by_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _baseline_env(monkeypatch)
        result = _run_cli([str(_BURGERS_SCRIPT), "--help"], env=env)
        assert result.returncode == 0, (
            "--help must succeed regardless of env (no diagnostics in play). "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )


        assert "RuntimeError" not in result.stderr
