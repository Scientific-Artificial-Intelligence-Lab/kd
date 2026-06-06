from __future__ import annotations

import argparse
import dataclasses
from typing import Any

import pytest





import kd.search.discover.tokens.prior
from kd.search.discover.config import DiscoverConfig


pytestmark = pytest.mark.unit





EXPECTED_TOKEN_LIST_FIELDS: tuple[str, ...] = (
    "diagnostic_scaffold_diffusion_tokens",
    "diagnostic_scaffold_reaction_tokens",
    "diagnostic_scaffold_root_tokens",
    "diagnostic_scaffold_neutral_tokens",
)


EXPECTED_FLAG_DESTS: tuple[str, ...] = (
    "diagnostic_scaffold",
    *EXPECTED_TOKEN_LIST_FIELDS,
)


def _default_namespace(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "diagnostic_scaffold": False,
        "diagnostic_scaffold_diffusion_tokens": None,
        "diagnostic_scaffold_reaction_tokens": None,
        "diagnostic_scaffold_root_tokens": None,
        "diagnostic_scaffold_neutral_tokens": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)





class TestModuleSurface:

    def test_module_exports(self) -> None:
        from kd.search.discover.runners import diagnostic_cli

        expected = {
            "DIAGNOSTIC_TOKEN_LIST_FIELDS",
            "add_diagnostic_scaffold_args",
            "enforce_diagnostic_env",
            "warn_orphan_token_lists",
            "build_scaffold_kwargs",
        }
        assert hasattr(diagnostic_cli, "__all__"), (
            "module must declare __all__ so import * stays explicit"
        )
        assert set(diagnostic_cli.__all__) == expected, (
            f"__all__ drift; expected {expected}, "
            f"got {set(diagnostic_cli.__all__)}"
        )
        for name in expected:
            assert hasattr(diagnostic_cli, name), (
                f"public name {name!r} listed in __all__ but missing"
            )

    def test_token_list_fields_canonical(self) -> None:
        from kd.search.discover.runners.diagnostic_cli import (
            DIAGNOSTIC_TOKEN_LIST_FIELDS,
        )

        assert isinstance(DIAGNOSTIC_TOKEN_LIST_FIELDS, tuple)
        assert DIAGNOSTIC_TOKEN_LIST_FIELDS == EXPECTED_TOKEN_LIST_FIELDS

    def test_token_list_fields_match_dataclass(self) -> None:
        from kd.search.discover.runners.diagnostic_cli import (
            DIAGNOSTIC_TOKEN_LIST_FIELDS,
        )

        dataclass_fields = {f.name for f in dataclasses.fields(DiscoverConfig)}
        for name in DIAGNOSTIC_TOKEN_LIST_FIELDS:
            assert name in dataclass_fields, (
                f"DIAGNOSTIC_TOKEN_LIST_FIELDS lists {name!r} but it is "
                "not a DiscoverConfig dataclass field — drift"
            )


        scaffold_token_fields = {
            name
            for name in dataclass_fields
            if name.startswith("diagnostic_scaffold_")
            and name.endswith("_tokens")
        }
        assert set(DIAGNOSTIC_TOKEN_LIST_FIELDS) == scaffold_token_fields, (
            "DIAGNOSTIC_TOKEN_LIST_FIELDS must mirror every "
            "diagnostic_scaffold_*_tokens dataclass field; "
            f"missing {scaffold_token_fields - set(DIAGNOSTIC_TOKEN_LIST_FIELDS)}, "
            f"extra {set(DIAGNOSTIC_TOKEN_LIST_FIELDS) - scaffold_token_fields}"
        )





class TestAddDiagnosticScaffoldArgs:

    def _build_parser(self) -> argparse.ArgumentParser:
        from kd.search.discover.runners.diagnostic_cli import (
            add_diagnostic_scaffold_args,
        )

        parser = argparse.ArgumentParser()
        add_diagnostic_scaffold_args(parser)
        return parser

    def test_registers_5_flags(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args([])
        for dest in EXPECTED_FLAG_DESTS:
            assert hasattr(args, dest), (
                f"--{dest.replace('_', '-')} did not register dest {dest!r}"
            )

    def test_main_flag_default_false(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args([])
        assert args.diagnostic_scaffold is False

    def test_token_list_default_None(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args([])
        for dest in EXPECTED_TOKEN_LIST_FIELDS:
            assert getattr(args, dest) is None, (
                f"{dest} default must be None (got {getattr(args, dest)!r})"
            )

    def test_help_text_includes_env_var_hint(self) -> None:
        parser = self._build_parser()
        actions_by_dest = {action.dest: action for action in parser._actions}
        for dest in EXPECTED_FLAG_DESTS:
            help_text = actions_by_dest[dest].help or ""
            assert "DISCOVER_ENABLE_DIAGNOSTICS=1" in help_text, (
                f"{dest} help missing env-var hint; got: {help_text!r}"
            )

    def test_neutral_tokens_nargs_star(self) -> None:
        parser = self._build_parser()
        actions_by_dest = {action.dest: action for action in parser._actions}
        assert (
            actions_by_dest["diagnostic_scaffold_neutral_tokens"].nargs == "*"
        ), "neutral-tokens must accept zero positional args"
        for dest in (
            "diagnostic_scaffold_diffusion_tokens",
            "diagnostic_scaffold_reaction_tokens",
            "diagnostic_scaffold_root_tokens",
        ):
            assert actions_by_dest[dest].nargs == "+", (
                f"{dest} nargs drift; expected '+', "
                f"got {actions_by_dest[dest].nargs!r}"
            )





class TestEnforceDiagnosticEnv:

    def test_no_flag_no_env_returns(self) -> None:
        from kd.search.discover.runners.diagnostic_cli import enforce_diagnostic_env


        enforce_diagnostic_env(_default_namespace(diagnostic_scaffold=False))

    def test_flag_set_env_one_returns(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import enforce_diagnostic_env

        monkeypatch.setenv("DISCOVER_ENABLE_DIAGNOSTICS", "1")
        enforce_diagnostic_env(_default_namespace(diagnostic_scaffold=True))

    def test_flag_set_no_env_exits_2(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import enforce_diagnostic_env

        monkeypatch.delenv("DISCOVER_ENABLE_DIAGNOSTICS", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            enforce_diagnostic_env(
                _default_namespace(diagnostic_scaffold=True)
            )
        assert exc_info.value.code == 2

    def test_flag_set_env_zero_exits_2(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import enforce_diagnostic_env

        monkeypatch.setenv("DISCOVER_ENABLE_DIAGNOSTICS", "0")
        with pytest.raises(SystemExit) as exc_info:
            enforce_diagnostic_env(
                _default_namespace(diagnostic_scaffold=True)
            )
        assert exc_info.value.code == 2

    def test_stderr_message_format(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import enforce_diagnostic_env

        monkeypatch.delenv("DISCOVER_ENABLE_DIAGNOSTICS", raising=False)
        with pytest.raises(SystemExit):
            enforce_diagnostic_env(
                _default_namespace(diagnostic_scaffold=True)
            )
        captured = capsys.readouterr()
        assert "DISCOVER_ENABLE_DIAGNOSTICS" in captured.err, (
            f"stderr must mention env var; got: {captured.err!r}"
        )

    def test_uses_shared_rationale_constant(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kd.search.discover.config import DIAGNOSTIC_GATE_RATIONALE
        from kd.search.discover.runners.diagnostic_cli import enforce_diagnostic_env

        monkeypatch.delenv("DISCOVER_ENABLE_DIAGNOSTICS", raising=False)
        with pytest.raises(SystemExit):
            enforce_diagnostic_env(
                _default_namespace(diagnostic_scaffold=True)
            )
        captured = capsys.readouterr()
        assert DIAGNOSTIC_GATE_RATIONALE in captured.err, (
            "stderr must reuse the shared rationale constant; "
            f"missing {DIAGNOSTIC_GATE_RATIONALE!r} in {captured.err!r}"
        )





class TestWarnOrphanTokenLists:

    def test_no_main_no_tokens_no_warn(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import warn_orphan_token_lists

        warn_orphan_token_lists(_default_namespace())
        captured = capsys.readouterr()
        assert captured.err == "", (
            f"no orphan flags ⇒ no warn; got stderr: {captured.err!r}"
        )

    def test_main_set_with_tokens_no_warn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import warn_orphan_token_lists

        monkeypatch.setenv("DISCOVER_ENABLE_DIAGNOSTICS", "1")
        ns = _default_namespace(
            diagnostic_scaffold=True,
            diagnostic_scaffold_diffusion_tokens=["diff2_x"],
            diagnostic_scaffold_reaction_tokens=["u"],
        )
        warn_orphan_token_lists(ns)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_main_off_diffusion_given_warns(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import warn_orphan_token_lists

        ns = _default_namespace(
            diagnostic_scaffold=False,
            diagnostic_scaffold_diffusion_tokens=["diff2_x"],
        )
        warn_orphan_token_lists(ns)
        captured = capsys.readouterr()
        assert "diagnostic-scaffold-diffusion-tokens" in captured.err, (
            f"stderr must name the orphan flag; got: {captured.err!r}"
        )

    def test_main_off_neutral_empty_warns(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import warn_orphan_token_lists

        ns = _default_namespace(
            diagnostic_scaffold=False,
            diagnostic_scaffold_neutral_tokens=[],
        )
        warn_orphan_token_lists(ns)
        captured = capsys.readouterr()
        assert "diagnostic-scaffold-neutral-tokens" in captured.err, (
            "empty list (nargs='*' with no values) must trigger warn; "
            f"got stderr: {captured.err!r}"
        )

    def test_main_off_multiple_orphan_lists(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import warn_orphan_token_lists

        ns = _default_namespace(
            diagnostic_scaffold=False,
            diagnostic_scaffold_diffusion_tokens=["diff2_x"],
            diagnostic_scaffold_reaction_tokens=["u"],
            diagnostic_scaffold_root_tokens=["add"],
            diagnostic_scaffold_neutral_tokens=["x"],
        )
        warn_orphan_token_lists(ns)
        captured = capsys.readouterr()
        for fragment in (
            "diagnostic-scaffold-diffusion-tokens",
            "diagnostic-scaffold-reaction-tokens",
            "diagnostic-scaffold-root-tokens",
            "diagnostic-scaffold-neutral-tokens",
        ):
            assert fragment in captured.err, (
                f"orphan-warn must mention {fragment}; "
                f"got stderr: {captured.err!r}"
            )





class TestBuildScaffoldKwargs:

    def test_main_flag_off_returns_empty_dict(self) -> None:
        from kd.search.discover.runners.diagnostic_cli import build_scaffold_kwargs

        ns = _default_namespace(diagnostic_scaffold=False)
        kwargs = build_scaffold_kwargs(ns)
        assert kwargs == {}, (
            "default args must yield empty kwargs so DiscoverConfig keeps "
            f"frozen defaults; got: {kwargs!r}"
        )

    def test_main_flag_on_no_overrides_uses_defaults(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import build_scaffold_kwargs

        monkeypatch.setenv("DISCOVER_ENABLE_DIAGNOSTICS", "1")
        ns = _default_namespace(diagnostic_scaffold=True)
        kwargs = build_scaffold_kwargs(ns)
        assert kwargs.get("diagnostic_scaffold") is True

        assert kwargs.get("diagnostic_scaffold_neutral_tokens") == (
            "u", "x", "y", "t",
        )
        assert kwargs.get("diagnostic_scaffold_root_tokens") == ("add", "sub")

        assert kwargs.get("diagnostic_scaffold_diffusion_tokens") == ()
        assert kwargs.get("diagnostic_scaffold_reaction_tokens") == ()

    def test_main_flag_on_with_overrides(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import build_scaffold_kwargs

        monkeypatch.setenv("DISCOVER_ENABLE_DIAGNOSTICS", "1")
        ns = _default_namespace(
            diagnostic_scaffold=True,
            diagnostic_scaffold_diffusion_tokens=["diff2_x", "diff2_y"],
            diagnostic_scaffold_reaction_tokens=["u", "n3"],
            diagnostic_scaffold_root_tokens=["add"],
            diagnostic_scaffold_neutral_tokens=["x"],
        )
        kwargs = build_scaffold_kwargs(ns)
        assert kwargs["diagnostic_scaffold_diffusion_tokens"] == (
            "diff2_x", "diff2_y",
        )
        assert kwargs["diagnostic_scaffold_reaction_tokens"] == ("u", "n3")
        assert kwargs["diagnostic_scaffold_root_tokens"] == ("add",)
        assert kwargs["diagnostic_scaffold_neutral_tokens"] == ("x",)


        for key in EXPECTED_TOKEN_LIST_FIELDS:
            assert isinstance(kwargs[key], tuple), (
                f"{key} must be a tuple in returned kwargs; "
                f"got {type(kwargs[key]).__name__}"
            )

    def test_kwargs_keys_match_dataclass_fields(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.runners.diagnostic_cli import build_scaffold_kwargs

        monkeypatch.setenv("DISCOVER_ENABLE_DIAGNOSTICS", "1")
        ns = _default_namespace(
            diagnostic_scaffold=True,
            diagnostic_scaffold_diffusion_tokens=["diff2_x"],
            diagnostic_scaffold_reaction_tokens=["u"],
        )
        kwargs = build_scaffold_kwargs(ns)
        dataclass_fields = {f.name for f in dataclasses.fields(DiscoverConfig)}
        unknown = set(kwargs) - dataclass_fields
        assert not unknown, (
            f"build_scaffold_kwargs returned keys not in DiscoverConfig: "
            f"{unknown}"
        )
