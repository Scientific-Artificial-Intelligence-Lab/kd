from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from kd.search.discover.config import DiscoverConfig
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import PriorContext, PriorSystem, ScaffoldPrior

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = (
    _PROJECT_ROOT
    / "scripts"
    / "discover"
    / "research"
    / "analysis"
    / "task4_parity_ablation.py"
)


def _load_parity_script() -> ModuleType:
    scripts_dir = str(_SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "test_p21_critical_fixes_parity_runner", _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parity_runner() -> ModuleType:
    return _load_parity_script()






@pytest.fixture(autouse=True)
def _enable_diagnostics_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCOVER_ENABLE_DIAGNOSTICS", "1")





@pytest.mark.unit
class TestCLIEndToEnd:

    def _make_args(
        self,
        parity_runner: ModuleType,
        extra_argv: list[str],
    ) -> object:
        old_argv = sys.argv
        try:
            sys.argv = [
                "task4_parity_ablation.py",
                "--config-name", "A",
                *extra_argv,
            ]
            return parity_runner.parse_args()
        finally:
            sys.argv = old_argv

    def test_diagnostic_override_threaded_through_build_configs(
        self, parity_runner: ModuleType,
    ) -> None:
        args = self._make_args(
            parity_runner,
            [
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x", "diff2_y",
                "--diagnostic-scaffold-reaction-tokens", "u", "n3",
            ],
        )
        diagnostic = parity_runner._resolve_diagnostic(args)
        assert diagnostic.scaffold is True
        assert diagnostic.scaffold_diffusion_tokens == ("diff2_x", "diff2_y")
        assert diagnostic.scaffold_reaction_tokens == ("u", "n3")
        assert diagnostic.scaffold_root_tokens == ("add", "sub")

        parity = parity_runner.CONFIGS["A"]
        budget = parity_runner._resolve_budget(args)
        config, _pinn = parity_runner._build_configs(
            parity, budget, config_name="A", diagnostic=diagnostic,
        )
        assert isinstance(config, DiscoverConfig)
        assert config.diagnostic_scaffold is True
        assert config.diagnostic_scaffold_diffusion_tokens == (
            "diff2_x", "diff2_y",
        )
        assert config.diagnostic_scaffold_reaction_tokens == ("u", "n3")
        assert config.diagnostic_scaffold_root_tokens == ("add", "sub")

    def test_absent_flag_leaves_scaffold_off(
        self, parity_runner: ModuleType,
    ) -> None:
        args = self._make_args(parity_runner, [])
        diagnostic = parity_runner._resolve_diagnostic(args)
        assert diagnostic is None
        config, _pinn = parity_runner._build_configs(
            parity_runner.CONFIGS["A"],
            parity_runner._resolve_budget(args),
            config_name="A",
            diagnostic=diagnostic,
        )
        assert config.diagnostic_scaffold is False

    def test_build_configs_without_diagnostic_preserves_default_off(
        self, parity_runner: ModuleType,
    ) -> None:
        args = self._make_args(parity_runner, [])
        config, _pinn = parity_runner._build_configs(
            parity_runner.CONFIGS["A"],
            parity_runner._resolve_budget(args),
            config_name="A",
        )
        assert config.diagnostic_scaffold is False
        assert config.diagnostic_scaffold_root_tokens == ("add", "sub")

    def test_end_to_end_scaffold_attached_when_flag_on(
        self, parity_runner: ModuleType,
    ) -> None:
        from kd.search.discover.builder import build_prior_system

        args = self._make_args(
            parity_runner,
            [
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x", "diff2_y",
                "--diagnostic-scaffold-reaction-tokens", "u", "n3",
            ],
        )
        diagnostic = parity_runner._resolve_diagnostic(args)
        config, _pinn = parity_runner._build_configs(
            parity_runner.CONFIGS["A"],
            parity_runner._resolve_budget(args),
            config_name="A",
            diagnostic=diagnostic,
        )
        library = Library.from_config(config.library)
        prior_system = build_prior_system(library, config)
        scaffolds = [
            p for p in prior_system.priors if isinstance(p, ScaffoldPrior)
        ]
        assert len(scaffolds) == 1, (
            "CRITICAL 1 regression: CLI flag must yield exactly one "
            f"ScaffoldPrior in the production PriorSystem, got "
            f"{[type(p).__name__ for p in prior_system.priors]}"
        )





_BINARY_TAIL_CONFIG = LibraryConfig(

    coord_vars=["x", "t"],
    state_vars=["u"],
    operators=["sub", "diff_x", "add", "mul"],
)


@pytest.fixture
def binary_tail_lib() -> Library:
    return Library.from_config(_BINARY_TAIL_CONFIG)


@pytest.mark.unit
class TestEmptyActionPaddingArity:

    def test_padding_sentinel_is_zero_arity(
        self, binary_tail_lib: Library,
    ) -> None:
        prior = ScaffoldPrior(
            binary_tail_lib,
            diffusion_tokens=["diff_x"],
            reaction_tokens=["u"],
        )
        assert prior._arities_with_pad.shape[0] == prior.n_choices + 1
        assert int(prior._arities_with_pad[-1]) == 0

    def test_padding_keeps_right_subtree_classification(
        self, binary_tail_lib: Library,
    ) -> None:
        prior = ScaffoldPrior(
            binary_tail_lib,
            diffusion_tokens=["diff_x"],
            reaction_tokens=["u"],
        )
        add_idx = binary_tail_lib.name_to_index("add")
        u_idx = binary_tail_lib.name_to_index("u")
        pad = binary_tail_lib.EMPTY_ACTION
        actions = np.array(
            [[add_idx, u_idx, pad, pad]], dtype=np.int32,
        )
        mask = prior._in_left_subtree(actions)
        assert mask.shape == (1,)
        assert bool(mask[0]) is False, (
            "right-subtree row misclassified — CRITICAL 3 regression"
        )

    def test_padding_only_row_stays_in_right_subtree(
        self, binary_tail_lib: Library,
    ) -> None:
        prior = ScaffoldPrior(
            binary_tail_lib,
            diffusion_tokens=["diff_x"],
            reaction_tokens=["u"],
        )
        add_idx = binary_tail_lib.name_to_index("add")
        pad = binary_tail_lib.EMPTY_ACTION
        actions = np.array([[add_idx, pad, pad]], dtype=np.int32)
        mask = prior._in_left_subtree(actions)
        assert mask.shape == (1,)
        assert bool(mask[0]) is False


@pytest.mark.unit
class TestRootTokenResolution:

    def test_all_unknown_root_tokens_raises(self) -> None:
        lib = Library.from_config(
            LibraryConfig(
                coord_vars=["x", "t"],
                state_vars=["u"],
                operators=["add", "sub", "diff_x"],
            )
        )
        with pytest.raises(ValueError, match="no root tokens resolved"):
            ScaffoldPrior(
                lib,
                diffusion_tokens=["diff_x"],
                reaction_tokens=["u"],
                root_tokens=("zzz_nope",),
            )

    def test_both_subtree_buckets_empty_warns(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        lib = Library.from_config(
            LibraryConfig(
                coord_vars=["x", "t"],
                state_vars=["u"],
                operators=["add", "sub", "diff_x"],
            )
        )
        with caplog.at_level("WARNING"):
            ScaffoldPrior(
                lib,
                diffusion_tokens=["zzz"],
                reaction_tokens=["yyy"],
            )
        assert any(
            "subtree disjointness will be inert" in r.getMessage()
            for r in caplog.records
        ), "expected loud warning on both-buckets-empty misconfig"





@pytest.mark.unit
class TestScaffoldStep1PropagatesThroughPriorSystem:

    def test_step1_forbid_survives_priorsystem_compose(self) -> None:
        from kd.search.discover.builder import build_prior_system

        cfg = DiscoverConfig(
            library=LibraryConfig(
                coord_vars=["x", "y", "t"],
                state_vars=["u"],
                operators=[
                    "add", "sub", "mul", "sin", "cos", "n2", "n3",
                    "diff_x", "diff2_x", "diff_y", "diff2_y", "diff_t",
                ],
            ),
            diagnostic_scaffold=True,
            diagnostic_scaffold_diffusion_tokens=("diff2_x", "diff2_y"),
            diagnostic_scaffold_reaction_tokens=("u", "n3"),
            diagnostic_scaffold_root_tokens=("add", "sub"),
        )
        lib = Library.from_config(cfg.library)
        ps: PriorSystem = build_prior_system(lib, cfg)

        for p in ps.priors:
            if isinstance(p, ScaffoldPrior):
                p.on_cycle_start(0)

        add_idx = lib.name_to_index("add")
        obs = np.array(
            [[add_idx, lib.parent_adjust[add_idx], lib.EMPTY_SIBLING, 1]],
            dtype=np.float32,
        )
        adjustment = ps.step(
            actions=np.array([[add_idx]], dtype=np.int32),
            obs=obs,
            step_idx=1,
        )
        reaction_indices = np.array(
            [lib.name_to_index(n) for n in ("u", "n3")],
            dtype=np.int32,
        )
        assert np.all(np.isneginf(adjustment[0, reaction_indices])), (
            "ScaffoldPrior step=1 -inf must survive PriorSystem._combine "
            f"(left-subtree forbid reaction): {adjustment[0, reaction_indices]}"
        )
        assert np.any(np.isfinite(adjustment[0])), (
            "PriorSystem would raise dead-end otherwise"
        )




_AC_LIKE_CONFIG_FOR_S1 = LibraryConfig(
    coord_vars=["x", "y", "t"],
    state_vars=["u"],
    operators=[
        "add", "sub", "mul", "sin", "cos", "n2", "n3",
        "diff_x", "diff2_x", "diff_y", "diff2_y", "diff_t",
    ],
)
_DIFFUSION_TOKENS_S1 = ["diff2_x", "diff2_y"]
_REACTION_TOKENS_S1 = ["u", "n3"]
_ROOT_TOKENS_S1 = ("add", "sub")
_NEUTRAL_TOKENS_S1 = ["u", "x", "y", "t"]


@pytest.fixture
def s1_ac_lib() -> Library:
    return Library.from_config(_AC_LIKE_CONFIG_FOR_S1)


def _make_neutral_ctx(
    lib: Library,
    history: list[int],
    *,
    step_idx: int,
    dangling: int = 1,
) -> PriorContext:
    actions = np.asarray([history], dtype=np.int32)
    parent = lib.parent_adjust[history[0]] if history else lib.EMPTY_PARENT
    return PriorContext(
        actions=actions,
        parent=np.array([parent], dtype=np.int32),
        sibling=np.array([lib.EMPTY_SIBLING], dtype=np.int32),
        dangling=np.array([dangling], dtype=np.int32),
        step_idx=step_idx,
        library=lib,
    )


@pytest.mark.unit
class TestScaffoldPriorNeutral:

    def test_neutral_allowed_in_left_subtree(self, s1_ac_lib: Library) -> None:
        prior = ScaffoldPrior(
            s1_ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_S1,
            reaction_tokens=_REACTION_TOKENS_S1,
            root_tokens=_ROOT_TOKENS_S1,
            neutral_tokens=_NEUTRAL_TOKENS_S1,
        )
        prior.on_cycle_start(0)
        add_idx = s1_ac_lib.name_to_index("add")
        ctx = _make_neutral_ctx(s1_ac_lib, [add_idx], step_idx=1)
        adjustment = prior(ctx)
        neutral_indices = np.asarray(
            [s1_ac_lib.name_to_index(n) for n in _NEUTRAL_TOKENS_S1],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(
            adjustment[0, neutral_indices],
            np.zeros(neutral_indices.size, dtype=np.float32),
        )

    def test_neutral_allowed_in_right_subtree(self, s1_ac_lib: Library) -> None:
        prior = ScaffoldPrior(
            s1_ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_S1,
            reaction_tokens=_REACTION_TOKENS_S1,
            root_tokens=_ROOT_TOKENS_S1,
            neutral_tokens=_NEUTRAL_TOKENS_S1,
        )
        prior.on_cycle_start(0)
        add_idx = s1_ac_lib.name_to_index("add")
        u_idx = s1_ac_lib.name_to_index("u")




        ctx = _make_neutral_ctx(
            s1_ac_lib, [add_idx, u_idx], step_idx=2,
        )
        adjustment = prior(ctx)
        diff_op_idx = s1_ac_lib.name_to_index("diff2_x")
        assert adjustment[0, diff_op_idx] == float("-inf"), (
            "diff2_x must be forbidden on right subtree even under S1"
        )
        neutral_indices = np.asarray(
            [s1_ac_lib.name_to_index(n) for n in _NEUTRAL_TOKENS_S1],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(
            adjustment[0, neutral_indices],
            np.zeros(neutral_indices.size, dtype=np.float32),
        )

    def test_diff2x_arg_can_be_u(self, s1_ac_lib: Library) -> None:
        prior = ScaffoldPrior(
            s1_ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_S1,
            reaction_tokens=_REACTION_TOKENS_S1,
            root_tokens=_ROOT_TOKENS_S1,
            neutral_tokens=_NEUTRAL_TOKENS_S1,
        )
        prior.on_cycle_start(0)
        add_idx = s1_ac_lib.name_to_index("add")
        diff2x_idx = s1_ac_lib.name_to_index("diff2_x")
        u_idx = s1_ac_lib.name_to_index("u")


        ctx = _make_neutral_ctx(
            s1_ac_lib, [add_idx, diff2x_idx], step_idx=2,
        )
        adjustment = prior(ctx)
        assert np.isfinite(adjustment[0, u_idx]), (
            "S1 loosening must keep ``u`` finite on the left subtree so "
            "``diff2_x(u)`` stays reachable (fixes P21 Phase A dead-end)"
        )
        n3_idx = s1_ac_lib.name_to_index("n3")
        assert adjustment[0, n3_idx] == float("-inf"), (
            "S1 must NOT weaken operator disjointness — n3 stays -inf"
        )

    def test_neutral_overlap_with_diffusion_accepted(
        self, s1_ac_lib: Library,
    ) -> None:
        prior = ScaffoldPrior(
            s1_ac_lib,
            diffusion_tokens=["diff2_x", "u"],
            reaction_tokens=["n3"],
            root_tokens=_ROOT_TOKENS_S1,
            neutral_tokens=["u"],
        )

        prior.on_cycle_start(0)
        add_idx = s1_ac_lib.name_to_index("add")



        u_idx = s1_ac_lib.name_to_index("u")
        ctx = _make_neutral_ctx(
            s1_ac_lib, [add_idx, u_idx], step_idx=2,
        )
        adjustment = prior(ctx)
        assert np.isfinite(adjustment[0, u_idx]), (
            "u declared in diffusion+neutral must remain finite on right"
        )

    def test_neutral_overlap_with_reaction_accepted(
        self, s1_ac_lib: Library,
    ) -> None:
        prior = ScaffoldPrior(
            s1_ac_lib,
            diffusion_tokens=["diff2_x"],
            reaction_tokens=["u", "n3"],
            root_tokens=_ROOT_TOKENS_S1,
            neutral_tokens=["u"],
        )
        prior.on_cycle_start(0)
        add_idx = s1_ac_lib.name_to_index("add")
        diff2x_idx = s1_ac_lib.name_to_index("diff2_x")
        u_idx = s1_ac_lib.name_to_index("u")

        ctx = _make_neutral_ctx(
            s1_ac_lib, [add_idx, diff2x_idx], step_idx=2,
        )
        adjustment = prior(ctx)
        assert np.isfinite(adjustment[0, u_idx]), (
            "u declared in reaction+neutral must remain finite on left "
            "(S1 AC-2D production scenario)"
        )

        n3_idx = s1_ac_lib.name_to_index("n3")
        assert adjustment[0, n3_idx] == float("-inf"), (
            "non-neutral reaction tokens still forbidden on left"
        )

    def test_neutral_overlap_with_root_raises(
        self, s1_ac_lib: Library,
    ) -> None:
        with pytest.raises(ValueError, match="neutral_tokens overlaps"):
            ScaffoldPrior(
                s1_ac_lib,
                diffusion_tokens=["diff2_x"],
                reaction_tokens=["n3"],
                root_tokens=("add", "sub"),
                neutral_tokens=["add"],
            )

    def test_unknown_neutral_warns_and_skips(
        self, s1_ac_lib: Library, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("WARNING"):
            ScaffoldPrior(
                s1_ac_lib,
                diffusion_tokens=_DIFFUSION_TOKENS_S1,
                reaction_tokens=_REACTION_TOKENS_S1,
                neutral_tokens=["zzz_unknown", "u"],
            )
        assert any(
            "unknown token name 'zzz_unknown'" in r.getMessage()
            and "neutral_tokens" in r.getMessage()
            for r in caplog.records
        ), "expected warn+skip for unknown neutral token name"

    def test_empty_neutral_preserves_strict_disjoint(
        self, s1_ac_lib: Library,
    ) -> None:
        prior = ScaffoldPrior(
            s1_ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_S1,
            reaction_tokens=_REACTION_TOKENS_S1,
            root_tokens=_ROOT_TOKENS_S1,
        )
        prior.on_cycle_start(0)
        add_idx = s1_ac_lib.name_to_index("add")
        diff2x_idx = s1_ac_lib.name_to_index("diff2_x")
        u_idx = s1_ac_lib.name_to_index("u")
        ctx = _make_neutral_ctx(
            s1_ac_lib, [add_idx, diff2x_idx], step_idx=2,
        )
        adjustment = prior(ctx)
        assert adjustment[0, u_idx] == float("-inf"), (
            "default ``neutral_tokens=()`` must keep u forbidden on the "
            "left subtree (pre-S1 behavior)"
        )

    def test_diff2x_u_end_to_end_priorsystem(self) -> None:
        from kd.search.discover.builder import build_prior_system

        cfg = DiscoverConfig(
            library=_AC_LIKE_CONFIG_FOR_S1,
            diagnostic_scaffold=True,
            diagnostic_scaffold_diffusion_tokens=("diff2_x", "diff2_y"),
            diagnostic_scaffold_reaction_tokens=("u", "n3"),
            diagnostic_scaffold_root_tokens=("add", "sub"),
            diagnostic_scaffold_neutral_tokens=("u", "x", "y", "t"),
        )
        library = Library.from_config(cfg.library)
        ps = build_prior_system(library, cfg)
        for p in ps.priors:
            if isinstance(p, ScaffoldPrior):
                p.on_cycle_start(0)
        add_idx = library.name_to_index("add")
        diff2x_idx = library.name_to_index("diff2_x")
        u_idx = library.name_to_index("u")
        obs = np.array(
            [[
                diff2x_idx,
                library.parent_adjust[diff2x_idx],
                library.EMPTY_SIBLING,
                1,
            ]],
            dtype=np.float32,
        )
        adjustment = ps.step(
            actions=np.array([[add_idx, diff2x_idx]], dtype=np.int32),
            obs=obs,
            step_idx=2,
        )
        assert np.isfinite(adjustment[0, u_idx]), (
            "S1 end-to-end: u must stay finite after [add, diff2_x] so "
            "diff2_x(u) can be emitted"
        )
        assert np.any(np.isfinite(adjustment[0])), (
            "no dead-end row — PriorSystem would otherwise raise"
        )


@pytest.mark.unit
class TestS1CLIEndToEnd:

    def _make_args(
        self,
        parity_runner: ModuleType,
        extra_argv: list[str],
    ) -> object:
        old_argv = sys.argv
        try:
            sys.argv = [
                "task4_parity_ablation.py",
                "--config-name", "P",
                *extra_argv,
            ]
            return parity_runner.parse_args()
        finally:
            sys.argv = old_argv

    def test_s1_cli_neutral_threaded_through_config(
        self, parity_runner: ModuleType,
    ) -> None:
        args = self._make_args(
            parity_runner,
            [
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x", "diff2_y",
                "--diagnostic-scaffold-reaction-tokens", "u", "n3",
                "--diagnostic-scaffold-neutral-tokens", "u", "x", "y", "t",
            ],
        )
        diagnostic = parity_runner._resolve_diagnostic(args)
        assert diagnostic.scaffold_neutral_tokens == ("u", "x", "y", "t")
        config, _pinn = parity_runner._build_configs(
            parity_runner.CONFIGS["P"],
            parity_runner._resolve_budget(args),
            config_name="P",
            diagnostic=diagnostic,
        )
        assert config.diagnostic_scaffold_neutral_tokens == (
            "u", "x", "y", "t",
        )

    def test_s1_cli_default_is_scalar_field_and_coords(
        self, parity_runner: ModuleType,
    ) -> None:
        args = self._make_args(
            parity_runner,
            [
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x", "diff2_y",
                "--diagnostic-scaffold-reaction-tokens", "u", "n3",
            ],
        )
        diagnostic = parity_runner._resolve_diagnostic(args)
        assert diagnostic.scaffold_neutral_tokens == ("u", "x", "y", "t")

    def test_s1_cli_empty_list_restores_strict_disjoint(
        self, parity_runner: ModuleType,
    ) -> None:
        args = self._make_args(
            parity_runner,
            [
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x", "diff2_y",
                "--diagnostic-scaffold-reaction-tokens", "u", "n3",
                "--diagnostic-scaffold-neutral-tokens",
            ],
        )
        diagnostic = parity_runner._resolve_diagnostic(args)
        assert diagnostic.scaffold_neutral_tokens == ()
