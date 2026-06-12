from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from kd.search.discover.builder import build_prior_system
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.tokens.library import Library, LibraryConfig

try:
    from kd.search.discover.tokens.prior import ScaffoldPrior
except ImportError:
    ScaffoldPrior = None

pytestmark = pytest.mark.skipif(
    ScaffoldPrior is None,
    reason="ScaffoldPrior symbol not exported — stub module missing",
)






@pytest.fixture(autouse=True)
def _enable_diagnostics_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCOVER_ENABLE_DIAGNOSTICS", "1")






_AC_LIKE_CONFIG = LibraryConfig(
    coord_vars=["x", "y", "t"],
    state_vars=["u"],
    operators=[
        "add", "sub", "mul", "sin", "cos", "n2", "n3",
        "diff_x", "diff2_x", "diff_y", "diff2_y", "diff_t",
    ],
)


_BURGERS_LIKE_CONFIG = LibraryConfig(
    coord_vars=["x", "t"],
    state_vars=["u"],
    operators=["add", "mul", "sub", "div", "diff_x", "diff2_x", "n2", "n3"],
)



_DIFFUSION_TOKENS_AC = ["diff2_x", "diff2_y"]
_REACTION_TOKENS_AC = ["u", "n3"]
_ROOT_TOKENS = ("add", "sub")
_SAMPLE_COUNT = 100


@pytest.fixture
def ac_lib() -> Library:
    return Library.from_config(_AC_LIKE_CONFIG)


@pytest.fixture
def burgers_lib() -> Library:
    return Library.from_config(_BURGERS_LIKE_CONFIG)


@pytest.fixture
def baseline_config() -> DiscoverConfig:
    return DiscoverConfig()


@pytest.fixture
def scaffold_enabled_config() -> DiscoverConfig:
    return DiscoverConfig(
        diagnostic_scaffold=True,
        diagnostic_scaffold_diffusion_tokens=tuple(_DIFFUSION_TOKENS_AC),
        diagnostic_scaffold_reaction_tokens=tuple(_REACTION_TOKENS_AC),
        diagnostic_scaffold_root_tokens=_ROOT_TOKENS,
    )




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
        "test_scaffold_prior_parity_runner", _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parity_runner() -> ModuleType:
    return _load_parity_script()







@pytest.mark.smoke
class TestStubSurface:

    def test_symbol_is_exported(self) -> None:


        from kd.search.discover.tokens.prior import (
            ScaffoldPrior as _Exported,
        )
        assert _Exported is not None

    def test_direct_scaffold_import_does_not_circular_import(self) -> None:
        code = (
            "from kd.search.discover.tokens.scaffold_prior import *\n"
            "from kd.search.discover.tokens.prior import ScaffoldPrior\n"
            "assert ScaffoldPrior.__name__ == 'ScaffoldPrior'\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_construct_is_live_after_impl(
        self, ac_lib: Library,
    ) -> None:
        prior = ScaffoldPrior(
            ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_AC,
            reaction_tokens=_REACTION_TOKENS_AC,
            root_tokens=_ROOT_TOKENS,
        )
        adjustment = prior.initial_adjustment(batch_size=2)
        np.testing.assert_array_equal(
            adjustment, np.zeros_like(adjustment),
        )







@pytest.mark.unit
class TestDefaultOff:

    def test_baseline_config_excludes_scaffold_prior(
        self, baseline_config: DiscoverConfig,
    ) -> None:
        assert baseline_config.diagnostic_scaffold is False, (
            "default config must have diagnostic_scaffold=False per §5 Rule 2"
        )

    def test_builder_omits_scaffold_prior_when_flag_off(
        self, baseline_config: DiscoverConfig,
    ) -> None:
        library = Library.from_config(baseline_config.library)
        prior_system = build_prior_system(library, baseline_config)

        scaffold_attached = [
            p for p in prior_system.priors if isinstance(p, ScaffoldPrior)
        ]
        assert len(scaffold_attached) == 0, (
            f"default build must not attach ScaffoldPrior, got "
            f"{[type(p).__name__ for p in prior_system.priors]}"
        )

    def test_enabled_config_appends_scaffold_prior(
        self, scaffold_enabled_config: DiscoverConfig,
    ) -> None:
        library = Library.from_config(scaffold_enabled_config.library)
        prior_system = build_prior_system(library, scaffold_enabled_config)
        scaffold_attached = [
            p for p in prior_system.priors if isinstance(p, ScaffoldPrior)
        ]
        assert len(scaffold_attached) == 1, (
            f"builder must attach exactly one ScaffoldPrior when the flag "
            f"is set, got {[type(p).__name__ for p in prior_system.priors]}"
        )







@pytest.mark.unit
class TestEnabledShallowRoot:

    def test_initial_adjustment_forbids_non_root_tokens(
        self, ac_lib: Library,
    ) -> None:
        prior = ScaffoldPrior(
            ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_AC,
            reaction_tokens=_REACTION_TOKENS_AC,
            root_tokens=_ROOT_TOKENS,
        )
        prior.on_cycle_start(0)
        adjustment = prior.initial_adjustment(batch_size=_SAMPLE_COUNT)

        root_indices = np.asarray(
            [ac_lib.name_to_index(name) for name in _ROOT_TOKENS],
            dtype=np.int32,
        )

        non_root_columns = np.setdiff1d(
            np.arange(len(ac_lib.tokens), dtype=np.int32),
            root_indices,
            assume_unique=True,
        )
        assert np.all(
            np.isneginf(adjustment[:, non_root_columns])
        ), "every non-root column must be -inf at cycle-0 step 0"


        assert np.all(
            np.isfinite(adjustment[:, root_indices])
        ), "root columns must be finite so the controller can sample them"

    def test_shape_matches_batch_and_n_choices(self, ac_lib: Library) -> None:
        prior = ScaffoldPrior(
            ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_AC,
            reaction_tokens=_REACTION_TOKENS_AC,
        )
        prior.on_cycle_start(0)
        adjustment = prior.initial_adjustment(batch_size=_SAMPLE_COUNT)
        assert adjustment.shape == (_SAMPLE_COUNT, len(ac_lib.tokens))
        assert adjustment.dtype == np.float32







@pytest.mark.unit
class TestEnabledLeftRightDisjoint:

    def test_step_1_after_root_forbids_opposite_set(self, ac_lib: Library) -> None:
        from kd.search.discover.tokens.prior import (
            PriorContext,
        )

        prior = ScaffoldPrior(
            ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_AC,
            reaction_tokens=_REACTION_TOKENS_AC,
            root_tokens=_ROOT_TOKENS,
        )
        prior.on_cycle_start(0)

        add_idx = ac_lib.name_to_index("add")
        actions = np.array([[add_idx]], dtype=np.int32)

        parent = np.array([ac_lib.parent_adjust[add_idx]], dtype=np.int32)
        sibling = np.array([ac_lib.EMPTY_SIBLING], dtype=np.int32)
        dangling = np.array([1], dtype=np.int32)
        ctx = PriorContext(
            actions=actions,
            parent=parent,
            sibling=sibling,
            dangling=dangling,
            step_idx=1,
            library=ac_lib,
        )
        adjustment = prior(ctx)

        diff_indices = np.asarray(
            [ac_lib.name_to_index(n) for n in _DIFFUSION_TOKENS_AC],
            dtype=np.int32,
        )
        reaction_indices = np.asarray(
            [ac_lib.name_to_index(n) for n in _REACTION_TOKENS_AC],
            dtype=np.int32,
        )
        reaction_all_forbidden = bool(
            np.all(np.isneginf(adjustment[0, reaction_indices]))
        )
        diffusion_all_forbidden = bool(
            np.all(np.isneginf(adjustment[0, diff_indices]))
        )

        assert reaction_all_forbidden ^ diffusion_all_forbidden, (
            "step-1 after root must hard-forbid exactly one subtree's "
            "token set (left/right symmetric). "
            f"reaction_forbidden={reaction_all_forbidden}, "
            f"diffusion_forbidden={diffusion_all_forbidden}"
        )







@pytest.mark.unit
class TestCycleGt0Noop:

    @pytest.mark.parametrize("cycle_idx", [1, 2, 3, 10])
    def test_later_cycles_return_zero_adjustment(
        self, ac_lib: Library, cycle_idx: int,
    ) -> None:
        prior = ScaffoldPrior(
            ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_AC,
            reaction_tokens=_REACTION_TOKENS_AC,
            root_tokens=_ROOT_TOKENS,
        )
        prior.on_cycle_start(cycle_idx)
        adjustment = prior.initial_adjustment(batch_size=_SAMPLE_COUNT)
        np.testing.assert_array_equal(adjustment, np.zeros_like(adjustment))

    def test_call_on_cycle_ge_1_returns_zeros(self, ac_lib: Library) -> None:
        from kd.search.discover.tokens.prior import PriorContext

        prior = ScaffoldPrior(
            ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_AC,
            reaction_tokens=_REACTION_TOKENS_AC,
        )
        prior.on_cycle_start(1)
        add_idx = ac_lib.name_to_index("add")
        actions = np.array([[add_idx]], dtype=np.int32)
        ctx = PriorContext(
            actions=actions,
            parent=np.array([ac_lib.parent_adjust[add_idx]], dtype=np.int32),
            sibling=np.array([ac_lib.EMPTY_SIBLING], dtype=np.int32),
            dangling=np.array([1], dtype=np.int32),
            step_idx=1,
            library=ac_lib,
        )
        adjustment = prior(ctx)
        np.testing.assert_array_equal(adjustment, np.zeros_like(adjustment))

    def test_cycle_reactivation_restores_scaffold(self, ac_lib: Library) -> None:
        prior = ScaffoldPrior(
            ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_AC,
            reaction_tokens=_REACTION_TOKENS_AC,
            root_tokens=_ROOT_TOKENS,
        )
        prior.on_cycle_start(0)
        prior.on_cycle_start(5)
        prior.on_cycle_start(0)

        root_indices = np.asarray(
            [ac_lib.name_to_index(n) for n in _ROOT_TOKENS], dtype=np.int32,
        )
        non_root_columns = np.setdiff1d(
            np.arange(len(ac_lib.tokens), dtype=np.int32),
            root_indices,
            assume_unique=True,
        )
        adjustment = prior.initial_adjustment(batch_size=4)
        assert np.all(np.isneginf(adjustment[:, non_root_columns])), (
            "re-entering cycle 0 must restore the root-enforcement regime"
        )







@pytest.mark.unit
class TestCoefUntouched:

    @pytest.mark.parametrize("cycle_idx, step_idx", [(0, 0), (0, 1), (0, 2)])
    def test_only_hard_forbid_or_zero(
        self, ac_lib: Library, cycle_idx: int, step_idx: int,
    ) -> None:
        from kd.search.discover.tokens.prior import PriorContext

        prior = ScaffoldPrior(
            ac_lib,
            diffusion_tokens=_DIFFUSION_TOKENS_AC,
            reaction_tokens=_REACTION_TOKENS_AC,
            root_tokens=_ROOT_TOKENS,
        )
        prior.on_cycle_start(cycle_idx)

        if step_idx == 0:
            adjustment = prior.initial_adjustment(batch_size=8)
        else:
            add_idx = ac_lib.name_to_index("add")
            history = [add_idx] + [ac_lib.name_to_index("diff2_x")] * (step_idx - 1)
            actions = np.asarray([history], dtype=np.int32)
            ctx = PriorContext(
                actions=actions,
                parent=np.array([ac_lib.parent_adjust[add_idx]], dtype=np.int32),
                sibling=np.array([ac_lib.EMPTY_SIBLING], dtype=np.int32),
                dangling=np.array([1], dtype=np.int32),
                step_idx=step_idx,
                library=ac_lib,
            )
            adjustment = prior(ctx)


        finite_mask = np.isfinite(adjustment)
        np.testing.assert_array_equal(
            adjustment[finite_mask],
            np.zeros(int(finite_mask.sum()), dtype=np.float32),
        )

        non_finite = adjustment[~finite_mask]
        assert np.all(np.isneginf(non_finite)), (
            f"ScaffoldPrior must only emit -inf or 0, found {non_finite}"
        )







@pytest.mark.integration
class TestCrossTaskNoRegressionBurgers:

    def test_burgers_library_adjustment_only_hard_or_zero(
        self, burgers_lib: Library,
    ) -> None:
        prior = ScaffoldPrior(
            burgers_lib,
            diffusion_tokens=["diff2_x"],
            reaction_tokens=["u", "n3"],
            root_tokens=_ROOT_TOKENS,
        )
        prior.on_cycle_start(0)
        adjustment = prior.initial_adjustment(batch_size=4)

        finite_vals = adjustment[np.isfinite(adjustment)]
        np.testing.assert_array_equal(
            finite_vals,
            np.zeros(finite_vals.size, dtype=np.float32),
        )
        non_finite = adjustment[~np.isfinite(adjustment)]
        assert non_finite.size == 0 or np.all(np.isneginf(non_finite)), (
            "Burgers-configured ScaffoldPrior must emit only -inf or 0"
        )

    def test_burgers_default_config_still_skips_scaffold(
        self, burgers_lib: Library,
    ) -> None:
        config = DiscoverConfig.burgers_preset()
        assert config.diagnostic_scaffold is False, (
            "burgers_preset() must keep ScaffoldPrior off by default"
        )
        prior_system = build_prior_system(burgers_lib, config)
        scaffold_attached = [
            p for p in prior_system.priors if isinstance(p, ScaffoldPrior)
        ]
        assert len(scaffold_attached) == 0







@pytest.mark.unit
@pytest.mark.skipif(
    not _SCRIPT_PATH.exists(),
    reason="requires scripts/discover research CLI (not shipped in the public tree)",
)
class TestCLIFlagExists:

    def test_scaffold_flag_registered(self, parity_runner: ModuleType) -> None:


        old_argv = sys.argv
        try:
            sys.argv = [
                "task4_parity_ablation.py",
                "--config-name", "A",
                "--diagnostic-scaffold",
                "--diagnostic-scaffold-diffusion-tokens", "diff2_x", "diff2_y",
                "--diagnostic-scaffold-reaction-tokens", "u", "n3",
            ]
            args = parity_runner.parse_args()
        finally:
            sys.argv = old_argv

        assert getattr(args, "diagnostic_scaffold", None) is True, (
            "--diagnostic-scaffold must be parsed into args.diagnostic_scaffold"
        )
        assert tuple(args.diagnostic_scaffold_diffusion_tokens) == (
            "diff2_x", "diff2_y",
        )
        assert tuple(args.diagnostic_scaffold_reaction_tokens) == ("u", "n3")

    def test_default_absent_flag_is_false(self, parity_runner: ModuleType) -> None:
        old_argv = sys.argv
        try:
            sys.argv = ["task4_parity_ablation.py", "--config-name", "A"]
            args = parity_runner.parse_args()
        finally:
            sys.argv = old_argv
        assert args.diagnostic_scaffold is False

    def test_help_text_contains_diagnostic_marker(
        self, parity_runner: ModuleType, capsys: pytest.CaptureFixture[str],
    ) -> None:
        old_argv = sys.argv
        try:
            sys.argv = ["task4_parity_ablation.py", "--help"]
            with pytest.raises(SystemExit):
                parity_runner.parse_args()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr().out
        assert "--diagnostic-scaffold" in captured
        assert "(diagnostic — not production)" in captured







@pytest.mark.integration
class TestPriorSystemIntegration:

    def test_scaffold_hard_forbid_propagates_through_priorsystem(
        self, scaffold_enabled_config: DiscoverConfig,
    ) -> None:
        library = Library.from_config(scaffold_enabled_config.library)
        ps = build_prior_system(library, scaffold_enabled_config)
        for p in ps.priors:
            if isinstance(p, ScaffoldPrior):
                p.on_cycle_start(0)
        adjustment = ps.step(
            actions=np.zeros((1, 0), dtype=np.int32),
            obs=np.array([[
                library.EMPTY_ACTION,
                library.EMPTY_PARENT,
                library.EMPTY_SIBLING,
                1,
            ]], dtype=np.float32),
            step_idx=0,
        )
        root_idx_set = {
            library.name_to_index(n)
            for n in scaffold_enabled_config.diagnostic_scaffold_root_tokens
        }
        for idx in range(len(library.tokens)):
            if idx not in root_idx_set:
                assert adjustment[0, idx] == float("-inf"), (
                    f"token {library.index_to_name(idx)} must be -inf "
                    f"under active ScaffoldPrior at step 0"
                )





