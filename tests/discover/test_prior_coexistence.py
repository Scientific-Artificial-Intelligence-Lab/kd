from __future__ import annotations

import numpy as np
import pytest

from kd.search.discover.builder import build_prior_system
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import PriorContext, ScaffoldPrior, TokenBiasPrior




_AC_LIKE_CONFIG = LibraryConfig(
    coord_vars=["x", "y", "t"],
    state_vars=["u"],
    operators=[
        "add", "sub", "mul", "sin", "cos", "n2", "n3",
        "diff_x", "diff2_x", "diff_y", "diff2_y", "diff_t",
    ],
)

_DIFFUSION_TOKENS = ("diff2_x", "diff2_y")
_REACTION_TOKENS = ("u", "n3")
_ROOT_TOKENS = ("add", "sub")
_TOKEN_BIAS_TARGETS = ("u",)
_TOKEN_BIAS_WEIGHT = 1.0
_BATCH_SIZE = 4
_INACTIVE_CYCLE_IDX = 1


@pytest.fixture
def library() -> Library:
    return Library.from_config(_AC_LIKE_CONFIG)


@pytest.fixture
def coexistence_config() -> DiscoverConfig:
    return DiscoverConfig(
        diagnostic_scaffold=True,
        diagnostic_scaffold_diffusion_tokens=_DIFFUSION_TOKENS,
        diagnostic_scaffold_reaction_tokens=_REACTION_TOKENS,
        diagnostic_scaffold_root_tokens=_ROOT_TOKENS,
        token_bias_tokens=_TOKEN_BIAS_TARGETS,
        token_bias_weight=_TOKEN_BIAS_WEIGHT,
    )






@pytest.fixture(autouse=True)
def _enable_diagnostics_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCOVER_ENABLE_DIAGNOSTICS", "1")


@pytest.mark.unit
class TestScaffoldAndTokenBiasCoexist:

    def test_both_priors_present_in_system(
        self, library: Library, coexistence_config: DiscoverConfig,
    ) -> None:
        prior_system = build_prior_system(library, coexistence_config)
        assert len(prior_system.priors) >= 2
        prior_types = [type(p).__name__ for p in prior_system.priors]
        assert "ScaffoldPrior" in prior_types
        assert "TokenBiasPrior" in prior_types

    def test_cycle_0_scaffold_mask_and_token_bias_both_apply(
        self, library: Library, coexistence_config: DiscoverConfig,
    ) -> None:
        prior_system = build_prior_system(library, coexistence_config)
        scaffold = next(
            p for p in prior_system.priors if isinstance(p, ScaffoldPrior)
        )


        scaffold.on_cycle_start(0)

        scaffold_only = scaffold.initial_adjustment(_BATCH_SIZE)
        token_bias = next(
            p for p in prior_system.priors if isinstance(p, TokenBiasPrior)
        )
        bias_only = token_bias.initial_adjustment(_BATCH_SIZE)

        u_idx = library.name_to_index("u")
        add_idx = library.name_to_index("add")


        assert np.isneginf(scaffold_only[:, u_idx]).all(), (
            "ScaffoldPrior must hard-forbid non-root tokens at step 0"
        )
        assert scaffold_only[:, add_idx].tolist() == [0.0] * _BATCH_SIZE, (
            "ScaffoldPrior must NOT mask root tokens at step 0"
        )
        assert bias_only[:, u_idx].tolist() == (
            [_TOKEN_BIAS_WEIGHT] * _BATCH_SIZE
        )
        assert bias_only[:, add_idx].tolist() == [0.0] * _BATCH_SIZE



        combined = scaffold_only + bias_only
        assert np.isneginf(combined[:, u_idx]).all(), (
            "Scaffold mask must dominate the finite TokenBias contribution"
        )
        assert combined[:, add_idx].tolist() == [0.0] * _BATCH_SIZE, (
            "Root column must remain finite after combining the two priors"
        )

    def test_cycle_ge_1_scaffold_inactive_token_bias_remains(
        self, library: Library, coexistence_config: DiscoverConfig,
    ) -> None:
        prior_system = build_prior_system(library, coexistence_config)
        scaffold = next(
            p for p in prior_system.priors if isinstance(p, ScaffoldPrior)
        )
        scaffold.on_cycle_start(_INACTIVE_CYCLE_IDX)

        scaffold_only = scaffold.initial_adjustment(_BATCH_SIZE)
        np.testing.assert_array_equal(
            scaffold_only, np.zeros_like(scaffold_only),
        )

        token_bias = next(
            p for p in prior_system.priors if isinstance(p, TokenBiasPrior)
        )
        bias_only = token_bias.initial_adjustment(_BATCH_SIZE)
        u_idx = library.name_to_index("u")

        assert bias_only[:, u_idx].tolist() == (
            [_TOKEN_BIAS_WEIGHT] * _BATCH_SIZE
        )


def _step0_ctx(library: Library, batch_size: int) -> PriorContext:
    return PriorContext(
        actions=np.zeros((batch_size, 0), dtype=np.int32),
        parent=np.full(batch_size, library.EMPTY_PARENT, dtype=np.int32),
        sibling=np.full(batch_size, library.EMPTY_SIBLING, dtype=np.int32),
        dangling=np.ones(batch_size, dtype=np.int32),
        step_idx=0,
        library=library,
    )


def _step1_ctx(library: Library, batch_size: int) -> PriorContext:
    add_idx = library.name_to_index("add")
    return PriorContext(
        actions=np.full((batch_size, 1), add_idx, dtype=np.int32),
        parent=np.full(batch_size, add_idx, dtype=np.int32),
        sibling=np.full(batch_size, library.EMPTY_SIBLING, dtype=np.int32),
        dangling=np.full(batch_size, 2, dtype=np.int32),
        step_idx=1,
        library=library,
    )


@pytest.mark.unit
class TestCombineViaEnginePath:

    def test_combine_step0_cycle0_sums_scaffold_and_token_bias(
        self, library: Library, coexistence_config: DiscoverConfig,
    ) -> None:
        prior_system = build_prior_system(library, coexistence_config)
        scaffold = next(
            p for p in prior_system.priors if isinstance(p, ScaffoldPrior)
        )
        scaffold.on_cycle_start(0)

        ctx = _step0_ctx(library, _BATCH_SIZE)
        combined = prior_system._combine(ctx)





        expected = np.zeros_like(combined)
        for prior in prior_system.priors:
            expected += prior_system._prior_adjustment(prior, ctx)
        np.testing.assert_array_equal(combined, expected)

        u_idx = library.name_to_index("u")
        add_idx = library.name_to_index("add")

        assert np.isneginf(combined[:, u_idx]).all()

        assert combined[:, add_idx].tolist() == [0.0] * _BATCH_SIZE

    def test_combine_step1_cycle_ge_1_drops_scaffold_keeps_token_bias(
        self, library: Library, coexistence_config: DiscoverConfig,
    ) -> None:
        prior_system = build_prior_system(library, coexistence_config)
        scaffold = next(
            p for p in prior_system.priors if isinstance(p, ScaffoldPrior)
        )
        scaffold.on_cycle_start(_INACTIVE_CYCLE_IDX)

        ctx = _step1_ctx(library, _BATCH_SIZE)
        combined = prior_system._combine(ctx)




        expected = np.zeros_like(combined)
        for prior in prior_system.priors:
            expected += prior_system._prior_adjustment(prior, ctx)
        np.testing.assert_array_equal(combined, expected)



        u_idx = library.name_to_index("u")
        assert np.isfinite(combined[:, u_idx]).all(), (
            "ScaffoldPrior mask must drop on cycle >= 1, leaving u "
            "finite at step >= 1 (Length terminal-mask only fires at step 0)"
        )


        no_bias_priors = [
            p for p in prior_system.priors
            if not isinstance(p, TokenBiasPrior)
        ]
        no_bias_sum = np.zeros_like(combined)
        for prior in no_bias_priors:
            no_bias_sum += prior_system._prior_adjustment(prior, ctx)
        delta = combined[:, u_idx] - no_bias_sum[:, u_idx]
        np.testing.assert_allclose(
            delta, [_TOKEN_BIAS_WEIGHT] * _BATCH_SIZE,
        )
