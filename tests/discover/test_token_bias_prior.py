from __future__ import annotations

import logging

import numpy as np
import pytest

from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    LengthConstraint,
    PriorSystem,
    SoftLengthPrior,
)

try:
    from kd.search.discover.tokens.prior import TokenBiasPrior
except ImportError:
    TokenBiasPrior = None

pytestmark = pytest.mark.skipif(
    TokenBiasPrior is None,
    reason="TokenBiasPrior not yet implemented",
)






_ALLEN_CAHN_2D_CONFIG = LibraryConfig(
    coord_vars=["x", "y", "t"],
    state_vars=["u"],
    operators=[
        "add", "sub", "mul", "sin", "cos", "n2", "n3",
        "lap", "diff_x", "diff2_x", "diff_y", "diff2_y", "diff_t",
    ],
)

SOFT_LOC = 6.0
SOFT_SCALE = 2.0


@pytest.fixture
def lib() -> Library:
    return Library.from_config(_ALLEN_CAHN_2D_CONFIG)


def _initial_obs(library: Library, batch_size: int = 1) -> np.ndarray:
    obs = np.empty((batch_size, 4), dtype=np.float32)
    obs[:, 0] = library.EMPTY_ACTION
    obs[:, 1] = library.EMPTY_PARENT
    obs[:, 2] = library.EMPTY_SIBLING
    obs[:, 3] = 1
    return obs


def _dummy_actions(batch_size: int, step_idx: int) -> np.ndarray:
    return np.zeros((batch_size, step_idx), dtype=np.int32)





class TestConstruction:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_construct_with_valid_tokens(self, lib: Library) -> None:
        prior = TokenBiasPrior(lib, token_names=["u", "n3"], bias=1.0)
        assert prior is not None


        assert prior.library is lib

    @pytest.mark.unit
    def test_unknown_tokens_warn_and_skip(
        self, lib: Library, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            prior = TokenBiasPrior(
                lib, token_names=["u", "not_a_token"], bias=1.0,
            )
        assert prior is not None, "constructor must not raise on unknown names"

        assert any(
            "not_a_token" in record.getMessage() for record in caplog.records
        ), (
            f"expected a WARNING mentioning 'not_a_token', "
            f"got {[r.getMessage() for r in caplog.records]}"
        )


        adjustment = prior._adjustment_for_step(batch_size=1, step_idx=0)
        u_idx = lib.name_to_index("u")
        assert adjustment[0, u_idx] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_empty_token_list_yields_all_zero(self, lib: Library) -> None:
        prior = TokenBiasPrior(lib, token_names=[], bias=5.0)
        adjustment = prior._adjustment_for_step(batch_size=4, step_idx=0)
        np.testing.assert_array_equal(adjustment, np.zeros_like(adjustment))





class TestAdjustmentSemantics:

    @pytest.mark.unit
    def test_zero_bias_is_noop(self, lib: Library) -> None:
        prior = TokenBiasPrior(lib, token_names=["u", "n3"], bias=0.0)
        adjustment = prior._adjustment_for_step(batch_size=3, step_idx=0)
        np.testing.assert_array_equal(adjustment, np.zeros_like(adjustment))

    @pytest.mark.unit
    def test_positive_bias_on_listed_tokens(self, lib: Library) -> None:
        bias = 2.5
        prior = TokenBiasPrior(lib, token_names=["u", "n3"], bias=bias)
        adjustment = prior._adjustment_for_step(batch_size=1, step_idx=0)

        u_idx = lib.name_to_index("u")
        n3_idx = lib.name_to_index("n3")
        assert adjustment[0, u_idx] == pytest.approx(bias)
        assert adjustment[0, n3_idx] == pytest.approx(bias)

    @pytest.mark.unit
    def test_non_listed_tokens_exactly_zero(self, lib: Library) -> None:
        prior = TokenBiasPrior(lib, token_names=["u"], bias=3.14)
        adjustment = prior._adjustment_for_step(batch_size=1, step_idx=0)

        listed = {lib.name_to_index("u")}
        for idx, name in enumerate(lib.names):
            if idx in listed:
                continue
            assert adjustment[0, idx] == 0.0, (
                f"non-listed token {name}: expected 0.0, "
                f"got {adjustment[0, idx]}"
            )

    @pytest.mark.unit
    def test_negative_bias_symmetric(self, lib: Library) -> None:
        prior = TokenBiasPrior(lib, token_names=["u", "n3"], bias=-5.0)
        adjustment = prior._adjustment_for_step(batch_size=1, step_idx=0)

        u_idx = lib.name_to_index("u")
        n3_idx = lib.name_to_index("n3")
        assert adjustment[0, u_idx] == pytest.approx(-5.0)
        assert adjustment[0, n3_idx] == pytest.approx(-5.0)

        add_idx = lib.name_to_index("add")
        assert adjustment[0, add_idx] == 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 32, 1000])
    def test_shape_matches_batch_and_n_choices(
        self, lib: Library, batch_size: int,
    ) -> None:
        prior = TokenBiasPrior(lib, token_names=["u"], bias=1.0)
        adjustment = prior._adjustment_for_step(batch_size, step_idx=0)
        assert adjustment.shape == (batch_size, len(lib.tokens))

    @pytest.mark.unit
    @pytest.mark.parametrize("step_idx", [0, 1, 3, 50, 255])
    def test_step_invariant(self, lib: Library, step_idx: int) -> None:
        prior = TokenBiasPrior(lib, token_names=["u", "n3"], bias=1.0)
        reference = prior._adjustment_for_step(batch_size=16, step_idx=0)
        current = prior._adjustment_for_step(batch_size=16, step_idx=step_idx)
        np.testing.assert_array_equal(reference, current)

    @pytest.mark.unit
    def test_dtype_is_float32(self, lib: Library) -> None:
        prior = TokenBiasPrior(lib, token_names=["u"], bias=1.0)
        adjustment = prior._adjustment_for_step(batch_size=2, step_idx=0)
        assert adjustment.dtype == np.float32

    @pytest.mark.unit
    def test_large_bias_stays_finite(self, lib: Library) -> None:
        prior = TokenBiasPrior(lib, token_names=["u", "n3"], bias=100.0)
        adjustment = prior._adjustment_for_step(batch_size=4, step_idx=0)
        assert np.all(np.isfinite(adjustment)), (
            "adjustment must be finite even for large bias values"
        )





class TestBatchRowBroadcasting:

    @pytest.mark.unit
    def test_all_rows_identical(self, lib: Library) -> None:
        prior = TokenBiasPrior(lib, token_names=["u"], bias=1.5)
        adjustment = prior._adjustment_for_step(batch_size=8, step_idx=0)
        for b in range(1, adjustment.shape[0]):
            np.testing.assert_array_equal(adjustment[b], adjustment[0])





class TestPriorSystemComposition:

    @pytest.mark.integration
    def test_adds_log_prob_in_composed_system(self, lib: Library) -> None:
        bias = 2.0
        u_idx = lib.name_to_index("u")

        baseline = PriorSystem(
            lib,
            priors=[SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)],
        )
        boosted = PriorSystem(
            lib,
            priors=[
                SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE),
                TokenBiasPrior(lib, token_names=["u"], bias=bias),
            ],
        )

        obs = _initial_obs(lib)
        actions = _dummy_actions(1, 2)

        adj_baseline = baseline.step(actions, obs, step_idx=2)
        adj_boosted = boosted.step(actions, obs, step_idx=2)


        assert adj_boosted[0, u_idx] == pytest.approx(
            float(adj_baseline[0, u_idx]) + bias,
        ), (
            f"expected u-column shift by {bias}: "
            f"baseline={adj_baseline[0, u_idx]}, boosted={adj_boosted[0, u_idx]}"
        )



        for idx, name in enumerate(lib.names):
            if idx == u_idx:
                continue
            assert adj_boosted[0, idx] == adj_baseline[0, idx], (
                f"token {name!r} changed unexpectedly: "
                f"baseline={adj_baseline[0, idx]}, boosted={adj_boosted[0, idx]}"
            )

    @pytest.mark.integration
    def test_hard_forbid_still_dominates(self, lib: Library) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                LengthConstraint(lib, min_=3, max_=10),
                TokenBiasPrior(lib, token_names=["u"], bias=5.0),
            ],
        )
        adjustment = ps.step(
            _dummy_actions(1, 0), _initial_obs(lib), step_idx=0,
        )
        u_idx = lib.name_to_index("u")
        assert adjustment[0, u_idx] == float("-inf"), (
            "hard LengthConstraint must dominate soft TokenBiasPrior at step 0"
        )

    @pytest.mark.integration
    def test_compute_batch_passes_through(self, lib: Library) -> None:
        bias = 1.5
        u_idx = lib.name_to_index("u")

        boosted = PriorSystem(
            lib,
            priors=[TokenBiasPrior(lib, token_names=["u"], bias=bias)],
        )


        add_idx = lib.name_to_index("add")
        actions = np.array([[add_idx, u_idx]], dtype=np.int32)

        seq_len = actions.shape[1]
        obs = np.zeros((1, 4, seq_len), dtype=np.float32)
        obs[:, 0,:] = lib.EMPTY_ACTION
        obs[:, 1,:] = lib.EMPTY_PARENT
        obs[:, 2,:] = lib.EMPTY_SIBLING
        obs[:, 3,:] = 1

        adjustment = boosted.compute_batch(actions, obs)
        assert adjustment.shape == (1, seq_len, len(lib.tokens))


        for t in range(seq_len):
            assert adjustment[0, t, u_idx] == pytest.approx(bias), (
                f"step {t}: expected u-column={bias}, "
                f"got {adjustment[0, t, u_idx]}"
            )
