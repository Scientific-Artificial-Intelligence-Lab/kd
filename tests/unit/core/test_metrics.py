
from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from kd.core.metrics import (
    ScorerFn,
    aic,
    aic_no_n,
    aicc,
    bic,
    make_aic_scorer,
    make_bic_scorer,
    make_sga_scorer,
    nmse,
)




positive_mse = st.floats(
    min_value=1e-12, max_value=1e8, allow_nan=False, allow_infinity=False
)


complexity_k = st.integers(min_value=0, max_value=50)


sample_n = st.integers(min_value=2, max_value=10000)


aic_ratio = st.floats(
    min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
)







class TestSmoke:

    @pytest.mark.smoke
    def test_aic_callable(self) -> None:
        result = aic(mse=1.0, k=2, n=100)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_aic_no_n_callable(self) -> None:
        result = aic_no_n(mse=1.0, k=2)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_aicc_callable(self) -> None:
        result = aicc(mse=1.0, k=2, n=100)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_bic_callable(self) -> None:
        result = bic(mse=1.0, k=2, n=100)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_nmse_callable(self) -> None:
        result = nmse(mse=0.5, target_var=1.0)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_scorer_fn_type_is_callable(self) -> None:

        assert ScorerFn is not None


        def my_scorer(mse: float, k: int) -> float:
            return mse + k

        scorer: ScorerFn = my_scorer
        assert scorer(1.0, 2) == 3.0

    @pytest.mark.smoke
    def test_make_aic_scorer_callable(self) -> None:
        scorer = make_aic_scorer(n=100)
        result = scorer(1.0, 2)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_make_sga_scorer_callable(self) -> None:
        scorer = make_sga_scorer(ratio=1.0)
        result = scorer(1.0, 2)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_make_bic_scorer_callable(self) -> None:
        scorer = make_bic_scorer(n=100)
        result = scorer(1.0, 2)
        assert isinstance(result, float)







class TestAICKnownValues:

    def test_mse_one_k_zero(self) -> None:
        result = aic(mse=1.0, k=0, n=100)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_mse_one_k_positive(self) -> None:
        result = aic(mse=1.0, k=5, n=50)
        assert result == pytest.approx(10.0, abs=1e-12)

    def test_mse_e_squared(self) -> None:
        e2 = math.e**2
        result = aic(mse=e2, k=3, n=50)
        expected = 50.0 * 2.0 + 2.0 * 3.0
        assert result == pytest.approx(expected, abs=1e-10)

    def test_k_zero_reduces_to_n_log_mse(self) -> None:
        result = aic(mse=0.01, k=0, n=200)
        expected = 200.0 * math.log(0.01)
        assert result == pytest.approx(expected, abs=1e-10)


class TestAICNoNKnownValues:

    def test_mse_one_ratio_one(self) -> None:
        result = aic_no_n(mse=1.0, k=3, ratio=1.0)
        assert result == pytest.approx(6.0, abs=1e-12)

    def test_k_zero(self) -> None:
        result = aic_no_n(mse=0.5, k=0, ratio=1.0)
        expected = 2.0 * math.log(0.5)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_custom_ratio(self) -> None:
        result = aic_no_n(mse=1.0, k=4, ratio=2.5)
        expected = 2.0 * 4 * 2.5 + 2.0 * math.log(1.0)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_default_ratio_is_one(self) -> None:
        with_default = aic_no_n(mse=2.0, k=3)
        with_explicit = aic_no_n(mse=2.0, k=3, ratio=1.0)
        assert with_default == pytest.approx(with_explicit, abs=1e-15)


class TestAICcKnownValues:

    def test_large_n_converges_to_aic(self) -> None:
        n = 10000
        k = 3
        mse = 0.5
        result_aicc = aicc(mse=mse, k=k, n=n)
        result_aic = aic(mse=mse, k=k, n=n)

        assert abs(result_aicc - result_aic) < 0.01

    def test_k_zero_equals_aic(self) -> None:
        result_aicc = aicc(mse=2.0, k=0, n=100)
        result_aic = aic(mse=2.0, k=0, n=100)
        assert result_aicc == pytest.approx(result_aic, abs=1e-12)

    def test_correction_positive(self) -> None:
        mse, k, n = 0.5, 5, 50
        result_aicc = aicc(mse=mse, k=k, n=n)
        result_aic = aic(mse=mse, k=k, n=n)
        assert result_aicc >= result_aic

    def test_independent_computation(self) -> None:
        mse, k, n = 0.3, 4, 20

        base = n * math.log(mse) + 2 * k

        correction = 2.0 * k * (k + 1) / (n - k - 1)
        expected = base + correction
        result = aicc(mse=mse, k=k, n=n)
        assert result == pytest.approx(expected, abs=1e-10)


class TestBICKnownValues:

    def test_mse_one_k_zero(self) -> None:
        result = bic(mse=1.0, k=0, n=100)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_k_zero_equals_aic(self) -> None:
        mse = 0.25
        n = 100
        result_bic = bic(mse=mse, k=0, n=n)
        result_aic = aic(mse=mse, k=0, n=n)
        assert result_bic == pytest.approx(result_aic, abs=1e-12)

    def test_bic_penalty_heavier_than_aic_for_large_n(self) -> None:
        mse = 0.5
        k = 5
        n = 100
        result_bic = bic(mse=mse, k=k, n=n)
        result_aic = aic(mse=mse, k=k, n=n)

        assert result_bic > result_aic

    def test_independent_computation(self) -> None:
        mse, k, n = 0.1, 3, 200
        expected = n * math.log(mse) + k * math.log(n)
        result = bic(mse=mse, k=k, n=n)
        assert result == pytest.approx(expected, abs=1e-10)


class TestNMSEKnownValues:

    def test_unit_variance(self) -> None:
        result = nmse(mse=0.3, target_var=1.0)
        assert result == pytest.approx(0.3, abs=1e-12)

    def test_normalization(self) -> None:
        result = nmse(mse=0.5, target_var=2.0)
        assert result == pytest.approx(0.25, abs=1e-12)

    def test_perfect_fit(self) -> None:
        result = nmse(mse=0.0, target_var=1.0)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_nmse_one_means_as_bad_as_mean_model(self) -> None:
        var = 3.14
        result = nmse(mse=var, target_var=var)
        assert result == pytest.approx(1.0, abs=1e-12)







class TestAICEdgeCases:

    def test_mse_zero_returns_inf_or_neg_inf(self) -> None:
        result = aic(mse=0.0, k=2, n=100)
        assert result == float("-inf")

    def test_mse_very_small_returns_neg_inf(self) -> None:
        result = aic(mse=1e-16, k=2, n=100)
        assert result == float("-inf")

    def test_mse_negative_returns_inf(self) -> None:
        result = aic(mse=-1.0, k=2, n=100)
        assert result == float("inf")

    def test_mse_nan_returns_inf(self) -> None:
        result = aic(mse=float("nan"), k=2, n=100)
        assert result == float("inf")

    def test_mse_inf_returns_inf(self) -> None:
        result = aic(mse=float("inf"), k=2, n=100)
        assert result == float("inf")

    def test_mse_neg_inf_returns_inf(self) -> None:
        result = aic(mse=float("-inf"), k=2, n=100)
        assert result == float("inf")


class TestAICNoNEdgeCases:

    def test_mse_zero_returns_inf(self) -> None:
        result = aic_no_n(mse=0.0, k=2)
        assert result == float("inf")

    def test_mse_negative_returns_inf(self) -> None:
        result = aic_no_n(mse=-0.5, k=1)
        assert result == float("inf")

    def test_mse_nan_returns_inf(self) -> None:
        result = aic_no_n(mse=float("nan"), k=1)
        assert result == float("inf")

    def test_mse_inf_returns_inf(self) -> None:
        result = aic_no_n(mse=float("inf"), k=1)
        assert result == float("inf")


class TestAICcEdgeCases:

    def test_n_equals_k_plus_one_returns_inf(self) -> None:
        result = aicc(mse=1.0, k=5, n=6)
        assert result == float("inf")

    def test_n_less_than_k_plus_one_returns_inf(self) -> None:
        result = aicc(mse=1.0, k=10, n=5)
        assert result == float("inf")

    def test_n_equals_k_plus_two(self) -> None:
        k = 3
        n = k + 2
        result = aicc(mse=1.0, k=k, n=n)

        assert math.isfinite(result)

    def test_mse_zero_returns_neg_inf_or_inf(self) -> None:

        result = aicc(mse=0.0, k=2, n=100)
        assert result == float("-inf")

    def test_mse_nan_returns_inf(self) -> None:
        result = aicc(mse=float("nan"), k=2, n=100)
        assert result == float("inf")


class TestBICEdgeCases:

    def test_mse_zero_returns_neg_inf(self) -> None:
        result = bic(mse=0.0, k=2, n=100)
        assert result == float("-inf")

    def test_mse_nan_returns_inf(self) -> None:
        result = bic(mse=float("nan"), k=2, n=100)
        assert result == float("inf")

    def test_mse_inf_returns_inf(self) -> None:
        result = bic(mse=float("inf"), k=2, n=100)
        assert result == float("inf")

    def test_mse_negative_returns_inf(self) -> None:
        result = bic(mse=-0.5, k=2, n=100)
        assert result == float("inf")


class TestNMSEEdgeCases:

    def test_target_var_zero_returns_mse(self) -> None:
        result = nmse(mse=0.5, target_var=0.0)
        assert result == pytest.approx(0.5, abs=1e-12)

    def test_target_var_below_eps_returns_mse(self) -> None:
        result = nmse(mse=0.3, target_var=1e-20, eps=1e-15)
        assert result == pytest.approx(0.3, abs=1e-12)

    def test_target_var_negative_returns_mse(self) -> None:
        result = nmse(mse=0.5, target_var=-1.0)
        assert result == pytest.approx(0.5, abs=1e-12)

    def test_mse_zero_target_var_positive(self) -> None:
        result = nmse(mse=0.0, target_var=5.0)
        assert result == pytest.approx(0.0, abs=1e-15)







class TestRankingEquivalence:

    def test_aic_no_n_equals_aic_at_n_2(self) -> None:
        cases = [
            (0.01, 1),
            (0.001, 5),
            (1.0, 0),
            (0.5, 3),
            (100.0, 10),
        ]
        for mse_val, k_val in cases:
            assert aic(mse=mse_val, k=k_val, n=2) == pytest.approx(
                aic_no_n(mse=mse_val, k=k_val, ratio=1.0)
            ), f"mse={mse_val}, k={k_val}"

    def test_ranking_can_diverge_for_large_n(self) -> None:


        mse_a, k_a = 0.01, 5
        mse_b, k_b = 0.1, 1


        n_large = 1000
        assert aic(mse=mse_a, k=k_a, n=n_large) < aic(mse=mse_b, k=k_b, n=n_large)


        assert aic_no_n(mse=mse_a, k=k_a) > aic_no_n(mse=mse_b, k=k_b)

    @given(
        mse_a=positive_mse,
        mse_b=positive_mse,
        k_a=st.integers(min_value=0, max_value=20),
        k_b=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=300)
    def test_aic_no_n_is_aic_at_n_2_property(
        self, mse_a: float, mse_b: float, k_a: int, k_b: int
    ) -> None:
        score_a_full = aic(mse=mse_a, k=k_a, n=2)
        score_b_full = aic(mse=mse_b, k=k_b, n=2)
        score_a_no_n = aic_no_n(mse=mse_a, k=k_a, ratio=1.0)
        score_b_no_n = aic_no_n(mse=mse_b, k=k_b, ratio=1.0)


        assume(score_a_full != score_b_full)
        assume(score_a_no_n != score_b_no_n)

        full_a_wins = score_a_full < score_b_full
        no_n_a_wins = score_a_no_n < score_b_no_n
        assert full_a_wins == no_n_a_wins







class TestMakeAICScorer:

    def test_returns_callable(self) -> None:
        scorer = make_aic_scorer(n=100)
        assert callable(scorer)

    def test_signature_matches_scorer_fn(self) -> None:
        scorer = make_aic_scorer(n=100)
        result = scorer(1.0, 3)
        assert isinstance(result, float)

    def test_binds_n_correctly(self) -> None:
        n = 250
        scorer = make_aic_scorer(n=n)
        mse, k = 0.3, 4
        assert scorer(mse, k) == pytest.approx(aic(mse, k, n), abs=1e-12)

    def test_different_n_different_result(self) -> None:
        scorer_100 = make_aic_scorer(n=100)
        scorer_200 = make_aic_scorer(n=200)
        mse, k = 0.5, 3

        assert scorer_100(mse, k) != scorer_200(mse, k)

    def test_edge_case_forwarded(self) -> None:
        scorer = make_aic_scorer(n=100)
        assert scorer(0.0, 2) == float("-inf")
        assert scorer(float("nan"), 2) == float("inf")


class TestMakeSGAScorer:

    def test_returns_callable(self) -> None:
        scorer = make_sga_scorer(ratio=1.0)
        assert callable(scorer)

    def test_binds_ratio_correctly(self) -> None:
        ratio = 1.5
        scorer = make_sga_scorer(ratio=ratio)
        mse, k = 0.4, 6
        assert scorer(mse, k) == pytest.approx(aic_no_n(mse, k, ratio), abs=1e-12)

    def test_default_ratio_one(self) -> None:
        scorer = make_sga_scorer()
        result = scorer(1.0, 3)
        expected = aic_no_n(1.0, 3, 1.0)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_different_ratio_different_result(self) -> None:
        scorer_1 = make_sga_scorer(ratio=1.0)
        scorer_2 = make_sga_scorer(ratio=2.0)

        assert scorer_1(0.5, 5) != scorer_2(0.5, 5)

    def test_edge_case_forwarded(self) -> None:
        scorer = make_sga_scorer(ratio=1.0)
        assert scorer(0.0, 2) == float("inf")
        assert scorer(float("nan"), 2) == float("inf")


class TestMakeBICScorer:

    def test_returns_callable(self) -> None:
        scorer = make_bic_scorer(n=100)
        assert callable(scorer)

    def test_binds_n_correctly(self) -> None:
        n = 300
        scorer = make_bic_scorer(n=n)
        mse, k = 0.2, 3
        assert scorer(mse, k) == pytest.approx(bic(mse, k, n), abs=1e-12)

    def test_edge_case_forwarded(self) -> None:
        scorer = make_bic_scorer(n=100)
        assert scorer(0.0, 2) == float("-inf")
        assert scorer(float("nan"), 2) == float("inf")







@pytest.mark.numerical
class TestAICProperties:

    @given(mse=positive_mse, k=complexity_k, n=sample_n)
    @settings(max_examples=500)
    def test_finite_output(self, mse: float, k: int, n: int) -> None:
        result = aic(mse=mse, k=k, n=n)
        assert math.isfinite(result)

    @given(mse=positive_mse, n=sample_n)
    @settings(max_examples=200)
    def test_monotonic_in_k(self, mse: float, n: int) -> None:
        scores = [aic(mse=mse, k=k, n=n) for k in range(10)]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    @given(k=complexity_k, n=sample_n)
    @settings(max_examples=200)
    def test_monotonic_in_mse(self, k: int, n: int) -> None:
        mse_values = [0.01, 0.1, 0.5, 1.0, 5.0, 100.0]
        scores = [aic(mse=m, k=k, n=n) for m in mse_values]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    @given(mse=positive_mse, k=complexity_k, n=sample_n)
    @settings(max_examples=200)
    def test_aic_decomposition(self, mse: float, k: int, n: int) -> None:
        result = aic(mse=mse, k=k, n=n)
        gof = n * math.log(mse)
        penalty = 2.0 * k
        assert result == pytest.approx(gof + penalty, abs=1e-8)


@pytest.mark.numerical
class TestAICNoNProperties:

    @given(mse=positive_mse, k=complexity_k, ratio=aic_ratio)
    @settings(max_examples=500)
    def test_finite_output(self, mse: float, k: int, ratio: float) -> None:
        result = aic_no_n(mse=mse, k=k, ratio=ratio)
        assert math.isfinite(result)

    @given(mse=positive_mse, ratio=aic_ratio)
    @settings(max_examples=200)
    def test_monotonic_in_k(self, mse: float, ratio: float) -> None:
        scores = [aic_no_n(mse=mse, k=k, ratio=ratio) for k in range(10)]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    @given(k=complexity_k, ratio=aic_ratio)
    @settings(max_examples=200)
    def test_monotonic_in_mse(self, k: int, ratio: float) -> None:
        mse_values = [0.01, 0.1, 0.5, 1.0, 5.0, 100.0]
        scores = [aic_no_n(mse=m, k=k, ratio=ratio) for m in mse_values]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


@pytest.mark.numerical
class TestAICcProperties:

    @given(
        mse=positive_mse,
        k=st.integers(min_value=0, max_value=10),
        n=st.integers(min_value=20, max_value=5000),
    )
    @settings(max_examples=300)
    def test_aicc_geq_aic(self, mse: float, k: int, n: int) -> None:
        assume(n > k + 1)
        result_aicc = aicc(mse=mse, k=k, n=n)
        result_aic = aic(mse=mse, k=k, n=n)
        assert result_aicc >= result_aic - 1e-10

    @given(mse=positive_mse, k=st.integers(min_value=1, max_value=10))
    @settings(max_examples=200)
    def test_correction_decreases_with_n(self, mse: float, k: int) -> None:
        n_values = [k + 5, k + 20, k + 100, k + 1000]
        corrections = []
        for n in n_values:
            diff = aicc(mse=mse, k=k, n=n) - aic(mse=mse, k=k, n=n)
            corrections.append(diff)

        for i in range(len(corrections) - 1):
            assert corrections[i] > corrections[i + 1]


@pytest.mark.numerical
class TestBICProperties:

    @given(mse=positive_mse, k=complexity_k, n=sample_n)
    @settings(max_examples=500)
    def test_finite_output(self, mse: float, k: int, n: int) -> None:
        result = bic(mse=mse, k=k, n=n)
        assert math.isfinite(result)

    @given(mse=positive_mse, n=sample_n)
    @settings(max_examples=200)
    def test_monotonic_in_k(self, mse: float, n: int) -> None:
        scores = [bic(mse=mse, k=k, n=n) for k in range(10)]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    @given(mse=positive_mse, k=st.integers(min_value=1, max_value=20))
    @settings(max_examples=200)
    def test_bic_penalty_scales_with_n(self, mse: float, k: int) -> None:
        n_small, n_large = 10, 1000
        score_small = bic(mse=mse, k=k, n=n_small)
        score_large = bic(mse=mse, k=k, n=n_large)


        penalty_diff = k * (math.log(n_large) - math.log(n_small))
        assert penalty_diff > 0

        gof_diff = (n_large - n_small) * math.log(mse)
        expected_diff = gof_diff + penalty_diff
        assert score_large - score_small == pytest.approx(expected_diff, abs=1e-8)


@pytest.mark.numerical
class TestNMSEProperties:

    @given(
        mse=st.floats(
            min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False
        ),
        var=st.floats(
            min_value=1e-6, max_value=1e8, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=300)
    def test_non_negative(self, mse: float, var: float) -> None:
        result = nmse(mse=mse, target_var=var)
        assert result >= 0.0

    @given(
        var=st.floats(
            min_value=1e-6, max_value=1e8, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=200)
    def test_monotonic_in_mse(self, var: float) -> None:
        mse_values = [0.0, 0.01, 0.1, 0.5, 1.0, 10.0]
        scores = [nmse(mse=m, target_var=var) for m in mse_values]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    @given(
        mse=st.floats(
            min_value=1e-10, max_value=1e4, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=200)
    def test_scales_inversely_with_variance(self, mse: float) -> None:
        var = 2.0
        result_1 = nmse(mse=mse, target_var=var)
        result_2 = nmse(mse=mse, target_var=2.0 * var)
        assert result_2 == pytest.approx(result_1 / 2.0, rel=1e-10)







class TestCrossMetricConsistency:

    def test_aic_and_bic_agree_on_simpler_better_at_same_mse(self) -> None:
        mse = 0.5
        n = 100
        simple = (aic(mse, k=1, n=n), bic(mse, k=1, n=n))
        complex_ = (aic(mse, k=5, n=n), bic(mse, k=5, n=n))

        assert simple[0] < complex_[0]
        assert simple[1] < complex_[1]

    def test_bic_more_conservative_than_aic(self) -> None:
        n = 100

        mse_simple = 0.12
        k_simple = 1

        mse_complex = 0.10
        k_complex = 5


        aic_diff = aic(mse_complex, k_complex, n) - aic(mse_simple, k_simple, n)

        bic_diff = bic(mse_complex, k_complex, n) - bic(mse_simple, k_simple, n)


        assert bic_diff > aic_diff

    def test_aicc_and_aic_agree_on_best_for_large_n(self) -> None:
        n = 10000
        candidates = [
            (0.01, 1),
            (0.005, 3),
            (0.001, 8),
            (0.1, 0),
        ]
        aic_scores = [aic(m, k, n) for m, k in candidates]
        aicc_scores = [aicc(m, k, n) for m, k in candidates]

        best_aic = min(range(len(candidates)), key=lambda i: aic_scores[i])
        best_aicc = min(range(len(candidates)), key=lambda i: aicc_scores[i])

        assert best_aic == best_aicc
