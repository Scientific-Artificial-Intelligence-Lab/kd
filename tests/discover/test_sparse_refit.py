
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from kd.search.discover.evaluation.reward import DEFAULT_ALPHA
from kd.search.discover.runners.sparse_refit import (
    SparseRefitResult,
    consolidate_signed_duplicates,
    refit_candidate,
    select_kept_indices,
)






@pytest.mark.unit
def test_select_kept_identity_no_decoys() -> None:
    kept, dropped, threshold = select_kept_indices(
        [10.0, 10.0, 10.0], eps_relative=1e-4,
    )
    assert kept == [0, 1, 2]
    assert dropped == []
    assert threshold == pytest.approx(10.0 * 1e-4)


@pytest.mark.unit
def test_select_kept_preserves_paper_small_coef_at_safe_eps() -> None:
    kept, dropped, _ = select_kept_indices(
        [-9.999, 0.0199, 9.999], eps_relative=1e-3,
    )
    assert kept == [0, 1, 2], (
        f"paper-D must survive at ε=1e-3; instead kept={kept}, dropped={dropped}"
    )
    assert dropped == []


@pytest.mark.unit
def test_select_kept_eps_1e2_drops_paper_d_TOO_AGGRESSIVE() -> None:
    kept, dropped, _ = select_kept_indices(
        [-9.999, 0.0199, 9.999], eps_relative=1e-2,
    )
    assert 1 in dropped, (
        f"paper-D gets dropped at TOO-AGGRESSIVE ε=1e-2; "
        f"kept={kept}, dropped={dropped}"
    )


@pytest.mark.unit
def test_select_kept_drops_obvious_decoys() -> None:
    coefs = [10.0, 10.0, 10.0, 10.0, 1e-6, 1e-7, 1e-8, 1e-9]
    kept, dropped, threshold = select_kept_indices(coefs, eps_relative=1e-4)
    assert kept == [0, 1, 2, 3]
    assert dropped == [4, 5, 6, 7]
    assert threshold == pytest.approx(10.0 * 1e-4)


@pytest.mark.unit
def test_select_kept_threshold_boundary_uses_ge_comparison() -> None:

    kept, dropped, threshold = select_kept_indices(
        [100.0, 1.0001, 1.0, 0.9999], eps_relative=0.01,
    )
    assert kept == [0, 1, 2]
    assert dropped == [3]
    assert threshold == pytest.approx(1.0)


@pytest.mark.unit
def test_select_kept_uses_abs_value_for_negative_coefs() -> None:

    kept, dropped, _ = select_kept_indices(
        [-10.0, -3.0, 7.0, -1.0, 5.0], eps_relative=0.5,
    )
    assert kept == [0, 2, 4]
    assert dropped == [1, 3]


@pytest.mark.unit
def test_select_kept_zero_eps_no_drops() -> None:
    kept, dropped, threshold = select_kept_indices(
        [10.0, 1e-10, 1e-20], eps_relative=0.0,
    )
    assert kept == [0, 1, 2]
    assert dropped == []
    assert threshold == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("bad_eps", [-0.1, -1.0, 1.0, 1.5, float("inf"), float("nan")])
def test_select_kept_rejects_invalid_eps(bad_eps: float) -> None:
    with pytest.raises(ValueError, match="eps_relative"):
        select_kept_indices([10.0, 1.0, 0.1], eps_relative=bad_eps)







@pytest.mark.unit
def test_consolidate_signed_duplicates_merges_u_and_neg_u() -> None:
    terms = ["u", "neg(u)", "u"]
    coefs = [3.0, -2.0, 5.0]
    unique_terms, net_coefs = consolidate_signed_duplicates(terms, coefs)

    assert unique_terms == ["u"]
    assert net_coefs == pytest.approx([10.0])


@pytest.mark.unit
def test_consolidate_signed_duplicates_no_match_is_identity() -> None:
    terms = ["u", "n2(u)", "diff2_x(u)"]
    coefs = [10.0, -5.0, 0.02]
    unique_terms, net_coefs = consolidate_signed_duplicates(terms, coefs)
    assert unique_terms == terms
    assert net_coefs == pytest.approx(coefs)


@pytest.mark.unit
def test_consolidate_signed_duplicates_signed_pair_uses_positive_canonical() -> None:

    unique_terms, net_coefs = consolidate_signed_duplicates(
        ["u", "neg(u)"], [3.0, 4.0],
    )
    assert unique_terms == ["u"]

    assert net_coefs == pytest.approx([-1.0])



    unique_terms2, net_coefs2 = consolidate_signed_duplicates(
        ["neg(u)", "u"], [3.0, 4.0],
    )
    assert unique_terms2 == ["u"]

    assert net_coefs2 == pytest.approx([1.0])




    unique_terms3, net_coefs3 = consolidate_signed_duplicates(
        ["neg(t)"], [5.0],
    )
    assert unique_terms3 == ["t"]
    assert net_coefs3 == pytest.approx([-5.0])







def _make_evaluator_mock(
    initial_terms: list[str],
    initial_coefs: list[float],
    initial_nmse: float,
    refit_coefs_by_terms: dict[tuple[str, ...], tuple[list[float], float]],
    is_valid: bool = True,
) -> MagicMock:
    evaluator = MagicMock()

    initial_result = MagicMock()
    initial_result.terms = initial_terms
    initial_result.coefficients = np.asarray(initial_coefs, dtype=float)
    initial_result.nmse = initial_nmse
    initial_result.complexity = len(initial_coefs)
    initial_result.is_valid = is_valid
    evaluator.evaluate_expression.return_value = initial_result

    def _evaluate_terms(kept_terms: list[str]) -> MagicMock:
        key = tuple(kept_terms)
        if key not in refit_coefs_by_terms:
            raise AssertionError(
                f"unexpected refit kept_terms {key!r}; "
                f"mock has {list(refit_coefs_by_terms.keys())!r}",
            )
        new_coefs, new_nmse = refit_coefs_by_terms[key]
        refit_result = MagicMock()
        refit_result.terms = list(kept_terms)
        refit_result.coefficients = np.asarray(new_coefs, dtype=float)
        refit_result.nmse = new_nmse
        refit_result.complexity = len(kept_terms)
        refit_result.is_valid = True
        return refit_result

    evaluator.evaluate_terms.side_effect = _evaluate_terms
    return evaluator


@pytest.mark.unit
def test_refit_candidate_drops_decoys_via_evaluator() -> None:
    initial_terms = ["t1", "t2", "t3", "t4", "d1", "d2", "d3", "d4"]
    initial_coefs = [10.0, 10.0, 10.0, 10.0, 1e-6, 1e-7, 1e-8, 1e-9]
    refit_coefs: dict[tuple[str, ...], tuple[list[float], float]] = {
        ("t1", "t2", "t3", "t4"): ([10.001, 9.999, 10.0, 10.0], 1e-5),
    }
    evaluator = _make_evaluator_mock(
        initial_terms, initial_coefs, initial_nmse=1e-4,
        refit_coefs_by_terms=refit_coefs,
    )
    result = refit_candidate(
        expression="some_expr_string",
        evaluator=evaluator,
        eps_relative=1e-4,
    )
    assert isinstance(result, SparseRefitResult)
    assert result.n_kept == 4
    assert result.n_dropped == 4
    assert result.sparse_terms == ["t1", "t2", "t3", "t4"]
    assert result.sparse_nmse == pytest.approx(1e-5)
    np.testing.assert_allclose(
        result.sparse_coefficients,
        [10.001, 9.999, 10.0, 10.0],
        rtol=1e-9,
    )


    evaluator.evaluate_expression.assert_called_once_with("some_expr_string")
    evaluator.evaluate_terms.assert_called_once_with(["t1", "t2", "t3", "t4"])


@pytest.mark.unit
def test_refit_candidate_uses_d016_reward_formula() -> None:
    initial_terms = ["t1", "t2", "t3", "t4", "d1", "d2", "d3", "d4"]
    initial_coefs = [10.0, 10.0, 10.0, 10.0, 1e-6, 1e-7, 1e-8, 1e-9]
    refit_coefs: dict[tuple[str, ...], tuple[list[float], float]] = {
        ("t1", "t2", "t3", "t4"): ([10.0, 10.0, 10.0, 10.0], 1e-5),
    }
    evaluator = _make_evaluator_mock(
        initial_terms, initial_coefs, initial_nmse=1e-4,
        refit_coefs_by_terms=refit_coefs,
    )
    result = refit_candidate(
        expression="expr",
        evaluator=evaluator,
        eps_relative=1e-4,
        alpha=0.01,
    )


    expected_reward = (1.0 - 0.01 * 4) / (1.0 + np.sqrt(1e-5))
    assert result.sparse_reward == pytest.approx(expected_reward, rel=1e-9)


@pytest.mark.unit
def test_refit_candidate_default_alpha_matches_d016() -> None:
    initial_terms = ["t1"]
    initial_coefs = [10.0]
    evaluator = _make_evaluator_mock(
        initial_terms, initial_coefs, initial_nmse=1e-4,
        refit_coefs_by_terms={},
    )
    result = refit_candidate(
        expression="expr",
        evaluator=evaluator,
        eps_relative=0.0,
    )

    assert result.alpha == DEFAULT_ALPHA == 0.01


@pytest.mark.unit
def test_refit_candidate_zero_eps_returns_input_unchanged_no_evaluate_terms() -> None:
    initial_terms = ["t1", "t2", "t3"]
    initial_coefs = [10.0, 1e-10, 1e-20]
    evaluator = _make_evaluator_mock(
        initial_terms, initial_coefs, initial_nmse=1e-4,
        refit_coefs_by_terms={},
    )
    result = refit_candidate(
        expression="expr",
        evaluator=evaluator,
        eps_relative=0.0,
    )
    assert result.n_kept == 3
    assert result.n_dropped == 0
    assert result.sparse_terms == initial_terms
    np.testing.assert_allclose(
        result.sparse_coefficients, initial_coefs, rtol=1e-12,
    )
    evaluator.evaluate_terms.assert_not_called()


@pytest.mark.unit
def test_refit_candidate_no_drops_when_all_above_threshold() -> None:
    initial_terms = ["t1", "t2", "t3"]
    initial_coefs = [10.0, 8.0, 9.0]
    evaluator = _make_evaluator_mock(
        initial_terms, initial_coefs, initial_nmse=1e-4,
        refit_coefs_by_terms={},
    )
    result = refit_candidate(
        expression="expr",
        evaluator=evaluator,
        eps_relative=1e-4,
    )
    assert result.n_kept == 3
    assert result.n_dropped == 0
    assert result.sparse_terms == initial_terms
    evaluator.evaluate_terms.assert_not_called()


@pytest.mark.unit
def test_refit_candidate_invalid_initial_returns_invalid_passthrough() -> None:
    evaluator = _make_evaluator_mock(
        initial_terms=[], initial_coefs=[], initial_nmse=float("inf"),
        refit_coefs_by_terms={},
        is_valid=False,
    )
    result = refit_candidate(
        expression="bad_expr",
        evaluator=evaluator,
        eps_relative=1e-4,
    )
    assert result.n_kept == 0
    assert result.n_dropped == 0
    assert result.sparse_reward == 0.0
    evaluator.evaluate_terms.assert_not_called()


@pytest.mark.unit
def test_refit_candidate_records_eps_and_threshold_in_result() -> None:
    initial_terms = ["t1", "t2", "t3"]
    initial_coefs = [10.0, 10.0, 1e-10]
    refit_coefs: dict[tuple[str, ...], tuple[list[float], float]] = {
        ("t1", "t2"): ([10.0, 10.0], 1e-5)
    }
    evaluator = _make_evaluator_mock(
        initial_terms, initial_coefs, initial_nmse=1e-4,
        refit_coefs_by_terms=refit_coefs,
    )
    result = refit_candidate(
        expression="expr",
        evaluator=evaluator,
        eps_relative=1e-3,
    )
    assert result.eps_relative == 1e-3
    assert result.abs_threshold == pytest.approx(10.0 * 1e-3)


@pytest.mark.unit
def test_refit_candidate_returns_json_serializable_coefs() -> None:
    import json
    initial_terms = ["t1", "t2", "t3", "t4", "d1"]
    initial_coefs = [10.0, 10.0, 10.0, 10.0, 1e-9]
    refit_coefs: dict[tuple[str, ...], tuple[list[float], float]] = {
        ("t1", "t2", "t3", "t4"): ([10.0, 10.0, 10.0, 10.0], 1e-5)
    }
    evaluator = _make_evaluator_mock(
        initial_terms, initial_coefs, initial_nmse=1e-4,
        refit_coefs_by_terms=refit_coefs,
    )
    result = refit_candidate(
        expression="expr", evaluator=evaluator, eps_relative=1e-4,
    )

    json_str = json.dumps({
        "sparse_terms": result.sparse_terms,
        "sparse_coefficients": result.sparse_coefficients,
        "sparse_nmse": result.sparse_nmse,
        "sparse_reward": result.sparse_reward,
        "eps_relative": result.eps_relative,
        "abs_threshold": result.abs_threshold,
        "alpha": result.alpha,
        "n_kept": result.n_kept,
        "n_dropped": result.n_dropped,
    })

    for c in result.sparse_coefficients:
        assert type(c) is float, (
            f"sparse_coefficients element has type {type(c)}, expected float"
        )
    assert "10.0" in json_str


@pytest.mark.unit
def test_refit_candidate_consolidates_signed_dupes_then_drops_decoys() -> None:
    initial_terms = [
        "n2(u)",
        "neg(t)",
        "mul(diff2_x(u), u)",
        "n2(add(diff2_x(u), add(t, x)))",
        "neg(n2(x))",
        "n2(diff_x(u))",
        "neg(u)",
        "u",
        "neg(u)",
    ]
    initial_coefs = [
        -10.00,
        -1.36e-4,
        0.0198,
        1.36e-6,
        -5.47e-4,
        0.0199,
        -3.33,
        3.33,
        -3.33,
    ]













    refit_coefs: dict[tuple[str, ...], tuple[list[float], float]] = {
        ("n2(u)", "mul(diff2_x(u), u)", "n2(diff_x(u))", "u"): (
            [-10.0, 0.02, 0.02, -3.33], 1.45e-5,
        ),
    }
    evaluator = _make_evaluator_mock(
        initial_terms, initial_coefs, initial_nmse=1.45e-5,
        refit_coefs_by_terms=refit_coefs,
    )
    result = refit_candidate(
        expression="fisher_nonlinear_seed123_hof2_expr",
        evaluator=evaluator,
        eps_relative=1e-3,
    )

    assert result.n_kept == 4, (
        f"expected 4-of-4 GT after consolidation+threshold; got {result.n_kept}"
    )



    assert result.n_dropped == 5, (
        f"expected 5 total removed (2 consolidation + 3 threshold); "
        f"got {result.n_dropped}"
    )
    assert set(result.sparse_terms) == {
        "n2(u)", "mul(diff2_x(u), u)", "n2(diff_x(u))", "u",
    }

    expected_reward = (1.0 - 0.01 * 4) / (1.0 + np.sqrt(1.45e-5))
    assert result.sparse_reward == pytest.approx(expected_reward, rel=1e-3)
