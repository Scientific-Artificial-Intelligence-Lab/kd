
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kd.search.discover.runners.multiseed import (
    aggregate_metrics,
    compute_combined_pass_rate,
    compute_pass_rate,
    compute_release_decision,
    compute_structural_pass_rate,
    extension_seeds,
    should_extend_v11,
    time_slice_indices,
)






def _seed_result(max_rel: float, l1_ratio: float = 0.0) -> dict[str, Any]:
    return {
        "ground_truth_fit": {
            "max_rel_coef_error": max_rel,
            "l1_ratio_error": l1_ratio,
        }
    }


class TestComputePassRate:
    def test_all_pass(self) -> None:
        results = [_seed_result(0.001), _seed_result(0.002), _seed_result(0.003)]
        assert compute_pass_rate(results, threshold=0.05) == (3, 3)

    def test_some_pass(self) -> None:
        results = [_seed_result(0.001), _seed_result(0.10), _seed_result(0.003)]
        assert compute_pass_rate(results, threshold=0.05) == (2, 3)

    def test_none_pass(self) -> None:
        results = [_seed_result(0.10), _seed_result(0.20)]
        assert compute_pass_rate(results, threshold=0.05) == (0, 2)

    def test_threshold_inclusive(self) -> None:
        results = [_seed_result(0.05)]
        assert compute_pass_rate(results, threshold=0.05) == (1, 1)

    def test_empty_results(self) -> None:
        assert compute_pass_rate([], threshold=0.05) == (0, 0)


class TestShouldExtendV11:
    def test_one_of_three_triggers(self) -> None:
        assert should_extend_v11(n_pass=1, n_total=3) is True

    def test_zero_of_three_no_extend(self) -> None:
        assert should_extend_v11(n_pass=0, n_total=3) is False

    def test_two_of_three_no_extend(self) -> None:
        assert should_extend_v11(n_pass=2, n_total=3) is False

    def test_three_of_three_no_extend(self) -> None:
        assert should_extend_v11(n_pass=3, n_total=3) is False

    def test_one_of_five_no_extend(self) -> None:

        assert should_extend_v11(n_pass=1, n_total=5) is False

    def test_one_of_four_no_extend(self) -> None:
        assert should_extend_v11(n_pass=1, n_total=4) is False


class TestExtensionSeeds:
    def test_default_extends_to_two_more(self) -> None:
        assert extension_seeds([0, 1, 2]) == [3, 4]

    def test_arbitrary_starting_point(self) -> None:
        assert extension_seeds([10, 11, 12]) == [13, 14]

    def test_non_contiguous_uses_max(self) -> None:

        assert extension_seeds([0, 5, 10]) == [11, 12]

    def test_empty_initial_raises(self) -> None:
        with pytest.raises(ValueError):
            extension_seeds([])


class TestTimeSliceIndices:

    def test_all_returns_full_range(self) -> None:
        idx = time_slice_indices("all", nx=4, ny=5, nt=6)
        assert tuple(idx.shape) == (4 * 5 * 6,)
        assert idx.tolist() == list(range(4 * 5 * 6))

    def test_middle_3_picks_t_indices_2_3_4(self) -> None:
        idx = time_slice_indices("middle-3", nx=4, ny=5, nt=100)

        assert tuple(idx.shape) == (60,)

        t_vals = set((idx % 100).tolist())
        assert t_vals == {2, 3, 4}

    def test_first_3_picks_t_0_1_2(self) -> None:
        idx = time_slice_indices("first-3", nx=4, ny=5, nt=100)
        t_vals = set((idx % 100).tolist())
        assert t_vals == {0, 1, 2}

    def test_interior_3_is_linspace_margin_2(self) -> None:

        idx = time_slice_indices("interior-3", nx=2, ny=2, nt=100)
        t_vals = set((idx % 100).tolist())
        assert t_vals == {2, 49, 97}

    def test_mid_3_picks_t_48_49_50(self) -> None:
        idx = time_slice_indices("mid-3", nx=2, ny=2, nt=100)
        t_vals = set((idx % 100).tolist())
        assert t_vals == {48, 49, 50}

    def test_late_3_picks_t_96_97_98(self) -> None:
        idx = time_slice_indices("late-3", nx=2, ny=2, nt=100)
        t_vals = set((idx % 100).tolist())
        assert t_vals == {96, 97, 98}

    def test_spread_3_picks_t_10_50_90(self) -> None:
        idx = time_slice_indices("spread-3", nx=2, ny=2, nt=100)
        t_vals = set((idx % 100).tolist())
        assert t_vals == {10, 50, 90}

    def test_unknown_slice_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown time_slice"):
            time_slice_indices("not-a-real-slice", nx=4, ny=5, nt=6)

    def test_flat_axis_order_x_Ny_Nt(self) -> None:
        idx = time_slice_indices("first-3", nx=2, ny=3, nt=4).tolist()







        expected = sorted(
            [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
        )
        assert sorted(idx) == expected

    def test_returns_sorted_int64(self) -> None:
        idx = time_slice_indices("middle-3", nx=3, ny=3, nt=100)
        import torch

        assert idx.dtype == torch.int64
        assert idx.dim() == 1

        diffs = idx[1:] - idx[:-1]
        assert (diffs > 0).all()

    def test_insufficient_nt_for_interior_3_raises(self) -> None:

        with pytest.raises(ValueError):
            time_slice_indices("interior-3", nx=2, ny=2, nt=4)


class TestAggregateMetrics:
    def test_basic_three_seeds(self) -> None:
        results = [
            _seed_result(0.01, 0.005),
            _seed_result(0.02, 0.010),
            _seed_result(0.03, 0.015),
        ]
        agg = aggregate_metrics(results)
        assert agg["max_rel"]["mean"] == pytest.approx(0.02)
        assert agg["max_rel"]["min"] == pytest.approx(0.01)
        assert agg["max_rel"]["max"] == pytest.approx(0.03)

        assert agg["max_rel"]["std"] == pytest.approx(0.01, rel=1e-3)
        assert agg["l1_ratio"]["mean"] == pytest.approx(0.010)
        assert agg["l1_ratio"]["std"] == pytest.approx(0.005, rel=1e-3)

    def test_single_seed_zero_std(self) -> None:
        results = [_seed_result(0.05, 0.03)]
        agg = aggregate_metrics(results)
        assert agg["max_rel"]["mean"] == pytest.approx(0.05)
        assert agg["max_rel"]["std"] == 0.0
        assert agg["l1_ratio"]["mean"] == pytest.approx(0.03)
        assert agg["l1_ratio"]["std"] == 0.0

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError):
            aggregate_metrics([])







def _seed_with_expr(expr: str, max_rel: float = 0.001) -> dict[str, Any]:
    return {
        "ground_truth_fit": {
            "max_rel_coef_error": max_rel,
            "l1_ratio_error": 0.0,
        },
        "mode1_run": {"best_expression": expr},
    }


class TestComputeStructuralPassRate:

    def test_correct_form_passes(self) -> None:
        results = [
            _seed_with_expr("add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))"),
        ]
        assert compute_structural_pass_rate(results) == (1, 1)

    def test_v10_paper_ic_cheating_form_fails(self) -> None:
        results = [
            _seed_with_expr(
                "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))"
            ),
        ]
        assert compute_structural_pass_rate(results) == (0, 1)

    def test_mixed(self) -> None:
        results = [
            _seed_with_expr("add(diff2_x(u), diff2_y(u))"),
            _seed_with_expr("sub(diff2_x(u), diff2_y(u))"),
            _seed_with_expr("add(diff2_y(u), diff2_x(u))"),
        ]
        assert compute_structural_pass_rate(results) == (2, 3)

    def test_empty_returns_zero(self) -> None:
        assert compute_structural_pass_rate([]) == (0, 0)

    def test_missing_best_expression_counts_as_fail(self) -> None:
        results: list[dict[str, Any]] = [{"mode1_run": {}, "ground_truth_fit": {}}]
        assert compute_structural_pass_rate(results) == (0, 1)

    def test_unparseable_best_expression_counts_as_fail(self) -> None:
        results = [_seed_with_expr("not_a_known_op(u)")]
        assert compute_structural_pass_rate(results) == (0, 1)


class TestComputeCombinedPassRate:

    def test_both_gates_pass(self) -> None:
        results = [
            _seed_with_expr(
                "add(diff2_x(u), diff2_y(u))",
                max_rel=0.001,
            )
        ]
        assert compute_combined_pass_rate(results, threshold=0.05) == (1, 1)

    def test_coef_pass_structural_fail(self) -> None:
        results = [
            _seed_with_expr(
                "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
                max_rel=0.003,
            )
        ]
        assert compute_combined_pass_rate(results, threshold=0.05) == (0, 1)

    def test_coef_fail_structural_pass(self) -> None:
        results = [
            _seed_with_expr(
                "add(diff2_x(u), diff2_y(u))", max_rel=0.50,
            )
        ]
        assert compute_combined_pass_rate(results, threshold=0.05) == (0, 1)

    def test_threshold_inclusive(self) -> None:
        results = [
            _seed_with_expr("add(diff2_x(u), diff2_y(u))", max_rel=0.05)
        ]
        assert compute_combined_pass_rate(results, threshold=0.05) == (1, 1)

    def test_mixed_three_seeds(self) -> None:
        results = [
            _seed_with_expr(
                "add(diff2_x(u), diff2_y(u))", max_rel=0.001
            ),
            _seed_with_expr(
                "sub(diff2_x(u), diff2_y(u))", max_rel=0.001
            ),
            _seed_with_expr(
                "add(diff2_x(u), diff2_y(u))", max_rel=0.50
            ),
        ]
        assert compute_combined_pass_rate(results, threshold=0.05) == (1, 3)

    def test_empty_returns_zero(self) -> None:
        assert compute_combined_pass_rate([], threshold=0.05) == (0, 0)


class TestComputeReleaseDecision:

    def test_strict_default_uses_combined_gate(self) -> None:

        results = [
            _seed_with_expr(
                "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
                max_rel=0.003,
            )
        ]
        passed, n_pass, n_total = compute_release_decision(
            results, threshold=0.05, min_pass_rate=1.0
        )
        assert passed is False
        assert (n_pass, n_total) == (0, 1)

    def test_strict_full_pass(self) -> None:
        results = [
            _seed_with_expr("add(diff2_x(u), diff2_y(u))", max_rel=0.001)
        ]
        passed, n_pass, n_total = compute_release_decision(
            results, threshold=0.05, min_pass_rate=1.0
        )
        assert passed is True
        assert (n_pass, n_total) == (1, 1)

    def test_legacy_mode_ignores_structural_gate(self) -> None:

        results = [
            _seed_with_expr(
                "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
                max_rel=0.003,
            )
        ]
        passed, n_pass, n_total = compute_release_decision(
            results, threshold=0.05, min_pass_rate=1.0, strict=False
        )
        assert passed is True
        assert (n_pass, n_total) == (1, 1)

    def test_min_pass_rate_partial(self) -> None:

        results = [
            _seed_with_expr("add(diff2_x(u), diff2_y(u))", max_rel=0.001),
            _seed_with_expr("add(diff2_x(u), diff2_y(u))", max_rel=0.001),
            _seed_with_expr("add(diff2_x(u), diff2_y(u))", max_rel=0.001),
            _seed_with_expr(
                "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
                max_rel=0.001,
            ),
            _seed_with_expr(
                "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
                max_rel=0.001,
            ),
        ]
        passed_strict_high, *_ = compute_release_decision(
            results, threshold=0.05, min_pass_rate=2 / 3
        )
        assert passed_strict_high is False
        passed_strict_low, *_ = compute_release_decision(
            results, threshold=0.05, min_pass_rate=0.5
        )
        assert passed_strict_low is True

    def test_min_pass_rate_inclusive(self) -> None:

        results = [
            _seed_with_expr("add(diff2_x(u), diff2_y(u))", max_rel=0.001),
            _seed_with_expr(
                "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
                max_rel=0.001,
            ),
        ]
        passed, *_ = compute_release_decision(
            results, threshold=0.05, min_pass_rate=0.5
        )
        assert passed is True

    def test_empty_results_fail(self) -> None:
        passed, n_pass, n_total = compute_release_decision(
            [], threshold=0.05, min_pass_rate=1.0
        )
        assert passed is False
        assert (n_pass, n_total) == (0, 0)






_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_PATH = _PROJECT_ROOT / "data" / "allen_cahn_2d_paper.npz"
_MIN_DATA_BYTES = 50_000


def _data_skip_reason() -> str | None:
    if not _DATA_PATH.exists():
        return f"Allen-Cahn paper data not found at {_DATA_PATH}"
    if _DATA_PATH.stat().st_size < _MIN_DATA_BYTES:
        return f"Allen-Cahn paper data too small at {_DATA_PATH}"
    return None


@pytest.mark.skipif(
    _data_skip_reason() is not None, reason=_data_skip_reason() or ""
)
class TestRunSingleSeedSmoke:

    def test_returns_v10_compatible_schema(self) -> None:
        from kd.search.discover.runners.multiseed import run_single_seed

        result = run_single_seed(
            seed=0,
            n_iterations=3,
            n_points=500,
            batch_size=50,
            data_path=_DATA_PATH,
        )
        assert result["seed"] == 0
        assert "ground_truth_fit" in result
        gtf = result["ground_truth_fit"]
        assert "max_rel_coef_error" in gtf
        assert "l1_ratio_error" in gtf
        assert "per_term_rel_error" in gtf

        assert set(gtf["per_term_rel_error"].keys()) == {
            "diff2_x",
            "diff2_y",
            "u",
            "n3_u",
        }
        assert "mode1_run" in result
        m1 = result["mode1_run"]
        assert "best_reward" in m1
        assert "best_expression" in m1

        assert isinstance(m1["best_reward"], float)
        assert isinstance(m1["best_expression"], str)

    def test_different_seeds_produce_different_search_paths(self) -> None:
        from kd.search.discover.runners.multiseed import run_single_seed

        result_a = run_single_seed(
            seed=0,
            n_iterations=3,
            n_points=500,
            batch_size=50,
            data_path=_DATA_PATH,
        )
        result_b = run_single_seed(
            seed=1,
            n_iterations=3,
            n_points=500,
            batch_size=50,
            data_path=_DATA_PATH,
        )



        assert result_a["mode1_run"]["best_expression"] != ""
        assert result_b["mode1_run"]["best_expression"] != ""
        assert result_a["mode1_run"]["best_reward"] > 0
        assert result_b["mode1_run"]["best_reward"] > 0




        assert (
            result_a["mode1_run"]["best_expression"]
            != result_b["mode1_run"]["best_expression"]
            or result_a["mode1_run"]["best_reward"]
            != result_b["mode1_run"]["best_reward"]
        )
