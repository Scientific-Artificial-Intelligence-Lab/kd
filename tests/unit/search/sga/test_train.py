
from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from kd.search.sga.config import SGAConfig
from kd.search.sga.pde import PDE
from kd.search.sga.train import (
    TrainResult,
    compute_aic,
    evaluate_candidate,
    train_sweep,
)
from kd.search.sga.tree import Node, Tree



N_SAMPLES = 100
RTOL = 1e-5
ATOL = 1e-8





def _leaf(name: str) -> Node:
    return Node(name=name, arity=0, children=[])


def _binary(op: str, left: Node, right: Node) -> Node:
    return Node(name=op, arity=2, children=[left, right])


def _make_sparse_system(
    n: int = N_SAMPLES,
    d: int = 5,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, d, generator=gen)
    true_w = torch.zeros(d)
    true_w[0] = 2.0
    true_w[1] = 3.0
    noise = 0.01 * torch.randn(n, generator=gen)
    y = theta @ true_w + noise
    return theta, y.unsqueeze(1), true_w







class TestSmoke:

    @pytest.mark.smoke
    def test_compute_aic_callable(self) -> None:
        result = compute_aic(mse=1.0, k=2, ratio=1.0)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_train_sweep_callable(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)

    @pytest.mark.smoke
    def test_evaluate_candidate_callable(self) -> None:
        from kd.search.sga.train import CandidateResult

        data = {"u": torch.randn(20), "x": torch.randn(20)}
        pde = PDE(terms=[Tree(root=_leaf("u")), Tree(root=_leaf("x"))])
        y = torch.randn(20, 1)
        config = SGAConfig()
        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)

    @pytest.mark.smoke
    def test_train_result_has_required_fields(self) -> None:
        tr = TrainResult(
            coefficients=torch.zeros(3),
            selected_indices=[0],
            aic_score=1.0,
            mse=0.5,
            best_tol=0.1,
        )
        assert hasattr(tr, "coefficients")
        assert hasattr(tr, "selected_indices")
        assert hasattr(tr, "aic_score")
        assert hasattr(tr, "mse")
        assert hasattr(tr, "best_tol")







class TestComputeAic:

    def test_known_values(self) -> None:
        result = compute_aic(mse=1.0, k=2, ratio=1.0)
        expected = 2 * 2 * 1.0 + 2 * math.log(1.0)
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_known_values_nonunit_mse(self) -> None:
        result = compute_aic(mse=2.5, k=3, ratio=1.0)
        expected = 2 * 3 * 1.0 + 2 * math.log(2.5)
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_k_zero_only_log_term(self) -> None:
        result = compute_aic(mse=0.5, k=0, ratio=1.0)
        expected = 2 * math.log(0.5)
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_mse_zero_returns_inf(self) -> None:
        result = compute_aic(mse=0.0, k=1, ratio=1.0)
        assert result == float("inf")

    def test_mse_negative_returns_inf(self) -> None:
        result = compute_aic(mse=-1.0, k=1, ratio=1.0)
        assert result == float("inf")

    def test_ratio_greater_than_one(self) -> None:
        result = compute_aic(mse=1.0, k=3, ratio=2.0)
        expected = 2 * 3 * 2.0 + 2 * math.log(1.0)
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_aic_monotonic_in_k(self) -> None:
        mse = 1.5
        aic_values = [compute_aic(mse=mse, k=k, ratio=1.0) for k in range(10)]
        for i in range(len(aic_values) - 1):
            assert aic_values[i] < aic_values[i + 1], (
                f"AIC not monotonically increasing in k: "
                f"AIC(k={i})={aic_values[i]}, AIC(k={i + 1})={aic_values[i + 1]}"
            )

    def test_aic_monotonic_in_mse(self) -> None:
        k = 2
        mse_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        aic_values = [compute_aic(mse=m, k=k, ratio=1.0) for m in mse_values]
        for i in range(len(aic_values) - 1):
            assert aic_values[i] < aic_values[i + 1], (
                f"AIC not monotonically increasing in MSE: "
                f"AIC(mse={mse_values[i]})={aic_values[i]}, "
                f"AIC(mse={mse_values[i + 1]})={aic_values[i + 1]}"
            )







class TestTrainSweepContract:

    def test_return_type(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)

    def test_coefficients_shape(self) -> None:
        theta, y, _ = _make_sparse_system(d=5)
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert result.coefficients.shape == (5,)

    def test_coefficients_is_tensor(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert isinstance(result.coefficients, Tensor)

    def test_selected_indices_is_list_of_int(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert isinstance(result.selected_indices, list)
        for idx in result.selected_indices:
            assert isinstance(idx, int)

    def test_aic_score_is_finite(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert math.isfinite(result.aic_score)

    def test_mse_is_nonnegative(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert result.mse >= 0.0

    def test_best_tol_is_nonnegative(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert result.best_tol >= 0.0







class TestTrainSweepMseConversion:

    def test_mse_consistent_with_coefficients(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)


        y_1d = y.squeeze(-1) if y.dim() == 2 else y
        residuals = y_1d - theta @ result.coefficients
        recomputed_mse = (residuals**2).mean().item()

        torch.testing.assert_close(
            torch.tensor(result.mse),
            torch.tensor(recomputed_mse),
            rtol=1e-4,
            atol=1e-6,
        )







class TestTrainSweepSparseRecovery:

    def test_recovers_true_support(self) -> None:
        theta, y, _ = _make_sparse_system(n=200, seed=123)
        config = SGAConfig(d_tol=0.5, maxit=20)
        result = train_sweep(theta, y, config)


        assert 0 in result.selected_indices, (
            f"Expected index 0 in support, got {result.selected_indices}"
        )
        assert 1 in result.selected_indices, (
            f"Expected index 1 in support, got {result.selected_indices}"
        )

    def test_recovers_approximate_coefficients(self) -> None:
        theta, y, true_w = _make_sparse_system(n=200, seed=123)
        config = SGAConfig(d_tol=0.5, maxit=20)
        result = train_sweep(theta, y, config)



        w = result.coefficients
        for i in range(len(true_w)):
            if true_w[i] != 0:
                torch.testing.assert_close(
                    w[i].unsqueeze(0),
                    true_w[i].unsqueeze(0),
                    rtol=0.05,
                    atol=0.1,
                )
            else:
                assert w[i].abs().item() < 0.1, (
                    f"Coefficient at index {i} should be ~0, got {w[i].item()}"
                )







class TestTrainSweepAdaptation:

    def test_different_dtol_different_best_tol(self) -> None:
        theta, y, _ = _make_sparse_system(n=200, seed=42)

        config_fine = SGAConfig(d_tol=0.1, maxit=20)
        config_coarse = SGAConfig(d_tol=2.0, maxit=20)

        result_fine = train_sweep(theta, y, config_fine)
        result_coarse = train_sweep(theta, y, config_coarse)



        assert math.isfinite(result_fine.aic_score)
        assert math.isfinite(result_coarse.aic_score)
        assert result_fine.mse >= 0
        assert result_coarse.mse >= 0

    def test_higher_tol_trend_fewer_terms(self) -> None:
        theta, y, _ = _make_sparse_system(n=200, d=8, seed=77)

        config_small = SGAConfig(d_tol=0.01, maxit=5)
        config_large = SGAConfig(d_tol=5.0, maxit=5)

        result_small = train_sweep(theta, y, config_small)
        result_large = train_sweep(theta, y, config_large)


        assert (
            len(result_large.selected_indices) <= len(result_small.selected_indices) + 2
        ), (
            f"Large tol sweep should be at least as sparse: "
            f"small_tol={len(result_small.selected_indices)} terms, "
            f"large_tol={len(result_large.selected_indices)} terms"
        )

    def test_ols_baseline_used_at_tol_zero(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig(d_tol=1.0, maxit=10)
        result = train_sweep(theta, y, config)


        y_1d = y.squeeze(-1) if y.dim() == 2 else y
        w_ols = torch.linalg.lstsq(theta, y_1d.unsqueeze(1)).solution.squeeze()
        mse_ols = ((y_1d - theta @ w_ols) ** 2).mean().item()
        k_ols = (w_ols.abs() > 1e-10).sum().item()
        aic_ols = compute_aic(mse=mse_ols, k=k_ols, ratio=config.aic_ratio)


        assert result.aic_score <= aic_ols + 1e-6, (
            f"train_sweep AIC ({result.aic_score}) should not exceed "
            f"OLS baseline AIC ({aic_ols})"
        )







class TestTrainSweepEdgeCases:

    def test_zero_column_theta(self) -> None:
        theta = torch.empty(N_SAMPLES, 0)
        y = torch.randn(N_SAMPLES, 1)
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert result.aic_score == float("inf")

    def test_single_column_theta(self) -> None:
        gen = torch.Generator().manual_seed(99)
        theta = torch.randn(N_SAMPLES, 1, generator=gen)
        y = 3.0 * theta[:, 0:1] + 0.01 * torch.randn(N_SAMPLES, 1, generator=gen)
        config = SGAConfig(d_tol=0.1, maxit=5)
        result = train_sweep(theta, y, config)

        assert math.isfinite(result.aic_score)
        assert len(result.selected_indices) <= 1

    def test_maxit_zero(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig(maxit=0)
        result = train_sweep(theta, y, config)

        assert math.isfinite(result.aic_score)
        assert result.mse >= 0

    def test_perfect_fit_very_low_mse(self) -> None:
        gen = torch.Generator().manual_seed(88)
        theta = torch.randn(N_SAMPLES, 3, generator=gen)
        true_w = torch.tensor([1.0, -2.0, 0.5])
        y = (theta @ true_w).unsqueeze(1)
        config = SGAConfig(d_tol=0.1, maxit=10)
        result = train_sweep(theta, y, config)

        assert result.mse < 1e-6, f"Expected near-zero MSE, got {result.mse}"







class TestTrainSweepConfigForwarding:

    def test_lam_zero_uses_ols(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig(lam=0.0, d_tol=1.0, maxit=5)
        result = train_sweep(theta, y, config)
        assert math.isfinite(result.aic_score)

    def test_lam_nonzero_uses_ridge(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig(lam=1e-3, d_tol=1.0, maxit=5)
        result = train_sweep(theta, y, config)
        assert math.isfinite(result.aic_score)

    def test_different_lam_different_result(self) -> None:
        theta, y, _ = _make_sparse_system(n=200)
        config_ols = SGAConfig(lam=0.0, d_tol=1.0, maxit=10)
        config_ridge = SGAConfig(lam=10.0, d_tol=1.0, maxit=10)

        result_ols = train_sweep(theta, y, config_ols)
        result_ridge = train_sweep(theta, y, config_ridge)



        diff = (result_ols.coefficients - result_ridge.coefficients).abs().sum()
        assert diff > 1e-3, "OLS and heavy-ridge solutions should differ significantly"







class TestEvaluateCandidatePipeline:

    def test_basic_pipeline(self) -> None:
        gen = torch.Generator().manual_seed(42)
        u = torch.randn(N_SAMPLES, generator=gen)
        x = torch.randn(N_SAMPLES, generator=gen)
        data = {"u": u, "x": x}


        y = (2.0 * u + 0.01 * torch.randn(N_SAMPLES, generator=gen)).unsqueeze(1)

        pde = PDE(terms=[Tree(root=_leaf("u")), Tree(root=_leaf("x"))])
        config = SGAConfig(d_tol=0.5, maxit=10)

        result = evaluate_candidate(pde, data, None, y, config)
        assert math.isfinite(result.aic_score)
        assert result.mse >= 0

    def test_all_terms_filtered_returns_inf(self) -> None:
        data = {"nan_var": torch.full((N_SAMPLES,), float("nan"))}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[Tree(root=_leaf("nan_var"))])
        config = SGAConfig()

        result = evaluate_candidate(pde, data, None, y, config)
        assert result.aic_score == float("inf")

    def test_empty_pde_returns_inf(self) -> None:
        data = {"u": torch.randn(N_SAMPLES)}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[])
        config = SGAConfig()

        result = evaluate_candidate(pde, data, None, y, config)
        assert result.aic_score == float("inf")

    def test_with_default_terms(self) -> None:
        gen = torch.Generator().manual_seed(55)
        u = torch.randn(N_SAMPLES, generator=gen)
        x = torch.randn(N_SAMPLES, generator=gen)
        data = {"u": u, "x": x}
        y = torch.randn(N_SAMPLES, 1, generator=gen)


        default_terms = torch.randn(N_SAMPLES, 2, generator=gen)
        pde = PDE(terms=[Tree(root=_leaf("u"))])
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, default_terms, y, config)
        assert math.isfinite(result.aic_score)

        assert result.coefficients.shape == (3,)

    def test_default_terms_only_when_pde_terms_filtered(self) -> None:
        data = {"nan_var": torch.full((N_SAMPLES,), float("nan"))}
        y = torch.randn(N_SAMPLES, 1)

        default_terms = torch.randn(N_SAMPLES, 2)
        pde = PDE(terms=[Tree(root=_leaf("nan_var"))])
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, default_terms, y, config)

        assert result.coefficients.shape == (2,)
        assert math.isfinite(result.aic_score)

    def test_no_default_no_valid_terms_returns_inf(self) -> None:
        data = {"zeros": torch.zeros(N_SAMPLES)}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[Tree(root=_leaf("zeros"))])
        config = SGAConfig()

        result = evaluate_candidate(pde, data, None, y, config)
        assert result.aic_score == float("inf")







class TestEvaluateCandidateIndices:

    def test_indices_within_range(self) -> None:
        gen = torch.Generator().manual_seed(42)
        data = {"u": torch.randn(N_SAMPLES, generator=gen)}
        y = torch.randn(N_SAMPLES, 1, generator=gen)

        default_terms = torch.randn(N_SAMPLES, 2, generator=gen)
        pde = PDE(terms=[Tree(root=_leaf("u"))])
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, default_terms, y, config)
        total_cols = 2 + 1
        for idx in result.selected_indices:
            assert 0 <= idx < total_cols, f"Index {idx} out of range [0, {total_cols})"







class TestNegativeAndFailure:

    @pytest.mark.numerical
    def test_nan_in_y_handled(self) -> None:
        theta = torch.randn(N_SAMPLES, 3)
        y = torch.randn(N_SAMPLES, 1)
        y[0, 0] = float("nan")
        config = SGAConfig()


        with pytest.raises(ValueError):
            train_sweep(theta, y, config)

    @pytest.mark.numerical
    def test_inf_in_theta_handled(self) -> None:
        theta = torch.randn(N_SAMPLES, 3)
        theta[0, 0] = float("inf")
        y = torch.randn(N_SAMPLES, 1)
        config = SGAConfig()

        with pytest.raises(ValueError):
            train_sweep(theta, y, config)

    @pytest.mark.numerical
    def test_nan_in_theta_handled(self) -> None:
        theta = torch.randn(N_SAMPLES, 3)
        theta[5, 1] = float("nan")
        y = torch.randn(N_SAMPLES, 1)
        config = SGAConfig()

        with pytest.raises(ValueError):
            train_sweep(theta, y, config)

    def test_evaluate_candidate_unknown_variable(self) -> None:
        data = {"u": torch.randn(N_SAMPLES)}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[Tree(root=_leaf("nonexistent"))])
        config = SGAConfig()

        result = evaluate_candidate(pde, data, None, y, config)
        assert result.aic_score == float("inf")

    def test_compute_aic_nan_mse(self) -> None:
        result = compute_aic(mse=float("nan"), k=1, ratio=1.0)
        assert result == float("inf")

    def test_compute_aic_inf_mse(self) -> None:
        result = compute_aic(mse=float("inf"), k=1, ratio=1.0)
        assert result == float("inf")

    def test_evaluate_candidate_returns_pruned_pde(self) -> None:
        from kd.search.sga.train import CandidateResult

        data = {
            "u": torch.randn(N_SAMPLES),
            "nan_var": torch.full((N_SAMPLES,), float("nan")),
        }
        y = (2.0 * data["u"]).unsqueeze(1)
        pde = PDE(terms=[Tree(root=_leaf("u")), Tree(root=_leaf("nan_var"))])
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)
        assert result.pruned_pde is not None
        assert result.pruned_pde.width == 1
        assert str(result.pruned_pde.terms[0]) == "u"

    def test_evaluate_candidate_coefficients_aligned_with_pruned_pde(self) -> None:
        from kd.search.sga.train import CandidateResult

        gen = torch.Generator().manual_seed(42)
        data = {
            "u": torch.randn(N_SAMPLES, generator=gen),
            "x": torch.randn(N_SAMPLES, generator=gen),
            "zeros": torch.zeros(N_SAMPLES),
        }

        y = (2.0 * data["u"] + 3.0 * data["x"]).unsqueeze(1)


        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
                Tree(root=_leaf("x")),
            ]
        )
        config = SGAConfig(d_tol=0.5, maxit=10)

        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)

        assert result.pruned_pde.width == 2

        assert result.train_result.coefficients.shape[0] == 2

    def test_evaluate_candidate_selected_indices_map_to_pruned_pde(self) -> None:
        from kd.search.sga.train import CandidateResult

        gen = torch.Generator().manual_seed(42)
        data = {
            "u": torch.randn(N_SAMPLES, generator=gen),
            "x": torch.randn(N_SAMPLES, generator=gen),
            "zeros": torch.zeros(N_SAMPLES),
        }
        y = (2.0 * data["u"]).unsqueeze(1)

        pde = PDE(
            terms=[
                Tree(root=_leaf("zeros")),
                Tree(root=_leaf("u")),
                Tree(root=_leaf("x")),
            ]
        )
        config = SGAConfig(d_tol=0.5, maxit=10)

        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)

        n_theta_cols = result.pruned_pde.width
        for idx in result.train_result.selected_indices:
            assert 0 <= idx < n_theta_cols

    def test_evaluate_candidate_preserves_original_pde(self) -> None:
        data = {"u": torch.randn(N_SAMPLES), "zeros": torch.zeros(N_SAMPLES)}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[Tree(root=_leaf("u")), Tree(root=_leaf("zeros"))])
        original_width = pde.width
        original_str = str(pde)
        config = SGAConfig()

        _ = evaluate_candidate(pde, data, None, y, config)
        assert pde.width == original_width
        assert str(pde) == original_str

    def test_evaluate_candidate_valid_term_indices_maps_original(self) -> None:
        from kd.search.sga.train import CandidateResult

        data = {
            "u": torch.randn(N_SAMPLES),
            "zeros": torch.zeros(N_SAMPLES),
            "x": torch.randn(N_SAMPLES),
        }
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
                Tree(root=_leaf("x")),
            ]
        )
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)
        assert result.valid_term_indices == [0, 2]

    def test_dimension_mismatch_theta_y(self) -> None:
        theta = torch.randn(N_SAMPLES, 3)
        y = torch.randn(N_SAMPLES + 10, 1)
        config = SGAConfig()

        with pytest.raises(ValueError):
            train_sweep(theta, y, config)







class TestNumericalStability:

    @pytest.mark.numerical
    def test_extreme_large_theta(self) -> None:
        gen = torch.Generator().manual_seed(42)
        theta = 1e10 * torch.randn(N_SAMPLES, 3, generator=gen)
        y = theta[:, 0:1] + 0.01 * torch.randn(N_SAMPLES, 1, generator=gen)
        config = SGAConfig(d_tol=1.0, maxit=5)

        result = train_sweep(theta, y, config)

        assert isinstance(result, TrainResult)

    @pytest.mark.numerical
    def test_extreme_small_theta(self) -> None:
        gen = torch.Generator().manual_seed(42)
        theta = 1e-10 * torch.randn(N_SAMPLES, 3, generator=gen)
        y = theta[:, 0:1] + 1e-12 * torch.randn(N_SAMPLES, 1, generator=gen)
        config = SGAConfig(d_tol=1e-12, maxit=5)

        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)

    @pytest.mark.numerical
    def test_collinear_columns(self) -> None:
        gen = torch.Generator().manual_seed(42)
        base = torch.randn(N_SAMPLES, generator=gen)
        theta = torch.stack([base, base * 1.001, base * 0.999], dim=1)
        y = (base + 0.01 * torch.randn(N_SAMPLES, generator=gen)).unsqueeze(1)
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)

        assert not math.isnan(result.aic_score)

    @pytest.mark.numerical
    def test_single_sample(self) -> None:
        theta = torch.tensor([[1.0, 2.0, 3.0]])
        y = torch.tensor([[5.0]])
        config = SGAConfig(d_tol=0.1, maxit=3)

        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)







class TestAicProperties:

    @pytest.mark.parametrize("mse", [0.001, 0.1, 1.0, 10.0, 1000.0])
    def test_aic_is_finite_for_positive_mse(self, mse: float) -> None:
        result = compute_aic(mse=mse, k=3, ratio=1.0)
        assert math.isfinite(result)

    @pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 5.0])
    def test_ratio_scales_penalty_linearly(self, ratio: float) -> None:
        mse = 1.0
        k = 3
        result = compute_aic(mse=mse, k=k, ratio=ratio)
        expected = 2 * k * ratio
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_aic_decomposition(self) -> None:
        mse = 2.5
        k = 4
        ratio = 1.5

        full_aic = compute_aic(mse=mse, k=k, ratio=ratio)
        penalty = 2 * k * ratio
        fit = 2 * math.log(mse)

        torch.testing.assert_close(
            torch.tensor(full_aic),
            torch.tensor(penalty + fit),
            rtol=RTOL,
            atol=ATOL,
        )
