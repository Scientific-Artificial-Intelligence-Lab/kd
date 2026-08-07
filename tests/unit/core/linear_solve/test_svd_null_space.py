
from __future__ import annotations

import pytest
import torch

from kd.core.linear_solve import SolveResult, SparseSolver, SVDNullSpaceSolver


def _exact_system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    coefficients = torch.tensor([2.0, -3.0], dtype=torch.float64)
    y = theta @ coefficients
    return theta, y, coefficients


class TestSVDNullSpaceSolverSmoke:
    @pytest.mark.smoke
    def test_importable_and_sparse_solver(self) -> None:
        solver = SVDNullSpaceSolver()
        assert isinstance(solver, SparseSolver)


class TestSVDNullSpaceSolver:
    @pytest.mark.unit
    def test_recovers_exact_coefficients(self) -> None:
        theta, y, expected = _exact_system()
        result = SVDNullSpaceSolver(eps=1e-12).solve(theta, y)

        assert isinstance(result, SolveResult)
        assert result.is_valid is True
        torch.testing.assert_close(result.coefficients, expected, rtol=1e-6, atol=1e-8)
        assert result.residual == pytest.approx(0.0, abs=1e-20)
        assert result.selected_indices == [0, 1]

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    def test_non_finite_input_returns_invalid(self, bad_value: float) -> None:
        theta, y, _ = _exact_system()
        theta = theta.clone()
        theta[0, 0] = bad_value

        result = SVDNullSpaceSolver().solve(theta, y)

        assert result.is_valid is False
        assert result.coefficients.shape == (theta.shape[1],)
        assert result.residual == float("inf")
        assert "finite" in result.error_message

    @pytest.mark.unit
    def test_svd_failure_returns_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        theta, y, _ = _exact_system()

        def _raise_svd(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("synthetic SVD failure")

        monkeypatch.setattr(torch.linalg, "svd", _raise_svd)

        result = SVDNullSpaceSolver().solve(theta, y)

        assert result.is_valid is False
        assert result.residual == float("inf")
        assert "SVD failed" in result.error_message

    @pytest.mark.unit
    def test_denominator_guard_returns_invalid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        theta, y, _ = _exact_system()

        def _fake_svd(
            matrix: torch.Tensor,
            *,
            full_matrices: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del full_matrices
            n_rows, n_cols = matrix.shape
            u = torch.zeros(n_rows, n_cols, dtype=matrix.dtype, device=matrix.device)
            s = torch.ones(n_cols, dtype=matrix.dtype, device=matrix.device)
            vh = torch.eye(n_cols, dtype=matrix.dtype, device=matrix.device)
            vh[-1, 0] = 0.0
            vh[-1, 1] = 1.0
            return u, s, vh

        monkeypatch.setattr(torch.linalg, "svd", _fake_svd)

        result = SVDNullSpaceSolver(eps=1e-10).solve(theta, y)

        assert result.is_valid is False
        assert "denominator" in result.error_message

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_denominator_guard_fires_on_real_degenerate_system(self) -> None:
        c0 = torch.tensor([1.0, 0.0, 0.0, 1.0, 2.0], dtype=torch.float64)
        c1 = torch.tensor([0.0, 1.0, 0.0, 1.0, -1.0], dtype=torch.float64)
        theta = torch.column_stack([c0, c1, c1.clone()])


        y = torch.tensor([1.0, 2.0, 5.0, 0.0, 3.0], dtype=torch.float64)




        augmented = torch.column_stack([y, theta])
        _u, s, vh = torch.linalg.svd(augmented, full_matrices=False)
        assert float(s[-1].item()) < 1e-10, "system is not rank-deficient as set up"
        assert abs(float(vh[-1, 0].item())) < 1e-12, "null vector v[0] is not ~0"

        result = SVDNullSpaceSolver(eps=1e-8).solve(theta, y)

        assert result.is_valid is False


        assert "denominator" in result.error_message
        assert "near zero" in result.error_message
        assert result.residual == float("inf")
        assert result.coefficients.shape == (theta.shape[1],)

    @pytest.mark.unit
    def test_lapack_failure_is_reported_as_degenerate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        theta, y, _ = _exact_system()

        def _fail(_matrix: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("linalg.cond: LAPACK failure")

        monkeypatch.setattr(torch.linalg, "cond", _fail)

        result = SVDNullSpaceSolver(eps=1e-12).solve(theta, y)

        assert result.condition_number == float("inf")

    @pytest.mark.unit
    def test_allocation_failure_is_not_reported_as_degenerate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        theta, y, _ = _exact_system()

        def _oom(_matrix: torch.Tensor) -> torch.Tensor:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        monkeypatch.setattr(torch.linalg, "cond", _oom)

        with pytest.raises(torch.cuda.OutOfMemoryError):
            SVDNullSpaceSolver(eps=1e-12).solve(theta, y)

    @pytest.mark.unit
    def test_cpu_allocation_failure_is_not_reported_as_degenerate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        theta, y, _ = _exact_system()

        def _oom(_matrix: torch.Tensor) -> torch.Tensor:
            raise RuntimeError(
                "[enforce fail at alloc_cpu.cpp:117]. DefaultCPUAllocator: "
                "can't allocate memory: you tried to allocate "
                "8000000000000 bytes."
            )

        monkeypatch.setattr(torch.linalg, "cond", _oom)

        with pytest.raises(RuntimeError, match="DefaultCPUAllocator"):
            SVDNullSpaceSolver(eps=1e-12).solve(theta, y)

    @pytest.mark.unit
    def test_shape_validation_rejects_mismatched_rows(self) -> None:
        theta, y, _ = _exact_system()
        with pytest.raises(ValueError, match="dimension mismatch"):
            SVDNullSpaceSolver().solve(theta, y[:-1])

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_perfect_fit_on_constant_target_returns_r2_one(self) -> None:
        torch.manual_seed(0)
        n = 10
        theta = torch.column_stack(
            [
                torch.ones(n, dtype=torch.float32),
                torch.randn(n, dtype=torch.float32),
            ]
        )

        y = torch.full((n,), 3.0, dtype=torch.float32)

        result = SVDNullSpaceSolver(eps=1e-6).solve(theta, y)

        assert result.is_valid is True

        assert float(result.coefficients[0].item()) == pytest.approx(3.0, rel=1e-3)

        assert abs(float(result.coefficients[1].item())) < 1e-3

        y_pred = theta @ result.coefficients
        ss_tot = float(((y - y.mean()) ** 2).sum().item())
        assert ss_tot < 1e-10, "test setup expects near-constant target"
        torch.testing.assert_close(y_pred, y, rtol=1e-5, atol=1e-8)

        assert result.r2 == pytest.approx(1.0, abs=1e-6)
