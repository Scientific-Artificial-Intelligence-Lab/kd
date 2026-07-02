
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kd.search.sga.train import _solve

N_REPEATS = 100


@pytest.fixture(autouse=True)
def _restore_determinism_state() -> object:
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def _near_collinear_system() -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(12345)
    x = torch.randn(2500, 8, generator=gen, dtype=torch.float64)
    noise = torch.randn(2500, generator=gen, dtype=torch.float64)
    x[:, 7] = x[:, 0] * 0.999999999 + noise * 1e-10
    y = torch.randn(2500, generator=gen, dtype=torch.float64)
    return x, y


class TestSolveBitwiseStability:

    def test_ols_path_bitwise_stable(self) -> None:
        x, y = _near_collinear_system()
        first = _solve(x, y, lam=0.0, d=x.shape[1])
        for _ in range(N_REPEATS - 1):
            again = _solve(x, y, lam=0.0, d=x.shape[1])
            assert torch.equal(first, again), (
                "lstsq jitter: identical input produced bitwise-different "
                "solutions -- seeded searches will diverge"
            )

    def test_ridge_path_bitwise_stable(self) -> None:
        x, y = _near_collinear_system()
        first = _solve(x, y, lam=1e-5, d=x.shape[1])
        for _ in range(N_REPEATS - 1):
            again = _solve(x, y, lam=1e-5, d=x.shape[1])
            assert torch.equal(first, again)


class TestSolveDeterministicMode:

    @pytest.mark.parametrize("lam", [0.0, 1e-5])
    def test_lstsq_called_with_deterministic_algorithms(
        self, lam: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x, y = _near_collinear_system()
        real_lstsq = torch.linalg.lstsq
        flag_during_call: list[bool] = []

        def spy(*args: object, **kwargs: object) -> object:
            flag_during_call.append(torch.are_deterministic_algorithms_enabled())
            return real_lstsq(*args, **kwargs)

        monkeypatch.setattr(torch.linalg, "lstsq", spy)
        _solve(x, y, lam=lam, d=x.shape[1])
        assert flag_during_call, "_solve never reached lstsq"
        assert all(flag_during_call)


class TestGlobalStatePreserved:

    @pytest.mark.parametrize(
        ("enabled", "warn_only"),
        [(False, False), (True, False), (True, True)],
    )
    def test_flag_state_preserved_after_solve(
        self, enabled: bool, warn_only: bool
    ) -> None:
        x, y = _near_collinear_system()
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        _solve(x, y, lam=0.0, d=x.shape[1])
        assert torch.are_deterministic_algorithms_enabled() == enabled
        assert (
            torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
        )

    def test_flag_state_restored_when_lstsq_raises(self) -> None:
        x, _ = _near_collinear_system()
        bad_y = torch.randn(7, dtype=torch.float64)
        torch.use_deterministic_algorithms(False)
        with pytest.raises(RuntimeError):
            _solve(x, bad_y, lam=0.0, d=x.shape[1])
        assert torch.are_deterministic_algorithms_enabled() is False
