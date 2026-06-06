
from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict, cast

import pytest
import torch
from torch import Tensor

from kd.core.expr import FunctionRegistry
from kd.data.schema import PDEDataset
from kd.search.discover.config import PINNConfig
from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset
from kd.search.discover.pinn.model import (
    PINNModel,
    TrainResult,
    _chunked_backward_residual_loss,
    _chunked_eval_residual_loss,
)

if TYPE_CHECKING:
    from kd.search.discover.pinn.model import _PINNBestState


class _EpochKit(TypedDict):

    model: PINNModel
    optimizer: torch.optim.Optimizer
    executor: PINNExecutor
    obs_coords: dict[str, Tensor]
    obs_targets: dict[str, Tensor]
    colloc: dict[str, Tensor]
    dataset_meta: PDEDataset
    config: PINNConfig
    best: _PINNBestState





COORD_NAMES = ["x", "t"]
FIELD_NAMES = ["u"]
HEAT_TERMS = ["diff2_x(u)"]
HEAT_COEFFICIENTS = [1.0]
_MODEL_SEED = 42


def _make_model(seed: int = _MODEL_SEED) -> PINNModel:
    config = PINNConfig(
        number_layer=2,
        n_hidden=10,
        activation="tanh",
        pretrain_epoch=1,
        pinn_epoch=1,
    )
    torch.manual_seed(seed)
    return PINNModel(COORD_NAMES, FIELD_NAMES, config)


def _make_coords(n: int, seed: int = 123) -> dict[str, Tensor]:
    torch.manual_seed(seed)
    return {
        "x": (
            (torch.rand(n, dtype=torch.float64) * 2 - 1)
            .detach()
            .requires_grad_(True)
        ),
        "t": (
            torch.rand(n, dtype=torch.float64).abs().detach().requires_grad_(True)
        ),
    }


@pytest.fixture
def model() -> PINNModel:
    m = _make_model()
    m.to(dtype=torch.float64)
    return m


@pytest.fixture
def executor() -> PINNExecutor:
    return PINNExecutor(FunctionRegistry.create_default())


@pytest.fixture
def dataset_meta() -> PDEDataset:
    return make_pinn_dataset(
        axis_names=COORD_NAMES,
        field_names=FIELD_NAMES,
        lhs_field="u",
        lhs_axis="t",
    )







def _zero_grads(m: PINNModel) -> None:
    for p in m.parameters():
        p.grad = None


def _grads_snapshot(m: PINNModel) -> list[Tensor]:
    return [
        (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for p in m.parameters()
    ]


def _clone_model_state(src: PINNModel, dst: PINNModel) -> None:
    dst.load_state_dict({k: v.detach().clone() for k, v in src.state_dict().items()})







class TestBackwardChunkedHelper:

    @pytest.mark.unit
    def test_value_equivalence_uneven_chunks(
        self,
        model: PINNModel,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        coords_full = _make_coords(n=23)
        coords_chunk = _make_coords(n=23)

        full_value = _chunked_backward_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_full, dataset_meta,
            weight=1.0, chunk_size=None,
        )
        _zero_grads(model)
        chunk_value = _chunked_backward_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_chunk, dataset_meta,
            weight=1.0, chunk_size=7,
        )
        assert chunk_value == pytest.approx(full_value, rel=1e-10, abs=1e-12)

    @pytest.mark.unit
    def test_uneven_chunks_n100_chunk33(
        self,
        model: PINNModel,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        coords_full = _make_coords(n=100)
        coords_chunk = _make_coords(n=100)

        full_value = _chunked_backward_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_full, dataset_meta,
            weight=1.0, chunk_size=None,
        )
        _zero_grads(model)
        chunk_value = _chunked_backward_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_chunk, dataset_meta,
            weight=1.0, chunk_size=33,
        )
        assert chunk_value == pytest.approx(full_value, rel=1e-10, abs=1e-12)

    @pytest.mark.unit
    def test_chunk_size_larger_than_n_falls_back(
        self,
        model: PINNModel,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        coords_a = _make_coords(n=50)
        coords_b = _make_coords(n=50)

        none_value = _chunked_backward_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_a, dataset_meta,
            weight=1.0, chunk_size=None,
        )
        _zero_grads(model)
        big_chunk_value = _chunked_backward_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_b, dataset_meta,
            weight=1.0, chunk_size=1000,
        )
        assert big_chunk_value == pytest.approx(none_value, rel=1e-12, abs=1e-14)







class TestBackwardChunkedGrads:

    @pytest.mark.unit
    def test_grad_equivalence_uneven_chunks(
        self,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:

        model_full = _make_model()
        model_full.to(dtype=torch.float64)
        model_chunk = _make_model()
        model_chunk.to(dtype=torch.float64)
        _clone_model_state(model_full, model_chunk)

        coords_full = _make_coords(n=23)
        coords_chunk = _make_coords(n=23)

        _zero_grads(model_full)
        _chunked_backward_residual_loss(
            model_full, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_full, dataset_meta,
            weight=1.0, chunk_size=None,
        )
        full_grads = _grads_snapshot(model_full)

        _zero_grads(model_chunk)
        _chunked_backward_residual_loss(
            model_chunk, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_chunk, dataset_meta,
            weight=1.0, chunk_size=7,
        )
        chunk_grads = _grads_snapshot(model_chunk)

        assert len(full_grads) == len(chunk_grads)
        for i, (gf, gc) in enumerate(
            zip(full_grads, chunk_grads, strict=False)
        ):
            assert torch.allclose(gf, gc, rtol=1e-9, atol=1e-12), (
                f"param[{i}] grad diff: max abs = "
                f"{(gf - gc).abs().max().item():.3e}"
            )

    @pytest.mark.unit
    def test_grad_equivalence_with_weight(
        self,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        model_full = _make_model()
        model_full.to(dtype=torch.float64)
        model_chunk = _make_model()
        model_chunk.to(dtype=torch.float64)
        _clone_model_state(model_full, model_chunk)

        coords_full = _make_coords(n=30)
        coords_chunk = _make_coords(n=30)

        _zero_grads(model_full)
        _chunked_backward_residual_loss(
            model_full, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_full, dataset_meta,
            weight=2.5, chunk_size=None,
        )
        full_grads = _grads_snapshot(model_full)

        _zero_grads(model_chunk)
        _chunked_backward_residual_loss(
            model_chunk, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_chunk, dataset_meta,
            weight=2.5, chunk_size=8,
        )
        chunk_grads = _grads_snapshot(model_chunk)

        for i, (gf, gc) in enumerate(
            zip(full_grads, chunk_grads, strict=False)
        ):
            assert torch.allclose(gf, gc, rtol=1e-9, atol=1e-12), (
                f"param[{i}] grad diff with weight=2.5: max abs = "
                f"{(gf - gc).abs().max().item():.3e}"
            )







class TestEvalChunkedHelper:

    @pytest.mark.unit
    def test_eval_value_equivalence(
        self,
        model: PINNModel,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        coords_full = _make_coords(n=23)
        coords_chunk = _make_coords(n=23)

        full_value = _chunked_eval_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_full, dataset_meta,
            weight=1.0, chunk_size=None,
        )
        chunk_value = _chunked_eval_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_chunk, dataset_meta,
            weight=1.0, chunk_size=7,
        )
        assert chunk_value == pytest.approx(full_value, rel=1e-10, abs=1e-12)

    @pytest.mark.unit
    def test_eval_does_not_accumulate_grads(
        self,
        model: PINNModel,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        _zero_grads(model)
        coords = _make_coords(n=20)
        _chunked_eval_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords, dataset_meta,
            weight=1.0, chunk_size=5,
        )
        for p in model.parameters():
            assert p.grad is None, "eval helper accidentally accumulated grad"

    @pytest.mark.unit
    def test_eval_chunk_size_larger_than_n_falls_back(
        self,
        model: PINNModel,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        coords_a = _make_coords(n=50)
        coords_b = _make_coords(n=50)

        none_value = _chunked_eval_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_a, dataset_meta,
            weight=1.0, chunk_size=None,
        )
        big_chunk_value = _chunked_eval_residual_loss(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords_b, dataset_meta,
            weight=1.0, chunk_size=1000,
        )
        assert big_chunk_value == pytest.approx(none_value, rel=1e-12, abs=1e-14)

    @pytest.mark.unit
    def test_short_circuit_on_zero_weight(
        self,
        model: PINNModel,
        dataset_meta: PDEDataset,
    ) -> None:
        class _RaisingExecutor:

            def compute_residual(self, **kwargs: object) -> Tensor:
                raise AssertionError(
                    "compute_residual must not run when weight=0.0"
                )

        coords = _make_coords(n=23)
        result = _chunked_eval_residual_loss(
            model,
            _RaisingExecutor(),
            HEAT_TERMS,
            HEAT_COEFFICIENTS,
            coords,
            dataset_meta,
            weight=0.0,
            chunk_size=None,
        )
        assert result == 0.0

        result_chunked = _chunked_eval_residual_loss(
            model,
            _RaisingExecutor(),
            HEAT_TERMS,
            HEAT_COEFFICIENTS,
            coords,
            dataset_meta,
            weight=0.0,
            chunk_size=7,
        )
        assert result_chunked == 0.0







def _heat_solution(x: Tensor, t: Tensor) -> Tensor:
    return torch.exp(-torch.pi**2 * t) * torch.sin(torch.pi * x)


def _make_obs(n: int = 30, seed: int = 7) -> tuple[
    dict[str, Tensor], dict[str, Tensor]
]:
    torch.manual_seed(seed)
    x = (torch.rand(n, dtype=torch.float64) * 2 - 1)
    t = torch.rand(n, dtype=torch.float64).abs()
    return {"x": x, "t": t}, {"u": _heat_solution(x, t)}


def _train_one_epoch_config(
    *,
    chunk_size: int | None,
    grad_clip_norm: float | None = None,
    pinn_epoch: int = 1,
    early_stop_patience: int = 1000,
    lr: float = 0.005,
) -> PINNConfig:
    return PINNConfig(
        number_layer=2,
        n_hidden=10,
        activation="tanh",
        pretrain_epoch=1,
        pinn_epoch=pinn_epoch,
        lr=lr,
        coef_pde=1.0,
        early_stop_patience=early_stop_patience,
        grad_clip_norm=grad_clip_norm,
        colloc_chunk_size=chunk_size,
    )







class TestRunPinnEpochChunkIntegration:

    @pytest.mark.unit
    def test_local_coords_none_path(
        self,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        torch.manual_seed(_MODEL_SEED)
        config = _train_one_epoch_config(chunk_size=7, pinn_epoch=2)
        model = PINNModel(COORD_NAMES, FIELD_NAMES, config)
        model.to(dtype=torch.float64)
        obs_coords, obs_targets = _make_obs(n=30)
        colloc = _make_coords(n=23)

        result = model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc,
            dataset_metadata=dataset_meta,
            config=config,
            local_coords=None,
        )
        assert isinstance(result, TrainResult)
        assert math.isfinite(result.total_loss)
        assert math.isfinite(result.physics_loss)

    @pytest.mark.unit
    def test_local_loss_chunked(
        self,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        obs_coords, obs_targets = _make_obs(n=30)
        colloc = _make_coords(n=23)
        local = _make_coords(n=15, seed=999)


        torch.manual_seed(_MODEL_SEED)
        config_full = _train_one_epoch_config(chunk_size=None, pinn_epoch=1)
        model_full = PINNModel(COORD_NAMES, FIELD_NAMES, config_full)
        model_full.to(dtype=torch.float64)
        result_full = model_full.train_pinn(
            terms=HEAT_TERMS, coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords, observation_targets=obs_targets,
            colloc_coords=colloc, dataset_metadata=dataset_meta,
            config=config_full, local_coords=local,
        )


        torch.manual_seed(_MODEL_SEED)
        config_chunk = _train_one_epoch_config(chunk_size=7, pinn_epoch=1)
        model_chunk = PINNModel(COORD_NAMES, FIELD_NAMES, config_chunk)
        model_chunk.to(dtype=torch.float64)
        result_chunk = model_chunk.train_pinn(
            terms=HEAT_TERMS, coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords, observation_targets=obs_targets,
            colloc_coords=colloc, dataset_metadata=dataset_meta,
            config=config_chunk, local_coords=local,
        )

        assert result_chunk.physics_loss == pytest.approx(
            result_full.physics_loss, rel=1e-9, abs=1e-12
        )

        for (n1, p1), (n2, p2) in zip(
            model_full.named_parameters(),
            model_chunk.named_parameters(),
            strict=False,
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2, rtol=1e-9, atol=1e-12), (
                f"param '{n1}' diverged after one step "
                f"(max abs = {(p1 - p2).abs().max().item():.3e})"
            )

    @pytest.mark.unit
    def test_grad_clip_post_accumulation(
        self,
        executor: PINNExecutor,
        dataset_meta: PDEDataset,
    ) -> None:
        obs_coords, obs_targets = _make_obs(n=30)
        colloc = _make_coords(n=23)

        torch.manual_seed(_MODEL_SEED)
        config_full = _train_one_epoch_config(
            chunk_size=None, grad_clip_norm=1.0, pinn_epoch=1, lr=0.01,
        )
        model_full = PINNModel(COORD_NAMES, FIELD_NAMES, config_full)
        model_full.to(dtype=torch.float64)
        model_full.train_pinn(
            terms=HEAT_TERMS, coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords, observation_targets=obs_targets,
            colloc_coords=colloc, dataset_metadata=dataset_meta,
            config=config_full, local_coords=None,
        )

        torch.manual_seed(_MODEL_SEED)
        config_chunk = _train_one_epoch_config(
            chunk_size=7, grad_clip_norm=1.0, pinn_epoch=1, lr=0.01,
        )
        model_chunk = PINNModel(COORD_NAMES, FIELD_NAMES, config_chunk)
        model_chunk.to(dtype=torch.float64)
        model_chunk.train_pinn(
            terms=HEAT_TERMS, coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords, observation_targets=obs_targets,
            colloc_coords=colloc, dataset_metadata=dataset_meta,
            config=config_chunk, local_coords=None,
        )

        for (n1, p1), (_n2, p2) in zip(
            model_full.named_parameters(),
            model_chunk.named_parameters(),
            strict=False,
        ):
            assert torch.allclose(p1, p2, rtol=1e-9, atol=1e-12), (
                f"param '{n1}' diverged with grad_clip "
                f"(max abs = {(p1 - p2).abs().max().item():.3e})"
            )







class _NaNInChunkExecutor:

    def __init__(
        self, real_executor: PINNExecutor, nan_call_index: int
    ) -> None:
        self._real = real_executor
        self.nan_call_index = nan_call_index
        self.call_count = 0

    def compute_residual(self, **kwargs: object) -> Tensor:
        residual = self._real.compute_residual(**kwargs)
        idx = self.call_count
        self.call_count += 1
        if idx == self.nan_call_index:

            return residual * float("nan")
        return residual


class _CountingExecutor:

    def __init__(self, real_executor: PINNExecutor) -> None:
        self._real = real_executor
        self.call_sizes: list[int] = []

    def compute_residual(self, **kwargs: object) -> Tensor:
        coords = cast("dict[str, Tensor]", kwargs["coords"])
        n = next(iter(coords.values())).shape[0]
        self.call_sizes.append(n)
        return self._real.compute_residual(**kwargs)


@pytest.mark.unit
def test_chunked_path_invoked_when_chunk_size_set(
    executor: PINNExecutor, dataset_meta: PDEDataset
) -> None:
    torch.manual_seed(_MODEL_SEED)
    config = _train_one_epoch_config(chunk_size=7, pinn_epoch=2)
    model = PINNModel(COORD_NAMES, FIELD_NAMES, config)
    model.to(dtype=torch.float64)
    obs_coords, obs_targets = _make_obs(n=30)
    colloc = _make_coords(n=23)

    counting = _CountingExecutor(executor)
    model.train_pinn(
        terms=HEAT_TERMS, coefficients=HEAT_COEFFICIENTS,
        pinn_executor=counting,
        observation_coords=obs_coords, observation_targets=obs_targets,
        colloc_coords=colloc, dataset_metadata=dataset_meta,
        config=config, local_coords=None,
    )

    assert counting.call_sizes, "compute_residual was never called"
    over_chunk = [n for n in counting.call_sizes if n > 7]
    assert not over_chunk, (
        f"chunk_size=7 ignored: saw call sizes {set(counting.call_sizes)}"
    )


@pytest.mark.unit
def test_nan_in_chunk_recovers(
    executor: PINNExecutor, dataset_meta: PDEDataset
) -> None:
    torch.manual_seed(_MODEL_SEED)
    config = _train_one_epoch_config(
        chunk_size=7, pinn_epoch=5, early_stop_patience=100,
    )
    model = PINNModel(COORD_NAMES, FIELD_NAMES, config)
    model.to(dtype=torch.float64)
    obs_coords, obs_targets = _make_obs(n=30)
    colloc = _make_coords(n=23)





    nan_exec = _NaNInChunkExecutor(executor, nan_call_index=2)

    result = model.train_pinn(
        terms=HEAT_TERMS, coefficients=HEAT_COEFFICIENTS,
        pinn_executor=nan_exec,
        observation_coords=obs_coords, observation_targets=obs_targets,
        colloc_coords=colloc, dataset_metadata=dataset_meta,
        config=config, local_coords=None,
    )


    with torch.no_grad():
        out = model(**obs_coords)
    assert torch.isfinite(out["u"]).all(), "model corrupted by NaN chunk"
    assert math.isfinite(result.total_loss), "total_loss leaked NaN"







class TestRunPinnEpochReturnShape:

    @pytest.fixture
    def epoch_kit(
        self, executor: PINNExecutor, dataset_meta: PDEDataset
    ) -> _EpochKit:
        from kd.search.discover.pinn.model import _capture_pinn_best

        torch.manual_seed(_MODEL_SEED)
        config = _train_one_epoch_config(chunk_size=7, pinn_epoch=10, lr=0.005)
        model = PINNModel(COORD_NAMES, FIELD_NAMES, config)
        model.to(dtype=torch.float64)
        obs_coords, obs_targets = _make_obs(n=30)
        colloc = _make_coords(n=23)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        best = _capture_pinn_best(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            obs_coords, obs_targets, colloc, dataset_meta, config, None,
        )
        return {
            "model": model,
            "optimizer": optimizer,
            "executor": executor,
            "obs_coords": obs_coords,
            "obs_targets": obs_targets,
            "colloc": colloc,
            "dataset_meta": dataset_meta,
            "config": config,
            "best": best,
        }

    @pytest.mark.unit
    def test_normal_branch_returns_4tuple(
        self, epoch_kit: _EpochKit
    ) -> None:
        from kd.search.discover.pinn.model import _run_pinn_epoch

        out = _run_pinn_epoch(
            epoch_kit["model"], epoch_kit["optimizer"], epoch_kit["executor"],
            HEAT_TERMS, HEAT_COEFFICIENTS,
            epoch_kit["obs_coords"], epoch_kit["obs_targets"],
            epoch_kit["colloc"], epoch_kit["dataset_meta"], epoch_kit["config"],
            epoch_kit["best"], None, epoch=1, epochs_without_improvement=0,
        )
        assert isinstance(out, tuple) and len(out) == 4
        best, ewi, result, nan_detected = out
        assert nan_detected is False
        assert ewi >= 0


    @pytest.mark.unit
    def test_nan_branch_returns_4tuple_nan_true(
        self, epoch_kit: _EpochKit
    ) -> None:
        from kd.search.discover.pinn.model import _run_pinn_epoch


        nan_exec = _NaNInChunkExecutor(
            epoch_kit["executor"], nan_call_index=0
        )
        out = _run_pinn_epoch(
            epoch_kit["model"], epoch_kit["optimizer"], nan_exec,
            HEAT_TERMS, HEAT_COEFFICIENTS,
            epoch_kit["obs_coords"], epoch_kit["obs_targets"],
            epoch_kit["colloc"], epoch_kit["dataset_meta"], epoch_kit["config"],
            epoch_kit["best"], None, epoch=1, epochs_without_improvement=0,
        )
        assert isinstance(out, tuple) and len(out) == 4
        _, _, result, nan_detected = out
        assert nan_detected is True
        assert result is None

    @pytest.mark.unit
    def test_early_stop_branch_returns_train_result(
        self, executor: PINNExecutor, dataset_meta: PDEDataset
    ) -> None:
        from kd.search.discover.pinn.model import _capture_pinn_best, _run_pinn_epoch


        torch.manual_seed(_MODEL_SEED)
        config = PINNConfig(
            number_layer=2, n_hidden=10, activation="tanh",
            pretrain_epoch=1, pinn_epoch=100, lr=0.0,
            coef_pde=1.0, early_stop_patience=2,
            grad_clip_norm=None, colloc_chunk_size=7,
            early_stop_warmup=0,
        )
        model = PINNModel(COORD_NAMES, FIELD_NAMES, config)
        model.to(dtype=torch.float64)
        obs_coords, obs_targets = _make_obs(n=30)
        colloc = _make_coords(n=23)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        best = _capture_pinn_best(
            model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            obs_coords, obs_targets, colloc, dataset_meta, config, None,
        )

        best.total_loss = -1.0
        best.physics_loss = -1.0
        best.data_loss = -1.0


        ewi = 0
        result = None
        for epoch in range(1, 5):
            best, ewi, result, nan_detected = _run_pinn_epoch(
                model, optimizer, executor,
                HEAT_TERMS, HEAT_COEFFICIENTS,
                obs_coords, obs_targets, colloc, dataset_meta, config,
                best, None, epoch=epoch, epochs_without_improvement=ewi,
            )
            assert nan_detected is False
            if result is not None:
                break
        assert isinstance(result, TrainResult), (
            f"early stop never fired; ewi={ewi}, result={result}"
        )

        assert result.total_loss == -1.0
        assert result.stopped_early is True







@pytest.mark.unit
def test_early_stop_restores_best_state_dict(
    executor: PINNExecutor, dataset_meta: PDEDataset
) -> None:
    from kd.search.discover.pinn.model import _capture_pinn_best, _run_pinn_epoch

    torch.manual_seed(_MODEL_SEED)

    config = PINNConfig(
        number_layer=2, n_hidden=10, activation="tanh",
        pretrain_epoch=1, pinn_epoch=100, lr=0.005,
        coef_pde=1.0, early_stop_patience=2,
        grad_clip_norm=None, colloc_chunk_size=7,
        early_stop_warmup=0,
    )
    model = PINNModel(COORD_NAMES, FIELD_NAMES, config)
    model.to(dtype=torch.float64)
    obs_coords, obs_targets = _make_obs(n=30)
    colloc = _make_coords(n=23)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best = _capture_pinn_best(
        model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
        obs_coords, obs_targets, colloc, dataset_meta, config, None,
    )

    best_state_snapshot = {
        k: v.detach().clone() for k, v in best.state_dict.items()
    }

    best.total_loss = -1.0
    best.physics_loss = -1.0
    best.data_loss = -1.0


    ewi = 0
    result = None
    for epoch in range(1, 5):
        best, ewi, result, nan_detected = _run_pinn_epoch(
            model, optimizer, executor,
            HEAT_TERMS, HEAT_COEFFICIENTS,
            obs_coords, obs_targets, colloc, dataset_meta, config,
            best, None, epoch=epoch, epochs_without_improvement=ewi,
        )
        if result is not None:
            break
    assert result is not None, "early stop never fired"


    current_state = model.state_dict()
    for key in best_state_snapshot:
        assert torch.allclose(
            current_state[key], best_state_snapshot[key], rtol=0, atol=0
        ), f"param '{key}' not restored from best after early stop"







@pytest.mark.unit
def test_recover_clears_grads(
    executor: PINNExecutor, dataset_meta: PDEDataset
) -> None:
    from kd.search.discover.pinn.model import _capture_pinn_best, _run_pinn_epoch

    torch.manual_seed(_MODEL_SEED)
    config = _train_one_epoch_config(chunk_size=7, pinn_epoch=5)
    model = PINNModel(COORD_NAMES, FIELD_NAMES, config)
    model.to(dtype=torch.float64)
    obs_coords, obs_targets = _make_obs(n=30)
    colloc = _make_coords(n=23)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best = _capture_pinn_best(
        model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
        obs_coords, obs_targets, colloc, dataset_meta, config, None,
    )

    nan_exec = _NaNInChunkExecutor(executor, nan_call_index=0)
    out_best, _, _, nan_detected = _run_pinn_epoch(
        model, optimizer, nan_exec,
        HEAT_TERMS, HEAT_COEFFICIENTS,
        obs_coords, obs_targets, colloc, dataset_meta, config,
        best, None, epoch=1, epochs_without_improvement=0,
    )
    assert nan_detected is True

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), (
            f"param '{name}' has NaN/Inf in .grad after recovery"
        )







class _OOMExecutor:

    def __init__(
        self, real_executor: PINNExecutor, oom_call_index: int
    ) -> None:
        self._real = real_executor
        self.oom_call_index = oom_call_index
        self.call_count = 0

    def compute_residual(self, **kwargs: object) -> Tensor:
        idx = self.call_count
        self.call_count += 1
        if idx == self.oom_call_index:
            raise torch.cuda.OutOfMemoryError(
                "synthetic OOM injected for v2.2 test #14"
            )
        return self._real.compute_residual(**kwargs)


@pytest.mark.unit
def test_chunk_oom_propagates_as_cuda_oom_type(
    model: PINNModel, dataset_meta: PDEDataset, executor: PINNExecutor,
) -> None:
    coords = _make_coords(n=23)
    oom_exec = _OOMExecutor(executor, oom_call_index=1)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _chunked_backward_residual_loss(
            model, oom_exec, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords, dataset_meta,
            weight=1.0, chunk_size=7,
        )


@pytest.mark.unit
def test_chunk_oom_propagates_through_cycle(
    executor: PINNExecutor, dataset_meta: PDEDataset
) -> None:
    torch.manual_seed(_MODEL_SEED)
    config = _train_one_epoch_config(chunk_size=7, pinn_epoch=2)
    model = PINNModel(COORD_NAMES, FIELD_NAMES, config)
    model.to(dtype=torch.float64)
    obs_coords, obs_targets = _make_obs(n=30)
    colloc = _make_coords(n=23)


    oom_exec = _OOMExecutor(executor, oom_call_index=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        model.train_pinn(
            terms=HEAT_TERMS, coefficients=HEAT_COEFFICIENTS,
            pinn_executor=oom_exec,
            observation_coords=obs_coords, observation_targets=obs_targets,
            colloc_coords=colloc, dataset_metadata=dataset_meta,
            config=config, local_coords=None,
        )







@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="memory regression test requires CUDA device",
)
@pytest.mark.unit
def test_memory_regression_chunked_under_full(
    executor: PINNExecutor, dataset_meta: PDEDataset
) -> None:
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(_MODEL_SEED)
    config_full = _train_one_epoch_config(chunk_size=None, pinn_epoch=1)
    model_full = PINNModel(COORD_NAMES, FIELD_NAMES, config_full, device=device)
    model_full.to(dtype=torch.float64)
    obs_coords, obs_targets = _make_obs(n=30)
    obs_coords = {k: v.to(device) for k, v in obs_coords.items()}
    obs_targets = {k: v.to(device) for k, v in obs_targets.items()}
    colloc = {k: v.to(device) for k, v in _make_coords(n=8000).items()}

    model_full.train_pinn(
        terms=HEAT_TERMS, coefficients=HEAT_COEFFICIENTS,
        pinn_executor=executor,
        observation_coords=obs_coords, observation_targets=obs_targets,
        colloc_coords=colloc, dataset_metadata=dataset_meta,
        config=config_full, local_coords=None,
    )
    full_peak = torch.cuda.max_memory_allocated(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(_MODEL_SEED)
    config_chunk = _train_one_epoch_config(chunk_size=1000, pinn_epoch=1)
    model_chunk = PINNModel(COORD_NAMES, FIELD_NAMES, config_chunk, device=device)
    model_chunk.to(dtype=torch.float64)
    model_chunk.train_pinn(
        terms=HEAT_TERMS, coefficients=HEAT_COEFFICIENTS,
        pinn_executor=executor,
        observation_coords=obs_coords, observation_targets=obs_targets,
        colloc_coords=colloc, dataset_metadata=dataset_meta,
        config=config_chunk, local_coords=None,
    )
    chunk_peak = torch.cuda.max_memory_allocated(device)
    assert chunk_peak < full_peak, (
        f"chunked peak {chunk_peak / 1e9:.2f} GB not less than "
        f"full-batch peak {full_peak / 1e9:.2f} GB"
    )










class TestAutoFallbackLargeBatch:

    @pytest.mark.unit
    def test_resolve_chunk_below_threshold_returns_none(self) -> None:
        from kd.search.discover.pinn.model import (
            _LARGE_BATCH_AUTO_THRESHOLD,
            _resolve_effective_chunk_size,
        )

        assert _resolve_effective_chunk_size(None, 1000) is None
        assert (
            _resolve_effective_chunk_size(None, _LARGE_BATCH_AUTO_THRESHOLD)
            is None
        )

    @pytest.mark.unit
    def test_resolve_chunk_above_threshold_auto_chunks(self) -> None:
        from kd.search.discover.pinn.model import (
            _LARGE_BATCH_AUTO_CHUNK,
            _LARGE_BATCH_AUTO_THRESHOLD,
            _resolve_effective_chunk_size,
        )

        result = _resolve_effective_chunk_size(
            None, _LARGE_BATCH_AUTO_THRESHOLD + 1
        )
        assert result == _LARGE_BATCH_AUTO_CHUNK

        assert (
            _resolve_effective_chunk_size(None, 6_881_280)
            == _LARGE_BATCH_AUTO_CHUNK
        )

    @pytest.mark.unit
    def test_resolve_chunk_explicit_never_rewritten(self) -> None:
        from kd.search.discover.pinn.model import _resolve_effective_chunk_size

        assert _resolve_effective_chunk_size(7, 10_000_000) == 7
        assert _resolve_effective_chunk_size(50_000, 6_881_280) == 50_000


        assert _resolve_effective_chunk_size(100, 100) == 100

    @pytest.mark.unit
    def test_maybe_empty_cache_cpu_is_noop(self) -> None:
        from kd.search.discover.pinn.model import _maybe_empty_cache


        _maybe_empty_cache(0)
        _maybe_empty_cache(3)
        _maybe_empty_cache(7)
        _maybe_empty_cache(123)

    @pytest.mark.unit
    def test_eval_auto_chunks_on_large_batch(
        self, model: PINNModel, executor: PINNExecutor, dataset_meta: PDEDataset
    ) -> None:
        from kd.search.discover.pinn.model import (
            _LARGE_BATCH_AUTO_CHUNK,
            _LARGE_BATCH_AUTO_THRESHOLD,
        )

        n = _LARGE_BATCH_AUTO_THRESHOLD + 1
        coords = _make_coords(n=n)
        counting = _CountingExecutor(executor)
        _chunked_eval_residual_loss(
            model, counting, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords, dataset_meta,
            weight=1.0, chunk_size=None,
        )
        assert counting.call_sizes, "compute_residual was never called"
        over_chunk = [m for m in counting.call_sizes if m > _LARGE_BATCH_AUTO_CHUNK]
        assert not over_chunk, (
            f"auto-fallback did not chunk; saw sizes {set(counting.call_sizes)}"
        )

    @pytest.mark.unit
    def test_backward_auto_chunks_on_large_batch(
        self, model: PINNModel, executor: PINNExecutor, dataset_meta: PDEDataset
    ) -> None:
        from kd.search.discover.pinn.model import (
            _LARGE_BATCH_AUTO_CHUNK,
            _LARGE_BATCH_AUTO_THRESHOLD,
        )

        n = _LARGE_BATCH_AUTO_THRESHOLD + 1
        coords = _make_coords(n=n)
        counting = _CountingExecutor(executor)
        _chunked_backward_residual_loss(
            model, counting, HEAT_TERMS, HEAT_COEFFICIENTS,
            coords, dataset_meta,
            weight=1.0, chunk_size=None,
        )
        assert counting.call_sizes, "compute_residual was never called"
        over_chunk = [m for m in counting.call_sizes if m > _LARGE_BATCH_AUTO_CHUNK]
        assert not over_chunk, (
            f"auto-fallback did not chunk; saw sizes {set(counting.call_sizes)}"
        )
