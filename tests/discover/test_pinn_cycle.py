
from __future__ import annotations

import logging
import math
from dataclasses import fields as dataclass_fields
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import (
    EvaluationResult,
    Evaluator,
)
from kd.core.executor.context import ExecutionContext
from kd.core.expr import (
    FunctionRegistry,
    PythonExecutor,
)
from kd.core.linear_solve.least_squares import (
    LeastSquaresSolver,
)
from kd.data.derivatives.finite_diff import (
    FiniteDiffProvider,
)
from kd.data.schema import (
    AxisInfo,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.discover.builder import build_engine
from kd.search.discover.config import DiscoverConfig, PINNConfig
from kd.search.discover.engine import CandidateSnapshot, EngineState
from kd.search.discover.pinn.cycle import PINNCycleResult, PINNCycleRunner
from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset
from kd.search.discover.pinn.model import PINNModel, PretrainResult
from kd.search.discover.stability import (
    StabilityCandidateStats,
    StabilitySelectionResult,
)
from kd.search.discover.tokens.library import LibraryConfig





SEED = 42
_N_GRID = 30
_N_OBS = 50
_N_COLLOC = 100
_N_ITER = 10
_PRETRAIN_EPOCH = 10
_PINN_EPOCH = 3

_OPERATORS = ["add", "mul", "diff_x", "diff2_x"]







def _heat_solution(x: Tensor, t: Tensor) -> Tensor:
    return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)


def _make_heat_dataset() -> PDEDataset:
    x = torch.linspace(0.01, 0.99, _N_GRID, dtype=torch.float64)
    t = torch.linspace(0.01, 0.99, _N_GRID, dtype=torch.float64)
    big_x, big_t = torch.meshgrid(x, t, indexing="ij")
    return PDEDataset(
        name="heat_cycle",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=_heat_solution(big_x, big_t))},
        lhs_field="u",
        lhs_axis="t",
    )


def _build_evaluator(dataset: PDEDataset) -> Evaluator:
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    u_t = provider.get_derivative("u", "t", order=1).flatten()
    return Evaluator(
        PythonExecutor(FunctionRegistry.create_default()),
        LeastSquaresSolver(),
        context,
        lhs=u_t,
    )


def _fast_config(
    n_cycles: int = 2, cycle_n_iterations: int | None = None
) -> DiscoverConfig:
    return DiscoverConfig(
        n_iterations=_N_ITER,
        batch_size=32,
        max_length=15,
        library=LibraryConfig(
            operators=_OPERATORS,
            state_vars=["u"],
            coord_vars=["x", "t"],
        ),
        num_units=8,
        num_layers=1,
        embedding_dim=4,
        pinn=PINNConfig(
            number_layer=2,
            n_hidden=8,
            pretrain_epoch=_PRETRAIN_EPOCH,
            pinn_epoch=_PINN_EPOCH,
            lr=0.001,
            n_cycles=n_cycles,
            cycle_n_iterations=cycle_n_iterations,
            n_collocation=_N_COLLOC,
            early_stop_patience=100,
        ),
    )


def _build_runner(
    evaluator: Any,
    config: DiscoverConfig,
    stability_seed: int | None = None,
    colloc_coords: dict[str, Tensor] | None = None,
    *,
    pretrain_split_seed: int = 0,
    local_sample_seed: int | None = None,
) -> PINNCycleRunner:
    assert config.pinn is not None
    torch.manual_seed(SEED)
    engine = build_engine(config)
    model = PINNModel(["x", "t"], ["u"], config.pinn)
    executor = PINNExecutor(FunctionRegistry.create_default())
    dataset_meta = make_pinn_dataset(
        axis_names=["x", "t"],
        field_names=["u"],
        lhs_field="u",
        lhs_axis="t",
    )

    rng = torch.Generator().manual_seed(SEED)
    x_obs = torch.rand(_N_OBS, generator=rng).float()
    t_obs = torch.rand(_N_OBS, generator=rng).float()
    obs_coords = {"x": x_obs, "t": t_obs}
    obs_targets = {"u": _heat_solution(x_obs, t_obs).float()}

    rng2 = torch.Generator().manual_seed(SEED + 1)
    colloc = colloc_coords or {
        "x": torch.rand(_N_COLLOC, generator=rng2).float(),
        "t": torch.rand(_N_COLLOC, generator=rng2).float(),
    }

    return PINNCycleRunner(
        engine=engine,
        pinn_model=model,
        pinn_executor=executor,
        initial_evaluator=evaluator,
        observation_coords=obs_coords,
        observation_targets=obs_targets,
        colloc_coords=colloc,
        dataset_metadata=dataset_meta,
        config=config,
        stability_seed=stability_seed,
        pretrain_split_seed=pretrain_split_seed,
        local_sample_seed=local_sample_seed,
    )


class _CorruptingTrainPinnError(RuntimeError):
    pass


class _AlwaysInvalidEvaluator:

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return EvaluationResult(
            mse=1e10,
            nmse=1e10,
            r2=-float("inf"),
            complexity=0,
            is_valid=False,
            expression=expr,
        )


class _AlwaysValidHeatEvaluator:

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return EvaluationResult(
            mse=0.01,
            nmse=0.01,
            r2=0.99,
            complexity=1,
            is_valid=True,
            expression=expr,
            terms=["diff2_x(u)"],
            coefficients=torch.tensor([1.0]),
        )


class _SelectionEvaluator:

    def __init__(self, results: dict[str, EvaluationResult]) -> None:
        self._results = results

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return self._results[expr]


def _snapshot(
    expression: str,
    reward: float,
    nmse: float,
    n_nodes: int,
    terms: list[str],
) -> CandidateSnapshot:
    return CandidateSnapshot(
        expression=expression,
        reward=reward,
        nmse=nmse,
        n_nodes=n_nodes,
        terms=terms,
    )







@pytest.fixture(scope="module")
def heat_dataset() -> PDEDataset:
    return _make_heat_dataset()


@pytest.fixture(scope="module")
def heat_evaluator(heat_dataset: PDEDataset) -> Evaluator:
    return _build_evaluator(heat_dataset)







class TestPINNCycleResult:

    def test_importable(self) -> None:
        from kd.search.discover.pinn.cycle import PINNCycleResult

        assert PINNCycleResult is not None

    def test_has_required_fields(self) -> None:
        names = {f.name for f in dataclass_fields(PINNCycleResult)}
        assert names >= {"final_state", "cycle_metrics", "pretrain_result"}







class TestCycleRun:

    def test_2_cycle_completes(self, heat_evaluator: Evaluator) -> None:
        result = _build_runner(heat_evaluator, _fast_config(n_cycles=2)).run()
        assert isinstance(result, PINNCycleResult)

    def test_final_state_type(self, heat_evaluator: Evaluator) -> None:
        result = _build_runner(heat_evaluator, _fast_config(n_cycles=1)).run()
        assert isinstance(result.final_state, EngineState)
        assert result.final_state.best_reward >= 0.0

    def test_pretrain_result_type(self, heat_evaluator: Evaluator) -> None:
        result = _build_runner(heat_evaluator, _fast_config(n_cycles=1)).run()
        assert isinstance(result.pretrain_result, PretrainResult)

        assert result.pretrain_result.epochs_run > 0

    def test_cycle_metrics_count(self, heat_evaluator: Evaluator) -> None:
        result = _build_runner(heat_evaluator, _fast_config(n_cycles=2)).run()
        assert len(result.cycle_metrics) == 2

    def test_metrics_have_best_reward(self, heat_evaluator: Evaluator) -> None:
        result = _build_runner(heat_evaluator, _fast_config(n_cycles=1)).run()
        for m in result.cycle_metrics:
            assert "best_reward" in m
            assert isinstance(m["best_reward"], float)
            assert math.isfinite(m["best_reward"])

    def test_metrics_have_pinn_losses(self) -> None:
        result = _build_runner(
            _AlwaysValidHeatEvaluator(),
            _fast_config(n_cycles=1),
        ).run()
        assert len(result.cycle_metrics) == 1
        for m in result.cycle_metrics:
            assert "data_loss" in m, f"PINN training did not run: {m}"
            assert math.isfinite(m["data_loss"])
            assert "physics_loss" in m
            assert math.isfinite(m["physics_loss"])







class TestNoValidExpression:

    def test_warns_on_skip(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = _build_runner(_AlwaysInvalidEvaluator(), _fast_config(n_cycles=1))
        with caplog.at_level(logging.WARNING):
            runner.run()
        assert any(
            "skip" in r.message.lower() or "no valid" in r.message.lower()
            for r in caplog.records
        ), f"Expected skip/no-valid warning; got {[r.message for r in caplog.records]}"

    def test_completes_with_metrics(self) -> None:
        result = _build_runner(
            _AlwaysInvalidEvaluator(),
            _fast_config(n_cycles=2),
        ).run()
        assert isinstance(result, PINNCycleResult)
        assert len(result.cycle_metrics) == 2

    def test_pinn_losses_absent_when_skipped(self) -> None:
        result = _build_runner(
            _AlwaysInvalidEvaluator(),
            _fast_config(n_cycles=1),
        ).run()
        for m in result.cycle_metrics:
            assert "data_loss" not in m, "data_loss present despite no valid expression"







class TestNoCycles:

    def test_returns_result(self, heat_evaluator: Evaluator) -> None:
        result = _build_runner(heat_evaluator, _fast_config(n_cycles=0)).run()
        assert isinstance(result, PINNCycleResult)

    def test_empty_cycle_metrics(self, heat_evaluator: Evaluator) -> None:
        result = _build_runner(heat_evaluator, _fast_config(n_cycles=0)).run()
        assert len(result.cycle_metrics) == 0

    def test_pretrain_still_runs(self, heat_evaluator: Evaluator) -> None:
        result = _build_runner(heat_evaluator, _fast_config(n_cycles=0)).run()
        assert isinstance(result.pretrain_result, PretrainResult)







class TestPlannedSearchIterations:

    def test_default_counts_the_final_search(self, heat_evaluator: Evaluator) -> None:

        runner = _build_runner(heat_evaluator, _fast_config(n_cycles=2))

        assert runner.planned_search_iterations() == 3 * _N_ITER

    def test_honors_cycle_n_iterations_override(
        self, heat_evaluator: Evaluator
    ) -> None:




        runner = _build_runner(
            heat_evaluator, _fast_config(n_cycles=2, cycle_n_iterations=7)
        )
        assert runner.planned_search_iterations() == _N_ITER + 2 * 7

    def test_n_cycles_zero_is_single_final_search(
        self, heat_evaluator: Evaluator
    ) -> None:
        runner = _build_runner(heat_evaluator, _fast_config(n_cycles=0))
        assert runner.planned_search_iterations() == _N_ITER







class TestConfigValidation:

    def test_requires_pinn_config(self, heat_evaluator: Evaluator) -> None:
        config_no_pinn = DiscoverConfig(pinn=None)
        torch.manual_seed(SEED)
        engine = build_engine(config_no_pinn)
        model = PINNModel(["x", "t"], ["u"], PINNConfig())
        with pytest.raises((ValueError, TypeError)):
            PINNCycleRunner(
                engine=engine,
                pinn_model=model,
                pinn_executor=PINNExecutor(FunctionRegistry.create_default()),
                initial_evaluator=heat_evaluator,
                observation_coords={"x": torch.zeros(5), "t": torch.zeros(5)},
                observation_targets={"u": torch.zeros(5)},
                colloc_coords={"x": torch.zeros(5), "t": torch.zeros(5)},
                dataset_metadata=make_pinn_dataset(
                    ["x", "t"],
                    ["u"],
                    lhs_field="u",
                    lhs_axis="t",
                ),
                config=config_no_pinn,
            )







class TestSplitObsData:

    def test_importable(self) -> None:
        from kd.search.discover.pinn.cycle import _split_obs_data

        assert callable(_split_obs_data)

    def test_preserves_total_count(self) -> None:
        from kd.search.discover.pinn.cycle import _split_obs_data

        data = {"x": torch.randn(100), "t": torch.randn(100)}
        train, val = _split_obs_data(data, val_ratio=0.2, seed=42)
        assert train["x"].shape[0] + val["x"].shape[0] == 100

    def test_same_indices_across_keys(self) -> None:
        from kd.search.discover.pinn.cycle import _split_obs_data

        n = 50

        data = {
            "x": torch.arange(n, dtype=torch.float32),
            "t": torch.arange(n, dtype=torch.float32) * 10,
        }
        train, val = _split_obs_data(data, val_ratio=0.2, seed=7)

        for i in range(train["x"].shape[0]):
            assert train["t"][i].item() == train["x"][i].item() * 10
        for i in range(val["x"].shape[0]):
            assert val["t"][i].item() == val["x"][i].item() * 10

    def test_respects_val_ratio(self) -> None:
        from kd.search.discover.pinn.cycle import _split_obs_data

        data = {"x": torch.randn(100), "t": torch.randn(100)}
        train, val = _split_obs_data(data, val_ratio=0.3, seed=0)
        assert val["x"].shape[0] in (29, 30, 31)
        assert train["x"].shape[0] in (69, 70, 71)

    def test_val_ratio_zero_gives_all_train(self) -> None:
        from kd.search.discover.pinn.cycle import _split_obs_data

        data = {"x": torch.randn(20), "t": torch.randn(20)}
        train, val = _split_obs_data(data, val_ratio=0.0, seed=0)
        assert train["x"].shape[0] == 20
        assert val["x"].shape[0] == 0

    def test_deterministic_with_same_seed(self) -> None:
        from kd.search.discover.pinn.cycle import _split_obs_data

        data = {"x": torch.randn(80)}
        t1, v1 = _split_obs_data(data, val_ratio=0.2, seed=99)
        t2, v2 = _split_obs_data(data, val_ratio=0.2, seed=99)
        assert torch.equal(t1["x"], t2["x"])
        assert torch.equal(v1["x"], v2["x"])


class TestPretrainSplit:

    def test_pretrain_receives_different_train_val(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import kd.search.discover.pinn.model as model_module

        captured: dict[str, Any] = {}
        original_pretrain = model_module.PINNModel.pretrain

        def spy_pretrain(
            self_model: Any,
            coords: Any,
            targets: Any,
            val_coords: Any,
            val_targets: Any,
            config: Any,
        ) -> Any:
            captured["coords"] = coords
            captured["val_coords"] = val_coords
            captured["targets"] = targets
            captured["val_targets"] = val_targets
            return original_pretrain(
                self_model,
                coords,
                targets,
                val_coords,
                val_targets,
                config,
            )

        monkeypatch.setattr(model_module.PINNModel, "pretrain", spy_pretrain)
        runner = _build_runner(heat_evaluator, _fast_config(n_cycles=0))
        runner.run()



        assert captured["coords"] is not captured["val_coords"]
        assert captured["targets"] is not captured["val_targets"]

        train_n = captured["coords"]["x"].shape[0]
        val_n = captured["val_coords"]["x"].shape[0]
        assert train_n + val_n == _N_OBS
        assert val_n > 0
        assert train_n > val_n

    def test_train_pinn_still_sees_full_obs(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import kd.search.discover.pinn.model as model_module

        captured_n: list[int] = []
        original_train_pinn = model_module.PINNModel.train_pinn

        def spy_train_pinn(self_model: Any, **kwargs: Any) -> Any:
            captured_n.append(kwargs["observation_coords"]["x"].shape[0])
            return original_train_pinn(self_model, **kwargs)

        monkeypatch.setattr(
            model_module.PINNModel,
            "train_pinn",
            spy_train_pinn,
        )
        runner = _build_runner(heat_evaluator, _fast_config(n_cycles=1))
        runner.run()


        for n in captured_n:
            assert n == _N_OBS

    def test_pretrain_val_ratio_config_respected(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import kd.search.discover.pinn.model as model_module

        captured: dict[str, int] = {}
        original_pretrain = model_module.PINNModel.pretrain

        def spy_pretrain(
            self_model: Any,
            coords: Any,
            targets: Any,
            val_coords: Any,
            val_targets: Any,
            config: Any,
        ) -> Any:
            captured["train_n"] = coords["x"].shape[0]
            captured["val_n"] = val_coords["x"].shape[0]
            return original_pretrain(
                self_model,
                coords,
                targets,
                val_coords,
                val_targets,
                config,
            )

        monkeypatch.setattr(model_module.PINNModel, "pretrain", spy_pretrain)

        config = _fast_config(n_cycles=0)
        assert config.pinn is not None

        from dataclasses import replace

        new_pinn = replace(config.pinn, pretrain_val_ratio=0.3)
        config = replace(config, pinn=new_pinn)

        runner = _build_runner(heat_evaluator, config)
        runner.run()


        assert captured["val_n"] in (14, 15, 16)
        assert captured["train_n"] + captured["val_n"] == _N_OBS


class TestPretrainSplitSeed:

    def _capture_permutation(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
        pretrain_split_seed: int | None,
    ) -> Tensor:
        import kd.search.discover.pinn.cycle as cycle_module
        import kd.search.discover.pinn.model as model_module

        permutations: list[Tensor] = []
        real_randperm = torch.randperm

        def spy_randperm(n: int, *args: Any, **kwargs: Any) -> Tensor:
            perm = real_randperm(n, *args, **kwargs)

            if n == _N_OBS:
                permutations.append(perm.detach().clone())
            return perm



        monkeypatch.setattr(
            cycle_module.torch,
            "randperm",
            spy_randperm,
        )


        original_pretrain = model_module.PINNModel.pretrain

        def fast_pretrain(
            self_model: Any,
            coords: Any,
            targets: Any,
            val_coords: Any,
            val_targets: Any,
            config: Any,
        ) -> Any:
            return PretrainResult(
                train_loss=0.0,
                val_loss=0.0,
                epochs_run=1,
                stopped_early=False,
            )

        monkeypatch.setattr(model_module.PINNModel, "pretrain", fast_pretrain)

        kwargs: dict[str, Any] = {}
        if pretrain_split_seed is not None:
            kwargs["pretrain_split_seed"] = pretrain_split_seed
        runner = _build_runner(heat_evaluator, _fast_config(n_cycles=0), **kwargs)
        runner._pretrain()


        monkeypatch.setattr(
            model_module.PINNModel,
            "pretrain",
            original_pretrain,
        )
        assert permutations, "spy never saw a randperm(N_OBS) call"
        return permutations[0]

    def test_pretrain_split_seed_default_preserves_zero(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        default_perm = self._capture_permutation(
            heat_evaluator,
            monkeypatch,
            pretrain_split_seed=None,
        )
        explicit_zero = self._capture_permutation(
            heat_evaluator,
            monkeypatch,
            pretrain_split_seed=0,
        )
        assert torch.equal(default_perm, explicit_zero)

    def test_pretrain_split_seed_changes_partition(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        perm_zero = self._capture_permutation(
            heat_evaluator,
            monkeypatch,
            pretrain_split_seed=0,
        )
        perm_one = self._capture_permutation(
            heat_evaluator,
            monkeypatch,
            pretrain_split_seed=1,
        )
        assert not torch.equal(perm_zero, perm_one)

    def test_pretrain_split_seed_same_value_bit_exact(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        perm_a = self._capture_permutation(
            heat_evaluator,
            monkeypatch,
            pretrain_split_seed=42,
        )
        perm_b = self._capture_permutation(
            heat_evaluator,
            monkeypatch,
            pretrain_split_seed=42,
        )
        assert torch.equal(perm_a, perm_b)


class TestDeriveCycleSeed:

    def test_deterministic_same_inputs(self) -> None:
        from kd.search.discover.pinn.cycle import _derive_cycle_seed

        seed_a = _derive_cycle_seed(42, 0, b"local_sample")
        seed_b = _derive_cycle_seed(42, 0, b"local_sample")
        assert seed_a == seed_b

    def test_cycle_idx_differs(self) -> None:
        from kd.search.discover.pinn.cycle import _derive_cycle_seed

        assert _derive_cycle_seed(42, 0, b"local_sample") != _derive_cycle_seed(
            42,
            1,
            b"local_sample",
        )

    def test_domain_separation(self) -> None:
        from kd.search.discover.pinn.cycle import _derive_cycle_seed

        assert _derive_cycle_seed(42, 0, b"local_sample") != _derive_cycle_seed(
            42,
            0,
            b"other_sub_task",
        )

    def test_collision_avoided_vs_naive_add(self) -> None:
        from kd.search.discover.pinn.cycle import _derive_cycle_seed

        assert _derive_cycle_seed(42, 1, b"local_sample") != _derive_cycle_seed(
            43,
            0,
            b"local_sample",
        )

    def test_returns_nonnegative_32bit_int(self) -> None:
        from kd.search.discover.pinn.cycle import _derive_cycle_seed

        for base in (0, 1, 42, 10_000_000):
            for cycle_idx in (0, 1, 5):
                seed = _derive_cycle_seed(base, cycle_idx, b"local_sample")
                assert 0 <= seed < 2**32


class TestLocalSampleSeed:

    def _capture_local_seeds(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
        local_sample_seed: int | None,
        *,
        n_cycles: int = 1,
    ) -> list[int | None]:
        import kd.search.discover.pinn.cycle as cycle_module
        import kd.search.discover.pinn.model as model_module

        captured_seeds: list[int | None] = []



        def fake_generate_local_samples(**kwargs: Any) -> dict[str, Tensor]:
            captured_seeds.append(kwargs.get("seed"))
            obs = kwargs["observation_coords"]
            return {k: v.detach().clone() for k, v in obs.items()}




        monkeypatch.setattr(
            "kd.search.discover.pinn.collocation.generate_local_samples",
            fake_generate_local_samples,
        )

        original_train_pinn = model_module.PINNModel.train_pinn


        def fast_train_pinn(self_model: Any, **kwargs: Any) -> Any:
            return original_train_pinn(self_model, **kwargs)

        config = _fast_config(n_cycles=n_cycles)
        assert config.pinn is not None

        from dataclasses import replace

        new_pinn = replace(config.pinn, local_sample=True, local_multiplier=4)
        config = replace(config, pinn=new_pinn)

        runner = _build_runner(
            heat_evaluator,
            config,
            local_sample_seed=local_sample_seed,
        )
        _ = cycle_module
        runner.run()
        return captured_seeds

    def test_local_sample_seed_none_passes_none_through(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seeds = self._capture_local_seeds(
            heat_evaluator,
            monkeypatch,
            local_sample_seed=None,
        )

        assert seeds, "generate_local_samples was never invoked"
        assert all(s is None for s in seeds)

    def test_local_sample_seed_derives_via_blake2b(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.pinn.cycle import _derive_cycle_seed

        seeds = self._capture_local_seeds(
            heat_evaluator,
            monkeypatch,
            local_sample_seed=7,
            n_cycles=2,
        )
        expected = [
            _derive_cycle_seed(7, cycle_idx, b"local_sample")
            for cycle_idx in range(len(seeds))
        ]
        assert seeds == expected

    def test_run_pinn_phase_threads_cycle_idx(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_cycle_idx: list[int] = []
        from kd.search.discover.pinn import cycle as cycle_module

        original_make = cycle_module.PINNCycleRunner._make_local_coords

        def spy_make(
            self_runner: Any,
            cycle_idx: int,
        ) -> dict[str, Tensor] | None:
            captured_cycle_idx.append(cycle_idx)
            return original_make(self_runner, cycle_idx)

        monkeypatch.setattr(
            cycle_module.PINNCycleRunner,
            "_make_local_coords",
            spy_make,
        )

        runner = _build_runner(
            heat_evaluator,
            _fast_config(n_cycles=2),
            local_sample_seed=42,
        )
        runner.run()




        assert captured_cycle_idx == [0, 1], (
            f"cycle_idx was not threaded: observed {captured_cycle_idx}"
        )


class TestSeedPlanArtifact:

    def test_seed_plan_default_values_in_extras(
        self,
        heat_evaluator: Evaluator,
    ) -> None:
        result = _build_runner(heat_evaluator, _fast_config(n_cycles=0)).run()
        extras = result.final_state.extras or {}
        assert "seed_plan" in extras
        plan = extras["seed_plan"]
        assert plan == {
            "pretrain_split_seed": 0,
            "local_sample_seed": None,
            "stability_seed": None,
        }

    def test_seed_plan_records_explicit_values(
        self,
        heat_evaluator: Evaluator,
    ) -> None:
        runner = _build_runner(
            heat_evaluator,
            _fast_config(n_cycles=0),
            stability_seed=11,
            pretrain_split_seed=22,
            local_sample_seed=33,
        )
        result = runner.run()
        plan = (result.final_state.extras or {}).get("seed_plan")
        assert plan == {
            "pretrain_split_seed": 22,
            "local_sample_seed": 33,
            "stability_seed": 11,
        }

    def test_seed_plan_preserves_existing_extras(
        self,
        heat_evaluator: Evaluator,
    ) -> None:



        from dataclasses import replace

        config = _fast_config(n_cycles=1)
        config = replace(config, stability_selection=3)
        runner = _build_runner(heat_evaluator, config, stability_seed=42)
        result = runner.run()

        extras = result.final_state.extras or {}

        assert "seed_plan" in extras
        assert "stability_selection" in extras

    def test_observability_path_also_attaches_seed_plan(
        self, heat_evaluator: Evaluator, tmp_path: Path
    ) -> None:
        from kd.search.discover.runners.pinn_cycle_observability import (
            run_pinn_cycle_with_observability,
        )

        runner = _build_runner(
            heat_evaluator,
            _fast_config(n_cycles=0),
            stability_seed=11,
            pretrain_split_seed=22,
            local_sample_seed=33,
        )
        result = run_pinn_cycle_with_observability(
            runner, checkpoint_dir=tmp_path / "ckpt"
        )
        plan = (result.final_state.extras or {}).get("seed_plan")
        assert plan == {
            "pretrain_split_seed": 22,
            "local_sample_seed": 33,
            "stability_seed": 11,
        }


class TestRebuildFromEvaluator:

    def test_rebuild_after_pinn_reuses_initial_evaluator_executor(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import kd.search.discover.pinn.cycle as cycle_module

        runner = _build_runner(heat_evaluator, _fast_config())

        captured: dict[str, object] = {}
        sentinel = object()

        def fake_rebuild(
            regen_data: object,
            executor: object,
            solver: object,
        ) -> object:
            captured["executor"] = executor
            captured["solver"] = solver
            return sentinel

        monkeypatch.setattr(cycle_module, "rebuild_evaluator", fake_rebuild)
        result = runner._rebuild_after_pinn()
        assert result is sentinel

        assert captured["executor"] is heat_evaluator.executor
        assert captured["solver"] is heat_evaluator.solver

    def test_rebuild_after_pinn_keeps_protocol_only_evaluator(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import kd.search.discover.pinn.cycle as cycle_module

        invalid_evaluator = _AlwaysInvalidEvaluator()
        runner = _build_runner(invalid_evaluator, _fast_config())

        def fail_rebuild(*args: object, **kwargs: object) -> object:
            raise AssertionError("rebuild_evaluator should not run for test stubs")

        monkeypatch.setattr(cycle_module, "rebuild_evaluator", fail_rebuild)

        rebuilt = runner._rebuild_after_pinn()





        assert cast(object, rebuilt) is invalid_evaluator

    def test_runner_stores_no_separate_rebuild_fields(
        self,
        heat_evaluator: Evaluator,
    ) -> None:
        runner = _build_runner(heat_evaluator, _fast_config())
        assert not hasattr(runner, "_rebuild_executor")
        assert not hasattr(runner, "_rebuild_solver")

    def test_extract_executor_helper_removed(self) -> None:
        import kd.search.discover.pinn.cycle as cycle_module

        assert not hasattr(cycle_module, "_extract_executor")

    def test_rebuild_executor_kwarg_rejected(
        self,
        heat_evaluator: Evaluator,
    ) -> None:
        config = _fast_config()
        torch.manual_seed(SEED)
        engine = build_engine(config)
        assert config.pinn is not None
        model = PINNModel(["x", "t"], ["u"], config.pinn)
        with pytest.raises(TypeError, match="rebuild_executor"):
            PINNCycleRunner(
                engine=engine,
                pinn_model=model,
                pinn_executor=PINNExecutor(FunctionRegistry.create_default()),
                initial_evaluator=heat_evaluator,
                observation_coords={"x": torch.zeros(5), "t": torch.zeros(5)},
                observation_targets={"u": torch.zeros(5)},
                colloc_coords={"x": torch.zeros(5), "t": torch.zeros(5)},
                dataset_metadata=make_pinn_dataset(
                    ["x", "t"],
                    ["u"],
                    lhs_field="u",
                    lhs_axis="t",
                ),
                config=config,
                rebuild_executor=PythonExecutor(FunctionRegistry.create_default()),
            )







class TestTrainPinnRollback:

    @pytest.mark.unit
    def test_model_restored_on_exception(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.pinn import model as model_module

        config = _fast_config(n_cycles=1)
        runner = _build_runner(heat_evaluator, config)


        runner._pretrain()

        runner._engine._best_expression = "diff2_x(u)"
        runner._engine._best_reward = 0.5


        pre_weights = {k: v.clone() for k, v in runner._pinn_model.state_dict().items()}


        def _corrupting_train_pinn(self_model: Any, **kwargs: Any) -> Any:

            with torch.no_grad():
                for p in self_model.parameters():
                    p.add_(torch.randn_like(p) * 10.0)
            raise _CorruptingTrainPinnError("simulated PINN failure")

        monkeypatch.setattr(
            model_module.PINNModel,
            "train_pinn",
            _corrupting_train_pinn,
        )


        evaluator = _build_evaluator(_make_heat_dataset())
        _metrics, pinn_ok = runner._run_pinn_phase(0, evaluator)

        assert pinn_ok is False, "Expected pinn_ok=False on exception"


        post_weights = runner._pinn_model.state_dict()
        for key in pre_weights:
            assert torch.equal(pre_weights[key], post_weights[key]), (
                f"Weight '{key}' not restored after train_pinn exception"
            )

    @pytest.mark.unit
    def test_exception_returns_skip_metrics(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.pinn import model as model_module

        config = _fast_config(n_cycles=1)
        runner = _build_runner(heat_evaluator, config)
        runner._pretrain()
        runner._engine._best_expression = "diff2_x(u)"
        runner._engine._best_reward = 0.5

        def _raising_train_pinn(self_model: Any, **kwargs: Any) -> Any:
            raise KeyError("unknown_symbol")

        monkeypatch.setattr(
            model_module.PINNModel,
            "train_pinn",
            _raising_train_pinn,
        )

        evaluator = _build_evaluator(_make_heat_dataset())
        metrics, pinn_ok = runner._run_pinn_phase(0, evaluator)

        assert pinn_ok is False
        assert "best_reward" in metrics

    @pytest.mark.unit
    def test_successful_train_updates_model(
        self,
        heat_evaluator: Evaluator,
    ) -> None:
        config = _fast_config(n_cycles=1)
        runner = _build_runner(heat_evaluator, config)
        runner._pretrain()


        evaluator = _build_evaluator(_make_heat_dataset())
        runner._engine.run_cycle(evaluator, _N_ITER)

        if not runner._engine.best_expression:
            pytest.skip("Engine found no valid expression in short run")

        pre_weights = {k: v.clone() for k, v in runner._pinn_model.state_dict().items()}

        _metrics, pinn_ok = runner._run_pinn_phase(0, evaluator)

        if not pinn_ok:
            pytest.skip("PINN training skipped (expression incompatible)")


        post_weights = runner._pinn_model.state_dict()
        any_changed = any(
            not torch.equal(pre_weights[k], post_weights[k]) for k in pre_weights
        )
        assert any_changed, (
            "No weights changed after successful train_pinn — "
            "rollback may be too aggressive"
        )







class TestTD060PINNConfigLocalMultiplier:

    @pytest.mark.unit
    def test_default_local_multiplier(self) -> None:
        cfg = PINNConfig()
        assert cfg.local_multiplier == 20

    @pytest.mark.unit
    def test_custom_local_multiplier(self) -> None:
        cfg = PINNConfig(local_multiplier=50)
        assert cfg.local_multiplier == 50


class TestTD060DomainBounds:

    @pytest.mark.unit
    def test_domain_bounds_accepted(self, heat_evaluator: Evaluator) -> None:
        config = _fast_config(n_cycles=1)
        assert config.pinn is not None
        torch.manual_seed(SEED)
        engine = build_engine(config)
        model = PINNModel(["x", "t"], ["u"], config.pinn)
        executor = PINNExecutor(FunctionRegistry.create_default())
        dataset_meta = make_pinn_dataset(
            ["x", "t"],
            ["u"],
            lhs_field="u",
            lhs_axis="t",
        )

        rng = torch.Generator().manual_seed(SEED)
        obs_coords = {
            "x": torch.rand(_N_OBS, generator=rng).float(),
            "t": torch.rand(_N_OBS, generator=rng).float(),
        }
        obs_targets = {"u": _heat_solution(obs_coords["x"], obs_coords["t"])}

        colloc = {
            "x": torch.rand(_N_COLLOC).float(),
            "t": torch.rand(_N_COLLOC).float(),
        }


        runner = PINNCycleRunner(
            engine=engine,
            pinn_model=model,
            pinn_executor=executor,
            initial_evaluator=heat_evaluator,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc,
            dataset_metadata=dataset_meta,
            config=config,
            domain_bounds={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        )
        assert runner._local_bounds == {"x": (0.0, 1.0), "t": (0.0, 1.0)}

    @pytest.mark.unit
    def test_domain_bounds_none_falls_back(
        self,
        heat_evaluator: Evaluator,
    ) -> None:
        config = _fast_config(n_cycles=1)
        runner = _build_runner(heat_evaluator, config)

        assert hasattr(runner, "_local_bounds")
        assert len(runner._local_bounds) == 2
        for name, (lo, hi) in runner._local_bounds.items():
            assert lo < hi, f"Inferred bounds for '{name}' are degenerate"

    @pytest.mark.unit
    def test_domain_bounds_none_expands_degenerate_axis(
        self,
        heat_evaluator: Evaluator,
    ) -> None:
        config = _fast_config(n_cycles=1)
        colloc = {
            "x": torch.full((_N_COLLOC,), 0.5, dtype=torch.float32),
            "t": torch.linspace(0.2, 0.8, _N_COLLOC, dtype=torch.float32),
        }
        runner = _build_runner(
            heat_evaluator,
            config,
            colloc_coords=colloc,
        )

        x_lo, x_hi = runner._local_bounds["x"]
        assert x_lo < x_hi
        assert math.isclose((x_lo + x_hi) / 2.0, 0.5, rel_tol=0.0, abs_tol=1e-6)

        local_coords = runner._make_local_coords(cycle_idx=0)

        assert local_coords is not None
        assert local_coords["x"].numel() > 0
        assert torch.isfinite(local_coords["x"]).all()


class TestTD060LocalMultiplierThreading:

    @pytest.mark.unit
    def test_make_local_coords_uses_config_multiplier(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.pinn import collocation as colloc_mod

        config = _fast_config(n_cycles=1)
        runner = _build_runner(heat_evaluator, config)

        captured_kwargs: dict[str, Any] = {}
        original_fn = colloc_mod.generate_local_samples

        def _spy(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return original_fn(**kwargs)

        monkeypatch.setattr(colloc_mod, "generate_local_samples", _spy)
        result = runner._make_local_coords(cycle_idx=0)

        assert result is not None, "local_sample should be True in config"
        assert "multiplier" in captured_kwargs, (
            "_make_local_coords must pass multiplier to generate_local_samples"
        )
        assert config.pinn is not None
        assert captured_kwargs["multiplier"] == config.pinn.local_multiplier

    @pytest.mark.unit
    def test_zero_local_multiplier_rejected_when_local_sampling_enabled(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="local_sample=True requires local_multiplier > 0",
        ):
            PINNConfig(local_sample=True, local_multiplier=0)

    @pytest.mark.unit
    def test_zero_local_multiplier_allowed_when_local_sampling_disabled(
        self,
    ) -> None:
        config = PINNConfig(local_sample=False, local_multiplier=0)

        assert config.local_sample is False
        assert config.local_multiplier == 0







class TestLhsFieldThreading:

    @pytest.mark.unit
    def test_compute_residual_receives_lhs_from_metadata(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.pinn import model as model_mod

        captured: dict[str, str] = {}
        original_back = model_mod._chunked_backward_residual_loss
        original_eval = model_mod._chunked_eval_residual_loss

        def _spy_back(*args: Any, **kwargs: Any) -> Any:
            ds = kwargs.get("dataset_metadata")
            if ds is None and len(args) > 5:
                ds = args[5]
            if ds is not None:
                captured["lhs_field"] = ds.lhs_field
                captured["lhs_axis"] = ds.lhs_axis
            return original_back(*args, **kwargs)

        def _spy_eval(*args: Any, **kwargs: Any) -> Any:
            ds = kwargs.get("dataset_metadata")
            if ds is None and len(args) > 5:
                ds = args[5]
            if ds is not None:
                captured["lhs_field"] = ds.lhs_field
                captured["lhs_axis"] = ds.lhs_axis
            return original_eval(*args, **kwargs)

        monkeypatch.setattr(model_mod, "_chunked_backward_residual_loss", _spy_back)
        monkeypatch.setattr(model_mod, "_chunked_eval_residual_loss", _spy_eval)
        runner = _build_runner(heat_evaluator, _fast_config(n_cycles=1))
        runner.run()
        assert captured.get("lhs_field") == "u"
        assert captured.get("lhs_axis") == "t"

    @pytest.mark.unit
    def test_dataset_metadata_lhs_fields_accessible(self) -> None:
        meta = make_pinn_dataset(
            axis_names=["x", "t"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        assert meta.lhs_field == "u"
        assert meta.lhs_axis == "t"

        meta2 = make_pinn_dataset(
            axis_names=["x", "t"],
            field_names=["v"],
            lhs_field="v",
            lhs_axis="x",
        )
        assert meta2.lhs_field == "v"
        assert meta2.lhs_axis == "x"

    @pytest.mark.unit
    def test_rebuild_passes_lhs_to_regenerate_metadata(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.pinn import cycle as cycle_mod

        captured_kwargs: list[dict[str, Any]] = []
        original_regen = cycle_mod.regenerate_metadata

        def _spy(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return original_regen(*args, **kwargs)

        monkeypatch.setattr(cycle_mod, "regenerate_metadata", _spy)
        runner = _build_runner(heat_evaluator, _fast_config(n_cycles=2))
        runner.run()



        assert captured_kwargs, (
            "regenerate_metadata was never called — engine found no valid "
            "expression in 2 cycles; increase _N_ITER or seed if this flakes"
        )
        assert any("lhs_field" in kw for kw in captured_kwargs), (
            "_rebuild_after_pinn must pass lhs_field to regenerate_metadata"
        )
        assert any("lhs_axis" in kw for kw in captured_kwargs), (
            "_rebuild_after_pinn must pass lhs_axis to regenerate_metadata"
        )







class TestMetadataConsistency:

    @pytest.mark.unit
    def test_matching_metadata_passes(self) -> None:
        from kd.search.discover.pinn.cycle import _validate_metadata_consistency

        source = make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="omega",
            lhs_axis="t",
        )
        pinn = make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="omega",
            lhs_axis="t",
        )

        _validate_metadata_consistency(source, pinn)

    @pytest.mark.unit
    def test_lhs_field_mismatch_raises(self) -> None:
        from kd.search.discover.pinn.cycle import _validate_metadata_consistency

        source = make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="omega",
            lhs_axis="t",
        )
        pinn = make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="u",
            lhs_axis="t",
        )
        with pytest.raises(ValueError, match="lhs_field mismatch"):
            _validate_metadata_consistency(source, pinn)

    @pytest.mark.unit
    def test_lhs_axis_mismatch_raises(self) -> None:
        from kd.search.discover.pinn.cycle import _validate_metadata_consistency

        source = make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="omega",
            lhs_axis="t",
        )
        pinn = make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="omega",
            lhs_axis="x",
        )
        with pytest.raises(ValueError, match="lhs_axis mismatch"):
            _validate_metadata_consistency(source, pinn)

    @pytest.mark.unit
    def test_axis_order_mismatch_raises(self) -> None:
        from kd.search.discover.pinn.cycle import _validate_metadata_consistency

        source = make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="omega",
            lhs_axis="t",
        )
        pinn = make_pinn_dataset(
            axis_names=["x", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="omega",
            lhs_axis="t",
        )
        with pytest.raises(ValueError, match="axis_order mismatch"):
            _validate_metadata_consistency(source, pinn)

    @pytest.mark.unit
    def test_runner_has_source_dataset_kwarg(self) -> None:
        import inspect

        sig = inspect.signature(PINNCycleRunner.__init__)
        assert "source_dataset" in sig.parameters
        assert sig.parameters["source_dataset"].default is None







class TestCycleIterationReduction:

    @pytest.mark.unit
    def test_cycle_n_iterations_field_exists(self) -> None:
        cfg = PINNConfig()
        assert cfg.cycle_n_iterations is None

    @pytest.mark.unit
    def test_cycle_n_iterations_accepts_int(self) -> None:
        cfg = PINNConfig(cycle_n_iterations=10)
        assert cfg.cycle_n_iterations == 10

    @pytest.mark.unit
    def test_cycle_n_iterations_validation(self) -> None:
        with pytest.raises(ValueError, match="cycle_n_iterations"):
            PINNConfig(cycle_n_iterations=0)
        with pytest.raises(ValueError, match="cycle_n_iterations"):
            PINNConfig(cycle_n_iterations=-1)

    @pytest.mark.unit
    def test_none_preserves_current_behavior(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.engine import DiscoverEngine

        iter_counts: list[int] = []
        original_run_cycle = DiscoverEngine.run_cycle

        def _spy(
            self_eng: Any,
            evaluator: Any,
            n_iterations: int = 1000,
            **kwargs: Any,
        ) -> Any:
            iter_counts.append(n_iterations)
            return original_run_cycle(
                self_eng,
                evaluator,
                n_iterations,
                **kwargs,
            )

        monkeypatch.setattr(DiscoverEngine, "run_cycle", _spy)
        runner = _build_runner(heat_evaluator, _fast_config(n_cycles=2))
        runner.run()

        assert all(c == _N_ITER for c in iter_counts), (
            f"Expected all iterations={_N_ITER}, got {iter_counts}"
        )

    @pytest.mark.unit
    def test_reduced_iterations_for_cycle_2_plus(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.engine import DiscoverEngine

        iter_counts: list[int] = []
        original_run_cycle = DiscoverEngine.run_cycle

        def _spy(
            self_eng: Any,
            evaluator: Any,
            n_iterations: int = 1000,
            **kwargs: Any,
        ) -> Any:
            iter_counts.append(n_iterations)
            return original_run_cycle(
                self_eng,
                evaluator,
                n_iterations,
                **kwargs,
            )

        monkeypatch.setattr(DiscoverEngine, "run_cycle", _spy)
        config = _fast_config(n_cycles=2)
        assert config.pinn is not None

        config = DiscoverConfig(
            **{
                **{
                    f.name: getattr(config, f.name)
                    for f in dataclass_fields(config)
                    if f.name != "pinn"
                },
                "pinn": PINNConfig(
                    **{
                        **{
                            f.name: getattr(config.pinn, f.name)
                            for f in dataclass_fields(config.pinn)
                        },
                        "cycle_n_iterations": 3,
                    },
                ),
            },
        )
        runner = _build_runner(heat_evaluator, config)
        runner.run()


        assert len(iter_counts) == 3, (
            f"Expected 3 run_cycle calls, got {len(iter_counts)}"
        )
        assert iter_counts[0] == _N_ITER, (
            f"Cycle 1 should use full iterations ({_N_ITER}), got {iter_counts[0]}"
        )
        assert iter_counts[1] == 3, (
            f"Cycle 2 should use reduced iterations (3), got {iter_counts[1]}"
        )
        assert iter_counts[2] == 3, (
            f"Final search should use reduced iterations (3), got {iter_counts[2]}"
        )







class TestStabilitySelectionIntegration:

    def _patched_runner(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
        config: DiscoverConfig,
        result_map: dict[str, EvaluationResult],
        stability_seed: int | None = None,
    ) -> PINNCycleRunner:
        runner = _build_runner(
            heat_evaluator,
            config,
            stability_seed=stability_seed,
        )
        monkeypatch.setattr(
            runner,
            "_pretrain",
            lambda: PretrainResult(0.0, 0.0, 0, False),
        )
        fake_evaluator = _SelectionEvaluator(result_map)
        monkeypatch.setattr(runner, "_rebuild_after_pinn", lambda: fake_evaluator)
        monkeypatch.setattr(
            runner,
            "_run_pinn_phase",
            lambda cycle_idx, evaluator: ({"best_reward": 0.0}, False),
        )
        return runner

    @pytest.mark.unit
    def test_stability_selection_uses_final_cycle_candidates_only(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = dataclass_replace(
            _fast_config(n_cycles=2),
            stability_selection=2,
            stability_queue_capacity=4,
        )
        result_map = {
            "final_selected": EvaluationResult(
                mse=0.1,
                nmse=0.1,
                r2=0.9,
                complexity=1,
                is_valid=True,
                expression="final_selected",
                terms=["u_xx", "u"],
                coefficients=torch.tensor([1.0, 0.0]),
                selected_indices=[0],
            ),
        }
        runner = self._patched_runner(heat_evaluator, monkeypatch, config, result_map)
        base_state = runner._engine.state
        states = [
            dataclass_replace(base_state, best_expression="cycle0", best_reward=0.2),
            dataclass_replace(base_state, best_expression="cycle1", best_reward=0.3),
            dataclass_replace(
                base_state,
                best_expression="final_prefilter",
                best_reward=0.9,
            ),
        ]
        candidate_sets = [
            [_snapshot("cycle0", 0.2, 0.2, 1, ["u"])],
            [_snapshot("cycle1", 0.3, 0.3, 1, ["u_x"])],
            [
                _snapshot("final_prefilter", 0.9, 0.1, 3, ["u_xx", "u"]),
                _snapshot("final_selected", 0.5, 0.12, 2, ["u_xx", "u"]),
            ],
        ]
        call_index = {"value": 0}

        def _run_cycle(
            evaluator: Any,
            n_iterations: int = 1000,
            **kwargs: Any,
        ) -> EngineState:
            idx = call_index["value"]
            runner._engine._cycle_candidates._candidates = list(candidate_sets[idx])
            call_index["value"] += 1
            return states[idx]

        captured: dict[str, Any] = {}

        def _select(
            candidates: list[CandidateSnapshot],
            evaluator: Any,
            *,
            top_k: int,
            n_bootstrap: int = 100,
            n_inner_bootstrap: int = 10,
            rng: Any = None,
        ) -> StabilitySelectionResult:
            captured["expressions"] = [candidate.expression for candidate in candidates]
            captured["top_k"] = top_k
            return StabilitySelectionResult(
                selected=candidates[1],
                candidates=[
                    StabilityCandidateStats(
                        candidate=candidates[0],
                        mse=torch.zeros(3).numpy(),
                        cv=torch.ones(3).numpy(),
                        score=torch.ones(3).numpy(),
                    ),
                    StabilityCandidateStats(
                        candidate=candidates[1],
                        mse=torch.zeros(3).numpy(),
                        cv=torch.zeros(3).numpy(),
                        score=torch.zeros(3).numpy(),
                    ),
                ],
                vote_counts=[0, 3],
            )

        monkeypatch.setattr(runner._engine, "run_cycle", _run_cycle)
        monkeypatch.setattr("kd.search.discover.pinn.cycle.stability_select", _select)

        result = runner.run()

        assert captured["expressions"] == ["final_prefilter", "final_selected"]
        assert captured["top_k"] == 2
        assert result.final_state.best_expression == "final_selected"
        assert result.final_state.best_reward == pytest.approx(0.5)
        assert result.final_state.best_result_terms == ["u_xx"]
        assert result.final_state.best_result_coefficients == pytest.approx([1.0])
        extras = result.final_state.extras
        assert extras is not None
        stability = extras["stability_selection"]
        assert stability["ran"] is True
        assert stability["pre_filter"]["expression"] == "final_prefilter"
        assert stability["selected"]["expression"] == "final_selected"
        assert stability["vote_counts"] == [0, 3]

    @pytest.mark.unit
    def test_stability_selection_uses_seeded_rng_when_configured(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = dataclass_replace(
            _fast_config(n_cycles=1),
            stability_selection=2,
            stability_queue_capacity=4,
        )
        runner = self._patched_runner(
            heat_evaluator,
            monkeypatch,
            config,
            {},
            stability_seed=123,
        )
        base_state = runner._engine.state
        final_state = dataclass_replace(
            base_state,
            best_expression="prefilter",
            best_reward=0.7,
        )
        runner._engine._cycle_candidates._candidates = [
            _snapshot("prefilter", 0.7, 0.2, 1, ["u"]),
            _snapshot("other", 0.6, 0.3, 1, ["u_x"]),
        ]
        monkeypatch.setattr(
            runner._engine,
            "run_cycle",
            lambda evaluator, n_iterations=1000, **kwargs: final_state,
        )
        captured: dict[str, Any] = {}

        def _select(
            candidates: list[CandidateSnapshot],
            evaluator: Any,
            *,
            top_k: int,
            n_bootstrap: int = 100,
            n_inner_bootstrap: int = 10,
            rng: Any = None,
        ) -> StabilitySelectionResult:
            assert rng is not None
            captured["first_int"] = int(rng.integers(0, 1_000_000))
            return StabilitySelectionResult(
                selected=candidates[0],
                candidates=[],
                vote_counts=[3, 0],
            )

        monkeypatch.setattr("kd.search.discover.pinn.cycle.stability_select", _select)

        runner.run()

        assert captured["first_int"] == 15440

    @pytest.mark.unit
    def test_stability_selection_disabled_is_no_op(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _fast_config(n_cycles=1)
        runner = self._patched_runner(
            heat_evaluator,
            monkeypatch,
            config,
            {},
        )
        base_state = runner._engine.state
        final_state = dataclass_replace(
            base_state,
            best_expression="prefilter",
            best_reward=0.7,
        )
        runner._engine._cycle_candidates._candidates = [
            _snapshot("prefilter", 0.7, 0.2, 1, ["u"]),
            _snapshot("other", 0.5, 0.3, 1, ["u_x"]),
        ]
        monkeypatch.setattr(
            runner._engine,
            "run_cycle",
            lambda evaluator, n_iterations=1000, **kwargs: final_state,
        )
        monkeypatch.setattr(
            "kd.search.discover.pinn.cycle.stability_select",
            lambda *args, **kwargs: pytest.fail("stability_select should not run"),
        )

        result = runner.run()

        assert result.final_state.best_expression == "prefilter"
        assert result.final_state.best_reward == pytest.approx(0.7)





        extras_without_seed_plan = dict(result.final_state.extras or {})
        extras_without_seed_plan.pop("seed_plan", None)
        assert extras_without_seed_plan == (final_state.extras or {})

    @pytest.mark.unit
    def test_single_candidate_skips_filter_but_records_no_op(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = dataclass_replace(
            _fast_config(n_cycles=1),
            stability_selection=2,
            stability_queue_capacity=4,
        )
        runner = self._patched_runner(
            heat_evaluator,
            monkeypatch,
            config,
            {},
        )
        base_state = runner._engine.state
        final_state = dataclass_replace(
            base_state,
            best_expression="only_candidate",
            best_reward=0.6,
        )
        runner._engine._cycle_candidates._candidates = [
            _snapshot("only_candidate", 0.6, 0.2, 1, ["u"]),
        ]
        monkeypatch.setattr(
            runner._engine,
            "run_cycle",
            lambda evaluator, n_iterations=1000, **kwargs: final_state,
        )
        monkeypatch.setattr(
            "kd.search.discover.pinn.cycle.stability_select",
            lambda *args, **kwargs: pytest.fail("stability_select should not run"),
        )

        result = runner.run()

        assert result.final_state.best_expression == "only_candidate"
        assert result.final_state.best_reward == pytest.approx(0.6)
        extras = result.final_state.extras
        assert extras is not None
        stability = extras["stability_selection"]
        assert stability["ran"] is False
        assert stability["pre_filter"]["expression"] == "only_candidate"
        assert stability["selected"]["expression"] == "only_candidate"
        assert stability["vote_counts"] == []

    @pytest.mark.unit
    def test_invalid_selected_candidate_falls_back_to_prefilter_result(
        self,
        heat_evaluator: Evaluator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = dataclass_replace(
            _fast_config(n_cycles=1),
            stability_selection=2,
            stability_queue_capacity=4,
        )
        result_map = {
            "bad_selected": EvaluationResult(
                mse=1e10,
                nmse=1e10,
                r2=-float("inf"),
                complexity=0,
                is_valid=False,
                expression="bad_selected",
            ),
        }
        runner = self._patched_runner(heat_evaluator, monkeypatch, config, result_map)
        base_state = runner._engine.state
        final_state = dataclass_replace(
            base_state,
            best_expression="prefilter",
            best_reward=0.7,
            best_result_terms=["u_xx"],
            best_result_coefficients=[1.0],
        )
        runner._engine._cycle_candidates._candidates = [
            _snapshot("prefilter", 0.7, 0.2, 1, ["u"]),
            _snapshot("bad_selected", 0.9, 0.1, 2, ["u_xx", "u"]),
        ]
        monkeypatch.setattr(
            runner._engine,
            "run_cycle",
            lambda evaluator, n_iterations=1000, **kwargs: final_state,
        )
        monkeypatch.setattr(
            "kd.search.discover.pinn.cycle.stability_select",
            lambda *args, **kwargs: StabilitySelectionResult(
                selected=_snapshot("bad_selected", 0.9, 0.1, 2, ["u_xx", "u"]),
                candidates=[],
                vote_counts=[0, 3],
            ),
        )

        result = runner.run()

        assert result.final_state.best_expression == "prefilter"
        assert result.final_state.best_reward == pytest.approx(0.7)
        assert result.final_state.best_result_terms == ["u_xx"]
        assert result.final_state.best_result_coefficients == pytest.approx([1.0])
        extras = result.final_state.extras
        assert extras is not None
        stability = extras["stability_selection"]
        assert stability["ran"] is False
        assert stability["error"] is not None
        assert stability["selected"]["expression"] == "prefilter"







class TestModelTensorSpec:

    @pytest.mark.unit
    def test_returns_device_and_dtype_for_normal_module(self) -> None:
        from torch import nn

        from kd.search.discover.pinn.cycle import _model_tensor_spec

        module = nn.Linear(3, 2)
        device, dtype = _model_tensor_spec(module)
        assert device.type in {"cpu", "cuda", "mps"}
        assert dtype == torch.float32

    @pytest.mark.unit
    def test_param_less_module_raises_value_error(self) -> None:
        from torch import nn

        from kd.search.discover.pinn.cycle import _model_tensor_spec

        class _Empty(nn.Module):
            pass

        with pytest.raises(ValueError, match="no parameters"):
            _model_tensor_spec(_Empty())







class TestChafeeAlignedStabilitySeedThreading:

    @pytest.mark.unit
    def test_chafee_script_passes_stability_seed(self) -> None:
        from pathlib import Path

        pipeline_path = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "kd"
            / "search"
            / "discover"
            / "runners"
            / "mode2_pipeline.py"
        )
        source = pipeline_path.read_text(encoding="utf-8")
        assert "stability_seed=seed" in source, (
            "discover/runners/mode2_pipeline.py must pass stability_seed=seed "
            "to PINNCycleRunner so --seed N produces deterministic "
            "stability_select tie-breaks."
        )

    @pytest.mark.unit
    def test_runner_preserves_stability_seed_kwarg(
        self, heat_evaluator: Evaluator
    ) -> None:
        runner = _build_runner(
            heat_evaluator, _fast_config(n_cycles=1), stability_seed=42
        )
        assert runner._stability_seed == 42

    @pytest.mark.unit
    def test_runner_default_stability_seed_is_none(
        self, heat_evaluator: Evaluator
    ) -> None:
        runner = _build_runner(heat_evaluator, _fast_config(n_cycles=1))
        assert runner._stability_seed is None

    @pytest.mark.unit
    def test_pipeline_passes_local_sample_seed(self) -> None:
        from pathlib import Path

        pipeline_path = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "kd"
            / "search"
            / "discover"
            / "runners"
            / "mode2_pipeline.py"
        )
        source = pipeline_path.read_text(encoding="utf-8")
        assert "local_sample_seed=seed" in source, (
            "discover/runners/mode2_pipeline.py must pass local_sample_seed=seed "
            "to PINNCycleRunner so per-cycle local-collocation derivation "
            "stays deterministic ( sub-task C, B4)."
        )
