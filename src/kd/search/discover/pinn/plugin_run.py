
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, cast

from kd.core.evaluator import Evaluator
from kd.core.jsonsafe import NO_MEASUREMENT
from kd.core.platform.builder import lower_target
from kd.search.discover.pinn.collocation import generate_collocation_points
from kd.search.discover.pinn.cycle import (
    cycle_iterations,
    finalize_cycle_state,
    make_local_coords,
    pretrain_split,
    rebuild_after_pinn,
    rebuild_evaluator,
    regenerate_metadata,
    regenerated_context,
    run_pinn_phase,
)
from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset_from
from kd.search.discover.pinn.model import PINNModel
from kd.search.discover.pinn.observations import domain_bounds, sample_observations
from kd.search.discover.sketch_backend import (
    SketchGatedEvaluator,
    assert_anchors_execute,
)
from kd.search.recorder import log_whitelisted_metrics

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import torch
    from torch import Tensor

    from kd.core.equation.sketch import Sketch
    from kd.core.expr import FunctionRegistry
    from kd.data.schema import PDEDataset
    from kd.search.discover.config import DiscoverConfig, PINNConfig
    from kd.search.discover.engine import DiscoverEngine, EngineState
    from kd.search.discover.pinn.model import PretrainResult, TrainResult
    from kd.search.discover.sketch_backend import DiscoverCompiled
    from kd.search.protocol import PlatformComponents
    from kd.search.recorder import VizRecorder

logger = logging.getLogger(__name__)



PRETRAIN_SPLIT_SEED = 0



PRETRAIN_SERIES: tuple[str, ...] = (
    "pinn_pretrain_train_loss",
    "pinn_pretrain_val_loss",
    "pinn_pretrain_epochs",
    "pinn_pretrain_stopped_early",
)
PHASE_SERIES: tuple[str, ...] = (
    "pinn_cycle_idx",
    "pinn_phase_ran",
    "pinn_best_reward",
    "pinn_data_loss",
    "pinn_physics_loss",
    "pinn_total_loss",
    "pinn_epochs_run",
    "pinn_train_seconds",
)


def _log_budget(
    pinn: PINNConfig,
    obs_coords: dict[str, Tensor],
    colloc_coords: dict[str, Tensor],
    device: torch.device,
) -> None:
    n_obs = next(iter(obs_coords.values())).numel()
    logger.info(
        "DISCOVER PINN mode: %d observations, %d collocation points, %d local "
        "points per PINN phase, pretrain_epoch=%d, pinn_epoch=%d, device=%s",
        n_obs,
        next(iter(colloc_coords.values())).numel(),
        (pinn.local_multiplier + 1) * n_obs if pinn.local_sample else 0,
        pinn.pretrain_epoch,
        pinn.pinn_epoch,
        device,
    )


def _pretrain(
    model: PINNModel,
    obs_coords: dict[str, Tensor],
    obs_targets: dict[str, Tensor],
    pinn: PINNConfig,
    recorder: VizRecorder | None,
) -> tuple[PretrainResult, float]:
    started = time.perf_counter()
    pretrained = pretrain_split(
        model, obs_coords, obs_targets, pinn, split_seed=PRETRAIN_SPLIT_SEED
    )
    seconds = time.perf_counter() - started
    log_whitelisted_metrics(
        recorder,
        PRETRAIN_SERIES,
        {
            "pinn_pretrain_train_loss": pretrained.train_loss,
            "pinn_pretrain_val_loss": pretrained.val_loss,
            "pinn_pretrain_epochs": pretrained.epochs_run,
            "pinn_pretrain_stopped_early": float(pretrained.stopped_early),
        },
    )
    return pretrained, seconds


def _cpu_copy(weights: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in weights.items()}


def _phase_measurements(train_result: TrainResult | None) -> dict[str, float]:
    if train_result is None:
        return dict.fromkeys(
            (
                "pinn_data_loss",
                "pinn_physics_loss",
                "pinn_total_loss",
                "pinn_epochs_run",
            ),
            NO_MEASUREMENT,
        )
    return {
        "pinn_data_loss": train_result.data_loss,
        "pinn_physics_loss": train_result.physics_loss,
        "pinn_total_loss": train_result.total_loss,
        "pinn_epochs_run": train_result.epochs_run,
    }


@dataclass(frozen=True, slots=True)
class PinnSketch:

    sketch: Sketch
    compiled: DiscoverCompiled
    registry: FunctionRegistry

    @property
    def pins(self) -> tuple[tuple[str, float], ...]:
        return tuple((pin.term_ir, pin.value) for pin in self.sketch.pinned)

    def gate(self, evaluator: Evaluator) -> SketchGatedEvaluator:
        return SketchGatedEvaluator(
            evaluator,
            sketch=self.sketch,
            compiled=self.compiled,
            registry=self.registry,
        )

    def assert_executes(
        self,
        model: PINNModel,
        colloc_coords: dict[str, Tensor],
        metadata: PDEDataset,
        platform: Evaluator,
    ) -> None:
        regen = regenerate_metadata(
            model,
            colloc_coords,
            metadata,
            lhs_field=metadata.lhs_field,
            lhs_axis=metadata.lhs_axis,
        )
        lower_target(
            regen.lhs_detached, self.pins, platform.executor, regenerated_context(regen)
        )
        assert_anchors_execute(
            rebuild_evaluator(regen, platform.executor, platform.solver),
            self.compiled.anchor_terms,
        )


def _check_field_order(stored: list[str] | None, field_names: list[str]) -> None:
    if stored is None and len(field_names) > 1:
        raise ValueError(
            "DISCOVER checkpoint's mode2 state does not record the field order "
            "of its network, and the dataset has the fields "
            f"{field_names}: the stored output columns cannot be matched to "
            "the fields. Start the PINN-mode run again."
        )


def _rebuild_on_network(
    model: PINNModel,
    colloc_coords: dict[str, Tensor],
    metadata: PDEDataset,
    platform: Evaluator,
    sketch: PinnSketch | None,
) -> Evaluator | SketchGatedEvaluator:
    if sketch is None:
        return rebuild_after_pinn(
            model,
            colloc_coords,
            metadata,
            executor=platform.executor,
            solver=platform.solver,
        )
    return sketch.gate(
        rebuild_after_pinn(
            model,
            colloc_coords,
            metadata,
            executor=platform.executor,
            solver=platform.solver,
            pins=sketch.pins,
        )
    )


@dataclass(slots=True)
class PinnRun:

    config: DiscoverConfig
    pinn: PINNConfig
    recorder: VizRecorder | None
    model: PINNModel
    executor: PINNExecutor
    platform: Evaluator
    evaluator: Evaluator | SketchGatedEvaluator
    sketch: PinnSketch | None
    obs_coords: dict[str, Tensor]
    obs_targets: dict[str, Tensor]
    colloc_coords: dict[str, Tensor]
    metadata: PDEDataset
    bounds: dict[str, tuple[float, float]]
    sampling_seed: int
    pretrain_result: dict[str, Any]
    train_seconds: float
    completed_generations: int = 0
    cycle_idx: int = 0
    cycle_metrics: list[dict[str, float]] = field(default_factory=list)
    phase_snapshot: dict[str, Tensor] | None = None

    @classmethod
    def _on_untrained_network(
        cls,
        config: DiscoverConfig,
        pinn: PINNConfig,
        components: PlatformComponents,
        sketch: PinnSketch | None,
        *,
        sampling_seed: int,
        train: Callable[
            [PINNModel, dict[str, Tensor], dict[str, Tensor]],
            tuple[dict[str, Any], float],
        ],
    ) -> PinnRun:
        if config.n_iterations < 1:


            raise ValueError(
                "the PINN mode needs n_iterations >= 1 (the first PINN phase "
                f"follows that generation), got {config.n_iterations}"
            )

        platform = cast(Evaluator, components.evaluator)
        device = platform.context.device
        dataset = components.dataset
        bounds = domain_bounds(dataset)
        obs_coords, obs_targets = sample_observations(
            dataset, obs_ratio=pinn.obs_ratio, seed=sampling_seed, device=device
        )
        model = PINNModel(list(obs_coords), list(obs_targets), pinn, device=device)
        colloc_coords = generate_collocation_points(
            bounds,
            pinn.n_collocation,
            pinn.collocation_cut_ratio,
            seed=sampling_seed,
            device=device,
        )
        _log_budget(pinn, obs_coords, colloc_coords, device)
        metadata = make_pinn_dataset_from(dataset)
        if sketch is not None:
            sketch.assert_executes(model, colloc_coords, metadata, platform)
        pretrain_result, train_seconds = train(model, obs_coords, obs_targets)
        return cls(
            config=config,
            pinn=pinn,
            recorder=components.recorder,
            model=model,
            executor=PINNExecutor(components.registry),
            platform=platform,
            evaluator=_rebuild_on_network(
                model, colloc_coords, metadata, platform, sketch
            ),
            sketch=sketch,
            obs_coords=obs_coords,
            obs_targets=obs_targets,
            colloc_coords=colloc_coords,
            metadata=metadata,
            bounds=bounds,
            sampling_seed=sampling_seed,
            pretrain_result=pretrain_result,
            train_seconds=train_seconds,
        )

    @classmethod
    def start(
        cls,
        config: DiscoverConfig,
        pinn: PINNConfig,
        engine: DiscoverEngine,
        components: PlatformComponents,
        sketch: PinnSketch | None,
    ) -> PinnRun:

        def pretrain(
            model: PINNModel,
            obs_coords: dict[str, Tensor],
            obs_targets: dict[str, Tensor],
        ) -> tuple[dict[str, Any], float]:
            pretrained, seconds = _pretrain(
                model, obs_coords, obs_targets, pinn, components.recorder
            )
            return asdict(pretrained), seconds

        run = cls._on_untrained_network(
            config, pinn, components, sketch, sampling_seed=config.seed, train=pretrain
        )
        engine.begin_cycle(run.evaluator, 0)
        return run

    @classmethod
    def restore(
        cls,
        config: DiscoverConfig,
        pinn: PINNConfig,
        components: PlatformComponents,
        block: Mapping[str, Any],
        sketch: PinnSketch | None,
    ) -> PinnRun:

        def load(
            model: PINNModel,
            obs_coords: dict[str, Tensor],
            obs_targets: dict[str, Tensor],
        ) -> tuple[dict[str, Any], float]:
            _check_field_order(block.get("field_names"), model.field_names)
            model.load_state_dict(block["pinn_state_dict"])
            return dict(block["pretrain_result"]), 0.0

        run = cls._on_untrained_network(
            config,
            pinn,
            components,
            sketch,
            sampling_seed=block["sampling_seed"],
            train=load,
        )
        run.completed_generations = block["completed_generations"]
        run.cycle_idx = block["cycle_idx"]
        run.cycle_metrics = [dict(metrics) for metrics in block["cycle_metrics"]]
        return run

    def resume_cycle(self, engine: DiscoverEngine, block: Mapping[str, Any]) -> None:
        boundary = self._owed_boundary()
        if boundary is not None:


            self._run_boundary(engine, boundary, deadline=None)
            return
        engine.notify_cycle_start(self.cycle_idx)
        engine.restore_cycle_candidates(
            [candidate["expression"] for candidate in block["stability_pool"]],
            self.evaluator,
            seen=block["stability_seen"],
        )

    def state_block(self, engine: DiscoverEngine) -> dict[str, Any]:
        return {
            "pinn_state_dict": _cpu_copy(
                self.model.state_dict()
                if self.phase_snapshot is None
                else self.phase_snapshot
            ),
            "field_names": list(self.model.field_names),
            "completed_generations": self.completed_generations,
            "cycle_idx": self.cycle_idx,
            "sampling_seed": self.sampling_seed,
            "pretrain_result": dict(self.pretrain_result),
            "cycle_metrics": [dict(metrics) for metrics in self.cycle_metrics],
            "stability_pool": [
                asdict(candidate) for candidate in engine.cycle_top_candidates
            ],
            "stability_seen": sorted(engine.cycle_seen_expressions),
        }

    def _owed_boundary(self) -> int | None:
        length = cycle_iterations(self.config, 1)
        past_first = self.completed_generations - cycle_iterations(self.config, 0)
        if past_first < 0 or past_first % length != 0:
            return None
        boundary = past_first // length
        return boundary if boundary >= self.cycle_idx else None

    def between_generations(
        self, engine: DiscoverEngine, *, deadline: float | None
    ) -> None:
        boundary = self._owed_boundary()
        if boundary is not None:
            self._run_boundary(engine, boundary, deadline=deadline)

    def _run_boundary(
        self, engine: DiscoverEngine, boundary: int, *, deadline: float | None
    ) -> None:
        self.phase_snapshot = _cpu_copy(self.model.state_dict())
        started = time.perf_counter()
        metrics, trained, train_result = run_pinn_phase(
            engine,
            self.model,
            self.executor,
            self.evaluator,
            obs_coords=self.obs_coords,
            obs_targets=self.obs_targets,
            colloc_coords=self.colloc_coords,
            dataset_metadata=self.metadata,
            config=self.config,
            cycle_idx=boundary,
            local_coords=make_local_coords(
                self.obs_coords,
                self.bounds,
                self.pinn,
                boundary,
                local_sample_seed=self.config.seed,
            ),
            deadline=deadline,
            pins=() if self.sketch is None else self.sketch.pins,
        )
        seconds = time.perf_counter() - started
        self.train_seconds += seconds
        if trained:
            self.evaluator = _rebuild_on_network(
                self.model,
                self.colloc_coords,
                self.metadata,
                self.platform,
                self.sketch,
            )
        log_whitelisted_metrics(
            self.recorder,
            PHASE_SERIES,
            {
                "pinn_cycle_idx": boundary,
                "pinn_phase_ran": float(trained),
                "pinn_best_reward": metrics["best_reward"],
                **_phase_measurements(train_result),
                "pinn_train_seconds": seconds,
            },
        )
        self.cycle_metrics.append(metrics)
        self.phase_snapshot = None
        self.cycle_idx = boundary + 1
        engine.begin_cycle(self.evaluator, self.cycle_idx)

    def finalize(self, engine: DiscoverEngine) -> EngineState:
        seed = self.config.seed
        return finalize_cycle_state(
            engine,
            engine.state,
            self.evaluator,
            self.config,
            stability_seed=seed,
            seed_plan={
                "pretrain_split_seed": PRETRAIN_SPLIT_SEED,
                "local_sample_seed": seed,
                "stability_seed": seed,
            },
        )


__all__ = ["PinnRun", "PinnSketch"]
