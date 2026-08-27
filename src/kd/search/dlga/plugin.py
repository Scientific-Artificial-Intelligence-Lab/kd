
from __future__ import annotations

import logging
import math
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
import torch.nn as nn
from torch import Tensor

from kd.core.equation import Form
from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.executor.surrogate_context import SurrogateContext
from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import FunctionRegistry
from kd.core.linear_solve import (
    LeastSquaresSolver,
    SparseSolver,
    SVDNullSpaceSolver,
)
from kd.core.platform.requirements import DerivativeReqs
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.schema import DataTopology, PDEDataset
from kd.search import surrogate_log as _surrogate_log
from kd.search._torch_module_artifact import (
    TORCH_MODULE_ARTIFACT_FORMAT,
    torch_module_artifact,
)
from kd.search.descriptor import (
    InstrumentDescriptor,
    InstrumentMode,
    Knob,
    Segmentation,
)
from kd.search.dlga import viz as _viz_helpers
from kd.search.dlga.config import DLGAConfig
from kd.search.dlga.genes import (
    Genome,
    crossover_population,
    genome_to_expression,
    mutate_add_module,
    mutate_delete_module,
    mutate_order,
    random_genome,
    random_population,
    select_survivors,
)
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder, log_whitelisted_metrics
from kd.search.result import invalid_evaluation_result
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_INVALID_FITNESS = float("inf")







_NMSE_FALLBACK_EPS: float = 1e-15










_NMSE_BLOAT_WARN_NMSE: float = 1e-2
_NMSE_BLOAT_WARN_LENGTH: int = 5
_NMSE_EPSILON_RETUNE_BAND: str = "1e-2 to 5e-2"






_LOGGED_METRICS = _viz_helpers.LOGGED_METRICS
_GEN_BEST_FITNESS_KEY = _viz_helpers.GEN_BEST_FITNESS_KEY
_GEN_MEAN_FITNESS_KEY = _viz_helpers.GEN_MEAN_FITNESS_KEY
_GEN_BEST_NMSE_KEY = _viz_helpers.GEN_BEST_NMSE_KEY
_N_VALID_KEY = _viz_helpers.N_VALID_KEY
_N_UNIQUE_KEY = _viz_helpers.N_UNIQUE_KEY
_GEN_MEAN_COMPLEXITY_KEY = _viz_helpers.GEN_MEAN_COMPLEXITY_KEY
_LHS_UT_KEY = _viz_helpers.LHS_UT_KEY
_LHS_UTT_KEY = _viz_helpers.LHS_UTT_KEY





_SURROGATE_METRICS = _surrogate_log.SURROGATE_METRICS


class DLGAPlugin:






    score_kind: ClassVar[str] = "DLGA fitness"
    score_direction: ClassVar[Literal["min", "max"]] = "min"
    headline_coefficient_source: ClassVar[Literal["native", "platform_refit"]] = (
        "platform_refit"
    )

    config_cls: ClassVar[type[DLGAConfig]] = DLGAConfig
    one_shot: ClassVar[bool] = False
    sketch_lower_owner: ClassVar[Literal["platform", "native"]] = "platform"

    descriptor: ClassVar[InstrumentDescriptor] = InstrumentDescriptor(
        algorithm="dlga",
        summary="Neural-surrogate genetic search for constant-coefficient PDEs.",
        cost_class="heavy",
        modes=(
            InstrumentMode(
                name="default",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="autograd",
            ),
        ),
        knobs=(
            Knob(
                "pop_size",
                "int",
                "Genetic population size.",
                resume_tier="resume_safe",
            ),
            Knob(
                "epsilon",
                "float",
                "Expression-length penalty.",
                resume_tier="init_only",
            ),
            Knob(
                "mutation_rate",
                "float",
                "Genome mutation probability.",
                resume_tier="resume_safe",
            ),
            Knob(
                "crossover_rate",
                "float",
                "Genome crossover probability.",
                resume_tier="resume_safe",
            ),
            Knob(
                "surrogate_lr",
                "float",
                "Neural-surrogate learning rate.",
                resume_tier="init_only",
            ),
        ),
        segmentation=Segmentation(
            archive="progress", unit="generations", reseed=True
        ),
    )

    def __init__(
        self,
        config: DLGAConfig | None = None,
        *,
        surrogate_model: nn.Module | None = None,
    ) -> None:
        self._config = config or DLGAConfig()
        self._require_constant_mode()
        self._provided_model = surrogate_model
        self._model: nn.Module | None = None
        self._provider: AutogradProvider | None = None
        self._context: SurrogateContext | None = None
        self._evaluators: dict[str, Evaluator] = {}
        self._lhs_targets: dict[str, Tensor] = {}
        self._executor: PythonExecutor | None = None
        self._registry: FunctionRegistry | None = None
        self._dataset: PDEDataset | None = None
        self._population: list[Genome] | None = None
        self._last_results: list[EvaluationResult] | None = None
        self._last_result_genomes: list[Genome | None] | None = None
        self._best_result: EvaluationResult | None = None
        self._best_genome: Genome | None = None
        self._best_score = float("inf")
        self._best_expression = ""
        self._best_lhs_name: str | None = None
        self._recorder: VizRecorder | None = None
        self._restore_pending: bool = False
        self._last_fitness: list[float] | None = None
        self._restored_model: nn.Module | None = None
        self._rng = torch.Generator()
        self._prepared = False

    @property
    def config(self) -> dict[str, Any]:
        config = asdict(self._config)
        if self._provided_model is not None:
            config["surrogate_model"] = {
                "artifact": "surrogate_model",
                "format": TORCH_MODULE_ARTIFACT_FORMAT,
            }
        return {"algorithm": "dlga", **config}

    @property
    def artifacts(self) -> dict[str, dict[str, str | int]] | None:
        if self._provided_model is None:
            return None
        return {
            "surrogate_model": torch_module_artifact(self._provided_model),
        }

    @property
    def runner_batch_size(self) -> int:
        return self._config.pop_size

    @property
    def derivative_requirements(self) -> DerivativeReqs:
        return DerivativeReqs(
            provider_kind="autograd",
            max_atomic_order=3,
            lhs_order=self._config.target_lhs_order,
            needs_surrogate=True,
            surrogate_model=(
                self._provided_model
                if self._provided_model is not None
                else self._restored_model
            ),
            surrogate_arch_kwargs={
                "hidden_sizes": list(self._config.surrogate_hidden_sizes),
                "activation": self._config.surrogate_activation,
            },
            surrogate_train_kwargs={
                "lr": self._config.surrogate_lr,
                "max_epochs": self._config.surrogate_max_epochs,
                "patience": self._config.surrogate_patience,
                "val_ratio": self._config.surrogate_val_ratio,
                "seed": self._config.seed,
                "restore_best": self._config.surrogate_restore_best,
            },
        )

    def prepare(self, components: PlatformComponents) -> None:
        self._prepared = False
        self._require_constant_mode()





        if not self._restore_pending:
            self._reset_search_state()
        self._restore_pending = False

        dataset = components.dataset
        self._validate_dataset(dataset)
        self._dataset = dataset
        self._registry = components.registry or FunctionRegistry.create_default()
        self._executor = components.executor or PythonExecutor(self._registry)






        if not isinstance(components.context, SurrogateContext):
            raise RuntimeError(
                "DLGAPlugin.prepare expected components.context to be a "
                f"SurrogateContext (built from DerivativeReqs(needs_surrogate="
                f"True)); got {type(components.context).__name__}. This "
                "indicates the facade did not wire DerivativeReqs through "
                "PlatformBuilder."
            )
        provider = components.context.derivative_provider
        if not isinstance(provider, AutogradProvider):
            raise RuntimeError(
                "DLGAPlugin.prepare expected SurrogateContext to wrap an "
                f"AutogradProvider; got {type(provider).__name__}."
            )
        self._provider = provider
        self._context = components.context


        self._model = getattr(provider, "model", None)






        self._restored_model = None
        self._build_evaluators(dataset)









        restored_fitness = self._last_fitness
        self._last_fitness = None
        if self._population is None:
            self._rng.manual_seed(self._config.seed)
            self._population = random_population(
                pop_size=self._config.pop_size,
                rng=self._rng,
                library_size=len(self._config.library),
                max_modules=self._config.max_modules,
                max_module_length=self._config.max_module_length,
                partial_prob=self._config.partial_prob,
                genes_prob=self._config.genes_prob,
            )
        elif restored_fitness is not None:
            self._evolve_population(self._population, restored_fitness)
        self._recorder = components.recorder



        _surrogate_log.log_surrogate_training(
            self._recorder,
            getattr(components.context, "training_result", None),
        )
        self._prepared = True

    def propose(self, n: int) -> list[str]:
        del n
        self._require_prepared()
        return [
            genome_to_expression(genome, self._config.library)
            for genome in (self._population or [])
        ]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        self._require_prepared()
        genomes = self._resolve_candidate_genomes(candidates)
        results = [
            self._evaluate_one(candidate, genome)
            for candidate, genome in zip(candidates, genomes, strict=True)
        ]
        self._last_results = results
        self._last_result_genomes = genomes
        return results

    def update(self, results: list[EvaluationResult]) -> None:
        genomes = self._last_result_genomes
        if genomes is None or len(genomes) != len(results):
            genomes = [None] * len(results)
        self._log_generation_metrics(results)
        valid = [
            (result, genome)
            for result, genome in zip(results, genomes, strict=True)
            if result.is_valid
        ]
        if valid:
            best, best_genome = min(valid, key=lambda item: _fitness(item[0]))
            score = _fitness(best)
            if score < self._best_score:
                self._best_score = score
                self._best_expression = best.expression
                self._best_lhs_name = best.lhs_name
                self._best_result = best
                self._best_genome = _clone_genome(best_genome)





        self._last_fitness = [_fitness(result) for result in results]

    def _log_generation_metrics(self, results: list[EvaluationResult]) -> None:
        recorder = self._recorder
        if recorder is None:
            return
        valid_results = [result for result in results if result.is_valid]
        if valid_results:
            best = min(valid_results, key=_fitness)
            gen_best_fitness = _fitness(best)
            gen_mean_fitness = sum(_fitness(r) for r in valid_results) / len(
                valid_results
            )
            gen_best_nmse = best.nmse
            gen_mean_complexity = sum(r.complexity for r in valid_results) / len(
                valid_results
            )
        else:
            gen_best_fitness = float("inf")
            gen_mean_fitness = float("inf")
            gen_best_nmse = float("inf")
            gen_mean_complexity = float("nan")
        metrics: dict[str, float | int] = {
            _GEN_BEST_FITNESS_KEY: gen_best_fitness,
            _GEN_MEAN_FITNESS_KEY: gen_mean_fitness,
            _GEN_BEST_NMSE_KEY: gen_best_nmse,
            _N_VALID_KEY: len(valid_results),
            _N_UNIQUE_KEY: len({result.expression for result in results}),
            _GEN_MEAN_COMPLEXITY_KEY: gen_mean_complexity,
            _LHS_UT_KEY: sum(1 for r in valid_results if r.lhs_name == "u_t"),
            _LHS_UTT_KEY: sum(1 for r in valid_results if r.lhs_name == "u_tt"),
        }
        log_whitelisted_metrics(recorder, _LOGGED_METRICS, metrics)

    def between_iterations(self) -> None:
        self._require_prepared()
        if self._population is None or self._last_fitness is None:
            return
        self._evolve_population(self._population, self._last_fitness)

    def _evolve_population(
        self,
        population: list[Genome],
        fitness: list[float],
    ) -> None:
        keep = max(1, self._config.pop_size // 2)
        survivors = select_survivors(population, fitness, keep=keep)
        while len(survivors) < self._config.pop_size:
            survivors.append(self._random_genome())
        crossed = crossover_population(
            survivors[: self._config.pop_size],
            rng=self._rng,
            pop_size=self._config.pop_size,
            crossover_rate=self._config.crossover_rate,
        )
        self._population = [self._mutate(genome) for genome in crossed]
        self._last_results = None
        self._last_result_genomes = None
        self._last_fitness = None

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_expression(self) -> str:
        return self._best_expression

    def build_final_result(self) -> EvaluationResult:
        if not self._best_expression:
            return self._invalid_result(
                "", "No best DLGA expression available", reason="no_candidate"
            )
        if not self._evaluators:
            return self._invalid_result(
                self._best_expression,
                "DLGA evaluators are not prepared",
                reason="evaluation_error",
            )
        try:
            result = self._evaluate_one(self._best_expression, self._best_genome)
        except Exception as exc:
            logger.debug("Failed to build DLGA final result", exc_info=exc)
            return self._invalid_result(
                self._best_expression,
                f"Final result evaluation failed: {exc}",
                reason="evaluation_error",
            )
        if result.is_valid:
            self._maybe_warn_expression_bloat(result)
            return result
        return self._invalid_result(
            self._best_expression,
            result.error_message or "No valid DLGA result",
            reason=result.invalid_reason or "unclassified",
        )

    def _maybe_warn_expression_bloat(self, result: EvaluationResult) -> None:
        if not result.is_valid:
            return


        expr_len = _length_penalty(self._best_genome, result)
        if result.nmse > _NMSE_BLOAT_WARN_NMSE and expr_len > _NMSE_BLOAT_WARN_LENGTH:
            logger.warning(
                "DLGA recovered expression has nmse=%.3g and length=%d "
                "tokens. The default epsilon=%.0e was tuned for raw-MSE-era "
                "fitness magnitudes; under the NMSE-era selector the "
                "length penalty contribution is "
                "smaller, so the GA may include small-coefficient noise "
                "terms. Truth structure is still recovered (see "
                "is_recovery_success superset match). To get a cleaner "
                "surface expression, raise DLGAConfig.epsilon to ~%s — "
                "see DLGAPlugin docstring for per-PDE recommendations.",
                result.nmse,
                expr_len,
                self._config.epsilon,
                _NMSE_EPSILON_RETUNE_BAND,
            )

    def build_result_target(self) -> Tensor:
        if self._best_lhs_name and self._best_lhs_name in self._lhs_targets:
            return self._lhs_targets[self._best_lhs_name].detach().clone()
        if "u_t" in self._lhs_targets:
            return self._lhs_targets["u_t"].detach().clone()
        raise RuntimeError("build_result_target called before prepare()")








    def list_plots(self) -> list[PlotInfo]:
        return _viz_helpers.list_plot_infos()

    def render_plot(self, name: str, ax: Axes) -> list[str]:
        return _viz_helpers.render(name, ax, self._recorder)

    def get_plot_data(self, name: str) -> dict[str, Any]:
        return _viz_helpers.get_data(name, self._recorder)

    @property
    def state(self) -> dict[str, Any]:
        return {
            "population": self._population,
            "best_score": self._best_score,
            "best_expression": self._best_expression,
            "best_lhs_name": self._best_lhs_name,
            "best_genome": self._best_genome,
            "rng_state": self._rng.get_state().numpy().tobytes(),
            "last_fitness": self._last_fitness,
            "surrogate_model": (
                self._model if self._model is not None else self._restored_model
            ),
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        if not value:
            self._reset_search_state()
            self._restore_pending = False
            return
        self._reset_search_state()
        self._population = value.get("population")
        self._last_fitness = value.get("last_fitness")
        self._restored_model = value.get("surrogate_model")
        self._best_score = value.get("best_score", float("inf"))
        self._best_expression = value.get("best_expression", "")
        self._best_lhs_name = value.get("best_lhs_name")
        self._best_genome = _clone_genome(value.get("best_genome"))
        rng_state = value.get("rng_state")
        if isinstance(rng_state, bytes):
            import numpy as np

            self._rng.set_state(
                torch.from_numpy(np.frombuffer(rng_state, dtype=np.uint8).copy())
            )
        elif isinstance(rng_state, Tensor):
            self._rng.set_state(rng_state)
        self._restore_pending = True

    def reseed(self) -> None:
        self._rng.manual_seed(self._config.seed)

    def _reset_search_state(self) -> None:
        self._population = None
        self._last_results = None
        self._last_result_genomes = None
        self._best_result = None
        self._best_genome = None
        self._best_score = float("inf")
        self._best_expression = ""
        self._best_lhs_name = None




        self._last_fitness = None
        self._restored_model = None

    def _build_evaluators(self, dataset: PDEDataset) -> None:
        if self._provider is None or self._context is None or self._executor is None:
            raise RuntimeError("DLGA internals not prepared")
        solver = self._make_solver()


        lhs_t = self._provider.get_derivative(
            dataset.lhs_field, dataset.lhs_axis, 1
        ).detach()
        self._lhs_targets = {"u_t": lhs_t.flatten().detach()}
        self._evaluators = {
            "u_t": Evaluator(
                self._executor,
                solver,
                self._context,
                lhs_t,
            )
        }
        if self._config.lhs_auto_select:
            lhs_tt = self._provider.get_derivative(
                dataset.lhs_field,
                dataset.lhs_axis,
                2,
            ).detach()
            self._lhs_targets["u_tt"] = lhs_tt.flatten().detach()
            self._evaluators["u_tt"] = Evaluator(
                self._executor,
                self._make_solver(),
                self._context,
                lhs_tt,
            )

    def _evaluate_one(
        self,
        expression: str,
        genome: Genome | None = None,
    ) -> EvaluationResult:
        cross_lhs_mode = len(self._evaluators) > 1
        choices: list[EvaluationResult] = []
        for lhs_name, evaluator in self._evaluators.items():
            result = evaluator.evaluate_expression(expression)
            result.expression = expression
            if result.is_valid:









                lhs_var = getattr(evaluator, "_lhs_var", math.inf)
                if cross_lhs_mode and lhs_var <= _NMSE_FALLBACK_EPS:
                    result = replace(result, nmse=float("inf"))
                length_penalty = _length_penalty(genome, result)
                fitness = result.nmse + self._config.epsilon * length_penalty
                result = replace(result, score=fitness, lhs_name=lhs_name)
            else:
                result = replace(result, lhs_name=lhs_name)
            choices.append(result)
        valid = [result for result in choices if result.is_valid]
        if not valid:
            return replace(choices[0], score=_INVALID_FITNESS)
        return min(
            valid,
            key=lambda result: (result.nmse, 0 if result.lhs_name == "u_tt" else 1),
        )

    def _mutate(self, genome: Genome) -> Genome:
        result = genome
        if _rand(self._rng) < self._config.add_rate:
            result = mutate_add_module(
                result,
                rng=self._rng,
                library_size=len(self._config.library),
                max_length=self._config.max_module_length,
                partial_prob=self._config.partial_prob,
                max_modules=self._config.max_modules,
            )
        if _rand(self._rng) < self._config.delete_rate:
            result = mutate_delete_module(result, rng=self._rng)
        if _rand(self._rng) < self._config.mutation_rate:
            result = mutate_order(
                result,
                rng=self._rng,
                library_size=len(self._config.library),
            )
        return result

    def _random_genome(self) -> Genome:
        return random_genome(
            rng=self._rng,
            library_size=len(self._config.library),
            max_modules=self._config.max_modules,
            max_module_length=self._config.max_module_length,
            partial_prob=self._config.partial_prob,
            genes_prob=self._config.genes_prob,
        )

    def _make_solver(self) -> SparseSolver:
        if self._config.solver == "svd_null_space":
            return SVDNullSpaceSolver()
        return LeastSquaresSolver()

    def _require_constant_mode(self) -> None:
        if self._config.mode != "constant":
            raise NotImplementedError(
                "DLGA Stage I implements only mode='constant'; "
                f"got mode={self._config.mode!r}."
            )







    def _resolve_candidate_genomes(self, candidates: list[str]) -> list[Genome | None]:
        if self._population is None:
            return [None] * len(candidates)
        available: dict[str, list[Genome]] = {}
        for genome in self._population:
            expression = genome_to_expression(genome, self._config.library)
            available.setdefault(expression, []).append(genome)
        genomes: list[Genome | None] = []
        for candidate in candidates:
            matches = available.get(candidate)
            if matches:
                genomes.append(_clone_genome(matches.pop(0)))
            else:
                genomes.append(None)
        return genomes

    @staticmethod
    def _validate_dataset(dataset: PDEDataset) -> None:
        if dataset.axis_order is None or dataset.axes is None:
            raise ValueError("DLGA requires a gridded dataset with axes")
        if dataset.fields is None or dataset.lhs_field not in dataset.fields:
            raise ValueError("DLGA requires dataset.fields containing lhs_field")
        if not dataset.lhs_axis:
            raise ValueError("DLGA requires dataset.lhs_axis")
        if dataset.lhs_axis not in dataset.axis_order:
            raise ValueError(
                f"DLGA dataset.lhs_axis={dataset.lhs_axis!r} must be in "
                f"axis_order={dataset.axis_order!r}"
            )

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before using DLGAPlugin")

    @staticmethod
    def _invalid_result(
        expression: str,
        message: str,
        *,
        reason: str = "unclassified",
    ) -> EvaluationResult:

        return invalid_evaluation_result(
            message,
            score=float("inf"),
            expression=expression,
            reason=reason,
        )


def _fitness(result: EvaluationResult) -> float:
    if not result.is_valid:
        return _INVALID_FITNESS






    if result.score is None:
        return _INVALID_FITNESS
    score = result.score
    return score if math.isfinite(score) else _INVALID_FITNESS


def _length_penalty(genome: Genome | None, result: EvaluationResult) -> int:
    if genome is None:
        return result.complexity
    return sum(len(module) for module in genome)


def _clone_genome(genome: Genome | None) -> Genome | None:
    if genome is None:
        return None
    return [list(module) for module in genome]


def _rand(rng: torch.Generator) -> float:
    return float(torch.rand((), generator=rng).item())
