
from __future__ import annotations

import ast
import logging
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from kd.core.equation import Form, LhsSpec
from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.sketch import Sketch, term_admission
from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.executor.surrogate_context import SurrogateContext
from kd.core.expr.executor import DIFF_OPERATOR_PATTERN, UNITY_TOKEN, PythonExecutor
from kd.core.expr.naming import parse_compound_derivative
from kd.core.expr.registry import FunctionRegistry
from kd.core.expr.terms import split_terms
from kd.core.interrupt import SearchInterrupted
from kd.core.linear_solve import (
    LeastSquaresSolver,
    SparseSolver,
    SVDNullSpaceSolver,
)
from kd.core.metrics import NMSE_EPS
from kd.core.oom import is_cpu_alloc_failure
from kd.core.platform.requirements import DerivativeReqs
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.schema import DataTopology, PDEDataset
from kd.models._training_precision import omit_default_training_dtype
from kd.models.field_model import FieldModel
from kd.models.serialize import StoredSurrogate, load_stored_surrogate
from kd.search import surrogate_log as _surrogate_log
from kd.search._torch_module_artifact import (
    TORCH_MODULE_ARTIFACT_FORMAT,
    torch_module_artifact,
)
from kd.search.descriptor import (
    SCORE_FRAME_SURROGATE_FIELD,
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
from kd.search.dlga.sketch_backend import DLGACompiled, compile_for_dlga
from kd.search.dlga.surrogate_store import (
    TrainedSurrogate,
    require_surrogate_identity,
    save_dlga_surrogate,
)
from kd.search.protocol import DiscoveryTask, PlatformComponents
from kd.search.recorder import VizRecorder, log_whitelisted_metrics
from kd.search.result import invalid_evaluation_result
from kd.search.trajectory import (
    active_fit_terms,
    log_search_trajectory,
    select_trajectory_candidates,
)
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.models.trainer import TrainingResult

logger = logging.getLogger(__name__)

_INVALID_FITNESS = float("inf")


def _stored_surrogate(
    surrogate_model: nn.Module | StoredSurrogate | str | Path | None,
) -> StoredSurrogate | None:
    if surrogate_model is None or isinstance(surrogate_model, nn.Module):
        return None
    stored = (
        surrogate_model
        if isinstance(surrogate_model, StoredSurrogate)
        else load_stored_surrogate(surrogate_model)
    )
    if stored.identity is None:
        raise ValueError(
            f"stored surrogate {stored.path} carries no identity, so it cannot "
            "be verified against the dataset; write it with "
            "save_dlga_surrogate (or train_surrogate(save_to=)), or inject "
            "its module directly to use it unverified"
        )
    return stored


def _unmodeled_field_terms(
    terms: tuple[str, ...], dataset: PDEDataset
) -> dict[str, set[str]]:
    assert dataset.fields is not None
    others = set(dataset.fields) - {dataset.lhs_field}
    found: dict[str, set[str]] = {}
    if not others:
        return found
    fields = set(dataset.fields)
    axes = set(dataset.axis_order or ())
    for term in terms:
        names = {
            node.id
            for node in ast.walk(ast.parse(term, mode="eval"))
            if isinstance(node, ast.Name)
        }
        used = names & others
        for name in names - others:
            parsed = parse_compound_derivative(
                name, known_fields=fields, known_axes=axes
            )
            if parsed is not None and parsed[0] in others:
                used.add(parsed[0])
        if used:
            found[term] = used
    return found







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
    constructor_kwargs: ClassVar[frozenset[str]] = frozenset(
        {"surrogate_model", "task"}
    )

    descriptor: ClassVar[InstrumentDescriptor] = InstrumentDescriptor(
        algorithm="dlga",
        summary="Neural-surrogate genetic search for constant-coefficient PDEs.",
        cost_class="heavy",
        modes=(
            InstrumentMode(
                name="default",
                order_params={
                    1: {"target_lhs_order": 1},
                    2: {"target_lhs_order": 2},
                },
                field_names=("u",),
                spatial_axes=("x", "y"),
                names_from="library",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="autograd",
                description=(
                    "The raw field and every derivative order are read from a "
                    "trained neural surrogate. Calibrated surrogate budget: "
                    "5000 epochs (surrogate_max_epochs), the smallest measured "
                    "budget at which the true law of a one-field u_t test case "
                    "fits the surrogate's own columns within 5% on every "
                    "coefficient (noise 0 and 0.05, three seeds); the config "
                    "default stays 50000."
                ),
                lhs_orders=frozenset({1, 2}),
                score_frame=SCORE_FRAME_SURROGATE_FIELD,
                sketch=SketchClauseLevels(
                    fixed_terms="lowered",
                    anchors="fit_enforced",
                    hole_count="exit_checked",
                    derivative_order="generation_enforced",
                    operator_set="generation_enforced",
                    field_axis_set="generation_enforced",
                ),
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











                resume_tier="resume_safe",
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
        segmentation=Segmentation(archive="progress", unit="generations", reseed=True),







        identity_breaking_fields=frozenset(
            {
                "library",
                "lhs_auto_select",
                "target_lhs_order",
                "surrogate_model",
                "training_dtype",
            }
        ),



        config_artifact_keys=frozenset({"surrogate_model"}),




        surrogate_fields=frozenset(
            {
                "surrogate_hidden_sizes",
                "surrogate_activation",
                "surrogate_lr",
                "surrogate_max_epochs",
                "surrogate_patience",
                "surrogate_val_ratio",
                "surrogate_restore_best",
                "training_dtype",
            }
        ),
    )

    def __init__(
        self,
        config: DLGAConfig | None = None,
        *,
        surrogate_model: nn.Module | StoredSurrogate | str | Path | None = None,
        task: DiscoveryTask | None = None,
    ) -> None:
        self._config = config or DLGAConfig()
        self._library = tuple(self._config.library)
        if task is not None:








            self._library = compile_for_dlga(task.sketch, self._config).library
        self._max_module_length = self._config.max_module_length
        self._sketch_compile_report: CompileReport | None = None



        self._sketch: Sketch | None = None if task is None else task.sketch
        self._sketch_compiled: DLGACompiled | None = None
        self._require_constant_mode()
        self._provided_stored = _stored_surrogate(surrogate_model)
        self._provided_model: nn.Module | None
        if self._provided_stored is not None:
            self._provided_model = self._provided_stored.module
        elif isinstance(surrogate_model, nn.Module):
            self._provided_model = surrogate_model
        else:
            self._provided_model = None
        self._model: nn.Module | None = None
        self._provider: AutogradProvider | None = None
        self._context: SurrogateContext | None = None
        self._evaluators: dict[str, Evaluator] = {}
        self._lhs_targets: dict[str, Tensor] = {}
        self._lhs_t_name = ""
        self._lhs_tt_name = ""
        self._executor: PythonExecutor | None = None
        self._registry: FunctionRegistry | None = None
        self._dataset: PDEDataset | None = None
        self._population: list[Genome] | None = None
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
        self._network_provenance: dict[str, Any] | None = None
        self._restored_provenance: dict[str, Any] | None = None
        self._rng = torch.Generator()
        self._prepared = False

    @property
    def config(self) -> dict[str, Any]:
        config = omit_default_training_dtype(asdict(self._config))
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
    def sketch_compile_report(self) -> CompileReport | None:
        return self._sketch_compile_report

    @property
    def trained_surrogate(self) -> TrainedSurrogate | None:
        if self._model is None or self._network_provenance is None:
            return None
        return TrainedSurrogate(
            module=cast("FieldModel", self._model),
            config=replace(self._config, seed=self._network_provenance["seed"]),
            stopped_at_deadline=self._network_provenance["stopped_at_deadline"],
            training=cast("SurrogateContext", self._context).training_result,
        )

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
                "training_dtype": self._config.training_dtype,
            },
        )

    def train_surrogate(
        self,
        dataset: PDEDataset,
        *,
        device: str | None = None,
        save_to: str | Path | None = None,
    ) -> tuple[nn.Module, TrainingResult]:




        from kd.core.platform.builder import PlatformBuilder

        reqs = replace(self.derivative_requirements, surrogate_model=None)
        model, result = PlatformBuilder(dataset, reqs, device).train_surrogate()
        if save_to is not None:


            save_dlga_surrogate(
                cast("FieldModel", model),
                save_to,
                dataset=dataset,
                config=self._config,
            )
        return model, result

    def prepare(self, components: PlatformComponents) -> None:
        self._prepared = False
        self._require_constant_mode()





        if not self._restore_pending:
            self._reset_search_state()
        self._restore_pending = False

        self._library = tuple(self._config.library)
        self._max_module_length = self._config.max_module_length
        self._sketch_compile_report = None
        self._sketch = components.task.sketch if components.task is not None else None
        self._sketch_compiled = None
        if self._sketch is not None:
            compiled = compile_for_dlga(self._sketch, self._config)
            self._library = compiled.library
            self._sketch_compile_report = compiled.report
            self._sketch_compiled = compiled
            if not compiled.allow_products:
                self._max_module_length = 1

        dataset = components.dataset
        self._validate_dataset(dataset)
        if self._provided_stored is not None:




            require_surrogate_identity(
                self._provided_stored, dataset, self._config, ignore=("seed",)
            )


        self.refuse_unmodeled_fields(dataset)
        self.refuse_lhs_terms(dataset)
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
        self._network_provenance = self._provenance_of(
            components.context.training_result
        )
        self._restored_provenance = None






        self._restored_model = None
        self._build_evaluators(dataset, self._lowered_target(components))


        self._reprice_best()









        restored_fitness = self._last_fitness
        self._last_fitness = None
        if self._population is None:
            self._rng.manual_seed(self._config.seed)
            self._population = random_population(
                pop_size=self._config.pop_size,
                rng=self._rng,
                library_size=len(self._library),
                max_modules=self._config.max_modules,
                max_module_length=self._max_module_length,
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

    def _provenance_of(self, training: TrainingResult | None) -> dict[str, Any] | None:
        if self._provided_model is not None:
            return None
        if training is not None:
            return {
                "seed": self._config.seed,
                "stopped_at_deadline": training.stopped_at_deadline,
            }
        return self._restored_provenance

    def propose(self, n: int) -> list[str]:
        del n
        self._require_prepared()
        return [
            genome_to_expression(genome, self._library)
            for genome in (self._population or [])
        ]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        self._require_prepared()
        genomes = self._resolve_candidate_genomes(candidates)
        results = [
            self._evaluate_one(candidate, genome)
            for candidate, genome in zip(candidates, genomes, strict=True)
        ]
        self._last_result_genomes = genomes
        return results

    def update(self, results: list[EvaluationResult]) -> None:
        genomes = self._last_result_genomes
        if genomes is None or len(genomes) != len(results):
            genomes = [None] * len(results)
        self._log_generation_metrics(results)
        self._log_search_trajectory(results)
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

    def _log_search_trajectory(self, results: list[EvaluationResult]) -> None:
        recorder = self._recorder
        if recorder is None or not recorder.enabled or recorder.trajectory_top_k == 0:
            return

        def project(index: int) -> tuple[str, tuple[str, ...], str]:
            result = results[index]

            assert result.terms is not None
            assert result.coefficients is not None
            assert result.lhs_name is not None
            terms = active_fit_terms(
                result.terms, result.coefficients, result.selected_indices
            )
            return result.expression, terms, result.lhs_name

        candidates = select_trajectory_candidates(
            [_fitness(result) if result.is_valid else None for result in results],
            project,
            top_k=recorder.trajectory_top_k,
            direction=self.score_direction,
        )
        log_search_trajectory(recorder, candidates)

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
            _LHS_UT_KEY: sum(
                1 for r in valid_results if r.lhs_name == self._lhs_t_name
            ),
            _LHS_UTT_KEY: sum(
                1 for r in valid_results if r.lhs_name == self._lhs_tt_name
            ),
        }
        log_whitelisted_metrics(recorder, _LOGGED_METRICS, metrics)

    def between_iterations(self) -> None:
        self._require_prepared()
        if self._population is None or self._last_fitness is None:
            return
        self._evolve_population(self._population, self._last_fitness)

    def _reprice_best(self) -> None:
        if self._best_genome is None:
            return
        result = self._evaluate_one(self._best_expression, self._best_genome)
        self._best_score = _fitness(result)
        self._best_lhs_name = result.lhs_name
        self._best_result = result

    def _evolve_population(
        self,
        population: list[Genome],
        fitness: list[float],
    ) -> None:
        keep = max(1, self._config.pop_size // 2)
        survivors = select_survivors(population, fitness, keep=keep)
        elite = survivors[0]
        while len(survivors) < self._config.pop_size:
            survivors.append(self._random_genome())
        crossed = crossover_population(
            survivors[: self._config.pop_size],
            rng=self._rng,
            pop_size=self._config.pop_size,
            crossover_rate=self._config.crossover_rate,
        )
        self._population = [self._mutate(genome) for genome in crossed]
        self._population[0] = elite
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
        except (SearchInterrupted, torch.cuda.OutOfMemoryError):



            raise
        except Exception as exc:
            if is_cpu_alloc_failure(exc):

                raise
            logger.debug("Failed to build DLGA final result", exc_info=exc)
            return self._invalid_result(
                self._best_expression,
                f"Final result evaluation failed: {exc}",
                reason="evaluation_error",
            )
        if result.is_valid:
            return result
        return self._invalid_result(
            self._best_expression,
            result.error_message or "No valid DLGA result",
            reason=result.invalid_reason or "unclassified",
        )

    def build_result_target(self) -> Tensor:
        if not self._lhs_targets:
            raise RuntimeError("build_result_target called before prepare()")

        lhs_name = self._best_lhs_name or next(iter(self._lhs_targets))
        return self._lhs_targets[lhs_name].detach().clone()








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
            "surrogate_provenance": (
                self._network_provenance
                if self._model is not None
                else self._restored_provenance
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
        self._restored_provenance = value.get("surrogate_provenance")
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
        self._last_result_genomes = None
        self._best_result = None
        self._best_genome = None
        self._best_score = float("inf")
        self._best_expression = ""
        self._best_lhs_name = None




        self._last_fitness = None
        self._restored_model = None
        self._restored_provenance = None

    def _lowered_target(self, components: PlatformComponents) -> Tensor | None:
        task = components.task
        if task is None or not task.sketch.pinned or task.compiled.closed:
            return None
        if task.sketch.lhs_spec.order != self._config.target_lhs_order:
            raise ValueError(
                f"pinned sketch LHS order {task.sketch.lhs_spec.order} differs "
                f"from target_lhs_order={self._config.target_lhs_order}, the "
                "order of the target the platform lowered"
            )

        assert components.evaluator is not None
        return components.evaluator.lhs_target

    def _build_evaluators(self, dataset: PDEDataset, lowered: Tensor | None) -> None:
        if self._provider is None or self._context is None or self._executor is None:
            raise RuntimeError("DLGA internals not prepared")
        self._lhs_t_name = render_lhs_label(
            LhsSpec(dataset.lhs_field, dataset.lhs_axis, 1)
        )
        self._lhs_tt_name = render_lhs_label(
            LhsSpec(dataset.lhs_field, dataset.lhs_axis, 2)
        )
        self._lhs_targets = {}
        self._evaluators = {}
        for order in self._fitted_orders():
            name = render_lhs_label(LhsSpec(dataset.lhs_field, dataset.lhs_axis, order))


            lhs = (
                self._provider.get_derivative(
                    dataset.lhs_field, dataset.lhs_axis, order
                ).detach()
                if lowered is None
                else lowered
            )
            self._lhs_targets[name] = lhs.flatten().detach()
            self._evaluators[name] = Evaluator(
                self._executor,
                self._make_solver(),
                self._context,
                lhs,
            )

    def _fitted_orders(self) -> tuple[int, ...]:
        if self._sketch is not None:
            return (self._sketch.lhs_spec.order,)
        if self._config.lhs_auto_select:
            return (1, 2)
        return (self._config.target_lhs_order,)

    def _admitted_terms(self, expression: str) -> list[str] | None:
        sketch, compiled = self._sketch, self._sketch_compiled
        assert sketch is not None and compiled is not None
        assert self._registry is not None
        try:
            terms = split_terms(expression, self._registry)
            for term in terms:
                reason = term_admission(
                    term,
                    sketch,
                    pinned_keys=compiled.pinned_keys,
                    pinned_monomials=compiled.reserved_monomials,
                    anchor_fingerprints=compiled.anchor_fingerprints,
                )
                if reason is not None:
                    return None
        except ValueError:
            return None
        return terms

    def _fit(
        self, evaluator: Evaluator, expression: str, terms: list[str] | None
    ) -> EvaluationResult:
        compiled = self._sketch_compiled
        if terms is None or compiled is None or not compiled.anchor_terms:
            return evaluator.evaluate_expression(expression)
        result = evaluator.evaluate_terms([*terms, *compiled.anchor_terms])
        if not result.is_valid:
            return result
        own = len(terms)
        selected = result.selected_indices
        if selected is None:
            return replace(result, complexity=own)
        kept = [index for index in selected if index < own]
        anchors = range(own, own + len(compiled.anchor_terms))
        return replace(result, selected_indices=[*kept, *anchors], complexity=len(kept))

    def _evaluate_one(
        self,
        expression: str,
        genome: Genome | None = None,
    ) -> EvaluationResult:
        terms: list[str] | None = None
        if self._sketch_compiled is not None and self._sketch_compiled.gated:
            terms = self._admitted_terms(expression)
            if terms is None:
                return self._invalid_result(
                    expression,
                    "Candidate contains a term outside the sketch",
                    reason="structural_reject",
                )
        cross_lhs_mode = len(self._evaluators) > 1
        choices: list[EvaluationResult] = []
        for lhs_name, evaluator in self._evaluators.items():
            result = self._fit(evaluator, expression, terms)
            result.expression = expression
            if result.is_valid:






                if cross_lhs_mode and evaluator.lhs_variance <= NMSE_EPS:
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
            key=lambda result: (
                result.nmse,
                0 if result.lhs_name == self._lhs_tt_name else 1,
            ),
        )

    def _mutate(self, genome: Genome) -> Genome:
        result = genome
        if _rand(self._rng) < self._config.add_rate:
            result = mutate_add_module(
                result,
                rng=self._rng,
                library_size=len(self._library),
                max_length=self._max_module_length,
                partial_prob=self._config.partial_prob,
                max_modules=self._config.max_modules,
            )
        if _rand(self._rng) < self._config.delete_rate:
            result = mutate_delete_module(result, rng=self._rng)
        if _rand(self._rng) < self._config.mutation_rate:
            result = mutate_order(
                result,
                rng=self._rng,
                library_size=len(self._library),
            )
        return result

    def _random_genome(self) -> Genome:
        return random_genome(
            rng=self._rng,
            library_size=len(self._library),
            max_modules=self._config.max_modules,
            max_module_length=self._max_module_length,
            partial_prob=self._config.partial_prob,
            genes_prob=self._config.genes_prob,
        )

    def _make_solver(self) -> SparseSolver:
        if self._config.solver == "svd_null_space":
            return SVDNullSpaceSolver()
        return LeastSquaresSolver()

    def refuse_unmodeled_fields(self, dataset: PDEDataset) -> None:
        fields = set(dataset.fields or {})
        axes = set(dataset.axis_order or ())
        terms = self._evaluated_terms()
        for term in terms:
            tree = ast.parse(term, mode="eval")
            heads = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            unknown = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Name):
                    continue
                if node.id in heads:
                    match = DIFF_OPERATOR_PATTERN.fullmatch(node.id)
                    if match is not None and match.group(2) not in axes:
                        unknown.add(node.id)
                elif (
                    node.id not in fields | axes | {UNITY_TOKEN}
                    and parse_compound_derivative(
                        node.id, known_fields=fields, known_axes=axes
                    )
                    is None
                ):
                    unknown.add(node.id)
            if unknown:
                raise ValueError(
                    f"DLGA library term {term!r} cannot resolve "
                    f"{sorted(unknown)} on dataset fields "
                    f"{sorted(fields)} and axes {sorted(axes)}; "
                    "set library from this input's fields and axes."
                )
        unmodeled = _unmodeled_field_terms(terms, dataset)
        if unmodeled:
            uses = "; ".join(
                f"{term!r} uses {sorted(fields)!r}"
                for term, fields in unmodeled.items()
            )
            raise ValueError(
                "DLGA fits a surrogate of the left-hand field "
                f"{dataset.lhs_field!r} only, so library or anchored terms on "
                f"other dataset fields cannot be evaluated: {uses}. Remove these "
                "terms, or use SGA, PySINDy or PySR, which read other fields "
                "from the data."
            )

    def _evaluated_terms(self) -> tuple[str, ...]:
        if self._sketch is None:
            return self._library
        return (*self._library, *(item.term_ir for item in self._sketch.anchored))

    def refuse_lhs_terms(self, dataset: PDEDataset) -> None:
        if self._sketch is not None:
            return
        field, axis = dataset.lhs_field, dataset.lhs_axis


        open_forms = {
            1: (f"diff_{axis}({field})", f"diff1_{axis}({field})"),
            2: (f"diff2_{axis}({field})",),
        }
        targets: dict[str, str] = {}
        for order in self._fitted_orders():
            label = render_lhs_label(LhsSpec(field, axis, order))
            targets |= dict.fromkeys((label, *open_forms[order]), label)
        for term in self._library:
            if term in targets:
                raise ValueError(
                    f"library contains LHS {targets[term]!r} as {term!r}: "
                    "tautology; remove that term"
                )

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
            expression = genome_to_expression(genome, self._library)
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
