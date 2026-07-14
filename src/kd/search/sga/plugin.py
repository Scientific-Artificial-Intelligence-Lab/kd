
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.core.linear_solve import R2_EPS_RES, R2_EPS_TOT, r2_score
from kd.core.metrics import nmse as metrics_nmse
from kd.core.platform.requirements import DerivativeReqs
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.derivatives.finite_diff import DX_ZERO_FLOOR, UNIFORM_GRID_RTOL
from kd.models.field_model import FieldModel
from kd.models.trainer import FieldModelTrainer, TrainingResult
from kd.search.dlga import surrogate_log as _surrogate_log
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder, log_whitelisted_metrics
from kd.search.result import invalid_evaluation_result
from kd.search.sga import tree_render as _tree_render
from kd.search.sga import viz as _viz_helpers
from kd.search.sga.config import OPS, ROOT, SGAConfig, build_den
from kd.search.sga.convert import pde_to_kd_expr, tree_to_kd_expr
from kd.search.sga.evaluate import DiffContext, build_theta, execute_pde
from kd.search.sga.pde import PDE
from kd.search.sga.train import CandidateResult, TrainResult, evaluate_candidate
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_INVALID_AIC = float("inf")

_AIC_LOWER_BOUND = -100.0

_MAX_RESAMPLE_PER_INDIVIDUAL = 50

_FAILED_EVAL_ERROR_MESSAGE = "Candidate evaluation failed"


@dataclass
class _ScoredPDE:

    pde: PDE
    score: float
    result: EvaluationResult


def _is_valid_aic(aic: float) -> bool:
    return math.isfinite(aic) and aic >= _AIC_LOWER_BOUND


def _aic_of(result: EvaluationResult) -> float:
    return result.score if result.score is not None else _INVALID_AIC








_LOGGED_METRICS: tuple[str, ...] = (
    "gen_best_aic",
    "gen_mean_aic",
    "gen_best_nmse",
    "n_valid",
    "n_unique",
    "gen_mean_complexity",
)









_SURROGATE_METRICS = _surrogate_log._SURROGATE_METRICS


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for s in shape:
        result *= s
    return result


def _rand_float(rng: torch.Generator) -> float:
    return float(torch.rand(1, generator=rng).item())


def _numeric_flatten(tensor: Tensor) -> Tensor:
    return tensor.detach().flatten()


def _safe_evaluate_aic(
    pde: PDE,
    data_dict: dict[str, Tensor],
    default_terms: Tensor | None,
    y: Tensor | None,
    config: SGAConfig,
    diff_ctx: DiffContext | None,
) -> tuple[float, PDE]:
    try:
        cr = evaluate_candidate(
            pde,
            data_dict,
            default_terms,
            y if y is not None else torch.zeros(1),
            config,
            diff_ctx=diff_ctx,
        )
        return cr.aic_score, cr.pruned_pde
    except Exception:
        logger.debug("Evaluation failed for PDE, assigning inf AIC")
        return _INVALID_AIC, pde


class SGAPlugin:




    score_kind: ClassVar[str] = "AIC"
    score_direction: ClassVar[Literal["min", "max"]] = "min"


    config_cls: ClassVar[type[SGAConfig]] = SGAConfig
    one_shot: ClassVar[bool] = False

    def __init__(self, config: SGAConfig | None = None) -> None:
        self._config = config or SGAConfig()
        self._population: list[PDE] | None = None
        self._scores: list[float] | None = None
        self._best_score: float = float("inf")
        self._best_expression: str = ""
        self._best_formatted_cache: str | None = None
        self._vars: list[str] = []
        self._data_dict: dict[str, Tensor] = {}
        self._den: tuple[tuple[str, int], ...] = ()
        self._diff_ctx: DiffContext | None = None
        self._default_terms: Tensor | None = None
        self._default_term_name: str | None = None
        self._y: Tensor | None = None
        self._rng: torch.Generator = torch.Generator()
        self._prepared: bool = False
        self._restore_pending: bool = False
        self._offspring: list[PDE] | None = None
        self._offspring_results: list[EvaluationResult] | None = None
        self._pending_population: list[PDE] | None = None
        self._pending_scores: list[float] | None = None
        self._recorder: VizRecorder | None = None
        self._autograd_provider: AutogradProvider | None = None

        self._surrogate_training_result: TrainingResult | None = None

        self._pde_lib: set[str] = set()

        self._repeat_cross: int = 0

        self._repeat_change: int = 0

    @property
    def _delta(self) -> dict[str, float]:
        if self._diff_ctx is None:
            return {}
        return self._diff_ctx.delta

    @property
    def _lhs_axis(self) -> str | None:
        if self._diff_ctx is None:
            return None
        return self._diff_ctx.lhs_axis

    @property
    def config(self) -> dict[str, Any]:
        return {
            "algorithm": "sga",
            **asdict(self._config),
        }

    @property
    def runner_batch_size(self) -> int:
        return self._config.num

    @property
    def derivative_requirements(self) -> DerivativeReqs:
        return DerivativeReqs(
            provider_kind="finite_diff",
            max_atomic_order=2,
            lhs_order=1,
            needs_surrogate=False,
        )



    def prepare(self, components: PlatformComponents) -> None:
        self._prepared = False
        self._clear_pending_generation()






        if not self._restore_pending:
            self._reset_search_state()
        self._restore_pending = False
        dataset = components.dataset
        context = components.context
        if context is None:




            raise TypeError(
                "SGAPlugin.prepare requires components.context (variable "
                "lookup); got None (a provider_kind='none' light bundle)."
            )
        self._recorder = components.recorder


        self._validate_naming(dataset)


        field_shape: tuple[int, ...] | None = None
        if dataset.fields is not None:
            for fd in dataset.fields.values():
                field_shape = tuple(fd.values.shape)
                break


        data_dict: dict[str, Tensor] = {}

        if dataset.fields is not None:
            for field_name in dataset.fields:
                data_dict[field_name] = _numeric_flatten(
                    context.get_variable(field_name)
                )

        axis_names = self._ordered_axes(dataset)
        self._den = build_den(axis_names, dataset.lhs_axis or "")
        delta = self._build_delta_map(dataset, axis_names)
        if field_shape is not None:
            axis_map = {axis_name: idx for idx, axis_name in enumerate(axis_names)}
            self._diff_ctx = DiffContext(
                field_shape=field_shape,
                axis_map=axis_map,
                delta=delta,
                lhs_axis=dataset.lhs_axis,
            )
        else:
            self._diff_ctx = None




        n_flat = _product(field_shape) if field_shape is not None else 0
        if dataset.axes is not None and field_shape is not None:
            for axis_name in dataset.axes:
                if axis_name == dataset.lhs_axis:
                    continue
                coord = context.get_variable(axis_name)
                if coord.numel() == n_flat:
                    data_dict[axis_name] = _numeric_flatten(coord)
                else:
                    broadcast = self._broadcast_coord(
                        coord, axis_name, dataset, field_shape
                    )
                    data_dict[axis_name] = _numeric_flatten(broadcast)



        self._autograd_provider = None
        self._surrogate_training_result = None
        if self._config.use_autograd:
            if field_shape is None:
                raise ValueError(
                    "use_autograd=True requires dataset.fields with grid shape."
                )
            self._autograd_provider = self._build_autograd_provider(
                dataset, field_shape
            )









            _surrogate_log.log_surrogate_training(
                self._recorder,
                self._surrogate_training_result,
            )

        self._add_derivatives(dataset, context, data_dict)
        self._data_dict = data_dict


        self._vars = sorted(data_dict.keys())



        self._y = self._extract_lhs_target(dataset, context)


        if dataset.lhs_field and dataset.lhs_field in data_dict:
            self._default_terms = data_dict[dataset.lhs_field].flatten().unsqueeze(1)
            self._default_term_name = dataset.lhs_field
        else:
            self._default_terms = None
            self._default_term_name = None


        if self._population is None:
            if not self._vars:
                raise ValueError(
                    "No variables available for SGA. "
                    "Check dataset fields, axes, and lhs configuration."
                )
            self._rng.manual_seed(self._config.seed)
            self._init_population()

        self._prepared = True

    def propose(self, n: int) -> list[str]:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before propose()")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        offspring = self._apply_genetic_ops()


        self._offspring = offspring
        return [pde_to_kd_expr(pde) for pde in offspring]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before evaluate()")
        if not candidates:
            return []

        if self._offspring_results is not None:
            return [
                self._offspring_results[i]
                if i < len(self._offspring_results)
                else self._invalid_result(expr_str)
                for i, expr_str in enumerate(candidates)
            ]

        offspring = self._offspring or []
        results: list[EvaluationResult] = []
        for i, expr_str in enumerate(candidates):
            if i < len(offspring):
                try:
                    cr = evaluate_candidate(
                        offspring[i],
                        self._data_dict,
                        self._default_terms,
                        self._y if self._y is not None else torch.zeros(1),
                        self._config,
                        diff_ctx=self._diff_ctx,
                    )

                    offspring[i] = cr.pruned_pde
                    results.append(self._to_eval_result(cr, expr_str))
                except Exception:
                    logger.debug("Evaluation failed for candidate %d", i)
                    results.append(self._invalid_result(expr_str))
            else:
                results.append(self._invalid_result(expr_str))
        return results

    def update(self, results: list[EvaluationResult]) -> None:
        if self._pending_population is not None and self._pending_scores is not None:
            self._commit_pending_generation()
            return

        if not results:
            return

        offspring = self._offspring or []
        population = list(self._population or [])
        scores = list(self._scores or [])

        for i, result in enumerate(results):
            if i < len(offspring):
                aic = result.score if result.score is not None else _INVALID_AIC
                if not result.is_valid or not math.isfinite(aic):
                    aic = _INVALID_AIC
                population.append(offspring[i])
                scores.append(aic)

        if population:
            paired = list(zip(scores, population, strict=True))
            paired.sort(key=lambda x: x[0])
            scores = [s for s, _ in paired]
            population = [p for _, p in paired]

            num = self._config.num
            self._population = population[:num]
            self._scores = scores[:num]

            if self._scores and self._scores[0] < self._best_score:
                self._best_score = self._scores[0]
                self._best_expression = pde_to_kd_expr(self._population[0])
                self._best_formatted_cache = None

        if self._recorder is not None:


            self._recorder.log("best_aic", self._best_score)

        self._offspring = None
        self._offspring_results = None







    def list_plots(self) -> list[PlotInfo]:
        return [*_viz_helpers.list_plot_infos(), _tree_render.genome_tree_info()]

    def render_plot(self, name: str, ax: Axes) -> None:
        if name == _tree_render.GENOME_TREE_INFO.name:
            _tree_render.render_genome_tree(ax, self._best_pde())
            return
        _viz_helpers.render(name, ax, self._recorder)

    def get_plot_data(self, name: str) -> dict[str, Any]:
        if name == _tree_render.GENOME_TREE_INFO.name:
            return _tree_render.genome_tree_data(self._best_pde())
        return _viz_helpers.get_data(name, self._recorder)

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_expression(self) -> str:
        if self._best_formatted_cache is None:
            self._best_formatted_cache = self._format_best_expression()
        return self._best_formatted_cache

    def build_final_result(self) -> EvaluationResult:
        best_pde = self._best_pde()
        if best_pde is None or self._y is None:
            return self._invalid_final_result("No best PDE available for final result")

        try:
            candidate = evaluate_candidate(
                best_pde,
                self._data_dict,
                self._default_terms,
                self._y,
                self._config,
                diff_ctx=self._diff_ctx,
            )
            predicted = self._predict_rhs(candidate.pruned_pde, candidate.coefficients)
        except Exception as exc:
            logger.debug("Failed to build final result", exc_info=exc)
            return self._invalid_final_result(f"Final result evaluation failed: {exc}")

        residuals = (predicted - self._y).detach()
        is_valid = _is_valid_aic(candidate.aic_score) and math.isfinite(candidate.mse)
        return EvaluationResult(
            mse=candidate.mse,
            nmse=metrics_nmse(candidate.mse, self._target_variance()),
            r2=self._compute_r2(predicted),
            score=candidate.aic_score,
            complexity=len(candidate.selected_indices),
            coefficients=candidate.coefficients.detach(),
            is_valid=is_valid,
            error_message="" if is_valid else "Invalid AIC or MSE",
            selected_indices=list(candidate.selected_indices),
            residuals=residuals,
            terms=self._build_term_list(candidate.pruned_pde),
            expression=self._best_expression,
        )

    def build_result_target(self) -> Tensor:
        if self._y is None:
            return torch.zeros(0)
        return self._y.detach().clone()

    @property
    def state(self) -> dict[str, Any]:
        return {
            "population": self._population,
            "scores": self._scores,
            "best_score": self._best_score,
            "best_expression": self._best_expression,
            "vars": self._vars,
            "rng_state": self._rng.get_state().numpy().tobytes(),
            "pde_lib": list(self._pde_lib),
            "repeat_cross": self._repeat_cross,
            "repeat_change": self._repeat_change,
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        if not value:
            self._reset_search_state()
            self._clear_pending_generation()
            self._restore_pending = False
            return
        self._reset_search_state()
        if "population" in value:
            self._population = value["population"]
        if "scores" in value:
            self._scores = value["scores"]
        if "best_score" in value:
            self._best_score = value["best_score"]
        if "best_expression" in value:
            self._best_expression = value["best_expression"]
            self._best_formatted_cache = None
        if "vars" in value:
            self._vars = value["vars"]
        if "rng_state" in value:
            rng_state = value["rng_state"]
            if isinstance(rng_state, bytes):
                import numpy as np

                rng_state = torch.from_numpy(
                    np.frombuffer(rng_state, dtype=np.uint8).copy()
                )
            self._rng.set_state(rng_state)
        self._pde_lib = set(value.get("pde_lib", []))
        self._repeat_cross = int(value.get("repeat_cross", 0))
        self._repeat_change = int(value.get("repeat_change", 0))
        self._clear_pending_generation()
        self._restore_pending = True



    def _reset_search_state(self) -> None:
        self._population = None
        self._scores = None
        self._best_score = float("inf")
        self._best_expression = ""
        self._best_formatted_cache = None
        self._pde_lib = set()
        self._repeat_cross = 0
        self._repeat_change = 0

    def _clear_pending_generation(self) -> None:
        self._offspring = None
        self._offspring_results = None
        self._pending_population = None
        self._pending_scores = None

    def _best_pde(self) -> PDE | None:
        population = self._population or []
        if not population:
            return None
        return population[0]

    def _predict_rhs(self, pde: PDE, coefficients: Tensor) -> Tensor:
        valid_terms, _ = execute_pde(pde, self._data_dict, self._diff_ctx)
        theta = build_theta(valid_terms, self._default_terms)
        if theta.shape[1] == 0:
            if self._y is None:
                return torch.zeros(0)
            return torch.zeros_like(self._y)
        return (theta @ coefficients).detach()

    def _build_term_list(self, pde: PDE) -> list[str]:
        terms = [tree_to_kd_expr(tree) for tree in pde.terms]
        if self._default_term_name is not None:
            return [self._default_term_name, *terms]
        return terms

    def _target_variance(self) -> float:
        if self._y is None or self._y.numel() < 2:
            return 0.0
        var = float(torch.var(self._y, correction=0).item())
        return var if math.isfinite(var) else 0.0

    def _format_best_expression(self) -> str:
        try:
            res = self.build_final_result()
        except Exception:
            res = None

        if (
            res is not None
            and res.is_valid
            and res.coefficients is not None
            and res.terms
        ):
            rhs = self._format_rhs_with_coefficients(res.coefficients, res.terms)
            if rhs:
                lhs = self._lhs_label_str()
                return f"{lhs} = {rhs}" if lhs else rhs

        if self._best_expression:
            return self._best_expression
        if self._default_term_name:
            return self._default_term_name
        return ""

    def _lhs_label_str(self) -> str:
        if self._default_term_name and self._lhs_axis:
            return f"{self._default_term_name}_{self._lhs_axis}"
        return ""

    @staticmethod
    def _format_rhs_with_coefficients(coefficients: Tensor, terms: list[str]) -> str:
        if coefficients.numel() == 0 or not terms:
            return ""
        n = min(coefficients.numel(), len(terms))
        active: list[tuple[float, str]] = []
        for i in range(n):
            c = float(coefficients[i].item())
            if abs(c) < 1e-10:
                continue
            active.append((c, terms[i]))
        if not active:
            return ""
        parts: list[str] = []
        for i, (c, name) in enumerate(active):
            mag = f"{abs(c):.4g}"
            if i == 0:
                parts.append(f"-{mag}*{name}" if c < 0 else f"{mag}*{name}")
            else:
                parts.append(f"{'-' if c < 0 else '+'} {mag}*{name}")
        return " ".join(parts)

    def _compute_r2(self, predicted: Tensor) -> float:
        if self._y is None:
            return -float("inf")
        return r2_score(predicted, self._y)

    def _invalid_final_result(self, error_message: str) -> EvaluationResult:
        return invalid_evaluation_result(
            error_message,
            score=float("inf"),
            expression=self._best_expression,
        )

    def _broadcast_coord(
        self,
        coord: Tensor,
        axis_name: str,
        dataset: Any,
        field_shape: tuple[int, ...],
    ) -> Tensor:
        if dataset.axis_order is None:
            return coord.flatten()
        axis_idx = dataset.axis_order.index(axis_name)
        shape = [1] * len(field_shape)
        shape[axis_idx] = coord.shape[0]
        return coord.view(*shape).expand(field_shape).contiguous()

    def _extract_lhs_target(
        self,
        dataset: Any,
        context: Any,
    ) -> Tensor:
        if dataset.lhs_field and dataset.lhs_axis:
            lhs_order = getattr(dataset, "lhs_order", 1)






            if lhs_order != 1:
                raise ValueError(
                    f"SGA only supports a first-order LHS (u_t); the dataset "
                    f"declares lhs_order={lhs_order} "
                    f"({dataset.lhs_field}_{dataset.lhs_axis * lhs_order}). "
                    f"Second-order LHS (u_tt) discovery is not supported by SGA "
                    f"(deferred to DATA-4)."
                )
            try:
                get_deriv = self._resolve_get_derivative(context)
                deriv: Tensor = get_deriv(
                    dataset.lhs_field, dataset.lhs_axis, lhs_order
                )
                return _numeric_flatten(deriv)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"LHS derivative {dataset.lhs_field}_{dataset.lhs_axis} "
                    f"not available. SGA requires the LHS target derivative "
                    f"to be computable from the dataset."
                ) from exc
        raise ValueError(
            "LHS field and axis must both be set for SGA. "
            f"Got lhs_field={dataset.lhs_field!r}, lhs_axis={dataset.lhs_axis!r}."
        )

    def _validate_naming(self, dataset: Any) -> None:
        field_names = list(dataset.fields.keys()) if dataset.fields else []
        axis_names = list(dataset.axes.keys()) if dataset.axes else []


        overlap = sorted(set(field_names) & set(axis_names))
        if overlap:
            raise ValueError(
                f"Field names conflict with axis names: {overlap}. "
                f"Fields and axes must have distinct names."
            )


        deriv_keys = {
            f"{field}_{axis * order}"
            for field in field_names
            for axis in axis_names
            for order in (1, 2)
        }
        deriv_overlap = sorted(set(field_names) & deriv_keys)
        if deriv_overlap:
            raise ValueError(
                f"Field names conflict with derivative keys: {deriv_overlap}. "
                f"Rename the field to avoid ambiguity."
            )


        lhs_field = dataset.lhs_field
        lhs_axis = dataset.lhs_axis
        if lhs_field and lhs_axis and len(lhs_axis) == 1:
            legacy_alias = f"{lhs_field}{lhs_axis}"
            if legacy_alias in field_names:
                raise ValueError(
                    f"Field name '{legacy_alias}' conflicts with LHS derivative "
                    f"alias ({lhs_field}_{lhs_axis}). This would allow the LHS "
                    f"derivative to leak into the RHS variable pool."
                )

    def _add_derivatives(
        self,
        dataset: Any,
        context: Any,
        data_dict: dict[str, Tensor],
    ) -> None:
        if dataset.fields is None or dataset.axes is None:
            return
        for field_name in dataset.fields:
            for axis_name in dataset.axes:
                if axis_name == dataset.lhs_axis:
                    continue
                key = f"{field_name}_{axis_name}"
                try:
                    get_deriv = self._resolve_get_derivative(context)
                    deriv = get_deriv(field_name, axis_name, 1)
                    data_dict[key] = _numeric_flatten(deriv)
                except (KeyError, ValueError):
                    logger.debug("Derivative %s not available", key)

    def _resolve_get_derivative(
        self,
        context: Any,
    ) -> Callable[[str, str, int], Tensor]:
        if self._autograd_provider is not None:
            return self._autograd_provider.get_derivative
        return context.get_derivative

    def _build_autograd_provider(
        self,
        dataset: Any,
        field_shape: tuple[int, ...],
    ) -> AutogradProvider:
        if dataset.fields is None or dataset.axes is None:
            raise ValueError(
                "use_autograd=True requires dataset.fields and dataset.axes."
            )

        field_names = list(dataset.fields.keys())
        coord_names = self._ordered_axes(dataset)
        if not coord_names:
            raise ValueError("use_autograd=True requires at least one ordered axis.")

        flat_coords: dict[str, Tensor] = {}
        for axis_name in coord_names:
            coord_1d = dataset.axes[axis_name].values
            broadcast = self._broadcast_coord(coord_1d, axis_name, dataset, field_shape)
            flat_coords[axis_name] = broadcast.flatten()

        flat_targets: dict[str, Tensor] = {
            name: fdata.values.flatten() for name, fdata in dataset.fields.items()
        }

        if self._config.field_model is not None:
            field_model: FieldModel = self._config.field_model
            self._validate_field_model(field_model, coord_names, field_names)
        else:
            field_model = FieldModel(
                coord_names=coord_names,
                field_names=field_names,
            )
            trainer = FieldModelTrainer(field_model, lr=self._config.autograd_train_lr)







            self._surrogate_training_result = trainer.fit(
                coords=flat_coords,
                targets=flat_targets,
                max_epochs=self._config.autograd_train_epochs,
                patience=self._config.autograd_train_patience,
                val_ratio=self._config.autograd_train_val_ratio,
                seed=self._config.seed,
            )

        grad_coords = {
            name: c.detach().clone().requires_grad_(True)
            for name, c in flat_coords.items()
        }
        return AutogradProvider(
            model=field_model,
            coords=grad_coords,
            dataset=dataset,
            max_order=1,
        )

    @staticmethod
    def _validate_field_model(
        field_model: FieldModel,
        coord_names: list[str],
        field_names: list[str],
    ) -> None:
        if list(field_model.coord_names) != list(coord_names):
            raise ValueError(
                f"field_model.coord_names {field_model.coord_names} != "
                f"dataset axes {coord_names}"
            )
        if list(field_model.field_names) != list(field_names):
            raise ValueError(
                f"field_model.field_names {field_model.field_names} != "
                f"dataset fields {field_names}"
            )

    def _ordered_axes(self, dataset: Any) -> list[str]:
        if dataset.axis_order is not None:
            return list(dataset.axis_order)
        if dataset.axes is None:
            return []
        return list(dataset.axes.keys())

    def _build_delta_map(
        self,
        dataset: Any,
        axis_names: list[str],
    ) -> dict[str, float]:
        if dataset.axes is None:
            return {}
        delta: dict[str, float] = {}
        for axis_name in axis_names:
            if axis_name not in dataset.axes:
                continue
            coord = dataset.axes[axis_name].values.reshape(-1)
            if coord.numel() < 2:
                continue
            diffs = torch.diff(coord)
            first_diff = float(diffs[0].item())
            if abs(first_diff) < DX_ZERO_FLOOR:
                raise ValueError(
                    f"Structured grid axis '{axis_name}' has degenerate spacing "
                    f"dx={first_diff:.6g}; finite-difference stencils require "
                    "nonzero dx."
                )
            max_diff = float(diffs.max().item())
            min_diff = float(diffs.min().item())
            if abs(max_diff - min_diff) > abs(first_diff) * UNIFORM_GRID_RTOL:
                raise ValueError(
                    f"Structured grid requires uniform spacing for axis '{axis_name}' "
                    f"(min spacing = {min_diff:.6e}, max spacing = {max_diff:.6e})."
                )
            if not (torch.all(diffs > 0) or torch.all(diffs < 0)):
                raise ValueError(
                    f"Structured grid axis '{axis_name}' must be strictly monotonic."
                )
            delta[axis_name] = first_diff
        return delta

    def _init_population(self) -> None:
        from kd.search.sga.genetic import random_pde

        population: list[PDE] = []
        scores: list[float] = []
        for i in range(self._config.num):
            pde = random_pde(self._config, self._vars, OPS, ROOT, self._den, self._rng)
            aic, pruned = _safe_evaluate_aic(
                pde,
                self._data_dict,
                self._default_terms,
                self._y,
                self._config,
                self._diff_ctx,
            )

            retries = 0
            while not _is_valid_aic(aic) and retries < _MAX_RESAMPLE_PER_INDIVIDUAL:
                logger.debug(
                    "Init individual %d: AIC=%s, resampling (retry %d/%d)",
                    i,
                    aic,
                    retries + 1,
                    _MAX_RESAMPLE_PER_INDIVIDUAL,
                )
                pde = random_pde(
                    self._config,
                    self._vars,
                    OPS,
                    ROOT,
                    self._den,
                    self._rng,
                )
                aic, pruned = _safe_evaluate_aic(
                    pde,
                    self._data_dict,
                    self._default_terms,
                    self._y,
                    self._config,
                    self._diff_ctx,
                )
                retries += 1

            if not _is_valid_aic(aic):
                raise RuntimeError(
                    f"Init population failed: individual {i} still has "
                    f"invalid AIC after {_MAX_RESAMPLE_PER_INDIVIDUAL} resample "
                    f"retries. All random candidates are pathological — "
                    f"check data, derivative quality, or STRidge config."
                )


            population.append(pruned)
            scores.append(aic)

        paired = list(zip(scores, population, strict=True))
        paired.sort(key=lambda x: x[0])
        self._scores = [s for s, _ in paired]
        self._population = [p for _, p in paired]


        if self._scores and math.isfinite(self._scores[0]):
            self._best_score = self._scores[0]
            self._best_expression = pde_to_kd_expr(self._population[0])
            self._best_formatted_cache = None

    def _apply_genetic_ops(self) -> list[PDE]:
        from kd.search.sga.config import OP1, OP2
        from kd.search.sga.genetic import crossover, mutate, replace



        self._validate_dedup_mode()

        population = list(self._population or [])
        if not population:
            self._pending_population = []
            self._pending_scores = []
            self._offspring_results = []
            return []

        self._repeat_cross = 0
        self._repeat_change = 0

        cfg = self._config
        scores = self._current_scores(len(population))
        population, scores = self._sort_truncate_population(population, scores)
        evaluated_offspring: list[_ScoredPDE] = []

        def bump_cross() -> None:
            self._repeat_cross += 1

        def bump_change() -> None:
            self._repeat_change += 1








        num_cross = int(len(population) * cfg.p_cro)
        if num_cross > 0:
            top = population[:num_cross]
            perm = torch.randperm(len(top), generator=self._rng).tolist()
            shuffled = [top[i] for i in perm]
            for orig, partner in zip(top, shuffled, strict=True):
                c1, c2 = crossover(orig, partner, self._rng)
                for child in (c1, c2):
                    scored = self._dedup_and_score(child, bump_cross)
                    if scored is not None:
                        evaluated_offspring.append(scored)

            population, scores = self._merge_sort_truncate(
                population,
                scores,
                evaluated_offspring,
            )


        mutation_offspring: list[_ScoredPDE] = []
        for i in range(1, len(population)):
            m = mutate(
                population[i],
                self._vars,
                OP1,
                OP2,
                self._den,
                cfg.p_mute,
                self._rng,
            )
            if _rand_float(self._rng) < cfg.p_rep:
                m = replace(
                    m,
                    self._vars,
                    OPS,
                    ROOT,
                    self._den,
                    cfg.depth,
                    cfg.p_var,
                    self._rng,
                )
            scored = self._dedup_and_score(m, bump_change)
            if scored is not None:
                mutation_offspring.append(scored)

        if mutation_offspring:
            evaluated_offspring.extend(mutation_offspring)
            population, scores = self._merge_sort_truncate(
                population,
                scores,
                mutation_offspring,
            )

        self._pending_population = population
        self._pending_scores = scores
        self._offspring_results = [item.result for item in evaluated_offspring]
        return [item.pde for item in evaluated_offspring]

    def _validate_dedup_mode(self) -> None:
        valid_modes = ("none", "pre_prune", "post_prune", "dual")
        if self._config.dedup_mode not in valid_modes:
            raise ValueError(
                f"Invalid dedup_mode: {self._config.dedup_mode!r}. "
                f"Must be one of {valid_modes}."
            )

    def _dedup_and_score(
        self,
        candidate_pde: PDE,
        on_duplicate: Callable[[], None],
    ) -> _ScoredPDE | None:
        mode = self._config.dedup_mode

        if mode == "none":
            return self._score_offspring(candidate_pde)

        pre_key: str | None = None
        if mode in ("pre_prune", "dual"):
            pre_key = pde_to_kd_expr(candidate_pde)
            if pre_key in self._pde_lib:
                on_duplicate()
                return None
            if mode == "pre_prune":
                self._pde_lib.add(pre_key)

        scored = self._score_offspring(candidate_pde)

        if mode in ("post_prune", "dual"):








            if scored.result.error_message == _FAILED_EVAL_ERROR_MESSAGE:
                return scored




            if mode == "dual" and pre_key is not None:
                self._pde_lib.add(pre_key)










            post_key = scored.result.expression
            if post_key in self._pde_lib:
                on_duplicate()
                return None
            self._pde_lib.add(post_key)

        return scored

    def _current_scores(self, population_len: int) -> list[float]:
        if self._scores is None or len(self._scores) != population_len:
            return [_INVALID_AIC] * population_len
        return list(self._scores)

    def _merge_sort_truncate(
        self,
        population: list[PDE],
        scores: list[float],
        offspring: list[_ScoredPDE],
    ) -> tuple[list[PDE], list[float]]:
        merged_population = [*population, *(item.pde for item in offspring)]
        merged_scores = [*scores, *(item.score for item in offspring)]
        return self._sort_truncate_population(merged_population, merged_scores)

    def _sort_truncate_population(
        self,
        population: list[PDE],
        scores: list[float],
    ) -> tuple[list[PDE], list[float]]:
        if not population:
            return [], []
        paired = list(zip(scores, population, strict=True))
        paired.sort(key=lambda x: x[0])
        paired = paired[: self._config.num]
        return [p for _, p in paired], [s for s, _ in paired]

    def _score_offspring(self, pde: PDE) -> _ScoredPDE:
        expression = pde_to_kd_expr(pde)
        if not self._prepared:
            return _ScoredPDE(
                pde,
                _INVALID_AIC,
                self._evaluation_failed_result(expression),
            )
        try:
            candidate = evaluate_candidate(
                pde,
                self._data_dict,
                self._default_terms,
                self._y if self._y is not None else torch.zeros(1),
                self._config,
                diff_ctx=self._diff_ctx,
            )
            pruned = candidate.pruned_pde
            result = self._to_eval_result(candidate, pde_to_kd_expr(pruned))
            aic = result.score if result.score is not None else _INVALID_AIC
            score = aic if result.is_valid and math.isfinite(aic) else _INVALID_AIC
            return _ScoredPDE(pruned, score, result)
        except Exception:
            logger.debug("Evaluation failed for staged offspring")
            return _ScoredPDE(
                pde,
                _INVALID_AIC,
                self._evaluation_failed_result(expression),
            )

    def _commit_pending_generation(self) -> None:
        self._population = self._pending_population
        self._scores = self._pending_scores

        if self._scores and self._scores[0] < self._best_score:
            self._best_score = self._scores[0]
            if self._population:
                self._best_expression = pde_to_kd_expr(self._population[0])
                self._best_formatted_cache = None

        if self._recorder is not None:
            self._recorder.log("best_aic", self._best_score)


            self._log_generation_metrics(self._offspring_results or [])

        self._clear_pending_generation()

    def _log_generation_metrics(self, results: list[EvaluationResult]) -> None:
        recorder = self._recorder
        if recorder is None:
            return
        valid_results = [result for result in results if result.is_valid]
        if valid_results:
            best = min(valid_results, key=_aic_of)
            gen_best_aic = _aic_of(best)
            gen_mean_aic = sum(_aic_of(r) for r in valid_results) / len(valid_results)
            gen_best_nmse = best.nmse
            gen_mean_complexity = sum(r.complexity for r in valid_results) / len(
                valid_results
            )
        else:
            gen_best_aic = _INVALID_AIC
            gen_mean_aic = _INVALID_AIC
            gen_best_nmse = float("inf")
            gen_mean_complexity = 0.0
        metrics: dict[str, float | int] = {
            "gen_best_aic": gen_best_aic,
            "gen_mean_aic": gen_mean_aic,
            "gen_best_nmse": gen_best_nmse,
            "n_valid": len(valid_results),
            "n_unique": len({result.expression for result in results}),
            "gen_mean_complexity": gen_mean_complexity,
        }
        log_whitelisted_metrics(recorder, _LOGGED_METRICS, metrics)

    def _r2_from_mse(self, mse: float) -> float:
        if not math.isfinite(mse):
            return -math.inf
        target_var = self._target_variance()
        if target_var < R2_EPS_TOT:
            return 1.0 if mse < R2_EPS_RES else 0.0
        return 1.0 - mse / target_var

    def _to_eval_result(
        self, result: CandidateResult | TrainResult, expression: str
    ) -> EvaluationResult:

        aic = result.aic_score
        mse = result.mse
        is_valid = _is_valid_aic(aic) and math.isfinite(mse)
        return EvaluationResult(
            mse=mse,
            nmse=metrics_nmse(mse, self._target_variance()),
            r2=self._r2_from_mse(mse),
            score=aic,
            complexity=len(result.selected_indices),
            coefficients=result.coefficients,
            is_valid=is_valid,
            error_message="" if is_valid else "Invalid AIC or MSE",
            selected_indices=result.selected_indices,
            residuals=None,
            terms=None,
            expression=expression,
        )

    def _invalid_result(self, expression: str) -> EvaluationResult:
        return invalid_evaluation_result(
            "No corresponding PDE for evaluation",
            score=float("inf"),
            expression=expression,
        )

    def _evaluation_failed_result(self, expression: str) -> EvaluationResult:
        result = self._invalid_result(expression)
        result.error_message = _FAILED_EVAL_ERROR_MESSAGE
        return result
