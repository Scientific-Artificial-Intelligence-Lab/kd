
from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from kd.core.equation import Form
from kd.core.evaluator import EvaluationResult
from kd.core.platform.requirements import DerivativeReqs
from kd.core.platform.sketch_compile import CompileReport
from kd.data.schema import DataTopology
from kd.llm import (
    BudgetedProvider,
    LLMBudgetExhausted,
    LLMParams,
    LLMRequest,
    OpenAICompatProvider,
    TapeRecordingProvider,
)
from kd.search.descriptor import (
    InstrumentDescriptor,
    InstrumentMode,
    Knob,
    Segmentation,
)
from kd.search.llm4ed import viz as _viz_helpers
from kd.search.llm4ed.columns import evaluate_ir_column
from kd.search.llm4ed.config import (
    ALGORITHM_NAME,
    Llm4edConfig,
    config_to_json_safe_dict,
)
from kd.search.llm4ed.fd import build_operand_columns_axes
from kd.search.llm4ed.filter_score import filter_score
from kd.search.llm4ed.parse import is_placeholder, parse_equation, reads_as_symbol
from kd.search.llm4ed.pool import ElitePool, PoolItem
from kd.search.llm4ed.prompts import (
    DEFAULT_PDE_OPERATORS,
    EVOLUTION,
    INITIALIZATION,
    OPTIMIZE,
    build_evolution_prompt,
    build_initialization_prompt,
    build_optimize_prompt,
    build_tabular_evolution_prompt,
    build_tabular_initialization_prompt,
    build_tabular_optimize_prompt,
    parse_response,
    parse_tabular_response,
    pde_operands,
    permute_terms,
)
from kd.search.llm4ed.score import (
    ERROR_ABNORMAL_COEF,
    ERROR_INEXPRESSIBLE,
    ERROR_LSTSQ,
    ERROR_NON_FINITE,
    ERROR_PARSE,
    ERROR_SKETCH,
    ERROR_UNDEFINED_OPERANDS,
    ERROR_UNDEFINED_OPERATORS,
    MAX_TERMS,
    EquationScore,
    score_equation,
)
from kd.search.llm4ed.sketch_backend import (
    SKETCH_LEVELS,
    Llm4edCompiled,
    compile_for_llm4ed,
)
from kd.search.recorder import log_whitelisted_metrics
from kd.search.result import invalid_evaluation_result
from kd.search.term_utils import fold_add
from kd.search.trajectory import (
    active_fit_terms,
    log_search_trajectory,
    select_trajectory_candidates,
)
from kd.viz.gap_notes import NO_MEASUREMENT

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.data.schema import PDEDataset
    from kd.llm import LLMProvider
    from kd.search.protocol import PlatformComponents
    from kd.search.recorder import VizRecorder
    from kd.viz.extension import PlotInfo

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]



SPATIAL_AXES: Final[tuple[str, ...]] = ("x", "y")

_STRUCTURAL_SCORE_ERRORS: Final[frozenset[str]] = frozenset(
    {
        ERROR_ABNORMAL_COEF,
        ERROR_INEXPRESSIBLE,
        ERROR_PARSE,
        ERROR_SKETCH,
        ERROR_UNDEFINED_OPERANDS,
        ERROR_UNDEFINED_OPERATORS,
    }
)


def _invalid_reason_for_score(error_type: str | None) -> str:
    if error_type in _STRUCTURAL_SCORE_ERRORS:
        return "structural_reject"
    if error_type == ERROR_NON_FINITE:
        return "non_finite"
    if error_type == ERROR_LSTSQ:
        return "evaluation_error"
    return "unclassified"


def _display_terms(score: EquationScore) -> tuple[str, ...]:
    assert score.coefficients is not None
    kept = [
        term
        for term, coef in zip(
            score.term_strs, np.ravel(score.coefficients), strict=True
        )
        if coef != 0
    ][:MAX_TERMS]
    return tuple(term[1:] if term.startswith("-") else term for term in kept)




_EDL_PHASE_ORDER: Final[tuple[str, str]] = (EVOLUTION, OPTIMIZE)








STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "population",
        "elite_pool",
        "phase_counter",
        "rng_state",
        "llm_seed_counter",
        "call_counts",
        "invalid_counts",
        "best",
    }
)








_LOGGED_METRICS: Final[tuple[str, ...]] = _viz_helpers.LOGGED_METRICS
_POOL_BEST_KEY: Final[str] = _viz_helpers.POOL_BEST_KEY
_POOL_MEDIAN_KEY: Final[str] = _viz_helpers.POOL_MEDIAN_KEY
_POOL_WORST_KEY: Final[str] = _viz_helpers.POOL_WORST_KEY
_N_INVALID_KEY: Final[str] = _viz_helpers.N_INVALID_KEY
_N_LLM_CALLS_KEY: Final[str] = _viz_helpers.N_LLM_CALLS_KEY
_N_VALID_KEY: Final[str] = _viz_helpers.N_VALID_KEY


_INVALID_SPREAD_METRICS: Final[tuple[str, ...]] = _viz_helpers.INVALID_SPREAD_METRICS
_INVALID_KEY_BY_LABEL: Final[dict[str, str]] = _viz_helpers.INVALID_KEY_BY_LABEL
_N_INVALID_OTHER_KEY: Final[str] = _viz_helpers.N_INVALID_OTHER_KEY









_NO_MEASUREMENT: Final[float] = NO_MEASUREMENT


@dataclass(frozen=True)
class _Member:

    score: float
    expression: str
    permutation: str
    terms: tuple[str, ...]


def _spatial_axes_of(dataset: PDEDataset) -> list[str]:
    lhs_field = dataset.lhs_field
    lhs_axis = dataset.lhs_axis
    if not lhs_field or not lhs_axis:
        raise ValueError(
            "Llm4edPlugin requires dataset.lhs_field and dataset.lhs_axis "
            "(the EDL FD pipeline differentiates the lhs field along the "
            "lhs axis)."
        )
    if not dataset.axis_order or not dataset.fields or not dataset.axes:
        raise ValueError(
            "Llm4edPlugin requires a grid dataset with axis_order, axes, "
            "and fields populated."
        )
    if lhs_field != "u":
        raise ValueError(
            f"Llm4edPlugin requires field 'u'; dataset has lhs_field={lhs_field!r}."
        )
    spatial_axes = [name for name in dataset.axis_order if name != lhs_axis]
    if not spatial_axes or any(name not in SPATIAL_AXES for name in spatial_axes):
        raise ValueError(
            f"Llm4edPlugin requires spatial axes drawn from {list(SPATIAL_AXES)} "
            f"plus the lhs axis {lhs_axis!r}; dataset has spatial axes "
            f"{spatial_axes!r}."
        )
    return [name for name in SPATIAL_AXES if name in spatial_axes]


def _dataset_columns(
    dataset: PDEDataset,
) -> tuple[FloatArray, dict[str, FloatArray], str]:
    spatial_axes = _spatial_axes_of(dataset)
    assert dataset.axis_order and dataset.fields and dataset.axes
    lhs_axis = dataset.lhs_axis
    axis_order = list(dataset.axis_order)
    axis_info = dataset.axes
    u = dataset.fields["u"].values.detach().cpu().numpy().astype(np.float64)
    perm = [axis_order.index(name) for name in (*spatial_axes, lhs_axis)]
    u = np.ascontiguousarray(np.transpose(u, perm))

    def values(name: str) -> FloatArray:
        return axis_info[name].values.detach().cpu().numpy().astype(np.float64)

    axes = {name: values(name) for name in spatial_axes}
    lhs, features = build_operand_columns_axes(u, axes, values(lhs_axis))
    return lhs, features, f"u_{lhs_axis}"


def _lower_target(
    lhs: FloatArray,
    pinned: tuple[tuple[str, float], ...],
    features: dict[str, FloatArray],
) -> FloatArray:
    lowered = lhs.reshape(-1).copy()
    for key, signed_value in pinned:
        column = evaluate_ir_column(key, features)
        if not np.isfinite(column).all():
            raise ValueError(
                f"pinned sketch term {key!r} evaluates to a non-finite column on "
                "the LLM4ED finite-difference footing; fix the pin or the dataset"
            )
        lowered = lowered - signed_value * column
    variance = float(np.var(lowered))
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "lowered LLM4ED sketch target variance must be finite and positive; "
            "the pinned terms leave no residual to regress"
        )
    return lowered.reshape(-1, 1)


def _tabular_columns(
    components: PlatformComponents,
) -> tuple[FloatArray, dict[str, FloatArray], str]:
    dataset = components.dataset
    context = components.context
    assert context is not None and dataset.fields is not None
    target = dataset.lhs_field

    def column(name: str) -> FloatArray:
        values = context.get_variable(name).detach().cpu().numpy()
        return np.ascontiguousarray(values, dtype=np.float64).reshape(-1)

    features = {name: column(name) for name in dataset.fields if name != target}
    return column(target), features, target


class Llm4edPlugin:


    score_kind: ClassVar[str] = "LLM4ED sparse reward"
    score_direction: ClassVar[Literal["min", "max"]] = "max"
    headline_coefficient_source: ClassVar[Literal["native", "platform_refit"]] = (
        "native"
    )

    config_cls: ClassVar[type[Llm4edConfig]] = Llm4edConfig
    one_shot: ClassVar[bool] = False


    sketch_lower_owner: ClassVar[Literal["platform", "native"]] = "native"
    constructor_kwargs: ClassVar[frozenset[str]] = frozenset({"provider", "mode"})

    descriptor: ClassVar[InstrumentDescriptor] = InstrumentDescriptor(
        algorithm="llm4ed",
        summary="Budgeted LLM proposal and sparse-reward PDE and tabular search.",
        cost_class="medium",
        modes=(
            InstrumentMode(
                name="default",
                field_names=("u",),
                spatial_axes=SPATIAL_AXES,
                description="LLM proposals over field u and spatial axes x and y.",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
                lhs_orders=frozenset({1}),
                refused_fields={"functions": "non_default"},
                sketch=SKETCH_LEVELS,
            ),
            InstrumentMode(
                name="tabular",
                description=(
                    "LLM formula proposals over the feature columns of a table."
                ),
                forms=frozenset({Form.REGRESSION}),
                topologies=frozenset({DataTopology.TABULAR}),
                provider_kind="none",
                lhs_orders=frozenset({0}),
            ),
        ),
        knobs=(
            Knob(
                "temperature",
                "float",
                "LLM sampling temperature.",
                resume_tier="resume_safe",
            ),
            Knob(
                "max_tokens",
                "int",
                "Maximum decoded tokens per request.",
                resume_tier="resume_safe",
            ),
            Knob(
                "stop_threshold",
                "float",
                "Reward threshold for completion.",
                resume_tier="resume_safe",
            ),
            Knob(
                "reward_limit",
                "float",
                "Minimum admitted sparse reward.",










                resume_tier="resume_safe",
            ),
            Knob(
                "pool_size",
                "int",
                "Elite-pool capacity.",









                resume_tier="resume_safe",
            ),
        ),
        segmentation=Segmentation(
            archive="progress", unit="rounds", reseed=True
        ),


        identity_breaking_fields=frozenset({"functions"}),



        config_artifact_keys=frozenset(),
        surrogate_fields=frozenset(),
    )

    def __init__(
        self,
        config: Llm4edConfig,
        *,
        provider: LLMProvider | None = None,
        mode: Literal["default", "tabular"] = "default",
    ) -> None:
        if mode not in ("default", "tabular"):
            raise ValueError(f"mode must be 'default' or 'tabular', got {mode!r}")
        if mode == "default" and config.functions:
            raise ValueError(
                f"functions={list(config.functions)} applies to the LLM4ED tabular "
                "mode only; the default PDE mode takes none"
            )
        self._mode = mode
        self._config = config
        self._provider = provider



        self._provider_self_built = False
        self._prepared = False
        self._restore_pending = False
        self._reseed_pending = False
        self._pending_state: dict[str, Any] | None = None

        self._components: PlatformComponents | None = None
        self._recorder: VizRecorder | None = None



        self._lhs: FloatArray | None = None
        self._features: dict[str, FloatArray] | None = None
        self._lhs_name: str | None = None


        self._operands: tuple[str, ...] = ()
        self._spatial_axes: tuple[str, ...] = ()

        self._sketch_compiled: Llm4edCompiled | None = None




        self._best_reward: float = 0.0
        self._best_expression: str = ""


        self._llm_seed_counter: int = 0
        self._population: list[_Member] = []
        self._pool: ElitePool = ElitePool(config.pool_size)
        self._phase_counter: int = 0
        self._rng: random.Random = random.Random(config.seed)
        self._call_counts: int = 0
        self._invalid_counts: dict[str, int] = {}





        self._organize_pool: list[PoolItem] = []



        self._display: dict[str, tuple[str, tuple[str, ...]]] = {}

        self._round_cache: dict[str, EquationScore] = {}
        self._round_invalid: int = 0




        self._round_invalid_by_label: dict[str, int] = {}
        self._round_llm_calls: int = 0






        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._usage_reporting_calls: int = 0
        self._completed_calls: int = 0


    def list_plots(self) -> list[PlotInfo]:
        return _viz_helpers.list_plot_infos()

    def render_plot(self, name: str, ax: Axes) -> list[str]:
        return _viz_helpers.render(name, ax, self._recorder)

    def get_plot_data(self, name: str) -> dict[str, Any]:
        return _viz_helpers.get_data(name, self._recorder)


    @property
    def derivative_requirements(self) -> DerivativeReqs:
        if self._mode == "tabular":
            return DerivativeReqs(
                provider_kind="none",
                lhs_source="field",
                lhs_order=0,
                supported_topologies=frozenset({DataTopology.TABULAR}),
            )
        return DerivativeReqs(
            provider_kind="finite_diff",
            max_atomic_order=3,
            lhs_order=1,
            needs_surrogate=False,
        )

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": ALGORITHM_NAME, **config_to_json_safe_dict(self._config)}

    @property
    def runner_batch_size(self) -> int:
        return self._config.samples_per_epoch

    @property
    def llm_token_totals(self) -> dict[str, int] | None:
        if self._completed_calls == 0:
            return None
        if self._usage_reporting_calls != self._completed_calls:
            return None
        return {"tokens_in": self._tokens_in, "tokens_out": self._tokens_out}


    def refuse_unsupported_names(self, dataset: PDEDataset) -> None:
        if self._mode == "tabular":
            for name in dataset.fields or {}:
                if is_placeholder(name):
                    raise ValueError(
                        f"LLM4ED tabular field {name!r} has the form of a constant "
                        "placeholder c<k>; rename the field"
                    )
                if not reads_as_symbol(name):
                    raise ValueError(
                        f"LLM4ED tabular field {name!r} is not read back as a plain "
                        "symbol by the equation parser; rename the field"
                    )
            return
        fields = set(dataset.fields or {})
        if fields != {"u"}:
            raise ValueError(
                f"LLM4ED requires fields ['u']; dataset fields={sorted(fields)}. "
                "Use an instrument that reads all of these fields."
            )

    def prepare(self, components: PlatformComponents) -> None:
        self.refuse_unsupported_names(components.dataset)
        self._components = components
        self._recorder = components.recorder
        self._lhs, self._features, self._lhs_name = (
            _tabular_columns(components)
            if self._mode == "tabular"
            else _dataset_columns(components.dataset)
        )
        self._operands = tuple(self._features)
        self._spatial_axes = (
            ()
            if self._mode == "tabular"
            else tuple(_spatial_axes_of(components.dataset))
        )
        self._prepare_sketch(components)





        self._reset_token_totals()

        is_restore = self._restore_pending and self._pending_state is not None



        initial_calls = 0
        if is_restore:
            assert self._pending_state is not None
            initial_calls = int(self._pending_state["call_counts"])
        if self._provider is None:
            self._provider = self._build_default_provider(
                initial_calls=initial_calls
            )
            self._provider_self_built = True
        elif self._provider_self_built:



            self._provider = self._build_default_provider(
                initial_calls=initial_calls
            )
        self._provider.prepare()

        if is_restore:
            assert self._pending_state is not None
            self._apply_state(self._pending_state)
            if self._reseed_pending:




                self._rng = random.Random(self._config.seed)
                self._llm_seed_counter = 0
        else:
            self._reset_search_state()
        self._restore_pending = False
        self._reseed_pending = False
        self._pending_state = None
        self._prepared = True

    def _prepare_sketch(self, components: PlatformComponents) -> None:
        self._sketch_compiled = None
        task = components.task
        if task is None:
            return
        assert self._lhs is not None and self._features is not None
        compiled = compile_for_llm4ed(task.sketch, self._operands)
        self._sketch_compiled = compiled
        self._operands = compiled.operands
        if not task.compiled.closed:
            self._lhs = _lower_target(self._lhs, compiled.pinned, self._features)

    @property
    def sketch_compile_report(self) -> CompileReport | None:
        compiled = self._sketch_compiled
        return None if compiled is None else compiled.report

    def propose(self, n: int) -> list[str]:
        self._require_prepared()
        phase = self._current_phase()
        prompt = self._build_prompt(phase, n)




        target = n

        self._round_cache = {}
        self._round_invalid = 0
        self._round_invalid_by_label = {}
        self._round_llm_calls = 0
        survivors: list[PoolItem] = []
        seen_scores: list[float] = []
        score_cache: dict[float, str] = {}
        while (
            len(survivors) < target
            and self._round_llm_calls < self._config.max_llm_calls_per_propose
        ):
            try:
                text = self._complete(prompt)
            except LLMBudgetExhausted as exc:


                logger.info(
                    "LLM budget exhausted mid-propose; keeping the partial "
                    "batch of %d survivor(s): %s",
                    len(survivors),
                    exc,
                )
                break
            kept = filter_score(
                self._score_response(text),
                self._config.reward_limit,
                seen_scores,
                score_cache,
            )
            seen_scores.extend(item.score for item in kept)
            survivors.extend(kept)

        if target < 1:
            return []


        batch = sorted(survivors, key=lambda item: item.score)[-target:]
        for item in batch:
            terms = _display_terms(self._round_cache[item.expression])
            permutation = permute_terms(list(terms), self._rng)
            self._display[item.expression] = (permutation, terms)
        return [item.expression for item in batch]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        self._require_prepared()
        results: list[EvaluationResult] = []
        for candidate in candidates:
            cached = self._round_cache.get(candidate)
            if cached is None:
                raise KeyError(
                    f"evaluate() replay miss: {candidate!r} was not scored by "
                    "the preceding propose() (P3 strict replay, no "
                    "recomputation)."
                )
            if not cached.valid:





                results.append(
                    invalid_evaluation_result(
                        cached.error_type or "invalid candidate",
                        score=None,
                        expression=candidate,
                        terms=list(cached.term_strs) or None,
                        reason=_invalid_reason_for_score(cached.error_type),
                    )
                )
                continue
            results.append(self._result_from_score(candidate, cached))
        return results

    def update(self, results: list[EvaluationResult]) -> None:
        self._require_prepared()
        admitted: list[PoolItem] = []
        members: list[_Member] = []
        for result in results:
            if not result.is_valid or result.score is None:
                continue
            item = PoolItem(score=float(result.score), expression=result.expression)
            admitted.append(item)
            members.append(self._member_for(item, result))
            if item.score > self._best_reward:
                self._best_reward = item.score
                self._best_expression = item.expression


        self._population = members









        self._organize_pool = self._pool.get_top_samples()
        self._pool.push(admitted)


        self._phase_counter += 1
        self._prune_display()

        pool_best, pool_median, pool_worst = self._pool_summary()
        log_whitelisted_metrics(
            self._recorder,
            _LOGGED_METRICS,
            {
                _POOL_BEST_KEY: pool_best,
                _POOL_MEDIAN_KEY: pool_median,
                _POOL_WORST_KEY: pool_worst,
                _N_INVALID_KEY: self._round_invalid,
                _N_LLM_CALLS_KEY: self._round_llm_calls,
                _N_VALID_KEY: len(members),


                **{
                    key: self._round_invalid_by_label.get(key, 0)
                    for key in _INVALID_SPREAD_METRICS
                },
            },
        )

        self._log_search_trajectory(results)

    def _log_search_trajectory(self, results: list[EvaluationResult]) -> None:
        recorder = self._recorder
        if recorder is None or not recorder.enabled or recorder.trajectory_top_k == 0:
            return
        operands = self._operands

        def project(index: int) -> tuple[str, tuple[str, ...], str]:
            result = results[index]
            assert result.terms is not None and result.coefficients is not None
            assert result.lhs_name is not None
            terms = active_fit_terms(
                result.terms, result.coefficients, result.selected_indices
            )
            tabular = self._mode == "tabular"
            parsed = parse_equation(
                result.expression,
                operands,
                functions=frozenset(self._config.functions),
                placeholders=tabular,
                constant_term=tabular,
                target=self._lhs_name if tabular else None,
            )
            expression = fold_add([term.ir for term in parsed.terms])
            return expression, terms, result.lhs_name

        candidates = select_trajectory_candidates(
            [result.score if result.is_valid else None for result in results],
            project,
            top_k=recorder.trajectory_top_k,
            direction=self.score_direction,
        )
        log_search_trajectory(recorder, candidates)



    def build_final_result(self) -> EvaluationResult:
        self._require_prepared()
        if not self._best_expression:

            return invalid_evaluation_result(
                "no candidates found",
                score=self._best_reward,
                reason="no_candidate",
            )
        rescored = self._score(self._best_expression)
        if not rescored.valid:

            return invalid_evaluation_result(
                f"best candidate re-scoring failed: {rescored.error_type or 'unknown'}",
                score=self._best_reward,
                expression=self._best_expression,
                reason=_invalid_reason_for_score(rescored.error_type),
            )
        return self._result_from_score(self._best_expression, rescored)

    def build_result_target(self) -> Tensor:
        self._require_prepared()
        assert self._lhs is not None
        return torch.from_numpy(self._lhs.reshape(-1).copy())


    @property
    def artifacts(self) -> dict[str, Any] | None:
        path = self._config.tape_record_path
        if path is None:
            return None
        artifacts: dict[str, Any] = {"llm_tape_path": path}
        tape = Path(path)
        if tape.is_file():
            raw = tape.read_bytes()
            artifacts["llm_tape_sha256"] = hashlib.sha256(raw).hexdigest()
            artifacts["llm_tape_entries"] = sum(
                1 for line in raw.decode("utf-8").splitlines() if line.strip()
            )
        return artifacts


    @property
    def best_score(self) -> float:
        return self._best_reward

    @property
    def best_expression(self) -> str:
        return self._best_expression


    @property
    def is_done(self) -> bool:
        return self._best_reward >= self._config.stop_threshold


    @property
    def state(self) -> dict[str, Any]:
        if self._prepared:
            state: dict[str, Any] = {
                "population": [
                    {
                        "score": member.score,
                        "expression": member.expression,
                        "permutation": member.permutation,
                        "terms": list(member.terms),
                    }
                    for member in self._population
                ],
                "elite_pool": {













                    "items": [
                        {
                            "score": item.score,
                            "expression": item.expression,
                            "permutation": self._display[item.expression][0],
                            "terms": list(self._display[item.expression][1]),
                        }
                        for item in self._pool.get_top_samples()
                    ],
                    "scores": list(self._pool.scores),
                },






                "organize_pool": [
                    {
                        "score": item.score,
                        "expression": item.expression,
                        "permutation": self._display[item.expression][0],
                        "terms": list(self._display[item.expression][1]),
                    }
                    for item in self._organize_pool
                ],
                "phase_counter": self._phase_counter,
                "rng_state": self._rng.getstate(),
                "llm_seed_counter": self._llm_seed_counter,
                "call_counts": self._call_counts,
                "invalid_counts": dict(self._invalid_counts),
                "best": {
                    "reward": self._best_reward,
                    "expression": self._best_expression,
                },
            }
            if self._mode == "tabular":
                state["mode"] = self._mode
                state["functions"] = list(self._config.functions)
            return state
        if self._pending_state is not None:
            return dict(self._pending_state)
        raise RuntimeError("prepare() must be called before reading state.")

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        if not value:
            self._pending_state = None
            self._restore_pending = False
            return
        archived_mode = value.get("mode", "default")
        if archived_mode != self._mode:
            raise ValueError(
                f"Cannot restore an LLM4ED {archived_mode!r}-mode checkpoint into "
                f"the {self._mode!r} mode"
            )
        if self._mode == "tabular":
            archived_functions = list(value["functions"])
            if archived_functions != list(self._config.functions):
                raise ValueError(
                    "Cannot restore an LLM4ED tabular checkpoint with "
                    f"functions={archived_functions} under "
                    f"functions={list(self._config.functions)}"
                )
        self._pending_state = dict(value)
        self._restore_pending = True

    def reseed(self) -> None:
        self._reseed_pending = True


    def _build_default_provider(self, *, initial_calls: int = 0) -> LLMProvider:
        if self._config.base_url is None:
            raise ValueError(
                "Llm4edConfig.base_url is required to assemble the default "
                "OpenAI-compatible provider chain; set base_url or inject a "
                "provider."
            )
        inner: LLMProvider = OpenAICompatProvider(
            model=self._config.model,
            base_url=self._config.base_url,
            extra_body=self._config.extra_body,
        )
        if self._config.tape_record_path is not None:
            inner = TapeRecordingProvider(
                inner, path=self._config.tape_record_path
            )
        return BudgetedProvider(
            inner,
            max_calls=self._config.max_llm_calls_per_run,
            initial_calls=initial_calls,
        )

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before using the plugin.")

    def _reset_token_totals(self) -> None:
        self._tokens_in = 0
        self._tokens_out = 0
        self._usage_reporting_calls = 0
        self._completed_calls = 0

    def _reset_search_state(self) -> None:
        self._population = []
        self._pool = ElitePool(self._config.pool_size)
        self._phase_counter = 0
        self._rng = random.Random(self._config.seed)
        self._llm_seed_counter = 0
        self._call_counts = 0
        self._invalid_counts = {}
        self._best_reward = 0.0
        self._best_expression = ""
        self._organize_pool = []
        self._display = {}
        self._round_cache = {}
        self._round_invalid = 0
        self._round_invalid_by_label = {}
        self._round_llm_calls = 0

    def _apply_state(self, state: dict[str, Any]) -> None:
        limit = self._config.reward_limit
        pool_state = state["elite_pool"]
        population_entries = [
            entry for entry in state["population"] if float(entry["score"]) > limit
        ]
        pool_entries = [
            entry for entry in pool_state["items"] if float(entry["score"]) > limit
        ][: self._config.pool_size]
        organize_entries = [
            entry for entry in state["organize_pool"] if float(entry["score"]) > limit
        ][: self._config.pool_size]
        self._population = [
            _Member(
                score=float(entry["score"]),
                expression=str(entry["expression"]),
                permutation=str(entry["permutation"]),
                terms=tuple(entry["terms"]),
            )
            for entry in population_entries
        ]
        self._pool = ElitePool.from_state(
            self._config.pool_size,
            [
                PoolItem(score=float(entry["score"]), expression=entry["expression"])
                for entry in pool_entries
            ],
            [float(score) for score in pool_state["scores"]],
        )



        self._organize_pool = [
            PoolItem(score=float(entry["score"]), expression=entry["expression"])
            for entry in organize_entries
        ]
        display: dict[str, tuple[str, tuple[str, ...]]] = {
            member.expression: (member.permutation, member.terms)
            for member in self._population
        }
        for entry in [*pool_entries, *organize_entries]:
            display[entry["expression"]] = (
                str(entry["permutation"]),
                tuple(entry["terms"]),
            )
        self._display = display
        self._phase_counter = int(state["phase_counter"])
        self._llm_seed_counter = int(state["llm_seed_counter"])
        self._call_counts = int(state["call_counts"])
        self._invalid_counts = {
            str(label): int(count)
            for label, count in state["invalid_counts"].items()
        }
        self._best_reward = float(state["best"]["reward"])
        self._best_expression = str(state["best"]["expression"])
        self._rng = random.Random()
        self._rng.setstate(state["rng_state"])
        self._round_cache = {}
        self._round_invalid = 0
        self._round_invalid_by_label = {}
        self._round_llm_calls = 0

    def _current_phase(self) -> str:
        if self._phase_counter == 0:
            return INITIALIZATION
        return _EDL_PHASE_ORDER[
            (self._phase_counter - 1) % len(_EDL_PHASE_ORDER)
        ]

    def _build_prompt(self, phase: str, n: int) -> str:
        if self._mode == "tabular":
            return self._build_tabular_prompt(phase, n)
        compiled = self._sketch_compiled
        operators = DEFAULT_PDE_OPERATORS if compiled is None else compiled.operators
        library = {"operators": operators, "operands": pde_operands(self._operands)}
        hint = "" if compiled is None else compiled.hint
        if phase == INITIALIZATION:
            prompt = build_initialization_prompt(
                self._config.init_num, spatial_axes=self._spatial_axes, **library
            )
        else:
            history = self._assemble_history(phase)
            if phase == EVOLUTION:
                prompt = build_evolution_prompt(history, n, **library)
            else:
                prompt = build_optimize_prompt(history, n, **library)
        return prompt + hint

    def _build_tabular_prompt(self, phase: str, n: int) -> str:
        assert self._lhs_name is not None and self._features is not None
        target = self._lhs_name
        variables = tuple(self._features)
        functions = self._config.functions
        if phase == INITIALIZATION:
            return build_tabular_initialization_prompt(
                self._config.init_num,
                target=target,
                variables=variables,
                functions=functions,
            )
        history = self._assemble_history(phase)
        build = (
            build_tabular_evolution_prompt
            if phase == EVOLUTION
            else build_tabular_optimize_prompt
        )
        return build(
            history, n, target=target, variables=variables, functions=functions
        )

    def _assemble_history(self, phase: str) -> str:
        members = list(self._population)
        population_scores = {member.score for member in members}
        for item in self._organize_pool:
            if item.score in population_scores:
                continue
            permutation, terms = self._display[item.expression]
            members.append(_Member(item.score, item.expression, permutation, terms))
        members.sort(key=lambda member: member.score)

        lines: list[str] = []
        for index, member in enumerate(members):
            if phase == EVOLUTION:
                term_set = "{" + ", ".join(member.terms) + "}"
                lines.append(f"{index}: {term_set}\n")
            else:
                lines.append(
                    f"{index}: {member.permutation} score: {member.score}\n"
                )
        return "".join(lines)

    def _complete(self, prompt: str) -> str:
        assert self._provider is not None
        request = LLMRequest(
            prompt=prompt,
            seed=self._config.seed + self._llm_seed_counter,
            params=LLMParams(
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            ),
        )
        self._llm_seed_counter += 1
        response = self._provider.complete(request)
        self._call_counts += 1
        self._round_llm_calls += 1




        self._completed_calls += 1
        if response.usage is not None:
            self._tokens_in += response.usage.prompt_tokens
            self._tokens_out += response.usage.completion_tokens
            self._usage_reporting_calls += 1
        return response.text

    def _score_response(self, text: str) -> list[PoolItem]:
        scored: list[PoolItem] = []
        lines = (
            parse_tabular_response(text)
            if self._mode == "tabular"
            else parse_response(text)
        )
        for candidate in lines:
            result = self._score(candidate)
            self._round_cache[candidate] = result
            if not result.valid:
                self._round_invalid += 1
                label = result.error_type or "unknown"
                self._invalid_counts[label] = self._invalid_counts.get(label, 0) + 1
                key = _INVALID_KEY_BY_LABEL.get(label, _N_INVALID_OTHER_KEY)
                self._round_invalid_by_label[key] = (
                    self._round_invalid_by_label.get(key, 0) + 1
                )
                continue
            assert result.reward is not None
            scored.append(PoolItem(score=result.reward, expression=candidate))
        return scored

    def _score(self, equation: str) -> EquationScore:
        assert self._lhs is not None and self._features is not None
        tabular = self._mode == "tabular"
        return score_equation(
            equation,
            self._lhs,
            self._features,
            operands=self._operands,
            functions=frozenset(self._config.functions),
            placeholders=tabular,
            constant_term=tabular,
            elementwise_dedup=tabular or len(self._spatial_axes) > 1,
            target=self._lhs_name if tabular else None,
            distinct_skeletons=tabular,
            gate=self._sketch_compiled,
        )

    def _member_for(self, item: PoolItem, result: EvaluationResult) -> _Member:
        display = self._display.get(item.expression)
        if display is None:
            terms = tuple(result.terms) if result.terms else (item.expression,)
            display = (permute_terms(list(terms), self._rng), terms)
            self._display[item.expression] = display
        return _Member(item.score, item.expression, display[0], display[1])

    def _prune_display(self) -> None:
        keep = {member.expression for member in self._population}
        keep.update(item.expression for item in self._pool.get_top_samples())
        keep.update(item.expression for item in self._organize_pool)
        self._display = {
            expression: display
            for expression, display in self._display.items()
            if expression in keep
        }

    def _pool_summary(self) -> tuple[float, float, float]:
        top = self._pool.get_top_samples()
        if not top:
            return _NO_MEASUREMENT, _NO_MEASUREMENT, _NO_MEASUREMENT
        best = top[0].score
        worst = top[-1].score
        mid = len(top) // 2
        if len(top) % 2 == 1:
            median = top[mid].score
        else:
            median = (top[mid - 1].score + top[mid].score) / 2.0
        return best, median, worst

    def _result_from_score(
        self, expression: str, score: EquationScore
    ) -> EvaluationResult:
        assert score.valid and score.reward is not None
        assert score.y_hat is not None and score.coefficients is not None
        assert self._lhs is not None
        lhs_flat = self._lhs.reshape(-1)
        residuals = score.y_hat.reshape(-1) - lhs_flat
        mse = float(np.mean(residuals**2))
        variance = float(np.var(lhs_flat))
        if variance > 0.0 and math.isfinite(mse):
            nmse = mse / variance
            r2 = 1.0 - nmse
        else:
            nmse = float("inf")
            r2 = -float("inf")
        raw = np.asarray(score.coefficients, dtype=np.float64).reshape(-1)






        factors = np.asarray(score.term_coeffs, dtype=np.float64).reshape(-1)
        assert raw.shape == factors.shape and len(score.term_irs) == raw.shape[0], (
            "llm4ed boundary misalignment: coefficients, term_coeffs and "
            f"term_irs must be 1:1 (got {raw.shape[0]}, {factors.shape[0]}, "
            f"{len(score.term_irs)})"
        )
        coefficients = raw * factors
        selected = [
            index
            for index, value in enumerate(coefficients)
            if value != 0.0
        ]
        return EvaluationResult(
            mse=mse,
            nmse=nmse,
            r2=r2,
            score=score.reward,
            complexity=score.n_terms,
            coefficients=torch.from_numpy(coefficients.copy()),
            is_valid=True,
            error_message="",
            selected_indices=selected,
            residuals=torch.from_numpy(residuals.copy()),
            terms=list(score.term_irs),
            expression=expression,
            lhs_name=self._lhs_name,
        )


__all__ = ["ALGORITHM_NAME", "STATE_KEYS", "Llm4edPlugin"]
