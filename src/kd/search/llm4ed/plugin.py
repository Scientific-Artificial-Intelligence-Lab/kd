
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

from kd.core.evaluator import EvaluationResult
from kd.core.platform.requirements import DerivativeReqs
from kd.llm import (
    BudgetedProvider,
    LLMBudgetExhausted,
    LLMParams,
    LLMRequest,
    OpenAICompatProvider,
    TapeRecordingProvider,
)
from kd.search.llm4ed import viz as _viz_helpers
from kd.search.llm4ed.config import (
    ALGORITHM_NAME,
    Llm4edConfig,
    config_to_json_safe_dict,
)
from kd.search.llm4ed.fd import build_operand_columns
from kd.search.llm4ed.filter_score import filter_score
from kd.search.llm4ed.pool import ElitePool, PoolItem
from kd.search.llm4ed.prompts import (
    EVOLUTION,
    INITIALIZATION,
    OPTIMIZE,
    build_evolution_prompt,
    build_initialization_prompt,
    build_optimize_prompt,
    parse_response,
    permute_terms,
)
from kd.search.llm4ed.score import EquationScore, score_equation
from kd.search.recorder import log_whitelisted_metrics
from kd.search.result import invalid_evaluation_result

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.data.schema import PDEDataset
    from kd.llm import LLMProvider
    from kd.search.protocol import PlatformComponents
    from kd.search.recorder import VizRecorder
    from kd.viz.extension import PlotInfo

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]



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





_LOGGED_METRICS: Final[tuple[str, ...]] = (
    "pool_best",
    "pool_median",
    "pool_worst",
    "n_invalid",
    "n_llm_calls",
    "n_valid",
)


@dataclass(frozen=True)
class _Member:

    score: float
    expression: str
    permutation: str
    terms: tuple[str, ...]


def _dataset_columns(
    dataset: PDEDataset,
) -> tuple[FloatArray, dict[str, FloatArray], str]:
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
    axis_order = list(dataset.axis_order)
    spatial_axes = [name for name in axis_order if name != lhs_axis]
    if len(spatial_axes) != 1:
        raise ValueError(
            "Llm4edPlugin supports exactly one spatial axis plus the lhs "
            f"axis {lhs_axis!r}; dataset has axes {axis_order!r}."
        )
    spatial_axis = spatial_axes[0]
    u = dataset.fields[lhs_field].values.detach().cpu().numpy().astype(np.float64)
    if axis_order[0] == lhs_axis:
        u = np.ascontiguousarray(u.T)
    x = dataset.axes[spatial_axis].values.detach().cpu().numpy().astype(np.float64)
    t = dataset.axes[lhs_axis].values.detach().cpu().numpy().astype(np.float64)
    lhs, features = build_operand_columns(u, x, t)
    return lhs, features, f"{lhs_field}_{lhs_axis}"


class Llm4edPlugin:


    score_kind: ClassVar[str] = "LLM4ED sparse reward"
    score_direction: ClassVar[Literal["min", "max"]] = "max"

    config_cls: ClassVar[type[Llm4edConfig]] = Llm4edConfig
    one_shot: ClassVar[bool] = False

    def __init__(
        self, config: Llm4edConfig, *, provider: LLMProvider | None = None
    ) -> None:
        self._config = config
        self._provider = provider



        self._provider_self_built = False
        self._prepared = False
        self._restore_pending = False
        self._pending_state: dict[str, Any] | None = None

        self._components: PlatformComponents | None = None
        self._recorder: VizRecorder | None = None


        self._lhs: FloatArray | None = None
        self._features: dict[str, FloatArray] | None = None
        self._lhs_name: str | None = None




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
        self._round_llm_calls: int = 0


    def list_plots(self) -> list[PlotInfo]:
        return _viz_helpers.list_plot_infos()

    def render_plot(self, name: str, ax: Axes) -> None:
        _viz_helpers.render(name, ax, self._recorder)

    def get_plot_data(self, name: str) -> dict[str, Any]:
        return _viz_helpers.get_data(name, self._recorder)


    @property
    def derivative_requirements(self) -> DerivativeReqs:
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


    def prepare(self, components: PlatformComponents) -> None:
        self._components = components
        self._recorder = components.recorder
        self._lhs, self._features, self._lhs_name = _dataset_columns(
            components.dataset
        )

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
        else:
            self._reset_search_state()
        self._restore_pending = False
        self._pending_state = None
        self._prepared = True

    def propose(self, n: int) -> list[str]:
        self._require_prepared()
        phase = self._current_phase()
        prompt = self._build_prompt(phase, n)




        target = n

        self._round_cache = {}
        self._round_invalid = 0
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
            equation_score = self._round_cache[item.expression]
            permutation = permute_terms(list(equation_score.term_strs), self._rng)
            self._display[item.expression] = (permutation, equation_score.term_strs)
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
                "pool_best": pool_best,
                "pool_median": pool_median,
                "pool_worst": pool_worst,
                "n_invalid": self._round_invalid,
                "n_llm_calls": self._round_llm_calls,
                "n_valid": len(members),
            },
        )


    def build_final_result(self) -> EvaluationResult:
        self._require_prepared()
        if not self._best_expression:
            return invalid_evaluation_result(
                "no candidates found", score=self._best_reward
            )
        assert self._lhs is not None and self._features is not None
        rescored = score_equation(self._best_expression, self._lhs, self._features)
        if not rescored.valid:
            return invalid_evaluation_result(
                "best candidate re-scoring failed: "
                f"{rescored.error_type or 'unknown'}",
                score=self._best_reward,
                expression=self._best_expression,
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
            return {
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
        if self._pending_state is not None:
            return dict(self._pending_state)
        raise RuntimeError("prepare() must be called before reading state.")

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        if not value:
            self._pending_state = None
            self._restore_pending = False
            return
        self._pending_state = dict(value)
        self._restore_pending = True


    def _build_default_provider(self, *, initial_calls: int = 0) -> LLMProvider:
        if self._config.base_url is None:
            raise ValueError(
                "Llm4edConfig.base_url is required to assemble the default "
                "OpenAI-compatible provider chain; set base_url or inject a "
                "provider."
            )
        inner: LLMProvider = OpenAICompatProvider(
            model=self._config.model, base_url=self._config.base_url
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
        self._round_llm_calls = 0

    def _apply_state(self, state: dict[str, Any]) -> None:
        self._population = [
            _Member(
                score=float(entry["score"]),
                expression=str(entry["expression"]),
                permutation=str(entry["permutation"]),
                terms=tuple(entry["terms"]),
            )
            for entry in state["population"]
        ]
        pool_state = state["elite_pool"]
        self._pool = ElitePool.from_state(
            self._config.pool_size,
            [
                PoolItem(score=float(entry["score"]), expression=entry["expression"])
                for entry in pool_state["items"]
            ],
            [float(score) for score in pool_state["scores"]],
        )



        self._organize_pool = [
            PoolItem(score=float(entry["score"]), expression=entry["expression"])
            for entry in state["organize_pool"]
        ]
        display: dict[str, tuple[str, tuple[str, ...]]] = {
            member.expression: (member.permutation, member.terms)
            for member in self._population
        }
        for entry in [*pool_state["items"], *state["organize_pool"]]:
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
        self._round_llm_calls = 0

    def _current_phase(self) -> str:
        if self._phase_counter == 0:
            return INITIALIZATION
        return _EDL_PHASE_ORDER[
            (self._phase_counter - 1) % len(_EDL_PHASE_ORDER)
        ]

    def _build_prompt(self, phase: str, n: int) -> str:
        if phase == INITIALIZATION:
            return build_initialization_prompt(self._config.init_num)
        history = self._assemble_history(phase)
        if phase == EVOLUTION:
            return build_evolution_prompt(history, n)
        return build_optimize_prompt(history, n)

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
        return response.text

    def _score_response(self, text: str) -> list[PoolItem]:
        assert self._lhs is not None and self._features is not None
        scored: list[PoolItem] = []
        for candidate in parse_response(text):
            result = score_equation(candidate, self._lhs, self._features)
            self._round_cache[candidate] = result
            if not result.valid:
                self._round_invalid += 1
                label = result.error_type or "unknown"
                self._invalid_counts[label] = self._invalid_counts.get(label, 0) + 1
                continue
            assert result.reward is not None
            scored.append(PoolItem(score=result.reward, expression=candidate))
        return scored

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
            return 0.0, 0.0, 0.0
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
