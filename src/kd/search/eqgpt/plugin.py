
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import torch
from torch import Tensor

from kd.core.equation import Form
from kd.core.evaluator import EvaluationResult
from kd.core.platform.requirements import DerivativeReqs
from kd.core.term_cache import TermColumnCache
from kd.data.schema import DataTopology
from kd.search.descriptor import InstrumentDescriptor, InstrumentMode, Knob
from kd.search.eqgpt import _scoring
from kd.search.eqgpt import steady_viz as _steady_viz_helpers
from kd.search.eqgpt import viz as _viz_helpers
from kd.search.eqgpt._multicase import (
    WAVE_LHS_ORDER,
    WAVE_MAX_ATOMIC_ORDER,
    MultiCaseEvaluator,
)
from kd.search.eqgpt._steady import STEADY_MAX_ATOMIC_ORDER, SteadyEvaluator
from kd.search.eqgpt.artifacts import resolve_weights_fingerprint_path
from kd.search.eqgpt.backend import GPTBackend, RealGPTBackend
from kd.search.eqgpt.config import EqGPTConfig, config_to_json_safe_dict
from kd.search.eqgpt.gates import should_zero_reward
from kd.search.eqgpt.gpt import GPTConfig
from kd.search.eqgpt.ir_map import (
    MalformedSentenceError,
    UnmappedTokenError,
    ir_inexpressible_tokens,
    order_masked_tokens,
    sentence_to_rhs_terms,
)
from kd.search.eqgpt.pool import dedup_sentence, merge_top_k
from kd.search.eqgpt.sampling import (
    Sampler,
    SamplingConfig,
    dimension_masked_tokens,
)
from kd.search.eqgpt.vocab import VOCAB_SHA256, Vocab, load_vocab
from kd.search.protocol import PlatformComponents
from kd.search.recorder import log_whitelisted_metrics
from kd.viz.extension import PlotInfo
from kd.viz.gap_notes import NO_MEASUREMENT

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder
    from kd.search.result import ExperimentResult




STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "backend_state",
        "optimizer_state",
        "top_k",
        "rng_state",
        "reward_history",
        "fingerprints",
    }
)

ALGORITHM_NAME: Final[str] = "eqgpt"







_LOGGED_METRICS: Final[tuple[str, ...]] = _viz_helpers.LOGGED_METRICS
_POOL_BEST_KEY: Final[str] = _viz_helpers.POOL_BEST_KEY
_POOL_MEDIAN_KEY: Final[str] = _viz_helpers.POOL_MEDIAN_KEY
_POOL_WORST_KEY: Final[str] = _viz_helpers.POOL_WORST_KEY
_FINETUNE_LOSS_KEY: Final[str] = _viz_helpers.FINETUNE_LOSS_KEY










_NO_MEASUREMENT: Final[float] = NO_MEASUREMENT


class EqGPTPlugin:

    score_kind: ClassVar[str] = "EqGPT reward"
    score_direction: ClassVar[Literal["min", "max"]] = "max"
    headline_coefficient_source: ClassVar[Literal["native", "platform_refit"]] = (
        "native"
    )

    config_cls: ClassVar[type[EqGPTConfig]] = EqGPTConfig
    one_shot: ClassVar[bool] = False
    sketch_lower_owner: ClassVar[Literal["platform", "native"]] = "platform"

    descriptor: ClassVar[InstrumentDescriptor] = InstrumentDescriptor(
        algorithm="eqgpt",
        summary="GPT-guided equation discovery across wave and steady modes.",
        cost_class="heavy",
        modes=(
            InstrumentMode(
                name="single_wave",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
                description="Default single-case evolution-equation path.",
            ),
            InstrumentMode(
                name="wave_multicase",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset(
                    {DataTopology.GRID, DataTopology.SCATTERED}
                ),
                provider_kind="none",
                description="Private multi-case wave evaluation path.",
            ),
            InstrumentMode(
                name="steady",
                forms=frozenset({Form.HOMOGENEOUS}),
                topologies=frozenset({DataTopology.SCATTERED}),
                provider_kind="none",
                description="Private steady homogeneous-equation path.",
            ),
        ),
        knobs=(
            Knob(
                "samples_per_epoch",
                "int",
                "GPT samples per Runner iteration.",
                resume_tier="resume_safe",
            ),
            Knob(
                "top_k",
                "int",
                "Elite-pool and fine-tune slice size.",
                resume_tier="init_only",
            ),
            Knob(
                "sparsity_alpha",
                "float",
                "Problem-specific sparsity weight.",
                resume_tier="init_only",
            ),
            Knob(
                "finetune_lr",
                "float",
                "GPT fine-tuning learning rate.",




                resume_tier="resume_safe",
            ),
            Knob(
                "exploration_rate",
                "float",
                "Sampling exploration rate.",
                resume_tier="resume_safe",
            ),
        ),
    )

    def __init__(
        self, config: EqGPTConfig, *, backend: GPTBackend | None = None
    ) -> None:
        self._config = config
        self._backend = backend


        self._backend_self_built = False
        self._prepared = False
        self._restore_pending = False
        self._pending_state: dict[str, Any] | None = None

        self._components: PlatformComponents | None = None





        self._multicase: MultiCaseEvaluator | None = None


        self._steady: SteadyEvaluator | None = None
        self._steady_viz_cache: (
            tuple[object | None, _steady_viz_helpers.SteadyVizData] | None
        ) = None


        self._recorder: VizRecorder | None = None
        self._vocab: Vocab | None = None
        self._sampler: Sampler | None = None
        self._start_len_no_s = 0
        self._variables: tuple[str, ...] = ()
        self._lhs_flat: Tensor | None = None
        self._optimizer: torch.optim.Optimizer | None = None

        self._pool_rewards: list[float] = []
        self._pool_sentences: list[list[int]] = []
        self._reward_history: list[list[float]] = []
        self._rng: torch.Generator = torch.Generator().manual_seed(config.seed)



        self._pending_cache: dict[str, list[int]] = {}




        self._term_cache: TermColumnCache = TermColumnCache()


    @property
    def derivative_requirements(self) -> DerivativeReqs:
        if self._config.is_steady:
            return DerivativeReqs(
                provider_kind="none",
                max_atomic_order=STEADY_MAX_ATOMIC_ORDER,
                lhs_order=0,
                needs_surrogate=False,
                supported_topologies=frozenset({DataTopology.SCATTERED}),
            )
        if self._config.is_wave_multicase:
            return DerivativeReqs(
                provider_kind="none",
                max_atomic_order=WAVE_MAX_ATOMIC_ORDER,
                lhs_order=WAVE_LHS_ORDER,
                needs_surrogate=False,
                supported_topologies=frozenset(
                    {DataTopology.GRID, DataTopology.SCATTERED}
                ),
            )
        return DerivativeReqs(
            provider_kind="finite_diff",
            max_atomic_order=WAVE_MAX_ATOMIC_ORDER,
            lhs_order=WAVE_LHS_ORDER,
            needs_surrogate=False,




        )

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": ALGORITHM_NAME, **config_to_json_safe_dict(self._config)}

    @property
    def runner_batch_size(self) -> int:
        return self._config.samples_per_epoch


    def prepare(self, components: PlatformComponents) -> None:
        dataset = components.dataset
        self._steady_viz_cache = None
        self._components = components
        self._recorder = components.recorder
        self._variables = self._resolve_variables(dataset)

        if self._config.is_wave_multicase:








            self._steady = None
            self._multicase = MultiCaseEvaluator.from_config(self._config)
            self._lhs_flat = None
            self._assert_primary_dataset(dataset)
        elif self._config.is_steady:





            self._multicase = None
            self._steady = None
            self._lhs_flat = None
        else:
            self._multicase = None
            self._steady = None
            evaluator = components.evaluator
            if evaluator is None:
                raise ValueError(
                    "EqGPTPlugin requires components.evaluator (builds the LHS "
                    "regression target this plugin scores candidates against); "
                    "assemble the platform with an evaluator (PlatformBuilder "
                    "default)."
                )
            self._lhs_flat = evaluator.lhs_target.detach().clone().reshape(-1)

        vocab = load_vocab()
        self._vocab = vocab
        is_restore = self._restore_pending and self._pending_state is not None
        if self._backend is None:
            self._backend = self._build_default_backend(vocab)
            self._backend_self_built = True
        elif self._backend_self_built and not is_restore:





            self._backend = self._build_default_backend(vocab)
        backend_max_pos = self._backend.max_pos()
        if self._config.max_length >= backend_max_pos:
            raise ValueError(
                f"config.max_length ({self._config.max_length}) must be < "
                f"backend.max_pos ({backend_max_pos})"
            )

        if self._config.is_steady:




            self._steady = SteadyEvaluator.from_components(
                dataset,
                components.executor,
                self._config,
            )

        start_tokens = tuple(vocab.word2id[word] for word in self._config.start_words)
        self._start_len_no_s = len(start_tokens) - 1
        masked_tokens = (
            dimension_masked_tokens(vocab, self._variables)
            | order_masked_tokens(vocab, self.derivative_requirements.max_atomic_order)
            | ir_inexpressible_tokens(vocab)
            | frozenset(self._config.masked_tokens)
        )
        sampling_config = SamplingConfig(
            start_tokens=start_tokens,
            masked_tokens=masked_tokens,
            exploration_rate=self._config.exploration_rate,
            max_length=self._config.max_length,
        )
        self._sampler = Sampler(self._backend, vocab, sampling_config)
        self._pending_cache = {}


        self._term_cache = TermColumnCache()

        if self._restore_pending and self._pending_state is not None:
            self._apply_state(self._pending_state)
        else:
            self._reset_search_state()

        self._optimizer = torch.optim.Adam(
            self._backend.parameters(), lr=self._config.finetune_lr
        )
        if self._restore_pending and self._pending_state is not None:
            optimizer_state = self._pending_state.get("optimizer_state")
            if optimizer_state:
                self._optimizer.load_state_dict(optimizer_state)





                for group in self._optimizer.param_groups:
                    group["lr"] = self._config.finetune_lr

        self._restore_pending = False
        self._pending_state = None
        self._prepared = True

    def propose(self, n: int) -> list[str]:
        self._require_prepared()
        assert self._sampler is not None and self._vocab is not None

        seed = self._next_seed()
        raw_batch = self._sampler.sample_batch(n, seed=seed)

        candidates: list[str] = []
        cache: dict[str, list[int]] = {}
        for raw in raw_batch:
            body = list(raw[1:])
            deduped = dedup_sentence(body)
            try:
                terms = sentence_to_rhs_terms(
                    self._vocab, deduped, start_len=self._start_len_no_s
                )
            except (MalformedSentenceError, UnmappedTokenError):


                continue
            candidate = " + ".join(terms)
            cache[candidate] = deduped
            candidates.append(candidate)

        self._pending_cache = cache
        return candidates

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        self._require_prepared()
        assert self._vocab is not None
        if self._multicase is not None:


            return [self._score_wave(candidate) for candidate in candidates]
        if self._steady is not None:
            return [self._score_steady(candidate) for candidate in candidates]
        assert self._lhs_flat is not None and self._components is not None
        context = self._components.context


        assert context is not None
        return [
            _scoring.score_candidate(
                candidate=candidate,
                sentence=self._pending_cache.get(candidate),
                vocab=self._vocab,
                variables=self._variables,
                executor=self._components.executor,
                context=context,
                lhs_flat=self._lhs_flat,
                sparsity_alpha=self._config.sparsity_alpha,
                term_cache=self._term_cache,
            )
            for candidate in candidates
        ]

    def _score_wave(self, candidate: str) -> EvaluationResult:
        assert self._vocab is not None and self._multicase is not None
        terms = candidate.split(" + ") if candidate else []
        sentence = self._pending_cache.get(candidate)
        if sentence is None:
            return _scoring.invalid_result(
                candidate, terms, "candidate not found in propose() cache"
            )
        words = self._vocab.decode(sentence)
        if should_zero_reward(words, self._variables):
            return _scoring.gate_zero_result(candidate, terms)
        return self._multicase.score_candidate(candidate=candidate, terms=terms)

    def _score_steady(self, candidate: str) -> EvaluationResult:
        assert self._vocab is not None and self._steady is not None
        terms = candidate.split(" + ") if candidate else []
        sentence = self._pending_cache.get(candidate)
        if sentence is None:
            result = _scoring.invalid_result(
                candidate, terms, "candidate not found in propose() cache"
            )
            return replace(result, form=Form.HOMOGENEOUS)
        words = self._vocab.decode(sentence)
        if should_zero_reward(words, self._variables):
            return replace(
                _scoring.gate_zero_result(candidate, terms),
                form=Form.HOMOGENEOUS,
            )
        return self._steady.score_candidate(candidate=candidate, terms=terms)

    def update(self, results: list[EvaluationResult]) -> None:
        self._require_prepared()
        new_rewards: list[float] = []
        new_sentences: list[list[int]] = []
        for result in results:
            if not result.is_valid:
                continue
            sentence = self._pending_cache.get(result.expression)
            if sentence is None:
                continue
            reward = result.score if result.score is not None else 0.0
            new_rewards.append(float(reward))
            new_sentences.append(list(sentence))

        self._pool_rewards, self._pool_sentences = merge_top_k(
            self._pool_rewards,
            self._pool_sentences,
            new_rewards,
            new_sentences,
            self._config.top_k,
        )
        self._reward_history.append(list(self._pool_rewards))
        assert self._backend is not None and self._optimizer is not None
        finetune_loss = _scoring.finetune(
            self._backend,
            self._optimizer,
            self._pool_sentences,
            self._config.finetune_steps,
        )
        log_whitelisted_metrics(
            self._recorder,
            _LOGGED_METRICS,
            self._epoch_metrics(finetune_loss),
        )


    def build_final_result(self) -> EvaluationResult:
        self._require_prepared()
        assert self._vocab is not None
        best_reward = self._pool_rewards[0] if self._pool_rewards else 0.0
        if not self._pool_sentences:
            if self._steady is not None:

                return replace(
                    _scoring.invalid_final_result(
                        "no candidates found", best_reward, reason="no_candidate"
                    ),
                    form=Form.HOMOGENEOUS,
                )
            return _scoring.invalid_final_result(
                "no candidates found", best_reward, reason="no_candidate"
            )

        try:
            terms = sentence_to_rhs_terms(
                self._vocab, self._pool_sentences[0], start_len=self._start_len_no_s
            )
        except (MalformedSentenceError, UnmappedTokenError) as exc:
            result = _scoring.invalid_final_result(
                f"best candidate malformed: {exc}",
                best_reward,
                reason="structural_reject",
            )
            if self._steady is not None:
                return replace(result, form=Form.HOMOGENEOUS)
            return result

        if self._multicase is not None:




            return self._multicase.build_final_result(terms, best_reward=best_reward)
        if self._steady is not None:
            return self._steady.build_final_result(terms, best_reward=best_reward)

        assert self._lhs_flat is not None and self._components is not None
        context = self._components.context
        assert context is not None
        return _scoring.refit_final(
            terms=terms,
            executor=self._components.executor,
            context=context,
            target=self._lhs_flat.detach(),
            best_reward=best_reward,
            term_cache=self._term_cache,
        )

    def build_result_target(self) -> Tensor:
        self._require_prepared()
        if self._multicase is not None:
            return self._multicase.result_target()
        if self._steady is not None:
            return self._steady.result_target(self._best_terms())
        assert self._lhs_flat is not None
        return self._lhs_flat.detach().clone()


    @property
    def best_score(self) -> float:
        return self._pool_rewards[0] if self._pool_rewards else 0.0

    @property
    def best_expression(self) -> str:
        terms = self._best_terms()
        return " + ".join(terms) if terms is not None else ""

    def _best_terms(self) -> list[str] | None:
        if not self._pool_sentences or self._vocab is None:
            return None
        try:
            return sentence_to_rhs_terms(
                self._vocab, self._pool_sentences[0], start_len=self._start_len_no_s
            )
        except (MalformedSentenceError, UnmappedTokenError):
            return None


    @property
    def state(self) -> dict[str, Any]:
        if self._prepared:
            assert self._backend is not None and self._optimizer is not None
            return {
                "backend_state": self._backend.state_dict(),
                "optimizer_state": self._optimizer.state_dict(),
                "top_k": {
                    "rewards": list(self._pool_rewards),
                    "sentences": [list(s) for s in self._pool_sentences],
                },
                "rng_state": self._rng.get_state(),
                "reward_history": [list(h) for h in self._reward_history],
                "fingerprints": self._build_fingerprints(),
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


    @property
    def artifacts(self) -> dict[str, Any]:
        from kd.search.eqgpt.artifacts import build_run_artifacts
        from kd.search.eqgpt.vocab import vocab_asset_path

        return build_run_artifacts(
            vocab_path=vocab_asset_path(),
            weights_path=resolve_weights_fingerprint_path(
                self._backend, self._config.weights_path
            ),
            variables=self._variables or (self._config.variables or ()),
            **self._wave_artifact_kwargs(),
        )

    def _wave_artifact_kwargs(self) -> dict[str, Any]:
        if not self._config.is_wave_multicase or self._multicase is None:
            return {}
        from kd.data.loaders.wave_breaking import (
            default_wave_pkl_path,
            wave_surrogate_checkpoint_path,
        )

        surrogate_paths = {
            name: wave_surrogate_checkpoint_path(name, self._config.v1_asset_dir)
            for name in self._multicase.case_names
        }
        wave_pkl = self._config.wave_pkl_path or default_wave_pkl_path()
        return {
            "surrogate_paths": surrogate_paths,
            "wave_pkl_path": wave_pkl,
            "grid_params": {
                "reward_points_per_window": self._config.reward_points_per_window,
                "coeff_points_per_window": self._config.coeff_points_per_window,
            },
        }


    def list_plots(self) -> list[PlotInfo]:
        infos = _viz_helpers.list_plot_infos()
        if self._multicase is not None:
            source = _viz_helpers.PER_CASE_PLOT_INFO
            infos.insert(
                0,
                PlotInfo(
                    name=source.name,
                    title=source.title,
                    description=source.description,
                ),
            )
        return infos

    def render_plot(self, name: str, ax: Axes) -> list[str]:
        if name == _viz_helpers.PER_CASE_PLOT_INFO.name and self._multicase is not None:
            return _viz_helpers.render_per_case_reward(ax, self._per_case_rewards())
        return _viz_helpers.render(name, ax, self._recorder)

    def get_plot_data(self, name: str) -> dict[str, Any]:
        if name == _viz_helpers.PER_CASE_PLOT_INFO.name and self._multicase is not None:
            return _viz_helpers.per_case_data(self._per_case_rewards())
        return _viz_helpers.get_data(name, self._recorder)

    def list_homogeneous_plots(self) -> list[PlotInfo]:
        if not self._config.is_steady:
            return []
        return [replace(info) for info in _steady_viz_helpers.PLOT_INFOS]

    def render_homogeneous_plot(
        self, name: str, ax: Axes, result: ExperimentResult
    ) -> list[str]:
        data = _steady_viz_helpers.SteadyVizData()
        if self._steady is not None:
            cache_key = result.equation
            if (
                self._steady_viz_cache is None
                or self._steady_viz_cache[0] is not cache_key
            ):
                data = self._steady.build_viz_data(result.equation)
                self._steady_viz_cache = (cache_key, data)
            else:
                data = self._steady_viz_cache[1]
        return _steady_viz_helpers.render(name, ax, data)

    def _per_case_rewards(self) -> dict[str, float]:
        assert self._multicase is not None
        terms = self._best_terms()
        if not terms:
            return {}
        return self._multicase.per_case_rewards(terms)



    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before using the plugin.")

    def _assert_primary_dataset(self, dataset: Any) -> None:
        from kd.data.loaders.wave_breaking import wave_dataset_name

        assert self._multicase is not None
        expected = wave_dataset_name(self._multicase.primary_case)
        actual = getattr(dataset, "name", None)
        if actual != expected:
            raise ValueError(
                f"wave multi-case primary dataset mismatch: config primary_case "
                f"resolves to {self._multicase.primary_case!r} (expected dataset "
                f"name {expected!r}), but the facade dataset is named {actual!r}. "
                f"Build it with wave_breaking_case_to_dataset(cases[primary_case]) "
                f"so the reproducibility manifest carries the right identity."
            )

    def _resolve_variables(self, dataset: Any) -> tuple[str, ...]:
        if dataset.axis_order:
            dataset_axes: tuple[str, ...] = tuple(dataset.axis_order)
        elif dataset.axes:
            dataset_axes = tuple(dataset.axes.keys())
        else:
            dataset_axes = ()
        if self._config.variables is None:
            return dataset_axes
        if set(self._config.variables) != set(dataset_axes):
            raise ValueError(
                f"EqGPTConfig.variables {self._config.variables} does not "
                f"match dataset axes {dataset_axes}."
            )
        return tuple(self._config.variables)

    def _build_default_backend(self, vocab: Vocab) -> GPTBackend:
        gpt_config = GPTConfig(vocab_size=vocab.size)
        return RealGPTBackend.from_assets(
            gpt_config,
            weights_path=self._config.weights_path,
            asset_dir=self._config.asset_dir,
        )

    def _reset_search_state(self) -> None:
        self._pool_rewards = []
        self._pool_sentences = []
        self._reward_history = []
        self._rng = torch.Generator().manual_seed(self._config.seed)

    def _apply_state(self, state: dict[str, Any]) -> None:
        assert self._backend is not None
        backend_state = state.get("backend_state")
        if backend_state:
            self._backend.load_state_dict(backend_state)
        top_k = state.get("top_k") or {}
        self._pool_rewards = [float(r) for r in top_k.get("rewards", [])]
        self._pool_sentences = [list(s) for s in top_k.get("sentences", [])]
        self._reward_history = [list(h) for h in state.get("reward_history", [])]
        rng_state = state.get("rng_state")
        self._rng = torch.Generator()
        if rng_state is not None:
            self._rng.set_state(rng_state)
        else:
            self._rng.manual_seed(self._config.seed)

    def _epoch_metrics(self, finetune_loss: float | None) -> dict[str, float]:
        if not self._pool_rewards:
            pool_best = pool_median = pool_worst = _NO_MEASUREMENT
        else:
            pool_best = self._pool_rewards[0]
            pool_worst = self._pool_rewards[-1]
            mid = len(self._pool_rewards) // 2
            if len(self._pool_rewards) % 2 == 1:
                pool_median = self._pool_rewards[mid]
            else:
                pool_median = (
                    self._pool_rewards[mid - 1] + self._pool_rewards[mid]
                ) / 2.0
        return {
            _POOL_BEST_KEY: float(pool_best),
            _POOL_MEDIAN_KEY: float(pool_median),
            _POOL_WORST_KEY: float(pool_worst),
            _FINETUNE_LOSS_KEY: (
                float(finetune_loss) if finetune_loss is not None else _NO_MEASUREMENT
            ),
        }

    def _build_fingerprints(self) -> dict[str, Any]:
        weights_path = resolve_weights_fingerprint_path(
            self._backend, self._config.weights_path
        )
        fingerprints: dict[str, Any] = {
            "vocab_sha256": VOCAB_SHA256,
            "weights_path": str(weights_path) if weights_path else None,
        }







        if self._config.is_wave_multicase and self._multicase is not None:
            fingerprints["wave_cases"] = list(self._multicase.case_names)
            fingerprints["v1_asset_dir"] = (
                str(self._config.v1_asset_dir)
                if self._config.v1_asset_dir is not None
                else None
            )
            fingerprints["reward_points_per_window"] = (
                self._config.reward_points_per_window
            )
            fingerprints["coeff_points_per_window"] = (
                self._config.coeff_points_per_window
            )
        if self._config.is_steady and self._steady is not None:
            fingerprints["steady_activation"] = self._config.steady_activation
            fingerprints["steady_train_points"] = self._config.steady_train_points
            fingerprints["steady_validate_points"] = (
                self._config.steady_validate_points
            )
            fingerprints["steady_train_iters"] = self._config.steady_train_iters
            fingerprints["steady_surrogate_seed"] = (
                self._config.steady_surrogate_seed
            )
            fingerprints["steady_boundary_delete_num"] = (
                self._config.steady_boundary_delete_num
            )
            fingerprints["steady_polar_eval"] = self._config.steady_polar_eval
            fingerprints["steady_constant_column"] = (
                self._config.steady_constant_column
            )
            fingerprints["steady_dataset_fingerprint"] = (
                self._steady.dataset_fingerprint
            )
        return fingerprints

    def _next_seed(self) -> int:
        return int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())


__all__ = ["ALGORITHM_NAME", "STATE_KEYS", "EqGPTPlugin"]
