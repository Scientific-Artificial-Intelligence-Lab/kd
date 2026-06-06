
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from kd.core.evaluator import EvaluationResult
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.engine import DiscoverEngine
from kd.search.discover.evaluation.dedup import Deduplicator
from kd.search.discover.evaluation.reward import compute_reward
from kd.search.discover.tokens.library import Library
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    DiffDescendantConstraint,
    InverseUnaryConstraint,
    LengthConstraint,
    Prior,
    PriorSystem,
    RepeatConstraint,
    ScaffoldPrior,
    SoftLengthPrior,
    TokenBiasPrior,
    TrigConstraint,
)
from kd.search.discover.tokens.validator import CandidateValidator
from kd.search.discover.training.strategy import RSPGStrategy


def build_library(config: DiscoverConfig) -> Library:
    return Library.from_config(config.library)


def build_prior_system(library: Library, config: DiscoverConfig) -> PriorSystem:
    priors: list[Prior] = [
        LengthConstraint(
            library,
            min_=config.min_length,
            max_=config.max_length,
        ),
    ]

    if config.use_diff_child_prior:
        priors.append(DiffChildConstraint(library))
    if config.use_repeat_prior:
        repeat_token_indices = np.array(
            [library.name_to_index(name) for name in config.repeat_tokens],
            dtype=np.int32,
        )
        priors.append(
            RepeatConstraint(
                library,
                tokens=repeat_token_indices,
                max_=config.repeat_max,
            )
        )
    if config.use_trig_prior:
        priors.append(TrigConstraint(library))
    if config.use_inverse_prior:
        priors.append(InverseUnaryConstraint(library))
    if config.use_diff_descendant_prior:
        priors.append(DiffDescendantConstraint(library))
    if config.soft_length_loc is not None:
        priors.append(
            SoftLengthPrior(
                library,
                loc=config.soft_length_loc,
                scale=config.soft_length_scale,
            )
        )



    if config.diagnostic_scaffold:
        priors.append(
            ScaffoldPrior(
                library,
                diffusion_tokens=list(config.diagnostic_scaffold_diffusion_tokens),
                reaction_tokens=list(config.diagnostic_scaffold_reaction_tokens),
                root_tokens=tuple(config.diagnostic_scaffold_root_tokens),
                neutral_tokens=tuple(
                    config.diagnostic_scaffold_neutral_tokens
                ),
            )
        )
    if config.token_bias_weight != 0.0 and config.token_bias_tokens:
        priors.append(
            TokenBiasPrior(
                library,
                token_names=list(config.token_bias_tokens),
                bias=config.token_bias_weight,
            )
        )
    return PriorSystem(library, priors)


def build_controller(
    library: Library,
    prior_system: PriorSystem,
    config: DiscoverConfig,
) -> LSTMController:
    return LSTMController(
        library=library,
        prior_system=prior_system,
        num_units=config.num_units,
        num_layers=config.num_layers,
        embedding_dim=config.embedding_dim,
        observe_parent=config.observe_parent,
        observe_sibling=config.observe_sibling,
        observe_action=config.observe_action,
        observe_dangling=config.observe_dangling,
        use_embedding=config.use_embedding,
        attention=config.attention,
        attn_length=config.attn_length,
        initializer=config.initializer,
    )


def build_strategy(config: DiscoverConfig) -> RSPGStrategy:
    return RSPGStrategy(
        epsilon=config.epsilon,
        baseline=config.baseline,
        entropy_weight=config.entropy_weight,
        gamma=config.gamma,
        entropy_gamma=config.entropy_gamma,
        learning_rate=config.learning_rate,
    )


def _make_reward_adapter(alpha: float) -> Callable[[EvaluationResult], float]:

    def adapter(result: EvaluationResult) -> float:
        return compute_reward(result, alpha=alpha)

    return adapter


def build_engine(config: DiscoverConfig) -> DiscoverEngine:
    library = build_library(config)
    prior_system = build_prior_system(library, config)
    controller = build_controller(library, prior_system, config)
    strategy = build_strategy(config)
    validator = CandidateValidator(
        library,
        max_length=config.max_length,
        max_diff_order=config.max_diff_order,
        min_length=config.min_length,
    )
    deduplicator = Deduplicator(library)
    return DiscoverEngine(
        generator=controller,
        strategy=strategy,
        reward_adapter=_make_reward_adapter(config.reward_alpha),
        validator=validator,
        deduplicator=deduplicator,
        batch_size=config.batch_size,
        cycle_candidate_capacity=(
            config.stability_queue_capacity
            if config.stability_selection > 0
            else 0
        ),
    )


__all__ = [
    "build_library",
    "build_prior_system",
    "build_controller",
    "build_strategy",
    "build_engine",
]
