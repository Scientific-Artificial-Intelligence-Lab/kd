"""Canonical configuration for the DISCOVER search algorithm.

This module provides the shared ``DiscoverConfig`` used by both the
standalone entry point and the kd plugin adapter. It includes the
standalone-only ``n_iterations`` field for outer-loop control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from kd.search.discover.tokens.library import LibraryConfig










DIAGNOSTIC_ENV_VAR = "DISCOVER_ENABLE_DIAGNOSTICS"
DIAGNOSTIC_ENV_ENABLED_VALUE = "1"








DIAGNOSTIC_GATE_RATIONALE = (
    "diagnostic-only prior leaks ground-truth structural "
    "assumptions; dev/audit use only"
)


def _require_diagnostics_enabled() -> None:
    """Gate diagnostic-only features behind an explicit env opt-in.

    Raises ``RuntimeError`` with an actionable message when
    ``DISCOVER_ENABLE_DIAGNOSTICS`` is not exactly ``"1"``. The message
    includes the env var name verbatim (greppable in stderr/logs), the
    activation hint (``=1``), and the diagnostic-intent rationale so
    users understand why the gate exists.
    """
    value = os.environ.get(DIAGNOSTIC_ENV_VAR)
    if value != DIAGNOSTIC_ENV_ENABLED_VALUE:
        raise RuntimeError(
            f"diagnostic_scaffold=True requires "
            f"{DIAGNOSTIC_ENV_VAR}={DIAGNOSTIC_ENV_ENABLED_VALUE} "
            f"({DIAGNOSTIC_GATE_RATIONALE}). Set {DIAGNOSTIC_ENV_VAR}="
            f"{DIAGNOSTIC_ENV_ENABLED_VALUE} to enable."
        )






DEFAULT_N_ITERATIONS = 2000
DEFAULT_SEED: int = 0
DEFAULT_MIN_LENGTH = 2
DEFAULT_MAX_LENGTH = 15
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_UNITS = 16
DEFAULT_NUM_LAYERS = 1
DEFAULT_EMBEDDING_DIM = 4
DEFAULT_OBSERVE_PARENT = True
DEFAULT_OBSERVE_SIBLING = True
DEFAULT_OBSERVE_ACTION = False
DEFAULT_OBSERVE_DANGLING = False
DEFAULT_USE_EMBEDDING = False
DEFAULT_ATTENTION = False
DEFAULT_ATTN_LENGTH = 10
DEFAULT_INITIALIZER: Literal["xavier", "zeros"] = "xavier"
DEFAULT_EPSILON = 0.05
DEFAULT_BASELINE = "R_e"
DEFAULT_ENTROPY_WEIGHT = 0.005
DEFAULT_GAMMA = 0.5
DEFAULT_REWARD_ALPHA = 0.01
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_ENTROPY_GAMMA = 1.0
DEFAULT_MAX_DIFF_ORDER: int | None = 4
DEFAULT_USE_REPEAT_PRIOR = True
DEFAULT_REPEAT_TOKENS = ["add"]
DEFAULT_REPEAT_MAX = 5
DEFAULT_USE_TRIG_PRIOR = True
DEFAULT_USE_INVERSE_PRIOR = True
DEFAULT_USE_DIFF_DESCENDANT_PRIOR = True
DEFAULT_USE_DIFF_CHILD_PRIOR = True
DEFAULT_SOFT_LENGTH_LOC: float | None = None
DEFAULT_SOFT_LENGTH_SCALE = 5.0
DEFAULT_STABILITY_SELECTION = 0
DEFAULT_STABILITY_QUEUE_CAPACITY = 10





DEFAULT_MAGNITUDE_FILTER: bool = False





DEFAULT_DIAGNOSTIC_SCAFFOLD: bool = False
DEFAULT_DIAGNOSTIC_SCAFFOLD_DIFFUSION_TOKENS: tuple[str, ...] = ()
DEFAULT_DIAGNOSTIC_SCAFFOLD_REACTION_TOKENS: tuple[str, ...] = ()
DEFAULT_DIAGNOSTIC_SCAFFOLD_ROOT_TOKENS: tuple[str, ...] = ("add", "sub")



DEFAULT_DIAGNOSTIC_SCAFFOLD_NEUTRAL_TOKENS: tuple[str, ...] = ()


DEFAULT_TOKEN_BIAS_TOKENS: tuple[str, ...] = ()
DEFAULT_TOKEN_BIAS_WEIGHT: float = 0.0

DEFAULT_OPERATORS = ["add", "mul", "sub", "div", "sin", "cos", "diff_x", "diff2_x"]
DEFAULT_STATE_VARS = ["u"]
DEFAULT_COORD_VARS = ["x", "t"]


@dataclass(frozen=True, slots=True)
class PINNConfig:
    """PINN hyperparameters (defaults from config_pde_pinn.json)."""

    number_layer: int = 8
    n_hidden: int = 20
    activation: Literal["tanh", "sin", "relu"] = "tanh"
    pretrain_epoch: int = 200_000
    pinn_epoch: int = 1_000
    lr: float = 0.001

























    coef_pde: float = 0.0
    n_cycles: int = 2
    n_collocation: int = 50_000
    local_sample: bool = True
    local_multiplier: int = 20
    early_stop_patience: int = 500






    early_stop_warmup: int | None = None
    pretrain_val_ratio: float = 0.2
    grad_clip_norm: float | None = 1.0
    max_nan_recoveries: int = 3
    cycle_n_iterations: int | None = None
    colloc_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.cycle_n_iterations is not None and self.cycle_n_iterations <= 0:
            raise ValueError(
                f"cycle_n_iterations must be positive or None, "
                f"got {self.cycle_n_iterations}"
            )
        if self.early_stop_warmup is not None and self.early_stop_warmup < 0:
            raise ValueError(
                f"early_stop_warmup must be >= 0 or None, got {self.early_stop_warmup}"
            )
        if not 0.0 <= self.pretrain_val_ratio < 1.0:
            raise ValueError(
                f"pretrain_val_ratio must be in [0.0, 1.0), "
                f"got {self.pretrain_val_ratio}"
            )
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {self.grad_clip_norm}"
            )
        if self.local_multiplier < 0:
            raise ValueError(
                f"local_multiplier must be >= 0, got {self.local_multiplier}"
            )
        if self.local_sample and self.local_multiplier == 0:
            raise ValueError(
                "local_sample=True requires local_multiplier > 0; "
                "set local_sample=False to disable local sampling"
            )
        if self.colloc_chunk_size is not None and self.colloc_chunk_size <= 0:
            raise ValueError(
                f"colloc_chunk_size must be positive or None, "
                f"got {self.colloc_chunk_size}"
            )


def _default_library_config() -> LibraryConfig:
    """Return the default Burgers vocabulary."""
    return LibraryConfig(
        operators=list(DEFAULT_OPERATORS),
        state_vars=list(DEFAULT_STATE_VARS),
        coord_vars=list(DEFAULT_COORD_VARS),
    )


@dataclass(frozen=True, slots=True)
class DiscoverConfig:
    """Configuration for a DISCOVER (RL + optional PINN) search run.

    Default values are a smoke-test configuration; to reproduce paper
    results use ``burgers_preset()`` / ``chafee_preset()``. Paper-comparison
    experiments MUST go through presets.

    DISCOVER trains an LSTM controller with risk-seeking policy gradient to
    propose candidate equations, optionally backed by a PINN surrogate for
    derivatives. It has many knobs (controller, reward shaping, structural
    priors, PINN) — the handful you'll actually reach for:

    - ``library``: the token vocabulary to search over. Defines the search
      space — the single most impactful setting.
    - ``batch_size``: equations sampled per iteration. Larger = steadier
      policy-gradient signal, more compute per step.
    - ``learning_rate`` / ``entropy_weight``: controller step size and the
      exploration bonus (raise ``entropy_weight`` if it converges prematurely).
    - ``pinn``: set a ``PINNConfig`` to enable the PINN surrogate (MODE2); its
      ``pretrain_epoch`` is the main speed/accuracy knob there. Leave ``None``
      for MODE1 (finite-diff derivatives). **MODE2 runs only on the standalone
      path — the kd.Model facade rejects a non-None pinn (it would silently
      run MODE1 instead).**

    FACADE GOTCHA: via ``kd.Model(algorithm="discover")`` the search length
    comes from ``Model(generations=...)``, NOT from ``n_iterations`` here
    (that field only drives the standalone runner — see the note below).

    Remaining fields (controller architecture, reward shaping, structural
    priors, diagnostic scaffolds) have validated defaults and rarely change.
    """






    n_iterations: int = DEFAULT_N_ITERATIONS
    """Number of search iterations the standalone entry point runs, one controller batch
    per iteration. Ignored when the search is driven through ``kd.Model``, where
    ``Model(generations=...)`` sets the loop length.
    """
    seed: int = DEFAULT_SEED
    """Random seed of the search: ``torch.manual_seed`` is called with this value when
    the algorithm is constructed and again before the engine is built, so it governs
    both the controller's weight initialization and every token sampled from it. It is
    the only seeding entry and overrides a ``torch.manual_seed`` the caller made
    beforehand.
    """
    library: LibraryConfig = field(default_factory=_default_library_config)
    """Ordered vocabulary of tokens that candidate terms are built from, defaulting to
    ``["u", "u_x", "u_xx", "u_xxx"]``. Each term is a product of tokens drawn from
    this list, and the ordering matters: one of the mutation operators shifts a token
    to an adjacent entry.
    """
    min_length: int = DEFAULT_MIN_LENGTH
    """Minimum length of a sampled expression, in tokens. The sampler is forbidden from
    ending an expression before this many tokens, and any candidate that still comes
    out shorter is rejected before evaluation.
    """
    max_length: int = DEFAULT_MAX_LENGTH
    """Maximum number of tokens in a sampled sentence, counting the pinned
    ``start_words`` prefix; sampling stops there if no end token is drawn first. It
    must exceed ``len(start_words)`` and stay below the model's context length, and
    the default 49 is one below the pretrained context.
    """
    batch_size: int = DEFAULT_BATCH_SIZE
    """Number of candidate expressions sampled per search iteration, and the batch the
    policy-gradient update is computed from. Larger values give a steadier training
    signal at more compute per iteration.
    """

    num_units: int = DEFAULT_NUM_UNITS
    """Number of hidden units in each LSTM layer of the controller. It also sets the
    width of the attention projections when ``attention`` is enabled.
    """
    num_layers: int = DEFAULT_NUM_LAYERS
    """Number of stacked LSTM layers in the controller. Layers beyond the first take the
    previous layer's hidden state as their input.
    """
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    """Width of the learned embedding used for each categorical observation channel. It
    has no effect unless ``use_embedding`` is True, which replaces the one-hot inputs
    with embeddings of this size.
    """
    observe_parent: bool = DEFAULT_OBSERVE_PARENT
    """When True, the parent token of the position about to be sampled is added to the
    controller input, as a one-hot vector or as an embedding when ``use_embedding``
    is set. At least one of the four observation channels must be enabled.
    """
    observe_sibling: bool = DEFAULT_OBSERVE_SIBLING
    """When True, the left sibling token of the position about to be sampled is added to
    the controller input, as a one-hot vector or as an embedding when
    ``use_embedding`` is set. At least one of the four observation channels must be
    enabled.
    """
    observe_action: bool = DEFAULT_OBSERVE_ACTION
    """When True, the token sampled at the previous step is added to the controller
    input, as a one-hot vector or as an embedding when ``use_embedding`` is set.
    """
    observe_dangling: bool = DEFAULT_OBSERVE_DANGLING
    """When True, the count of unfilled argument slots in the partial expression tree is
    appended to the controller input as a single numeric value. This channel is
    always a raw count, so ``use_embedding`` does not apply to it.
    """
    use_embedding: bool = DEFAULT_USE_EMBEDDING
    """When True, the categorical observation channels enter the controller as learned
    embeddings of width ``embedding_dim`` instead of one-hot vectors. The dangling-
    slot count is unaffected and stays a single numeric input.
    """
    attention: bool = DEFAULT_ATTENTION
    """When True, the controller applies additive (Bahdanau) attention over a sliding
    window of its own recent outputs before producing token logits. The window length
    is ``attn_length``.
    """
    attn_length: int = DEFAULT_ATTN_LENGTH
    """Length of the attention window, in past controller steps. It has no effect when
    ``attention`` is False.
    """
    initializer: Literal["xavier", "zeros"] = DEFAULT_INITIALIZER
    """Parameter initialization scheme, either ``"xavier"`` or ``"zeros"``. ``"xavier"``
    draws all parameters from a Xavier/uniform baseline; ``"zeros"`` also zeroes the
    LSTM cell parameters and the output bias, making the initial token distribution
    uniform.
    """

    epsilon: float = DEFAULT_EPSILON
    """Risk-seeking quantile: the policy-gradient update keeps the batch rewards at
    or above the ``1 - epsilon`` quantile (``RSPGStrategy``), so a smaller value is a
    greedier update off fewer samples. Ties at the quantile are all kept, so the
    retained share can exceed ``epsilon`` (an all-equal batch keeps everything). Must
    lie in ``(0, 1]``. The packaged PDE presets use 0.01-0.02; the delta registry
    records the lineage of the ``0.05`` default in.
    """
    baseline: str = DEFAULT_BASELINE
    """Value subtracted from the rewards in the policy-gradient loss. ``"R_e"`` (the
    default) uses the risk-seeking reward quantile itself, ``"ewma_R"`` a moving
    average of the mean kept reward, and ``"combined"`` the quantile plus a moving
    average of the gap between the two.
    """
    entropy_weight: float = DEFAULT_ENTROPY_WEIGHT
    """Weight of the entropy bonus added to the policy-gradient loss, which rewards a
    less peaked sampling distribution. Must be non-negative; raise it when the
    controller commits to one expression family too early, set 0.0 to drop the bonus.
    """
    gamma: float = DEFAULT_GAMMA
    """Decay of the moving-average baseline: the running value keeps weight ``gamma``
    and the new batch contributes ``1 - gamma``. In the range 0 to 1, and only read
    when ``baseline`` is ``"ewma_R"`` or ``"combined"``.
    """
    reward_alpha: float = DEFAULT_REWARD_ALPHA
    """Weight of the complexity penalty in the reward ``(1 - reward_alpha * complexity)
    / (1 + sqrt(nmse))``. Larger values push the search toward shorter equations;
    since the reward is clipped at 0, a large value flattens long candidates to zero
    reward.
    """
    learning_rate: float = DEFAULT_LEARNING_RATE
    """Step size of the Adam optimizer that updates the controller network. A run
    resumed from a checkpoint adopts this value rather than the one saved with the
    checkpoint.
    """
    entropy_gamma: float = DEFAULT_ENTROPY_GAMMA
    """Per-position decay of the entropy bonus: the entropy at position ``t`` of a
    sampled sequence is weighted by ``entropy_gamma ** t``. The default 1.0 weights
    every position equally; values below 1 concentrate the exploration bonus on the
    first tokens.
    """
    max_diff_order: int | None = DEFAULT_MAX_DIFF_ORDER
    """Highest cumulative derivative order allowed in a sampled expression; candidates
    above it are rejected before evaluation. Orders accumulate along a chain, so
    ``diff2_x(diff_x(u))`` counts as 3, and ``None`` disables the check.
    """
    use_repeat_prior: bool = DEFAULT_USE_REPEAT_PRIOR
    """If True, stop a token from being sampled again once it has already appeared
    ``repeat_max`` times in the expression being built. The tokens subject to the
    limit are named in ``repeat_tokens``; with the defaults this caps ``add`` at five
    and so bounds the number of additive terms.
    """
    repeat_tokens: list[str] = field(
        default_factory=lambda: list(DEFAULT_REPEAT_TOKENS)
    )
    """Token names counted by the repeat limit, sharing one budget: their occurrences
    are pooled and compared against ``repeat_max``. Read only when
    ``use_repeat_prior`` is true, and every name must exist in the token library.
    """
    repeat_max: int = DEFAULT_REPEAT_MAX
    """Maximum combined number of times the ``repeat_tokens`` may appear in one
    expression; on reaching the count those tokens are removed from the sampling
    distribution for the rest of that expression. Read only when ``use_repeat_prior``
    is true.
    """
    use_trig_prior: bool = DEFAULT_USE_TRIG_PRIOR
    """If True, forbid trigonometric and derivative tokens anywhere inside the subtree
    of another trigonometric or derivative token. This rules out compositions such as
    ``sin(cos(u))`` and, because derivative tokens are included, nested derivatives
    such as ``diff_x(diff_x(u))``.
    """
    use_inverse_prior: bool = DEFAULT_USE_INVERSE_PRIOR
    """If True, forbid a unary token from being the direct child of its own inverse, so
    cancelling pairs such as ``exp(log(u))`` and ``sqrt(n2(u))`` are never sampled.
    Only pairs whose two tokens are both in the library are constrained.
    """
    use_diff_descendant_prior: bool = DEFAULT_USE_DIFF_DESCENDANT_PRIOR
    """If True, forbid ``add`` and ``sub`` anywhere inside the subtree of a derivative
    token, so no sampled candidate differentiates a sum.
    """
    use_diff_child_prior: bool = DEFAULT_USE_DIFF_CHILD_PRIOR
    """If True, restrict the child of a derivative token to a state variable or another
    derivative token; coordinate variables and every other operator are forbidden in
    that position. This keeps constant terms such as ``diff_x(x)`` out of the search.
    """
    soft_length_loc: float | None = DEFAULT_SOFT_LENGTH_LOC
    """Target expression length in tokens: past position ``soft_length_loc`` the
    sampling logits of operator tokens are reduced by ``(t - soft_length_loc)**2 / (2
    * soft_length_scale)``, so sampling tends to terminate near that length.
    ``None``, the default, leaves the prior off.
    """
    soft_length_scale: float = DEFAULT_SOFT_LENGTH_SCALE
    """Width of the length penalty past ``soft_length_loc``, which is ``(t -
    soft_length_loc)**2 / (2 * soft_length_scale)``: larger values make the pull
    toward the target length gentler. Must be positive, and is read only when
    ``soft_length_loc`` is set.
    """
    stability_selection: int = DEFAULT_STABILITY_SELECTION
    """Number of distinct top-reward candidates from the final search cycle that enter
    bootstrap stability selection, which re-fits each candidate on resampled rows and
    keeps the one that wins the most resamples. ``0`` (the default) skips the step
    and keeps the best-reward candidate; only a run driven by the PINN surrogate
    applies it.
    """
    stability_queue_capacity: int = DEFAULT_STABILITY_QUEUE_CAPACITY
    """Maximum number of distinct candidates kept in the reward-ordered per-cycle pool
    that stability selection draws from. Read only when ``stability_selection`` is
    greater than 0, and must be at least as large as it.
    """





    magnitude_filter: bool = DEFAULT_MAGNITUDE_FILTER
    """When ``True``, a fit whose active coefficients include any magnitude below
    ``5e-5`` or above ``1e4`` is marked invalid, so it scores zero reward and is
    dropped from controller training. Defaults to ``False``, which accepts a fit at
    any coefficient magnitude.
    """

    diagnostic_scaffold: bool = DEFAULT_DIAGNOSTIC_SCAFFOLD
    """Attach the diagnostic scaffold prior, which restricts sampling in the first
    search cycle to a root token from ``diagnostic_scaffold_root_tokens`` whose two
    branches draw on disjoint token sets. Off by default; ``True`` also requires
    ``DISCOVER_ENABLE_DIAGNOSTICS=1`` in the environment, because the scaffold builds
    an assumed equation shape into the search.
    """






    diagnostic_scaffold_diffusion_tokens: tuple[str, ...] = (
        DEFAULT_DIAGNOSTIC_SCAFFOLD_DIFFUSION_TOKENS
    )
    """Token names the scaffold keeps to the left branch of the root: each one is
    excluded from sampling while the right branch is filled. Read only when
    ``diagnostic_scaffold`` is ``True``, and a name also listed as neutral is exempt.
    """
    diagnostic_scaffold_reaction_tokens: tuple[str, ...] = (
        DEFAULT_DIAGNOSTIC_SCAFFOLD_REACTION_TOKENS
    )
    """Token names the scaffold keeps to the right branch of the root: each one is
    excluded from sampling while the left branch is filled. Read only when
    ``diagnostic_scaffold`` is ``True``, and a name also listed as neutral is exempt.
    """
    diagnostic_scaffold_root_tokens: tuple[str, ...] = (
        DEFAULT_DIAGNOSTIC_SCAFFOLD_ROOT_TOKENS
    )
    """Token names the scaffold allows at the root of a sampled expression; at the first
    sampling step every other token is excluded, so the tree starts from one of
    these. Defaults to ``("add", "sub")`` and is read only when
    ``diagnostic_scaffold`` is ``True``; a set in which no name exists in the token
    library fails when the search is built.
    """
    diagnostic_scaffold_neutral_tokens: tuple[str, ...] = (
        DEFAULT_DIAGNOSTIC_SCAFFOLD_NEUTRAL_TOKENS
    )
    """Token names the scaffold never excludes, so they stay samplable in both branches
    even when they also appear in the diffusion or reaction list. Read only when
    ``diagnostic_scaffold`` is ``True``; a name shared with
    ``diagnostic_scaffold_root_tokens`` is rejected.
    """
    token_bias_tokens: tuple[str, ...] = DEFAULT_TOKEN_BIAS_TOKENS
    """Token names whose sampling probability is shifted by ``token_bias_weight``,
    identically at every step of every sampled expression. The bias applies only when
    this list is non-empty and ``token_bias_weight`` is not ``0.0``; a name absent
    from the token library is skipped.
    """
    token_bias_weight: float = DEFAULT_TOKEN_BIAS_WEIGHT
    """Amount added to the log-probability of every token in ``token_bias_tokens``, the
    same at each sampling step. Positive values make those tokens more likely and
    negative values less likely; ``0.0`` (the default) leaves the bias off.
    """

    pinn: PINNConfig | None = None
    """Settings for the optional PINN surrogate, which alternates symbolic search with
    training a network on the data and takes derivatives from that trained network.
    ``None`` (the default) searches on finite-difference derivatives; ``kd.Model``
    rejects a non-``None`` value because it never runs the PINN cycle.
    """

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.stability_selection < 0:
            raise ValueError(
                f"stability_selection must be >= 0, got {self.stability_selection}"
            )
        if self.stability_queue_capacity < 0:
            raise ValueError(
                "stability_queue_capacity must be >= 0, "
                f"got {self.stability_queue_capacity}"
            )
        if (
            self.stability_selection > 0
            and self.stability_queue_capacity < self.stability_selection
        ):
            raise ValueError(
                "stability_queue_capacity must be >= stability_selection, "
                f"got {self.stability_queue_capacity} < "
                f"{self.stability_selection}"
            )




        if self.diagnostic_scaffold:
            _require_diagnostics_enabled()



    @classmethod
    def burgers_preset(cls, **overrides: Any) -> DiscoverConfig:
        """Config matching DISCOVER reference for the Burgers equation.

        All hyperparameters are taken from the upstream DISCOVER
        reference configuration (``config_pde_Burgers.json`` layered on
        ``config_common.json``).

        Operator names use our unary-diff convention:
        ``diff`` -> ``diff_x``, ``diff2`` -> ``diff2_x``, etc.

        Pass ``**overrides`` to customize individual fields, e.g.
        ``DiscoverConfig.burgers_preset(batch_size=64, n_iterations=500)``.
        """
        defaults: dict[str, Any] = {

            "n_iterations": 100,
            "library": LibraryConfig(
                operators=[
                    "add",
                    "mul",
                    "div",
                    "diff_x",
                    "diff2_x",
                    "diff3_x",
                    "n2",
                    "n3",
                ],
                state_vars=["u"],
                coord_vars=["x", "t"],
            ),
            "batch_size": 500,
            "epsilon": 0.02,
            "learning_rate": 0.0025,
            "entropy_weight": 0.03,
            "entropy_gamma": 0.7,
            "num_units": 32,
            "attention": True,
            "attn_length": 10,
            "initializer": "zeros",



            "min_length": 2,
            "max_length": 64,
            "soft_length_loc": 10.0,
            "soft_length_scale": 5.0,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def chafee_preset(cls, **overrides: Any) -> DiscoverConfig:
        """Config matching DISCOVER reference for the Chafee-Infante equation.

        Identical to :meth:`burgers_preset` except ``max_length=256``
        (reference: ``config_pde_Chafee.json``).
        """
        chafee_overrides: dict[str, Any] = {"max_length": 256}
        chafee_overrides.update(overrides)
        return cls.burgers_preset(**chafee_overrides)


__all__ = [
    "DIAGNOSTIC_ENV_ENABLED_VALUE",
    "DIAGNOSTIC_ENV_VAR",
    "DIAGNOSTIC_GATE_RATIONALE",
    "DiscoverConfig",
    "PINNConfig",
]




