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
    seed: int = DEFAULT_SEED
    library: LibraryConfig = field(default_factory=_default_library_config)
    min_length: int = DEFAULT_MIN_LENGTH
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE

    num_units: int = DEFAULT_NUM_UNITS
    num_layers: int = DEFAULT_NUM_LAYERS
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    observe_parent: bool = DEFAULT_OBSERVE_PARENT
    observe_sibling: bool = DEFAULT_OBSERVE_SIBLING
    observe_action: bool = DEFAULT_OBSERVE_ACTION
    observe_dangling: bool = DEFAULT_OBSERVE_DANGLING
    use_embedding: bool = DEFAULT_USE_EMBEDDING
    attention: bool = DEFAULT_ATTENTION
    attn_length: int = DEFAULT_ATTN_LENGTH
    initializer: Literal["xavier", "zeros"] = DEFAULT_INITIALIZER

    epsilon: float = DEFAULT_EPSILON
    baseline: str = DEFAULT_BASELINE
    entropy_weight: float = DEFAULT_ENTROPY_WEIGHT
    gamma: float = DEFAULT_GAMMA
    reward_alpha: float = DEFAULT_REWARD_ALPHA
    learning_rate: float = DEFAULT_LEARNING_RATE
    entropy_gamma: float = DEFAULT_ENTROPY_GAMMA
    max_diff_order: int | None = DEFAULT_MAX_DIFF_ORDER
    use_repeat_prior: bool = DEFAULT_USE_REPEAT_PRIOR
    repeat_tokens: list[str] = field(
        default_factory=lambda: list(DEFAULT_REPEAT_TOKENS)
    )
    repeat_max: int = DEFAULT_REPEAT_MAX
    use_trig_prior: bool = DEFAULT_USE_TRIG_PRIOR
    use_inverse_prior: bool = DEFAULT_USE_INVERSE_PRIOR
    use_diff_descendant_prior: bool = DEFAULT_USE_DIFF_DESCENDANT_PRIOR
    use_diff_child_prior: bool = DEFAULT_USE_DIFF_CHILD_PRIOR
    soft_length_loc: float | None = DEFAULT_SOFT_LENGTH_LOC
    soft_length_scale: float = DEFAULT_SOFT_LENGTH_SCALE
    stability_selection: int = DEFAULT_STABILITY_SELECTION
    stability_queue_capacity: int = DEFAULT_STABILITY_QUEUE_CAPACITY





    magnitude_filter: bool = DEFAULT_MAGNITUDE_FILTER

    diagnostic_scaffold: bool = DEFAULT_DIAGNOSTIC_SCAFFOLD






    diagnostic_scaffold_diffusion_tokens: tuple[str, ...] = (
        DEFAULT_DIAGNOSTIC_SCAFFOLD_DIFFUSION_TOKENS
    )
    diagnostic_scaffold_reaction_tokens: tuple[str, ...] = (
        DEFAULT_DIAGNOSTIC_SCAFFOLD_REACTION_TOKENS
    )
    diagnostic_scaffold_root_tokens: tuple[str, ...] = (
        DEFAULT_DIAGNOSTIC_SCAFFOLD_ROOT_TOKENS
    )
    diagnostic_scaffold_neutral_tokens: tuple[str, ...] = (
        DEFAULT_DIAGNOSTIC_SCAFFOLD_NEUTRAL_TOKENS
    )
    token_bias_tokens: tuple[str, ...] = DEFAULT_TOKEN_BIAS_TOKENS
    token_bias_weight: float = DEFAULT_TOKEN_BIAS_WEIGHT

    pinn: PINNConfig | None = None

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




