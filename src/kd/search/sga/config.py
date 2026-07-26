"""SGAConfig and operator pool constants for the SGA search algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kd.models.field_model import FieldModel

DedupMode = Literal["none", "pre_prune", "post_prune", "dual"]
"""Allowed values for ``SGAConfig.dedup_mode``"""

OperatorPool = tuple[tuple[str, int], ...]
"""Convenience alias for name/arity operator pools."""






OPS: OperatorPool = (
    ("+", 2),
    ("-", 2),
    ("*", 2),
    ("/", 2),
    ("^2", 1),
    ("^3", 1),
    ("d", 2),
    ("d^2", 2),
)
"""All operators: arithmetic, powers, and derivative operators."""

ROOT: OperatorPool = (
    ("*", 2),
    ("/", 2),
    ("^2", 1),
    ("^3", 1),
    ("d", 2),
    ("d^2", 2),
)
"""Root-eligible operators: NO +/- (addition semantics handled by PDE terms)."""

OP1: OperatorPool = (
    ("^2", 1),
    ("^3", 1),
)
"""Unary operators for mutation."""

OP2: OperatorPool = (
    ("+", 2),
    ("-", 2),
    ("*", 2),
    ("/", 2),
    ("d", 2),
    ("d^2", 2),
)
"""Binary operators for mutation."""


def build_den(axes: list[str], lhs_axis: str) -> tuple[tuple[str, int], ...]:
    """Build the allowed derivative denominator pool from dataset axes."""
    den = tuple((axis, 0) for axis in axes if axis != lhs_axis)
    if not den:
        raise ValueError("No RHS derivative axes available after filtering lhs_axis.")
    return den


@dataclass
class SGAConfig:
    """Configuration for the SGA search algorithm.

    This config covers GA parameters and tree structure constraints.
    It does NOT contain ``generations`` or ``sga_run`` -- the Runner
    is the sole loop owner.
    """


    num: int = 20
    """Population size (number of PDE candidates)."""

    p_var: float = 0.5
    """Probability that a node is a variable (vs. operator)."""

    p_mute: float = 0.3
    """Mutation probability per node."""

    p_cro: float = 0.5
    """Crossover probability between PDEs."""

    p_rep: float = 1.0
    """Replace probability (chance of replacing a term)."""

    seed: int = 0
    """Random seed for reproducibility."""


    depth: int = 4
    """Maximum tree depth for each term."""

    width: int = 5
    """Maximum number of terms per PDE."""


    aic_ratio: float = 1.0
    """AIC penalty ratio."""

    lam: float = 0.0
    """Ridge lambda (0 = OLS)."""

    d_tol: float = 1.0
    """Tolerance step size for STRidge sweep."""

    maxit: int = 10
    """Number of tolerance sweep iterations."""

    str_iters: int = 10
    """STRidge internal iterations per tolerance level."""

    normalize: int = 2
    """Column norm order for STRidge normalization."""


    dedup_mode: DedupMode = "pre_prune"
    """Deduplication strategy for genetic offspring.

    Controls how ``_pde_lib`` keys are computed in
    ``SGAPlugin._apply_genetic_ops``:

    - ``"none"``: no dedup (accept duplicate offspring; ``_pde_lib`` stays
      empty, counters stay 0).
    - ``"pre_prune"``: dedup using ``pde_to_kd_expr`` of the **raw** offspring
      genotype (current default; matches the predecessor / paper SGA-PDE behavior).
    - ``"post_prune"``: dedup using ``pde_to_kd_expr`` of the **pruned** PDE
      after ``evaluate_candidate``. Closes the pre-prune-key/post-prune-storage
      gap.
    - ``"dual"``: pre-prune cheap filter + post-prune strict check. Most
      expensive, strictest dedup.

    Default ``"pre_prune"`` is conservative; the optimal mode is decided by
    a dedicated dedup-mode ablation experiment.
    """


    use_autograd: bool = False
    """If True, train (or reuse) a FieldModel surrogate and use AutogradProvider
    for Layer 2 terminals (u_x, u_t). Layer 1 (raw u leaf) and Layer 3 (tree
    d / d^2 operators) are unchanged

    Note: enabling this triggers FieldModel training in ``prepare()`` unless a
    pre-trained ``field_model`` is provided, which can take seconds to minutes.
    Does NOT replace ``components.context.derivative_provider`` — the surrogate
    is used only inside SGA's Layer 2 lookups."""

    field_model: FieldModel | None = None
    """Optional pre-trained FieldModel surrogate (skips auto-training when
    ``use_autograd=True``). Must have matching ``coord_names`` /
    ``field_names``. Ignored when ``use_autograd=False``.

    The plugin aligns this model **in place** (``nn.Module.to``) to the
    dataset's device and dtype when they differ — e.g. a CPU float32 model
    passed with cuda float64 data ends up on cuda:0 in float64 after
    ``fit()``. Pass a dedicated copy if the original placement matters."""

    autograd_train_epochs: int = 1000
    """Maximum training epochs for the auto-trained FieldModel. Only used when
    ``use_autograd=True`` and ``field_model is None``."""

    autograd_train_lr: float = 1e-3
    """Learning rate for the auto-trained FieldModel. Only used when
    ``use_autograd=True`` and ``field_model is None``."""

    autograd_train_patience: int | None = None
    """Early-stopping patience for the auto-trained FieldModel (epochs without
    validation improvement). ``None`` (default) disables early stopping, so the
    surrogate trains the full ``autograd_train_epochs`` budget — the v1 / paper
    reference semantics (``sgapde/metann.py`` fixed-step training, no early stop).
    Requires ``autograd_train_val_ratio > 0`` (there is no validation signal to
    monitor otherwise). Only used when ``use_autograd=True`` and
    ``field_model is None``; ignored in finite-diff mode (same handling as
    ``autograd_train_epochs``)."""

    autograd_train_val_ratio: float = 0.0
    """Fraction of data held out for validation while auto-training the
    FieldModel. ``0.0`` (default) trains on ALL data — the v1 / paper reference
    semantics (full-data training, no val split). Set ``> 0`` only when using
    ``autograd_train_patience`` for early stopping. Only used when
    ``use_autograd=True`` and ``field_model is None``; ignored in finite-diff
    mode (same handling as ``autograd_train_epochs``)."""

    def __post_init__(self) -> None:
        """Validate the autograd training-budget fields (fail-loud).

        Deliberate style exception: ``SGAConfig`` is otherwise a plain dataclass
        with no ``__post_init__``, but the autograd budget fields gate a silent
        failure mode (a patience-with-no-val-signal config that quietly does
        nothing) — so this mirrors ``DLGAConfig``'s runtime validation. The
        config layer is intentionally STRICTER than the trainer (which only
        *warns* on patience+val_ratio=0); a misconfigured budget must not reach
        training. ``use_autograd=False`` paths ignore these fields, but the
        validation still runs (cheap, and keeps the constraint honest if the
        config is later switched to autograd).
        """
        if (
            self.autograd_train_patience is not None
            and self.autograd_train_val_ratio == 0.0
        ):
            raise ValueError(
                "autograd_train_patience requires autograd_train_val_ratio > 0 "
                "(early stopping needs a validation signal); got patience="
                f"{self.autograd_train_patience}, val_ratio="
                f"{self.autograd_train_val_ratio}."
            )
        if (
            self.autograd_train_patience is not None
            and self.autograd_train_patience < 1
        ):
            raise ValueError(
                "autograd_train_patience must be None or >= 1, got "
                f"{self.autograd_train_patience}."
            )



        if math.isnan(self.autograd_train_val_ratio) or not (
            0.0 <= self.autograd_train_val_ratio < 1.0
        ):
            raise ValueError(
                "autograd_train_val_ratio must be in [0, 1), got "
                f"{self.autograd_train_val_ratio}."
            )
