"""SGAConfig and operator pool constants for the SGA search algorithm."""

from __future__ import annotations

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
    ``field_names``. Ignored when ``use_autograd=False``."""

    autograd_train_epochs: int = 1000
    """Maximum training epochs for the auto-trained FieldModel. Only used when
    ``use_autograd=True`` and ``field_model is None``."""

    autograd_train_lr: float = 1e-3
    """Learning rate for the auto-trained FieldModel. Only used when
    ``use_autograd=True`` and ``field_model is None``."""
