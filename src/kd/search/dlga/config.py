"""Configuration for the DLGA Stage I plugin."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DLGAConfig:
    """DLGA Stage I (constant-coefficient) configuration.

    DLGA evolves gene-encoded operator combinations with a genetic algorithm,
    scoring each candidate against derivatives from a trained NN surrogate
    (NN_1). Most fields have defaults aligned with the Xu 2020 reference and
    rarely need touching — the handful you'll actually reach for:

    - ``library``: operator vocabulary to search over (default
      ``["u", "u_x", "u_xx", "u_xxx"]``). Widen to admit more candidate terms.
    - ``pop_size``: GA population size (default 400). Larger = broader search,
      slower; drop to ~50-100 for a quick demo.
    - ``epsilon``: length penalty in ``fitness = nmse + epsilon * length``
      (default 1e-3). Raise to bias toward simpler expressions.
    - ``surrogate_max_epochs``: NN_1 training budget (default 50000). The main
      speed/accuracy knob — 50000 matches Xu's paper-grade recovery but is
      slow; ~2000 is plenty for a fast demo (coarser derivatives).

    Fields below are grouped: core (search space / size / scoring), GA
    operators (advanced), and NN_1 surrogate (advanced).
    """


    mode: Literal["constant", "adaptive", "auto"] = field(
        default="constant",
        metadata={
            "doc": (
                "Legal values are 'constant', 'adaptive', and 'auto'. "
                "Stage I implements only 'constant'; 'adaptive' and 'auto' "
                "are reserved and rejected by DLGAPlugin."
            )
        },
    )
    """Coefficient mode of the search; only ``"constant"`` (constant-coefficient
    equations) is implemented. The values ``"adaptive"`` and ``"auto"`` pass
    configuration validation but raise ``NotImplementedError`` when the algorithm is
    constructed.
    """
    library: list[str] = field(default_factory=lambda: ["u", "u_x", "u_xx", "u_xxx"])
    """Ordered vocabulary of tokens that candidate terms are built from, defaulting to
    ``["u", "u_x", "u_xx", "u_xxx"]``. Each term is a product of tokens drawn from
    this list, and the ordering matters: one of the mutation operators shifts a token
    to an adjacent entry.
    """
    solver: Literal["svd_null_space", "ols"] = "svd_null_space"
    """Solver used to fit each candidate's coefficients against the left-hand side:
    ``"svd_null_space"`` takes the null space of the augmented system (total least
    squares), ``"ols"`` uses ordinary least squares.
    """
    lhs_auto_select: bool = True
    """When True, every candidate is fitted against both the first time derivative
    (``u_t``) and the second (``u_tt``), and the branch with the lower NMSE is kept.
    When False, only ``u_t`` is built and used as the left-hand side.
    """






    target_lhs_order: int = 1
    """Order of the time derivative this configuration targets as the left-hand side: 1
    for ``u_t``, 2 for ``u_tt``. It must equal the dataset's own left-hand-side order
    or the fit is rejected before it starts, and order 2 additionally requires
    ``lhs_auto_select=True``.
    """






























    epsilon: float = 1e-3
    """Complexity penalty in the genetic fitness ``NMSE + epsilon * length``, where
    ``length`` is the total number of tokens across the candidate's terms. Raise it
    to push the search toward shorter equations; the useful value is problem-
    dependent, and the packaged presets span 1e-6 to 1e-3.
    """
    pop_size: int = 400
    """Number of candidate equations in each generation of the genetic search, which is
    also how many candidates the platform evaluates per iteration.
    """
    seed: int = 0
    """Random seed of the genetic search: it seeds the generator behind the initial
    population and every crossover, mutation, add and delete draw, and is also
    forwarded to the surrogate network's training. Runs differing only in this value
    explore different candidates.
    """







    auto_upgrade_threshold: float = field(
        default=1e-3,
        metadata={
            "doc": (
                "Reserved for Stage II — the mode='auto' upgrade trigger: "
                "when Stage I (constant-coefficient) best NMSE exceeds this "
                "threshold, escalate to the Stage II adaptive (variable-"
                "coefficient, Xu 2021) search. Inert while mode='constant' "
                "(the only Stage I mode); Stage II is not yet implemented. "
                "Default 1e-3 is an unvalidated placeholder."
            )
        },
    )
    """Reserved for the planned ``mode="auto"`` escalation: the best NMSE above which
    the search would hand over to an adaptive, variable-coefficient stage. It has no
    effect today, because ``"constant"`` is the only implemented mode, and the 1e-3
    default is an untuned placeholder.
    """


    max_modules: int = 5
    """Maximum number of additive terms allowed in one candidate equation."""
    max_module_length: int = 5
    """Maximum number of tokens multiplied together inside a single term."""
    partial_prob: float = 0.6
    """Probability, from 0 to 1, of extending a randomly built term with one more
    factor. The term stops growing once it reaches ``max_module_length``.
    """
    genes_prob: float = 0.6
    """Probability, from 0 to 1, of adding one more term while a random candidate
    equation is built. Generation stops once the candidate reaches ``max_modules``
    terms.
    """
    crossover_rate: float = 0.8
    """Probability that a pair of surviving candidates exchanges one term during
    crossover, from 0 to 1. Survivors are paired off in order, and each pair either
    swaps one randomly chosen term or passes through unchanged.
    """
    mutation_rate: float = 0.4
    """Probability that a mutation shifts one factor of one term along ``library``, from
    0 to 1. The shift is to an adjacent entry, except for a factor sitting at the
    first entry, which is redrawn from the whole ``library``.
    """
    add_rate: float = 0.4
    """Probability that a newly built random term is appended to a candidate, from 0 to
    1. The term is added only if the candidate is below ``max_modules`` and does not
    already contain it.
    """
    delete_rate: float = 0.5
    """Probability that one term is dropped from a candidate, from 0 to 1. A candidate
    that is down to a single term is left unchanged. The draw is independent of
    ``add_rate``, so one candidate can gain and lose a term in the same generation.
    """












    surrogate_hidden_sizes: list[int] = field(
        default_factory=lambda: [50, 50, 50, 50, 50]
    )
    """Widths of the hidden layers in the neural network fitted to the field data, one
    hidden layer per entry (default five layers of 50 units). The search
    differentiates this network for the derivatives it scores, so its capacity bounds
    their accuracy.
    """
    surrogate_activation: Literal["tanh", "sin", "relu"] = "sin"
    """Activation applied after each hidden layer of the fitted network: ``tanh``,
    ``sin``, or ``relu``. The default ``sin`` stays smooth under the repeated
    differentiation the search relies on, which ``relu`` does not.
    """
    surrogate_lr: float = 1e-3
    """Learning rate of the Adam optimizer used to fit the network to the field data."""
    surrogate_max_epochs: int = 50000
    """Maximum number of epochs to spend fitting the network, one full-batch Adam step
    per epoch. It trades run time against derivative accuracy: dropping the default
    50000 to a few thousand finishes much sooner with coarser derivatives.
    """







    surrogate_patience: int | None = None
    """Number of consecutive epochs without validation-loss improvement after which the
    fit stops early. ``None``, the default, disables early stopping and runs the full
    ``surrogate_max_epochs`` budget; other values need ``surrogate_val_ratio`` above
    0 to take effect.
    """
    surrogate_val_ratio: float = 0.2
    """Fraction of the samples held out to measure validation loss while fitting the
    network, from 0 up to but not including 1. Setting it to 0 trains on every sample
    and leaves ``surrogate_patience`` and ``surrogate_restore_best`` with no signal
    to act on.
    """
    surrogate_restore_best: bool = True
    """If True, the weights from the epoch with the lowest validation loss are restored
    at the end of the fit instead of keeping the last epoch's weights. Has no effect
    when ``surrogate_val_ratio`` is 0, since there is no validation loss to rank
    epochs by.
    """

    def __post_init__(self) -> None:
        if self.mode not in {"constant", "adaptive", "auto"}:
            raise ValueError(f"mode must be constant/adaptive/auto, got {self.mode!r}")
        if self.solver not in {"svd_null_space", "ols"}:
            raise ValueError(
                f"solver must be svd_null_space or ols, got {self.solver!r}"
            )
        if not self.library:
            raise ValueError("library must not be empty")
        if self.pop_size < 2:
            raise ValueError(f"pop_size must be >= 2, got {self.pop_size}")
        if self.target_lhs_order not in (1, 2):
            raise ValueError(
                "target_lhs_order must be 1 (u_t) or 2 (u_tt) — DLGA only "
                f"materializes these evaluators, got {self.target_lhs_order}"
            )
        if self.target_lhs_order == 2 and not self.lhs_auto_select:
            raise ValueError(
                "target_lhs_order=2 (u_tt) requires lhs_auto_select=True: the "
                "u_tt evaluator is only built under auto-select (see "
                "DLGAPlugin._build_evaluators), so declaring order 2 with "
                "auto-select disabled would pass the LHS-order gate but fit "
                "u_t — a silent wrong-order ('trusted but wrong') result"
            )
        if self.max_modules < 1:
            raise ValueError(f"max_modules must be >= 1, got {self.max_modules}")
        if self.max_module_length < 1:
            raise ValueError(
                f"max_module_length must be >= 1, got {self.max_module_length}"
            )
        for name, value in (
            ("epsilon", self.epsilon),
            ("auto_upgrade_threshold", self.auto_upgrade_threshold),
            ("surrogate_lr", self.surrogate_lr),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            ("partial_prob", self.partial_prob),
            ("genes_prob", self.genes_prob),
            ("crossover_rate", self.crossover_rate),
            ("mutation_rate", self.mutation_rate),
            ("add_rate", self.add_rate),
            ("delete_rate", self.delete_rate),
            ("surrogate_val_ratio", self.surrogate_val_ratio),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")









    @classmethod
    def burgers_preset(cls, **overrides: Any) -> DLGAConfig:
        """DLGAConfig for the Burgers equation (``epsilon=1e-3``).

        Uses the default ``epsilon=1e-3``, validated on 5/5 seeds for both
        clean (0%) and 15%-noise Burgers. Surrogate budget / library are the
        DLGA defaults.
        Pass ``**overrides`` to customize, e.g.
        ``burgers_preset(surrogate_max_epochs=2000)`` for a faster demo.
        """
        defaults: dict[str, Any] = {"epsilon": 1e-3}
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def kdv_preset(cls, **overrides: Any) -> DLGAConfig:
        """DLGAConfig for the KdV equation (``epsilon=1e-6``).

        KdV's small coefficients need a tight ``epsilon=1e-6`` — the default
        ``1e-3`` lets a single-token genome win and recovery fails (validated
        5/5 seeds under the NMSE-era selector). This is the preset's core value.

        Note: ``1e-6`` was tuned in the raw-MSE era and re-validated (not
        re-optimized) under NMSE; the plugin's NMSE recommendation table lists
        a looser range — override if you want to explore it. Pass
        ``**overrides`` to customize (e.g. ``surrogate_max_epochs=2000``).
        """
        defaults: dict[str, Any] = {"epsilon": 1e-6}
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def wave_preset(cls, **overrides: Any) -> DLGAConfig:
        """DLGAConfig for the wave equation (u_tt LHS, ``epsilon=1e-3``).

        Uses the default ``epsilon=1e-3`` (validated 5/5 seeds). Wave exercises
        the dual-LHS auto-select (u_t vs u_tt) and relies on the paper-grade
        surrogate budget to pick the u_tt branch — keep ``surrogate_max_epochs``
        high. Pass ``**overrides`` to customize.
        """
        defaults: dict[str, Any] = {"epsilon": 1e-3, "target_lhs_order": 2}
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def kg_preset(cls, **overrides: Any) -> DLGAConfig:
        """DLGAConfig for the Klein-Gordon equation (u_tt LHS, ``epsilon=1e-3``).

        Klein-Gordon is ``u_tt = 0.5*u_xx - 5*u`` — a two-term second-order
        target. Like wave it relies on the dual-LHS auto-select picking the
        ``u_tt`` branch, so keep ``surrogate_max_epochs`` high. ``epsilon=1e-3``
        was validated on the real EqGPT ``KG_Exp.mat`` (recovers ``u_xx`` + ``u``
        with ``lhs=u_tt``, coefficients within tolerance and the -5*u sign
        correct); it happens to match wave's epsilon but is kept a distinct
        preset so KG-specific tuning stays decoupled. Pass ``**overrides`` to
        customize.
        """
        defaults: dict[str, Any] = {"epsilon": 1e-3, "target_lhs_order": 2}
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def chafee_preset(cls, **overrides: Any) -> DLGAConfig:
        """DLGAConfig for Chafee-Infante (``epsilon=1e-5``).

        WARNING: Chafee-Infante is the hardest recovery PDE — the data is
        underdetermined and recovery succeeds on only ~1/5 seeds. This preset
        locks the ``1e-5`` epsilon that gives the best shot but does
        NOT promise paper-grade recovery; expect to run multiple seeds. Pass
        ``**overrides`` to customize.
        """
        defaults: dict[str, Any] = {"epsilon": 1e-5}
        defaults.update(overrides)
        return cls(**defaults)
