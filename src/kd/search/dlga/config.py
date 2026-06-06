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
    library: list[str] = field(default_factory=lambda: ["u", "u_x", "u_xx", "u_xxx"])
    solver: Literal["svd_null_space", "ols"] = "svd_null_space"
    lhs_auto_select: bool = True






























    epsilon: float = 1e-3
    pop_size: int = 400
    seed: int = 0







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


    max_modules: int = 5
    max_module_length: int = 5
    partial_prob: float = 0.6
    genes_prob: float = 0.6
    crossover_rate: float = 0.8
    mutation_rate: float = 0.4
    add_rate: float = 0.4
    delete_rate: float = 0.5












    surrogate_hidden_sizes: list[int] = field(
        default_factory=lambda: [50, 50, 50, 50, 50]
    )
    surrogate_activation: Literal["tanh", "sin", "relu"] = "sin"
    surrogate_lr: float = 1e-3
    surrogate_max_epochs: int = 50000







    surrogate_patience: int | None = None
    surrogate_val_ratio: float = 0.2
    surrogate_restore_best: bool = True

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
        defaults: dict[str, Any] = {"epsilon": 1e-3}
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
