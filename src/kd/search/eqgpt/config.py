"""Configuration for the EqGPT search plugin.

Deliberately carries NO loop-count field (no ``optimize_epochs`` / ``generations``):
the Runner owns the loop (facade ``generations`` -> ``max_iterations``),
mirroring ``SGAConfig`` having no ``generations``. A loop field here would be a
silent no-op under the Runner, so it is excluded. ``sparsity_alpha`` is required:
it is a per-problem hyperparameter, not a constant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from kd.search.eqgpt.reward import WAVE_SPARSITY_ALPHA


@dataclass(frozen=True)
class EqGPTConfig:
    """Static configuration for the EqGPT plugin.

    ``variables`` defaults to ``None`` -> derived from the dataset axes at
    ``prepare()`` (the dataset is the single axis source); an explicit
    value is validated against the dataset axes there (fail-loud on mismatch).
    ``masked_tokens`` optionally overrides / augments the sampling mask.
    ``weights_path`` / ``asset_dir`` resolve the pretrained GPT weights. NO loop
    field -- the Runner drives epochs.

    ``top_k`` is the reference implementation's single pool knob: it is BOTH the
    running pool cap (the top ``top_k`` distinct-reward candidates kept across
    all epochs) AND the fine-tune slice size (the reference fine-tunes on the
    whole pool). The reference hardcodes this as 10; kd parameterizes it
    (default 10, reference-faithful). ``samples_per_epoch`` is only the
    per-epoch sampling / topk scan window -- NOT the pool length (the
    reference's ``find_min_no_repeat`` scans ``samples_per_epoch`` but breaks
    at 10 distinct).
    """

    sparsity_alpha: float
    """Weight of the term-count penalty in the reward: a candidate's ``R^2`` is
    multiplied by ``1 - sparsity_alpha * log10(number of distinct terms)``. It has no
    default because the value is problem-specific; the Burgers and wave presets pin
    0.02 and the steady presets 1.0.
    """
    seed: int = 0
    """Random seed of candidate sampling: it seeds the generator that hands every
    sampling call its own sub-seed, exploration draws included. A run resumed from a
    checkpoint restores the saved generator state instead, and falls back to this
    value only when the checkpoint carries none.
    """
    samples_per_epoch: int = 400
    """Number of candidate sentences drawn from the model per iteration; the platform
    uses it as the batch size for each search iteration. Sentences that cannot be
    converted into a valid expression are dropped, so fewer candidates may reach
    scoring.
    """
    top_k: int = 10
    """Maximum size of the running pool of best candidates, which is held sorted by
    reward and deduplicated by reward value. The same pool is the corpus the model is
    fine-tuned on after every iteration, so this sets both the pool cap and the fine-
    tuning batch.
    """
    finetune_lr: float = 1e-5
    """Learning rate of the Adam optimizer that fine-tunes the model on the pool of best
    candidates. A resumed run adopts the current value: the restored optimizer keeps
    its moment estimates but takes its learning rate from this field.
    """
    finetune_steps: int = 5
    """Number of gradient steps taken on the pool of best candidates after each
    iteration. Each step is one full pass over the pool, and the reported fine-tuning
    loss is the mean over the steps.
    """
    exploration_rate: float = 0.2
    """Probability of drawing the next token uniformly from the legal tokens instead of
    from the model's distribution, from 0 to 1. The draw is made independently at
    every token position.
    """
    max_length: int = 49
    """Maximum number of tokens in a sampled sentence, counting the pinned
    ``start_words`` prefix; sampling stops there if no end token is drawn first. It
    must exceed ``len(start_words)`` and stay below the model's context length, and
    the default 49 is one below the pretrained context.
    """
    variables: tuple[str, ...] | None = None
    """Coordinate axis names the equation is allowed to mention; vocabulary tokens
    naming an axis outside this set are masked out of sampling. ``None`` (the
    default) takes the axes from the dataset, and an explicit value must match the
    dataset axes exactly or the run fails to start.
    """
    start_words: tuple[str, ...] = ("S", "ut", "+")
    """Vocabulary words pinned at the head of every sampled sentence, so generation
    always continues from the same prefix. The default ``("S", "ut", "+")`` fixes the
    left-hand side to the time derivative, and the prefix is removed from the right-
    hand side the algorithm reports.
    """
    masked_tokens: frozenset[int] = field(default_factory=frozenset)
    """Extra vocabulary token ids the sampler must never emit. They are added to the
    masks the algorithm derives on its own from the declared axes, the derivative
    order the data supports, and the tokens the platform can represent, so this
    parameter only ever widens the mask.
    """
    weights_path: Path | None = None
    """Path to the pretrained GPT checkpoint file, taking precedence over ``asset_dir``
    and the ``KD_EQGPT_ASSET_DIR`` environment variable. When set, the file must
    exist: a missing path raises ``FileNotFoundError`` instead of falling back to the
    other two sources.
    """
    asset_dir: Path | None = None
    """Directory holding the pretrained GPT checkpoint at
    ``gpt_model/PDEGPT_wave_breaking.pt``, consulted when ``weights_path`` is unset.
    With both unset the directory is read from the ``KD_EQGPT_ASSET_DIR`` environment
    variable; the weights are not distributed with the package, so a run with none of
    the three given raises ``FileNotFoundError``.
    """










    case_filter: str | None = None
    """Substring matched against the wave-breaking case names: every case whose name
    contains it is scored, and a candidate's reward is the mean over the cases that
    could be scored. Setting it selects multi-case wave mode; ``None`` (default)
    keeps the single-case path.
    """
    wave_pkl_path: Path | None = None
    """Path to the pickle file of wave-breaking cases that ``case_filter`` selects from.
    When ``None`` (default), it resolves to ``data/hf-
    knowledgediscover/WaveBreaking.pkl`` under the project root, and a missing file
    raises ``FileNotFoundError``.
    """
    v1_asset_dir: Path | None = None
    """Directory holding the pretrained per-case surrogate checkpoints used to evaluate
    candidate terms in multi-case wave mode; one checkpoint is loaded per selected
    case. When ``None`` (default), the directory is taken from the
    ``KD_V1_WAVE_ASSETS`` environment variable, and if neither is set preparation
    raises ``FileNotFoundError``.
    """
    reward_points_per_window: int = 50
    """Number of ``x`` sample points per camera window on the grid used to score
    candidates. The grid covers three camera windows, so each time slice contributes
    three times this value; it must be a positive integer.
    """
    coeff_points_per_window: int = 100
    """Number of ``x`` sample points per camera window on the grid used for the final
    coefficient fits. This grid stays separate from the reward grid
    (``reward_points_per_window``): scoring runs on the reward grid, the reported
    coefficients are fitted on this one.
    """
    primary_case: str | None = None
    """Name of the case, among those ``case_filter`` selects, whose coefficient fit is
    reported as the run's final result. When ``None`` (default) the first selected
    case in sorted order is used; a name outside the selection, or a value given
    without ``case_filter``, is rejected.
    """


    steady: bool = False
    """Run the time-independent EqGPT search, where each candidate is a homogeneous
    relation whose terms sum to zero rather than an equation with a fixed left-hand
    side. The first term's coefficient is pinned to 1 and the remaining coefficients
    are fitted by least squares on a surrogate network the algorithm trains itself;
    enabling it requires ``start_words=("S",)`` and ``steady_activation``, and
    excludes ``case_filter``.
    """
    steady_activation: Literal["sin", "rational"] | None = None
    """Activation function of the surrogate network trained in steady mode, either
    ``"sin"`` or ``"rational"``. The choice also selects that network's weight
    initialization; it is required when ``steady`` is True and must stay ``None``
    otherwise.
    """
    steady_boundary_delete_num: int | None = None
    """Width, in grid cells, of the border removed from the steady evaluation domain;
    ``None`` (the default) keeps every dataset point. A point survives only if it
    sits at least this many cells from every edge with a fully populated surrounding
    block, so the data must lie on a grid; it requires ``steady=True`` and excludes
    ``steady_polar_eval``.
    """
    steady_polar_eval: bool = False
    """Score steady candidates on a generated polar domain instead of the dataset's own
    points. The domain is 100 radii from 0.5 to 1.45 crossed with 100 angles over the
    full circle, converted to ``(x, y)``; it requires ``steady=True`` and excludes
    ``steady_boundary_delete_num``.
    """
    steady_constant_column: bool = False
    """Add a constant column to the term matrix used to score and refit steady
    candidates, letting the recovered relation carry a constant offset. Requires
    ``steady=True``.
    """
    steady_train_points: int = 10_000
    """Number of dataset points sampled to train the steady surrogate network, 10000 by
    default. The sample is capped at one below the number of available points so a
    validation point remains; a value other than the default requires
    ``steady=True``.
    """
    steady_validate_points: int = 1_000
    """Number of points held out from the training sample to validate the steady
    surrogate, 1000 by default. Training keeps the checkpoint with the lowest
    validation loss; the count is capped by whatever points ``steady_train_points``
    leaves, and a value other than the default requires ``steady=True``.
    """
    steady_train_iters: int = 50_000
    """Number of optimizer iterations used to train the steady surrogate over the full
    training sample, 50000 by default. The whole budget runs unless the loss becomes
    non-finite, in which case training rolls back to the last finite weights and
    stops; a value other than the default requires ``steady=True``.
    """
    steady_surrogate_seed: int = 525
    """Random seed for the steady surrogate; it fixes both the network's initial weights
    and the split of the data into training and validation points. It is separate
    from ``seed``, which drives candidate sampling, and a value other than the
    default 525 requires ``steady=True``.
    """

    @property
    def is_wave_multicase(self) -> bool:
        """True when the plugin runs in multi-case wave mode.

        The mode signal is ``case_filter is not None`` -- a wave config selects a
        family of cases; a single-case (Burgers/KdV) config leaves it ``None`` and
        keeps the current grid + platform-evaluator path.
        """
        return self.case_filter is not None

    @property
    def is_steady(self) -> bool:
        """True when the plugin runs its homogeneous free-pivot path."""
        return self.steady

    def __post_init__(self) -> None:
        for name in ("samples_per_epoch", "top_k", "finetune_steps", "max_length"):
            value = getattr(self, name)

            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {self.seed!r}")
        if (
            isinstance(self.finetune_lr, bool)
            or not isinstance(self.finetune_lr, (int, float))
            or self.finetune_lr <= 0.0
        ):
            raise ValueError(
                f"finetune_lr must be a number > 0, got {self.finetune_lr!r}"
            )
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError(
                f"exploration_rate must be in [0, 1], got {self.exploration_rate}"
            )
        if self.variables is not None and not self.variables:
            raise ValueError("variables, if given, must be non-empty")
        if self.max_length <= len(self.start_words):
            raise ValueError(
                f"max_length ({self.max_length}) must exceed len(start_words) "
                f"({len(self.start_words)})"
            )

        for name in ("reward_points_per_window", "coeff_points_per_window"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.case_filter is not None and not self.case_filter:
            raise ValueError("case_filter, if given, must be a non-empty string")
        if self.primary_case is not None and not self.primary_case:
            raise ValueError("primary_case, if given, must be a non-empty string")


        if self.primary_case is not None and self.case_filter is None:
            raise ValueError(
                "primary_case requires case_filter (multi-case wave mode); a "
                "single-case config has no case family to pick a primary from."
            )
        self._validate_steady()

    def _validate_steady(self) -> None:
        """Validate steady-mode exclusivity and its otherwise-inert knobs."""
        if type(self.steady) is not bool:
            raise ValueError(f"steady must be a bool, got {self.steady!r}")
        for name in ("steady_polar_eval", "steady_constant_column"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise ValueError(f"{name} must be a bool, got {value!r}")
        if self.steady and self.case_filter is not None:
            raise ValueError("steady and wave case_filter modes are mutually exclusive")
        if self.steady and self.start_words != ("S",):
            raise ValueError("steady requires start_words=('S',) for a free pivot")
        if self.steady and self.steady_activation is None:
            raise ValueError("steady_activation is required when steady=True")
        if self.steady_activation not in (None, "sin", "rational"):
            raise ValueError(
                "steady_activation must be 'sin' or 'rational', got "
                f"{self.steady_activation!r}"
            )
        if self.steady_polar_eval and self.steady_boundary_delete_num is not None:
            raise ValueError(
                "steady_polar_eval and steady_boundary_delete_num are mutually "
                "exclusive"
            )
        for name in (
            "steady_train_points",
            "steady_validate_points",
            "steady_train_iters",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if (
            type(self.steady_surrogate_seed) is not int
            or self.steady_surrogate_seed < 0
        ):
            raise ValueError(
                "steady_surrogate_seed must be a non-negative int, got "
                f"{self.steady_surrogate_seed!r}"
            )
        delete_num = self.steady_boundary_delete_num
        if delete_num is not None and (type(delete_num) is not int or delete_num < 1):
            raise ValueError(
                "steady_boundary_delete_num must be a positive int or None, got "
                f"{delete_num!r}"
            )
        if not self.steady:
            defaults = {
                "steady_activation": None,
                "steady_boundary_delete_num": None,
                "steady_polar_eval": False,
                "steady_constant_column": False,
                "steady_train_points": 10_000,
                "steady_validate_points": 1_000,
                "steady_train_iters": 50_000,
                "steady_surrogate_seed": 525,
            }
            for name, default in defaults.items():
                value = getattr(self, name)
                if value != default:
                    raise ValueError(f"{name} requires steady=True, got {value!r}")







    @classmethod
    def burgers_preset(cls, **overrides: Any) -> EqGPTConfig:
        """EqGPTConfig for the Burgers equation (``sparsity_alpha=0.02``).

        Probe-verified 2026-07-04: at the default config this recovers Burgers
        ``u_t = -u*u_x + 0.1*u_xx`` exactly across seeds 0/1/7. ``sparsity_alpha``
        is the only field the preset pins; every other field keeps its
        dataclass default, so the preset equals ``EqGPTConfig(sparsity_alpha=
        0.02)``.

        ``**overrides`` layer on top and win, INCLUDING ``sparsity_alpha``: KdV
        needs ``alpha ~ 0.001`` (per-problem), which is exactly WHY there is
        no global default and the pinned value stays overridable. Pass e.g.
        ``burgers_preset(seed=7, samples_per_epoch=100)`` for a faster demo.
        """
        defaults: dict[str, Any] = {"sparsity_alpha": 0.02}
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def wave_preset(cls, **overrides: Any) -> EqGPTConfig:
        """EqGPTConfig for the wave-breaking multi-case showcase.

        Pins ``sparsity_alpha=0.02`` (``WAVE_SPARSITY_ALPHA``) and
        ``case_filter="N"`` (the 12 paper experiments), activating multi-case
        wave mode (:attr:`is_wave_multicase`). Every other field keeps its
        default: ``start_words=("S", "ut", "+")`` already pins the evolution LHS
        ``u_t``, and the dual grids default to 50 (reward) / 100 (coeff).
        ``**overrides`` layer on top and win (e.g. ``wave_preset(primary_case=
        "N_G2Tp12A090_broad", seed=7)``). Asset paths (``wave_pkl_path`` /
        ``v1_asset_dir``) default to ``None`` -> resolved at ``prepare()`` from
        the repo-relative default + env vars.
        """
        defaults: dict[str, Any] = {
            "sparsity_alpha": WAVE_SPARSITY_ALPHA,
            "case_filter": "N",
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def steady_preset(
        cls,
        dataset: Literal["eitech", "smile", "disk"],
        **overrides: Any,
    ) -> EqGPTConfig:
        """Build one of the three published steady EqGPT configurations."""
        per_dataset: dict[str, dict[str, Any]] = {
            "eitech": {
                "steady_activation": "sin",
                "steady_boundary_delete_num": 8,
                "steady_polar_eval": False,
                "steady_constant_column": True,
            },
            "smile": {
                "steady_activation": "rational",
                "steady_boundary_delete_num": 8,
                "steady_polar_eval": False,
                "steady_constant_column": False,
            },
            "disk": {
                "steady_activation": "sin",
                "steady_boundary_delete_num": None,
                "steady_polar_eval": True,
                "steady_constant_column": False,
            },
        }
        if dataset not in per_dataset:
            raise ValueError(
                f"dataset must be one of 'eitech', 'smile', or 'disk', got {dataset!r}"
            )
        defaults: dict[str, Any] = {
            "sparsity_alpha": 1.0,
            "steady": True,
            "start_words": ("S",),
            **per_dataset[dataset],
        }
        defaults.update(overrides)
        return cls(**defaults)


def config_to_json_safe_dict(config: EqGPTConfig) -> dict[str, Any]:
    """``asdict(config)``, normalized to be actually JSON-safe.

    Plain ``dataclasses.asdict`` passes ``masked_tokens`` through as a
    ``frozenset`` and ``weights_path``/``asset_dir`` through as ``Path``
    objects -- neither survives a JSON round-trip cleanly (a bare ``asdict``
    would make ``ExperimentResult.save``'s ``make_json_safe`` warn on every
    save and round-trip ``masked_tokens`` as a junk ``str(frozenset(...))``).
    """
    raw = asdict(config)
    raw["masked_tokens"] = sorted(raw["masked_tokens"])
    for path_key in ("weights_path", "asset_dir", "wave_pkl_path", "v1_asset_dir"):
        raw[path_key] = str(raw[path_key]) if raw[path_key] is not None else None
    return raw
