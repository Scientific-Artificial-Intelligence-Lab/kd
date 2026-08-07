"""High-level PySR-style facade for kd.

Single-class entry point that wraps the standard wiring (executor +
evaluator + derivative provider + plugin + runner) into an sklearn-style
``Model`` with ``.fit(dataset)`` and post-fit ``best_expr_`` /
``best_score_`` / ``result_`` attributes.

Example:
    >>> import kd
    >>> m = kd.Model(algorithm="sga", generations=50, population=20)
    >>> m.fit(dataset)
    >>> print(m.best_expr_, m.best_score_)
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch

from kd.search import tool_schema
from kd.search.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    RunnerCallback,
)
from kd.search.checkpoint_manifest import (
    MANIFEST_FILENAME,
    CheckpointManifestError,
    load_checkpoint_manifest,
)
from kd.search.discover import DiscoverConfig, DISCOVERPlugin
from kd.search.discover.config import (
    DEFAULT_N_ITERATIONS,
    DEFAULT_STABILITY_QUEUE_CAPACITY,
    DEFAULT_STABILITY_SELECTION,
)
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.llm4ed import Llm4edConfig, Llm4edPlugin
from kd.search.protocol import FacadeWiringContract, PlatformComponents
from kd.search.pysindy.config import PySINDyConfig
from kd.search.pysindy.plugin import PySINDyPlugin
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.plugin import PySRPlugin
from kd.search.result import DEFAULT_SCORE_KIND
from kd.search.resume_policy import CONFIG_ARTIFACT_KEYS, check_resume_config
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin

if TYPE_CHECKING:
    from collections.abc import Callable

    from kd.core.evaluator import EvaluationResult
    from kd.core.platform.requirements import DerivativeReqs
    from kd.data.schema import PDEDataset
    from kd.llm import LLMProvider
    from kd.search.protocol import SearchAlgorithm
    from kd.search.result import ExperimentResult

__all__ = ["Model", "instrument_schemas"]


_ConfigT = TypeVar("_ConfigT")





















_PLUGIN_CLASS_BY_ALGORITHM: dict[str, type[FacadeWiringContract]] = {
    "sga": SGAPlugin,
    "dlga": DLGAPlugin,
    "discover": DISCOVERPlugin,
    "pysr": PySRPlugin,
    "eqgpt": EqGPTPlugin,
    "llm4ed": Llm4edPlugin,
    "pysindy": PySINDyPlugin,
}
_SUPPORTED_ALGORITHMS = tuple(_PLUGIN_CLASS_BY_ALGORITHM)


def instrument_schemas() -> list[dict[str, Any]]:
    """Return agent-facing schemas for facade plugins in registration order."""
    return [
        tool_schema(plugin_cls)
        for plugin_cls in _PLUGIN_CLASS_BY_ALGORITHM.values()
    ]






_PROGRESS_PREFIX = "[kd]"
_FIT_REQUIRED_MSG = "Model has not been fit. Call .fit(dataset) first."






_UNSET: Any = object()






_DEFAULT_GENERATIONS = 50
_DEFAULT_POPULATION = 20
_DEFAULT_DEPTH = 4
_DEFAULT_WIDTH = 5
_DEFAULT_AIC_RATIO = 1.0
_DEFAULT_DERIVATIVES = "finite_diff"
_DEFAULT_SEED = 0
_DEFAULT_CHECKPOINT_EVERY = 10

_VALID_DERIVATIVES = frozenset({"finite_diff", "autograd"})







_GENERATIONS_INTO_CONFIG = frozenset({"pysr"})




_SGA_FIELDS = frozenset(f.name for f in dataclasses.fields(SGAConfig))
_EXPLICITLY_MAPPED = frozenset(
    {"num", "depth", "width", "aic_ratio", "seed", "use_autograd"}
)
_ALLOWED_KWARGS = _SGA_FIELDS - _EXPLICITLY_MAPPED




_PRETTY_MAPPED = {"num": "population", "use_autograd": "derivatives"}







class _ProgressPrinter:
    """RunnerCallback that prints per-iteration progress to stdout.

    Prints ``[kd] Generation N/M | best <metric>=... | expr=...`` with an
    algorithm-aware metric label. Emits a final ``[kd] Done.`` line on
    experiment end. ``M`` is the run's EFFECTIVE iteration count (1 for a
    one-shot plugin), not the requested ``generations``, which for a one-shot
    run would advertise iterations the runner never drives.
    """

    def __init__(self, total_generations: int) -> None:
        self._total = total_generations

    @property
    def should_stop(self) -> bool:
        """Never requests stopping."""
        return False

    def on_experiment_start(self, algorithm: SearchAlgorithm) -> None:
        """No-op."""

    def on_iteration_start(self, iteration: int, algorithm: SearchAlgorithm) -> None:
        """No-op."""

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: SearchAlgorithm,
        candidates: list[str],
        results: list[EvaluationResult],
    ) -> None:
        """Print one progress line per iteration."""
        gen = iteration + 1





        label = _score_label(algorithm)
        print(
            f"{_PROGRESS_PREFIX} Generation {gen:>3}/{self._total} | "
            f"best {label}={algorithm.best_score:.4g} | "
            f"expr={algorithm.best_expression}"
        )

    def on_experiment_end(self, algorithm: SearchAlgorithm) -> None:
        """Print final summary line."""
        label = _score_label(algorithm)
        print(
            f"{_PROGRESS_PREFIX} Done. Best: {algorithm.best_expression} "
            f"({label}={algorithm.best_score:.4g})"
        )


def _score_label(algorithm: SearchAlgorithm) -> str:
    """Pick the user-facing score label for an algorithm.

    Reads the instance's ``ScoreContract`` ``score_kind`` declaration — the
    single source of truth, declared once on each plugin class: SGA -> "AIC";
    DLGA -> "DLGA fitness" (``nmse + epsilon*length`` — it is NOT an AIC,
    labeling it so would mislead); discover -> "reward"; pysr -> "NMSE";
    eqgpt -> "EqGPT reward" (maximized); llm4ed -> "LLM4ED sparse reward"
    (maximized). Algorithms without a declaration (external / fake plugins)
    fall back to ``"Score"`` for consistency with HTML reports.
    """
    return getattr(algorithm, "score_kind", DEFAULT_SCORE_KIND)







class Model:
    """High-level facade for PDE discovery (PySR-style API).

    The facade wraps the ``ExperimentRunner`` + plugin + components stack
    into a single class with sklearn-style ``.fit()`` and trailing-underscore
    post-fit attributes (``best_expr_``, ``best_score_``, ``result_``).

    Limitations:
        Most packaged engines fit a **first-order LHS** only (e.g.
        ``u_t = f(u, u_x, ...)``). A dataset carries its LHS order via
        ``dataset.lhs_order`` (the single source of truth — DATA-0; e.g.
        ``u_tt`` for the wave equation ``u_tt = c**2 * u_xx``), and an
        unsupported ``(algorithm, lhs_order)`` combination fails loud at
        ``fit`` time rather than silently fitting the wrong target.
        Second-order-in-time targets are covered by DLGA only: set
        ``DLGAConfig(target_lhs_order=2)``, or use the ``wave_preset()`` /
        ``kg_preset()`` constructors (recovery validated on the bundled
        ``wave`` and ``klein-gordon`` datasets). For every other engine,
        reduce to a first-order system manually (introduce ``v = u_t``, then
        discover ``u_t = v`` and ``v_t = ...`` separately).

    Args:
        algorithm: Search algorithm name. Supported: ``"sga"`` (default,
            full facade-parameter coverage), ``"dlga"`` (driven through
            ``config=DLGAConfig(...)`` + ``surrogate_model=`` only),
            ``"discover"`` (driven through ``config=DiscoverConfig(...)``
            only), ``"pysr"`` (driven through ``config=PySRConfig(...)``;
            ``generations`` maps to PySR's internal GP ``niterations``), or
            ``"eqgpt"`` (driven through ``config=EqGPTConfig(...)`` ONLY;
            ``config`` is MANDATORY because ``sparsity_alpha`` is a per-problem
            hyperparameter with no facade default, decision D5), or
            ``"llm4ed"`` (an LLM equation proposer, driven through
            ``config=Llm4edConfig(...)``; the real backend needs
            ``base_url=`` + an ``OPENAI_API_KEY`` env var, or inject an offline
            ``provider=`` for deterministic runs), or ``"pysindy"`` (native
            PySINDy STLSQ over the kd term library, driven through
            ``config=PySINDyConfig(...)``; ``generations`` does not map to the
            optimizer's ``max_iter``).
            For non-SGA algorithms the individual facade parameters below
            (population/depth/width/aic_ratio/derivatives) are SGA-only.
        generations: Search-loop length (default 50). The five iterative
            algorithms (sga / dlga / discover / eqgpt / llm4ed) spend it as the
            kd runner's ``max_iterations``. ``pysr`` is one-shot: the runner
            loop is pinned to 1 and ``generations`` becomes PySR's internal GP
            ``niterations`` instead (only when no ``config=PySRConfig(...)``
            supplies its own). ``pysindy`` consumes it NOWHERE (one STLSQ
            solve, whose ``max_iter`` is a convergence cap set through
            ``config=PySINDyConfig(...)``), so passing it explicitly there
            emits a ``UserWarning`` naming the algorithm rather than silently
            dropping the value.
        population: SGA population size (number of PDE candidates). SGA-only.
        depth: Maximum tree depth per term. SGA-only.
        width: Maximum number of terms per PDE. SGA-only.
        aic_ratio: AIC penalty ratio. SGA-only.
        derivatives: Derivative provider mode: ``"finite_diff"`` or
            ``"autograd"`` (forwards ``use_autograd=True`` to ``SGAConfig``).
            When ``"autograd"``, the SGA terminal derivatives (u_x, u_t) are
            routed through an autograd-trained surrogate, while the dataset's
            finite-difference provider remains in place for tree-internal
            derivative operators. SGA-only — DLGA always uses autograd,
            discover uses finite-diff.
        seed: Random seed for reproducibility. It threads into the default
            config for dlga / discover / pysr / pysindy (and SGA) when ``config=`` is
            unset, otherwise the config's own seed wins. For eqgpt the seed
            comes ONLY via ``EqGPTConfig(seed=...)``: passing the facade
            ``seed=`` together with ``config=`` is rejected by
            ``_validate_config_exclusivity``.
        verbose: When True, print per-iteration progress to stdout. The label
            is algorithm-aware (``AIC`` for SGA, ``DLGA fitness`` for DLGA,
            ``reward`` for discover, ``NMSE`` for pysr, ``EqGPT reward`` for
            eqgpt, ``LLM4ED sparse reward`` for llm4ed, and native STLSQ
            ``NMSE`` for pysindy, lower is better).
        config: Optional pre-built ``SGAConfig``, ``DLGAConfig``,
            ``DiscoverConfig``, ``PySRConfig``, ``EqGPTConfig``,
            ``Llm4edConfig``, or ``PySINDyConfig``. When provided, it is the
            single source of plugin settings; passing any non-default
            SGA-related facade
            parameters (``population``, ``depth``, ``width``, ``aic_ratio``,
            ``derivatives``, ``seed``) or extra ``kwargs`` raises
            ``ValueError`` (SGA path) / ``TypeError`` (DLGA + discover paths).
            Only ``algorithm``, ``generations``, ``verbose``, ``callbacks``,
            ``checkpoint_dir``/``checkpoint_every``, (for DLGA)
            ``surrogate_model``, and (for llm4ed) ``provider`` remain effective
            on the facade.
            Type must match ``algorithm`` (e.g. ``algorithm='dlga' +
            config=SGAConfig(...)`` raises). The config is deep-copied to
            prevent aliasing.
        callbacks: Optional list of additional ``RunnerCallback`` instances
            to attach to the runner. The verbose progress printer is appended
            automatically when ``verbose=True``. The list is shallow-copied
            so later ``.append()`` on the user's list cannot leak in, but
            the callback instances themselves are NOT deep-copied: reusing
            the same callback across fits will retain state from previous
            fits. This is intentional (some callbacks integrate over runs)
            but means stateful callbacks should be reset between fits if a
            clean slate is desired.
        surrogate_model: Optional pre-trained ``torch.nn.Module`` forwarded
            to ``DLGAPlugin(surrogate_model=...)``. DLGA-only — passing this
            with ``algorithm='sga'`` raises ``TypeError`` (silent drop is a
            bug-attractor; SGA's surrogate path is opt-in via
            ``derivatives='autograd'`` + internal ``FieldModel`` training).
            When ``None`` and ``algorithm='dlga'``, the platform builder
            trains a fresh ``FieldModel`` per the DLGAConfig surrogate
            settings (``surrogate_hidden_sizes``, ``surrogate_activation``,
            etc.).
        provider: Optional pre-built ``kd.llm.LLMProvider`` forwarded to
            ``Llm4edPlugin(provider=...)``. llm4ed-only — passing it with any
            other algorithm raises ``TypeError`` (silent drop is a
            bug-attractor). Carries the offline ``FakeProvider`` /
            ``TapeReplayProvider`` for deterministic runs, or a caller-built
            canonical chain. When ``None`` and ``algorithm='llm4ed'``, the
            plugin builds its default chain in ``prepare()`` from the config's
            transport knobs (``base_url`` + an ``OPENAI_API_KEY`` env var).
            The injected instance is REUSED across ``fit`` calls and NOT reset
            (like ``callbacks``, unlike the self-built default chain, which is
            rebuilt with a fresh budget on each fit): a second ``fit`` with a
            stateful provider carries its state over — a ``TapeReplayProvider``
            resumes from its exhausted cursor, an injected ``BudgetedProvider``
            keeps its spent budget. Build a fresh provider per fit if a clean
            slate is desired.
        checkpoint_dir: Optional directory for periodic checkpoints. Writing
            covers all seven algorithms (the payload is the plugin's own
            ``state``, part of the search protocol), but RESUME semantics split
            by plugin: the five iterative engines (sga / dlga / discover /
            eqgpt / llm4ed) restore the search state and keep searching, so a
            larger ``generations`` extends the run, while the two one-shot
            engines (pysr / pysindy) restore the finished fit without
            re-running the solver (recover-without-rerun; their runner loop is
            pinned to one iteration either way). When set, every ``fit``
            attaches a fresh ``CheckpointCallback`` writing
            ``checkpoint_{iteration:06d}.pt``
            every ``checkpoint_every`` iterations plus a
            ``checkpoint_final.pt`` at experiment end, alongside a
            ``manifest.json`` evidence ledger (``kd-ckptman-v1``): the ledger
            is the only entry point a controller uses to select a resumable
            checkpoint (R2.1-O4). The directory is manifest-managed and refuses
            reuse: pointing ``checkpoint_dir`` at a non-empty directory fails
            loud with a ``CheckpointManifestError`` (a ``ValueError``), so use a
            FRESH directory per run. ``None`` (default) disables checkpointing
            entirely. A user-supplied ``CheckpointCallback`` in ``callbacks=``
            must write to a DIFFERENT directory (a shared directory is now
            rejected, not interleaved).
        checkpoint_every: Save a per-iteration checkpoint every N
            iterations (0-indexed: saves at 0, N, 2N, ...; default 10).
            Requires ``checkpoint_dir`` — passing ``checkpoint_every``
            without it raises ``ValueError`` at construction (a silent
            no-op would be a bug-attractor), as does any value < 1.
        checkpoint_keep_last: Retention bound on PERIODIC checkpoints. ``None``
            (default) keeps all of them (存证优先); an ``int >= 1`` keeps only the
            N most recent periodic checkpoints, pruning older ones from both the
            manifest and disk. The final checkpoint is exempt (never counted,
            never pruned). Requires ``checkpoint_dir``: a value < 1, or passing
            it without ``checkpoint_dir``, raises ``ValueError`` at construction.
        **kwargs: Forwarded to ``SGAConfig`` if the field name matches an
            allowed (non-explicitly-mapped) field. Unknown or colliding
            kwargs raise ``TypeError``. SGA-only — DLGA and discover paths
            raise if any kwargs are supplied (drive those algorithms via
            ``config=DLGAConfig(...)`` / ``config=DiscoverConfig(...)``).
            Three such fields are agent-tunable training-budget knobs for the
            ``derivatives='autograd'`` surrogate (they only take effect in
            autograd mode, but their validation runs for every SGA config —
            an orphan ``autograd_train_patience`` raises even under
            finite-diff):

            - ``autograd_train_epochs`` (default ``1000``): the HONEST
              fixed-step training budget — the surrogate trains exactly this
              many epochs (no silent early stop), so the value directly trades
              runtime for derivative quality.
            - ``autograd_train_patience`` (default ``None``): early-stopping
              patience (epochs without validation improvement). ``None`` keeps
              the full fixed-step budget (the v1 / paper reference semantics).
              Requires ``autograd_train_val_ratio > 0``; otherwise
              ``ValueError`` is raised when ``fit()`` assembles the
              ``SGAConfig`` (a patience knob with no validation signal would
              silently do nothing).
            - ``autograd_train_val_ratio`` (default ``0.0``): fraction of data
              held out for validation. ``0.0`` trains on ALL data (reference
              semantics); set ``> 0`` only when using
              ``autograd_train_patience``. Must be in ``[0, 1)``.

            Honest-failure note: with weak surrogate features SGA now FAILS
            LOUD — degenerate candidates whose STRidge support is empty are
            invalid, so a hopeless search raises ``RuntimeError`` (init
            resample exhaustion) instead of returning a garbage equation.
            Raise ``autograd_train_epochs`` when that happens.
    """

    def __init__(
        self,
        algorithm: str = "sga",
        generations: int = _UNSET,
        population: int = _UNSET,
        depth: int = _UNSET,
        width: int = _UNSET,
        aic_ratio: float = _UNSET,
        derivatives: str = _UNSET,
        seed: int = _UNSET,
        verbose: bool = True,
        config: SGAConfig
        | DLGAConfig
        | DiscoverConfig
        | PySRConfig
        | PySINDyConfig
        | EqGPTConfig
        | Llm4edConfig
        | None = None,
        callbacks: list[RunnerCallback] | None = None,
        surrogate_model: torch.nn.Module | None = None,
        provider: LLMProvider | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = _UNSET,
        checkpoint_keep_last: int | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:




        generations_resolved = (
            _DEFAULT_GENERATIONS if generations is _UNSET else generations
        )
        population_resolved = (
            _DEFAULT_POPULATION if population is _UNSET else population
        )
        depth_resolved = _DEFAULT_DEPTH if depth is _UNSET else depth
        width_resolved = _DEFAULT_WIDTH if width is _UNSET else width
        aic_ratio_resolved = _DEFAULT_AIC_RATIO if aic_ratio is _UNSET else aic_ratio
        derivatives_resolved = (
            _DEFAULT_DERIVATIVES if derivatives is _UNSET else derivatives
        )
        seed_resolved = _DEFAULT_SEED if seed is _UNSET else seed
        checkpoint_every_resolved = (
            _DEFAULT_CHECKPOINT_EVERY
            if checkpoint_every is _UNSET
            else checkpoint_every
        )

        self._validate_checkpoint_params(
            checkpoint_dir, checkpoint_every, checkpoint_keep_last
        )
        self._validate_device(device)
        self._validate_derivatives(derivatives_resolved)
        self._validate_kwargs(kwargs)
        self._validate_field_model(kwargs, derivatives_resolved)


        self._validate_config_exclusivity(
            config, population, depth, width, aic_ratio, derivatives, seed, kwargs
        )
        self._validate_callbacks(callbacks)




        self._validate_algorithm_supported(algorithm)
        self._validate_early_stopping_direction(callbacks, algorithm)


        self._validate_algorithm_arg_exclusivity(
            algorithm,
            kwargs=kwargs,
            surrogate_model=surrogate_model,
            provider=provider,
            population=population,
            depth=depth,
            width=width,
            aic_ratio=aic_ratio,
            derivatives=derivatives,
        )




        self._validate_discover_config_facade_compat(algorithm, config)



        self._warn_if_generations_unused(algorithm, generations)

        self.algorithm = algorithm
        self.generations = generations_resolved
        self.population = population_resolved
        self.depth = depth_resolved
        self.width = width_resolved
        self.aic_ratio = aic_ratio_resolved
        self.derivatives = derivatives_resolved
        self.seed = seed_resolved
        self.verbose = verbose




        self.device: str | None = device
        self.checkpoint_dir: Path | None = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.checkpoint_every = checkpoint_every_resolved


        self.checkpoint_keep_last = checkpoint_keep_last





        self._config_override = copy.deepcopy(config) if config is not None else None



        self._user_callbacks: list[RunnerCallback] | None = (
            list(callbacks) if callbacks is not None else None
        )
        self._extra_kwargs: dict[str, Any] = kwargs




        self._surrogate_model = surrogate_model








        self._provider = provider


        self._fitted: bool = False
        self._result: ExperimentResult | None = None
        self._algorithm: SearchAlgorithm | None = None



    @staticmethod
    def _validate_checkpoint_params(
        checkpoint_dir: str | Path | None,
        checkpoint_every: Any,
        checkpoint_keep_last: int | None = None,
    ) -> None:
        """Validate the checkpoint parameters at construction.

        Receives the SENTINEL-bearing ``checkpoint_every`` so an explicit
        ``checkpoint_every=10`` (the literal default) without
        ``checkpoint_dir`` is also rejected — a silent no-op would be a
        bug-attractor. The ``>= 1`` range check fails here, at
        construction, rather than waiting for ``CheckpointCallback`` to
        re-validate at fit time.

        An empty-string ``checkpoint_dir`` is also rejected: ``""``
        is a common "disabled" sentinel in agent-generated configs, but
        ``Path("") == Path(".")`` would silently checkpoint into the CWD —
        the same silent-attractor the ``checkpoint_every`` orphan check
        guards against.

        ``checkpoint_keep_last`` (G2c retention): ``None`` = keep_all (the
        meaningful default, so NO ``_UNSET`` sentinel is needed). Passing it
        without ``checkpoint_dir`` is rejected (same silent-no-op rule as
        ``checkpoint_every``); a value ``< 1`` is rejected at construction.
        """
        if isinstance(checkpoint_dir, str) and checkpoint_dir == "":
            raise ValueError(
                "checkpoint_dir='' is not a valid directory (it would resolve "
                "to the current working directory). Pass a real path, or None "
                "to disable checkpointing."
            )
        if checkpoint_keep_last is not None and checkpoint_dir is None:
            raise ValueError(
                "checkpoint_keep_last was passed without checkpoint_dir; "
                "checkpointing is enabled by checkpoint_dir=..., so "
                "checkpoint_keep_last alone would be a silent no-op. "
                "Pass checkpoint_dir as well (or drop checkpoint_keep_last)."
            )



        if checkpoint_keep_last is not None and (
            type(checkpoint_keep_last) is not int or checkpoint_keep_last < 1
        ):
            raise ValueError(
                f"checkpoint_keep_last must be an int >= 1 or None, got "
                f"{checkpoint_keep_last!r}"
            )
        if checkpoint_every is _UNSET:
            return
        if checkpoint_dir is None:
            raise ValueError(
                "checkpoint_every was passed without checkpoint_dir; "
                "checkpointing is enabled by checkpoint_dir=..., so "
                "checkpoint_every alone would be a silent no-op. "
                "Pass checkpoint_dir as well (or drop checkpoint_every)."
            )
        if checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")

    @staticmethod
    def _validate_device(device: str | None) -> None:
        """Syntactically validate the ``device`` knob (None passes through).

        Only the SYNTAX is checked: a non-str is rejected, and a string that
        ``torch.device`` cannot parse raises ``ValueError``. Constructing
        ``"cuda"`` on a CUDA-less host is legal (so a plan can be built /
        validated anywhere); actual unavailability fails loud at fit time when
        a tensor ``.to(device)`` runs (GPU efficacy is validated per instrument).
        """
        if device is None:
            return
        if not isinstance(device, str):
            raise ValueError(
                f"device must be None or a torch device string; got {device!r}"
            )
        try:
            torch.device(device)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise ValueError(
                f"device must be None or a torch device string; got {device!r}"
            ) from exc

    @staticmethod
    def _validate_derivatives(derivatives: str) -> None:
        """Reject ``derivatives`` values outside the supported whitelist."""
        if derivatives not in _VALID_DERIVATIVES:
            raise ValueError(
                f"Invalid derivatives='{derivatives}'. "
                f"Must be one of {sorted(_VALID_DERIVATIVES)}."
            )

    @staticmethod
    def _validate_kwargs(kwargs: dict[str, Any]) -> None:
        """Reject unknown kwargs and kwargs that collide with facade params."""
        unknown = set(kwargs) - _SGA_FIELDS
        if unknown:
            raise TypeError(
                f"Unknown keyword arguments for kd.Model: {sorted(unknown)}. "
                f"Valid extra kwargs (forwarded to SGAConfig): "
                f"{sorted(_ALLOWED_KWARGS)}"
            )
        collision = set(kwargs) & _EXPLICITLY_MAPPED
        if collision:
            hints = ", ".join(
                f"{k} (use '{_PRETTY_MAPPED.get(k, k)}=' instead)"
                for k in sorted(collision)
            )
            raise TypeError(
                f"Keyword arguments collide with explicit facade parameters: {hints}"
            )

    @staticmethod
    def _validate_field_model(kwargs: dict[str, Any], derivatives: str) -> None:
        """Require ``derivatives='autograd'`` when a ``field_model`` is given."""
        if kwargs.get("field_model") is not None and derivatives != "autograd":
            raise ValueError(
                "Pre-trained 'field_model' was provided but "
                "derivatives='finite_diff'. Set derivatives='autograd' to "
                "use the field model."
            )

    @staticmethod
    def _validate_config_exclusivity(
        config: SGAConfig
        | DLGAConfig
        | DiscoverConfig
        | PySRConfig
        | PySINDyConfig
        | EqGPTConfig
        | Llm4edConfig
        | None,
        population: Any,
        depth: Any,
        width: Any,
        aic_ratio: Any,
        derivatives: Any,
        seed: Any,
        kwargs: dict[str, Any],
    ) -> None:
        """Reject mixing ``config=`` with overlapping SGA-related parameters.

        Receives the SENTINEL-bearing values from ``__init__`` so that
        ``Model(config=cfg, population=20)`` (where 20 happens to equal the
        default) is rejected — the user-set check uses ``is _UNSET`` rather
        than equality to the default value.
        """
        if config is None:
            return
        overrides: list[str] = []
        if population is not _UNSET:
            overrides.append("population")
        if depth is not _UNSET:
            overrides.append("depth")
        if width is not _UNSET:
            overrides.append("width")
        if aic_ratio is not _UNSET:
            overrides.append("aic_ratio")
        if derivatives is not _UNSET:
            overrides.append("derivatives")
        if seed is not _UNSET:
            overrides.append("seed")
        if kwargs:
            overrides.append(f"kwargs={sorted(kwargs)}")
        if overrides:
            raise ValueError(
                f"Cannot pass both 'config=' and algorithm-specific "
                f"parameters: {overrides}. When 'config' is provided, use "
                f"it as the single source of plugin settings; only "
                f"'generations' (except pysr, where the config's own "
                f"'niterations' wins), 'verbose', 'callbacks', "
                f"'checkpoint_dir'/'checkpoint_every' ('surrogate_model' "
                f"for DLGA, 'provider' for llm4ed) remain effective on the "
                f"facade."
            )

    @staticmethod
    def _validate_algorithm_arg_exclusivity(
        algorithm: str,
        *,
        kwargs: dict[str, Any],
        surrogate_model: torch.nn.Module | None,
        provider: LLMProvider | None,
        population: Any,
        depth: Any,
        width: Any,
        aic_ratio: Any,
        derivatives: Any,
    ) -> None:
        """Reject algorithm-incompatible parameter combinations at construction.

        - ``surrogate_model`` is meaningful only for ``algorithm='dlga'`` —
          SGA + ``surrogate_model=`` would silently drop the model, so raise
          instead of accepting a no-op argument.

        - ``provider`` is meaningful only for ``algorithm='llm4ed'`` — passing
          it on any other algorithm would silently drop it, so raise.

        - ``algorithm='dlga'`` + SGA-only facade params (``population``,
          ``depth``, ``width``, ``aic_ratio``, ``derivatives``) or extra
          kwargs (``lam``, ``p_var``, ``autograd_train_*``, ...) would
          also silently drop. Raise — DLGA users must drive non-default
          settings through ``DLGAConfig`` via ``config=``. The same applies
          to ``discover`` (drive via ``DiscoverConfig``) and ``pysr`` (drive
          via ``PySRConfig``).
        """
        if algorithm != "dlga" and surrogate_model is not None:
            raise TypeError(
                f"Model(algorithm={algorithm!r}, surrogate_model=...) is not "
                "supported. ``surrogate_model`` is consumed only by "
                "``algorithm='dlga'``; for SGA opt-in autograd derivatives use "
                "``derivatives='autograd'`` instead."
            )
        if algorithm != "llm4ed" and provider is not None:
            raise TypeError(
                f"Model(algorithm={algorithm!r}, provider=...) is not "
                "supported. ``provider`` is consumed only by "
                "``algorithm='llm4ed'`` (an injected LLMProvider); other "
                "algorithms have no LLM provider seam."
            )
        Model._reject_sga_only_params_for_non_sga(
            algorithm,
            kwargs=kwargs,
            population=population,
            depth=depth,
            width=width,
            aic_ratio=aic_ratio,
            derivatives=derivatives,
        )

    @staticmethod
    def _reject_sga_only_params_for_non_sga(
        algorithm: str,
        *,
        kwargs: dict[str, Any],
        population: Any,
        depth: Any,
        width: Any,
        aic_ratio: Any,
        derivatives: Any,
    ) -> None:
        """Raise if SGA-only facade params ride a non-SGA algorithm path.

        DLGA / DISCOVER / PySR are driven exclusively via their own
        ``config=`` object; the individual SGA facade knobs
        (``population``/``depth``/``width``/``aic_ratio``/``derivatives``) and
        any extra ``kwargs`` are not exposed for them and would otherwise be
        silently dropped (a classic bug-attractor). The error names the offending
        params and the config class to use instead. No-op for ``sga`` (whose
        knobs are first-class) and any algorithm not in the registry.
        """




        plugin_class = _PLUGIN_CLASS_BY_ALGORITHM.get(algorithm)
        if plugin_class is None or algorithm == "sga":
            return
        config_class = plugin_class.config_cls.__name__
        sga_only_set: list[str] = []
        if population is not _UNSET:
            sga_only_set.append("population")
        if depth is not _UNSET:
            sga_only_set.append("depth")
        if width is not _UNSET:
            sga_only_set.append("width")
        if aic_ratio is not _UNSET:
            sga_only_set.append("aic_ratio")
        if derivatives is not _UNSET:
            sga_only_set.append("derivatives")
        if kwargs:
            sga_only_set.extend(sorted(kwargs))
        if sga_only_set:
            raise TypeError(
                f"Model(algorithm={algorithm!r}, ...): the following SGA-only "
                f"parameters cannot be passed alongside {algorithm}: "
                f"{sorted(set(sga_only_set))}. Drive {algorithm} via "
                f"``config={config_class}(...)`` instead — the facade does not "
                f"expose individual {algorithm} fields as ``Model(...)`` "
                "parameters."
            )

    @staticmethod
    def _validate_discover_config_facade_compat(
        algorithm: str,
        config: SGAConfig
        | DLGAConfig
        | DiscoverConfig
        | PySRConfig
        | PySINDyConfig
        | EqGPTConfig
        | Llm4edConfig
        | None,
    ) -> None:
        """Guard DiscoverConfig fields that have no effect under the facade.

        The facade drives the search loop via ``generations`` and single-steps
        DISCOVER's engine through ExperimentRunner — it never runs the
        standalone loop (``n_iterations``) or the MODE2 PINN cycle (``pinn``).

        ``pinn`` is a hard error: silently falling back to MODE1 when the user
        asked for the PINN surrogate yields wrong scientific conclusions.
        ``n_iterations`` / ``stability_selection`` only ever shorten the run or
        skip a post-hoc filter (never wrong science), so they warn rather than
        raise — this also keeps the ``DiscoverConfig.burgers_preset()`` /
        ``chafee_preset()`` reference configs (which carry a non-default
        ``n_iterations``) usable under the facade.

        No-op unless ``algorithm == "discover"`` with a ``DiscoverConfig``.
        These fields apply on the standalone / MODE2 entry points
        (the standalone research runner); the check belongs in the facade layer,
        not ``DiscoverConfig.__post_init__`` (the config is shared by all paths).

        Reads the caller's ``config`` *before* ``__init__`` deep-copies it —
        safe only because ``DiscoverConfig`` is frozen and the guarded fields
        are top-level immutable scalars (no post-construction mutate window).
        If they ever become mutable, move this check after the deep-copy.
        """
        if algorithm != "discover" or not isinstance(config, DiscoverConfig):
            return

        if config.pinn is not None:
            raise TypeError(
                "Model(algorithm='discover', config=DiscoverConfig(pinn=...)): "
                "MODE2 PINN is unreachable via the facade and would be silently "
                "ignored (the run falls back to MODE1 finite-diff). Use the "
                "standalone MODE2 research entry point (not part of the packaged "
                "API), or drop pinn."
            )


        ignored: list[str] = []
        if config.n_iterations != DEFAULT_N_ITERATIONS:
            ignored.append(
                f"n_iterations={config.n_iterations} (the facade sets search "
                "length via Model(generations=...))"
            )
        if (
            config.stability_selection != DEFAULT_STABILITY_SELECTION
            or config.stability_queue_capacity != DEFAULT_STABILITY_QUEUE_CAPACITY
        ):
            ignored.append(
                "stability_selection / stability_queue_capacity (stability "
                "selection runs only on the MODE2 cycle path)"
            )
        if ignored:
            warnings.warn(
                "Model(algorithm='discover', config=DiscoverConfig(...)): these "
                f"fields have no effect under the facade and are ignored: "
                f"{ignored}.",
                stacklevel=3,
            )

    @staticmethod
    def _validate_callbacks(callbacks: list[RunnerCallback] | None) -> None:
        """Reject callbacks that are not ``RunnerCallback`` instances.

        Catches the misuse early at construction time instead of crashing
        mid-fit with an opaque ``AttributeError`` from inside the runner.
        """
        if callbacks is None:
            return
        for i, cb in enumerate(callbacks):
            if not isinstance(cb, RunnerCallback):
                raise TypeError(
                    f"callbacks[{i}] is not a RunnerCallback (got {type(cb).__name__})"
                )

    @staticmethod
    def _validate_algorithm_supported(algorithm: str) -> None:
        """Reject unsupported algorithm names at construction time.

        The fit-time loop also re-checks ``self.algorithm`` (which can be
        mutated post-construction), so this guard is the fast-fail leg of
        defense-in-depth. The check is case-sensitive against
        ``_SUPPORTED_ALGORITHMS`` because plugin dispatch, recorder field
        labelling, and score-label lookup are all lowercase-keyed — and we
        prefer a clear "not supported" error over silently coercing
        ``"DISCOVER"`` to ``"discover"`` (silent normalization is the kind
        of magic that bites future readers).
        """
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise NotImplementedError(
                f"Algorithm '{algorithm}' is not implemented. "
                f"Supported algorithms: {list(_SUPPORTED_ALGORITHMS)}"
            )

    @staticmethod
    def _warn_if_generations_unused(algorithm: str, generations: Any) -> None:
        """Warn when an EXPLICIT ``generations`` reaches no knob of ``algorithm``.

        A one-shot plugin runs exactly one runner iteration, so ``generations``
        only means something if the facade routes it into the plugin's own
        config (PySR's GP ``niterations``). For a one-shot plugin outside
        ``_GENERATIONS_INTO_CONFIG`` the value is a pure no-op, which is the
        silent drop every other facade parameter is guarded against. It warns
        rather than raises because the drop is harmless to the science: the run
        is correct, only the user's iteration budget is imaginary.

        Receives the SENTINEL-bearing value: an omitted ``generations`` carries
        no user intent, so only an explicit one is worth a warning.
        """
        if generations is _UNSET or algorithm in _GENERATIONS_INTO_CONFIG:
            return
        if not _PLUGIN_CLASS_BY_ALGORITHM[algorithm].one_shot:
            return
        config_name = _PLUGIN_CLASS_BY_ALGORITHM[algorithm].config_cls.__name__
        warnings.warn(
            f"generations={generations} has no effect for "
            f"algorithm='{algorithm}': it runs a single one-shot solve. "
            f"Set its native iteration cap through config={config_name}(...).",
            UserWarning,
            stacklevel=3,
        )

    @staticmethod
    def _validate_early_stopping_direction(
        callbacks: list[RunnerCallback] | None,
        algorithm: str,
    ) -> None:
        """Reject an EarlyStoppingCallback whose mode opposes the algorithm.

        SGA/DLGA minimize (AIC / GA fitness) -> mode must be ``"min"``;
        DISCOVER maximizes a reward -> mode must be ``"max"``. A mismatched
        mode makes the patience counter treat real improvements as
        regressions, so early stopping fires prematurely (or never resets).
        The original guard only rejected discover+min, letting sga/dlga+max
        pass silently; this checks both directions symmetrically.

        The required mode is the plugin class's ``ScoreContract``
        ``score_direction`` declaration (read from
        ``_PLUGIN_CLASS_BY_ALGORITHM`` — no plugin instance exists yet at
        construction time), replacing the old hand-maintained
        ``_EARLY_STOP_MODE_BY_ALGORITHM`` table.
        """
        if callbacks is None:
            return
        plugin_class = _PLUGIN_CLASS_BY_ALGORITHM.get(algorithm)
        if plugin_class is None:
            return
        required = plugin_class.score_direction
        direction = "minimizes" if required == "min" else "maximizes"
        for cb in callbacks:
            if isinstance(cb, EarlyStoppingCallback) and cb.mode != required:
                raise TypeError(
                    f"EarlyStoppingCallback(mode={cb.mode!r}) is incompatible "
                    f"with algorithm={algorithm!r} which {direction} its score; "
                    f"use mode={required!r}. A mismatched mode treats real "
                    f"improvements as regressions and triggers early stopping "
                    f"prematurely."
                )



    def fit(
        self,
        dataset: PDEDataset,
        resume_from: str | Path | None = None,
    ) -> Model:
        """Run the search and populate post-fit attributes.

        Args:
            dataset: The PDE dataset to discover an equation for.
            resume_from: Optional path to a checkpoint file written by a
                previous run (``checkpoint_*.pt``). The checkpoint restores
                **search state** (population / controller weights / best);
                generations, algorithm settings and seed come from THIS Model.
                The checkpoint's algorithm must match this Model's (legacy
                checkpoints without the recorded name load unchecked).

                Resume is tier-gated (G2b) against a ``kd-config-v1`` config
                snapshot the checkpoint now carries. Each config field has a
                resume tier and a changed field is handled by its tier:

                - ``resume_safe`` fields (e.g. SGA ``population`` / DISCOVER
                  ``entropy_weight``) may change and the NEW value takes effect.
                - ``init_only`` fields raise ``ValueError`` naming the field(s):
                  they are only settable on a fresh fit (drop ``resume_from``).
                - ``identity_breaking`` fields (a science-identity change:
                  catalog / derivative semantics) raise ``ValueError`` pointing
                  at a new lineage — a fresh fit in a NEW checkpoint directory;
                  the checkpoint cannot be resumed under the changed config.

                An injected model (SGA ``field_model`` / DLGA ``surrogate_model``)
                is content-hashed for its resume identity AS SUPPLIED, before any
                in-place data alignment; supply the same dtype as the original run
                (DLGA aligns to the data's float64 by default) or the byte-same
                model is rejected as a new lineage.

                Because ``generations`` maps into ``max_iterations`` (not the
                config) for every iterative plugin, "resume with more
                generations" keeps working — EXCEPT for the one-shot engines:
                for PySR ``generations`` maps into ``niterations``
                (``init_only``), so changing it is rejected by the diff; for
                PySINDy it reaches no knob at all (accepted but inert, warned
                at construction). A controller must special-case one-shot
                instruments (no "add more generations" action for
                pysr/pysindy).

                A checkpoint written under a DIFFERENT kd config schema (a field
                was added or removed since it was written) is fail-closed: the
                diff sees the field on only one side and raises; the remedy is a
                fresh fit / new lineage, not a field revert (there is no value to
                revert to). If THIS Model's config cannot be canonicalized, the
                guard raises ``ValueError`` (``ConfigCanonicalizationError``)
                before the expensive component build.

                Legacy checkpoints predating the snapshot key carry no config
                and load unchecked (the tier gate is skipped). Iteration
                numbering restarts at 0. A ``checkpoint_dir`` is now a
                manifest-managed evidence directory (``manifest.json``,
                ``kd-ckptman-v1``) that refuses reuse: a fit pointing
                ``checkpoint_dir`` at a non-empty directory fails loud with a
                ``CheckpointManifestError`` (a ``ValueError``), so each run uses
                a FRESH directory. Each run's final manifest entry records a
                ``final_status`` (``completed`` vs ``crashed``), so a
                ``checkpoint_final.pt`` written by the runner's finally-block on a
                CRASH is distinguishable from a clean final and cannot
                masquerade as a good resume point; and because reuse is refused
                there is no prior good final in the same directory for a failed
                resume to clobber. ``final_status`` qualifies the CHECKPOINT
                only, not the segment: the flag wraps just the iteration loop, so
                a post-loop raise (result building) can leave ``fit()`` raising
                while the final entry reads ``completed``. Whether ``fit()``
                raised is the segment verdict; ``final_status`` is only the
                checkpoint's own qualifier. When ``resume_from`` lives in a
                manifest-managed directory it must be listed in that manifest (a
                controller selects checkpoints ONLY via the manifest, else
                ``CheckpointManifestError``); a directory with no
                ``manifest.json`` (legacy / pre-G2c) resumes exactly as before.
                A *structural* config change on a LEGACY checkpoint (e.g. a
                DISCOVER controller ``num_layers`` / hidden size differing from
                the checkpoint) still raises ``RuntimeError`` from the state_dict
                load — but for a modern (snapshot-bearing) checkpoint that field
                is ``init_only``, so the tier gate rejects it with a named
                ``ValueError`` first. Cross-DATA resume re-prices the best-score
                gate onto the new data for DISCOVER only; SGA/DLGA keep the
                restored best as their ratchet baseline.

        Returns:
            ``self`` for sklearn-style chaining.

        Raises:
            NotImplementedError: If ``algorithm`` is not supported.
            ValueError: If the dataset is missing the required LHS field
                or axis; or if ``resume_from`` is not a kd checkpoint payload
                / has a mismatched version or algorithm / is a corrupt or
                truncated file; or if a resumed config changes an
                ``init_only`` / ``identity_breaking`` field, carries a foreign
                config-canon scheme, or THIS Model's config cannot be
                canonicalized (G2b resume guard); or if ``checkpoint_dir`` /
                ``resume_from`` lives in a manifest-managed directory that is
                torn or that does not list the resumed file
                (``CheckpointManifestError``, a ``ValueError``).
            FileNotFoundError: If ``resume_from`` does not exist.
            IsADirectoryError: If ``resume_from`` points at a directory.
            RuntimeError: If a LEGACY ``resume_from`` (no config snapshot) was
                written by a structurally different config (state_dict shape
                mismatch on restore); a snapshot-bearing checkpoint hits the
                ``ValueError`` tier gate first.
        """








        self._fitted = False
        self._result = None
        self._algorithm = None

        if self.algorithm not in _SUPPORTED_ALGORITHMS:
            raise NotImplementedError(
                f"Algorithm '{self.algorithm}' is not implemented. "
                f"Supported algorithms: {list(_SUPPORTED_ALGORITHMS)}"
            )







        if (
            self.checkpoint_dir is not None
            and self.checkpoint_dir.is_dir()
            and any(self.checkpoint_dir.iterdir())
        ):
            raise CheckpointManifestError(
                "checkpoint directory is not empty (reuse is not supported; "
                f"use a fresh directory per run): {self.checkpoint_dir}"
            )




        plugin, batch_size = self._build_plugin()
        self._algorithm = plugin

        max_iterations = self._effective_max_iterations()
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=max_iterations,
            batch_size=batch_size,
            callbacks=self._build_callbacks(),
        )



















        if resume_from is not None:
            config_guard = functools.partial(
                check_resume_config,
                algorithm=self.algorithm,
                plugin_cls=_PLUGIN_CLASS_BY_ALGORITHM[self.algorithm],
                live_config=plugin.config,





                live_artifacts=(
                    getattr(plugin, "artifacts", None)
                    if CONFIG_ARTIFACT_KEYS.get(self.algorithm)
                    else None
                ),
            )








            resume_path = Path(resume_from)
            manifest_parent = resume_path.parent
            if (manifest_parent / MANIFEST_FILENAME).exists():
                entries = load_checkpoint_manifest(manifest_parent)
                if resume_path.name not in {entry.filename for entry in entries}:
                    raise CheckpointManifestError(
                        f"{resume_path.name!r} is not listed in the checkpoint "
                        f"manifest at {manifest_parent}; a controller must "
                        "select a resumable checkpoint via the manifest "
                        "(charter §4.3), not by naming an unlisted file."
                    )
            runner.load_checkpoint(resume_path, config_guard=config_guard)
        build_started = time.perf_counter()
        components = self._build_components(dataset)
        preprocessing_seconds = time.perf_counter() - build_started
        self._result = runner.run(
            components,
            preprocessing_seconds=preprocessing_seconds,
        )
        self._fitted = True
        return self








    _CONFIG_BUILDER_BY_ALGORITHM: dict[str, str] = {
        "sga": "_build_config",
        "dlga": "_build_dlga_config",
        "discover": "_build_discover_config",
        "pysr": "_build_pysr_config",
        "eqgpt": "_build_eqgpt_config",
        "llm4ed": "_build_llm4ed_config",
        "pysindy": "_build_pysindy_config",
    }

    def _build_plugin(self) -> tuple[SearchAlgorithm, int]:
        """Construct the search plugin for the configured ``algorithm``.

        Generic over the ``_PLUGIN_CLASS_BY_ALGORITHM`` registry: look up the
        plugin class, resolve its config via the facade-side builder dispatch,
        construct the plugin, and read the runner batch size from the plugin's
        own ``runner_batch_size`` declaration (FacadeWiringContract).

        Returns:
            ``(plugin, batch_size)`` — the plugin instance and the runner batch
            size (sourced from ``plugin.runner_batch_size``; SGA ``num``, DLGA
            ``pop_size``, DISCOVER ``batch_size``, PySR fixed ``1``, EqGPT
            ``samples_per_epoch``). This two-tuple signature is load-bearing:
            several facade tests call ``_build_plugin()`` directly.
        """
        plugin_cls = _PLUGIN_CLASS_BY_ALGORITHM.get(self.algorithm)
        if plugin_cls is None:
            raise NotImplementedError(
                f"Algorithm '{self.algorithm}' has no plugin builder."
            )
        builder = getattr(self, self._CONFIG_BUILDER_BY_ALGORITHM[self.algorithm])
        cfg = builder()



        plugin_factory = cast("Callable[..., FacadeWiringContract]", plugin_cls)
        if self.algorithm == "dlga":




            plugin = plugin_factory(cfg, surrogate_model=self._surrogate_model)
        elif self.algorithm == "llm4ed":





            plugin = plugin_factory(cfg, provider=self._provider)
        else:
            plugin = plugin_factory(cfg)
        return cast("SearchAlgorithm", plugin), plugin.runner_batch_size

    def _resolve_config(
        self,
        config_cls: type[_ConfigT],
        default_factory: Callable[[], _ConfigT],
    ) -> _ConfigT:
        """Shared config-resolution mechanics for every algorithm.

        Three-way policy, identical across algorithms (only the config class
        and the facade-default factory differ, which the wrappers supply):

        - user ``config=`` MATCHES the algorithm's class -> deep-copy it (the
          user config is the single source of truth; the copy prevents later
          mutations of the user's object from leaking into the fit);
        - user ``config=`` is a DIFFERENT config type -> fail fast with a
          ``TypeError`` so e.g. ``Model(algorithm='dlga', config=SGAConfig())``
          does not silently ignore the mismatched config;
        - no ``config=`` -> call ``default_factory`` (the facade default; some
          wrappers raise here instead, e.g. EqGPT has no default, D5).

        The ``requires {article} {ConfigName}`` article follows English
        initialism pronunciation: "an SGAConfig"/"an EqGPTConfig" (spoken
        "ess-", "ee-") vs "a DLGAConfig"/"a DiscoverConfig"/"a PySRConfig".
        """
        if isinstance(self._config_override, config_cls):
            return copy.deepcopy(self._config_override)
        if self._config_override is not None:
            name = config_cls.__name__



            article = "an" if name[0] in "AEIOUS" else "a"
            raise TypeError(
                f"Model(algorithm={self.algorithm!r}, config=...) requires "
                f"{article} {name}; got {type(self._config_override).__name__}."
            )
        return default_factory()

    def _build_dlga_config(self) -> DLGAConfig:
        """Resolve the DLGAConfig: user override (deep-copied) or facade default.

        Unlike SGA, the facade does not yet expose individual DLGA fields as
        ``Model(...)`` parameters — power users must pass a complete
        ``DLGAConfig`` via ``config=`` for non-default settings. The default
        path (``Model(algorithm='dlga')`` without ``config=``) yields
        ``DLGAConfig(seed=self.seed)`` which targets Xu 2020 Stage I baseline
        (sin+5×50, SVD null space, pop_size=400). See ``_resolve_config`` for
        the shared override/deep-copy/type-check mechanics.
        """
        return self._resolve_config(DLGAConfig, lambda: DLGAConfig(seed=self.seed))

    def _build_discover_config(self) -> DiscoverConfig:
        """Resolve the DiscoverConfig: user override (deep-copied) or facade default.

        Same pattern as ``_build_dlga_config``: the facade does not expose
        individual DISCOVER fields as ``Model(...)`` parameters — users must
        pass a complete ``DiscoverConfig`` via ``config=`` for non-default
        settings. The default path (``Model(algorithm='discover', seed=N)``
        without ``config=``) yields ``DiscoverConfig(seed=self.seed)`` so the
        facade ``seed=`` parameter threads through to the plugin (mirrors SGA's
        ``_build_config``). See ``_resolve_config`` for the shared mechanics.
        """
        return self._resolve_config(
            DiscoverConfig, lambda: DiscoverConfig(seed=self.seed)
        )

    def _build_pysr_config(self) -> PySRConfig:
        """Resolve the PySRConfig: user override (deep-copied) or facade default.

        Same pattern as ``_build_discover_config``. With a user
        ``config=PySRConfig(...)`` the config is the single source of truth and
        is deep-copied verbatim (its ``niterations`` is preserved). Without one,
        the facade ``generations`` knob maps to PySR's *internal* GP loop length
        via ``PySRConfig(niterations=self.generations)`` — note this is the GP
        loop, not the kd runner loop (which is pinned to 1 for the one-shot
        plugin; see ``fit``) — and the facade ``seed=`` parameter threads
        through as ``PySRConfig(seed=self.seed)`` (PySR's ``random_state``;
        mirrors SGA/DLGA/DISCOVER builders, and keeps the RunManifest seed
        truthful). See ``_resolve_config`` for the shared mechanics.
        """
        return self._resolve_config(
            PySRConfig,
            lambda: PySRConfig(niterations=self.generations, seed=self.seed),
        )

    def _build_pysindy_config(self) -> PySINDyConfig:
        """Resolve PySINDy config without mapping facade generations.

        ``max_iter`` is the native STLSQ convergence cap, not the kd search
        loop length. Configure it only through ``config=PySINDyConfig(...)``.
        The default path threads the facade seed into manifest bookkeeping.
        """
        return self._resolve_config(
            PySINDyConfig,
            lambda: PySINDyConfig(seed=self.seed),
        )

    def _build_eqgpt_config(self) -> EqGPTConfig:
        """Resolve the EqGPTConfig: user override (deep-copied), required.

        Unlike SGA/DLGA/DISCOVER/PySR, there is no facade default: EqGPTConfig
        requires ``sparsity_alpha`` (D5: per-problem, not a constant), so a
        bare ``Model(algorithm='eqgpt')`` with no ``config=`` cannot resolve
        one and the default factory raises rather than silently guessing a
        value. See ``_resolve_config`` for the shared mechanics.
        """
        return self._resolve_config(EqGPTConfig, self._no_eqgpt_default)

    @staticmethod
    def _no_eqgpt_default() -> EqGPTConfig:
        """Default factory for EqGPT: there is none (D5), so raise loud."""
        raise TypeError(
            "Model(algorithm='eqgpt') requires config=EqGPTConfig(...) "
            "(sparsity_alpha has no facade default, D5); e.g. "
            "config=EqGPTConfig(sparsity_alpha=0.02)."
        )

    def _build_llm4ed_config(self) -> Llm4edConfig:
        """Resolve the Llm4edConfig: user override (deep-copied), else default.

        Unlike EqGPT, every ``Llm4edConfig`` field carries an EDL-faithful (or
        kd-guardrail) default, so a bare ``Model(algorithm='llm4ed')`` resolves
        a runnable default config. The facade ``seed=`` threads through as
        ``Llm4edConfig(seed=self.seed)`` (mirrors the SGA/DLGA/DISCOVER/PySR
        builders, keeping the RunManifest seed truthful). Note the default
        config has ``base_url=None``: a bare ``fit`` with no injected provider
        fails loud in ``prepare()`` (the real backend needs an endpoint); CI
        drives llm4ed through an injected offline provider instead. See
        ``_resolve_config`` for the shared mechanics.
        """
        return self._resolve_config(
            Llm4edConfig, lambda: Llm4edConfig(seed=self.seed)
        )

    @property
    def best_expr_(self) -> str:
        """Best discovered expression string (post-fit only)."""
        return self._require_result().best_expression

    @property
    def best_score_(self) -> float:
        """Best score from the last fit (post-fit only).

        Algorithm-specific direction:

        - ``"sga"``: AIC, **lower is better**.
        - ``"dlga"``: GA fitness (``nmse + epsilon*length``; not an AIC),
          **lower is better**.
        - ``"discover"``: reward in roughly ``[0, 1]``, **higher is better**.
        - ``"pysr"``: kd re-fit NMSE of the best expression, **lower is better**
          (PySR's own loss is discarded; kd re-scores on the term library).
        - ``"eqgpt"``: EqGPT reward in roughly ``[0, 1]``, **higher is better**.
        - ``"llm4ed"``: LLM4ED sparse reward in roughly ``(0, 1]``, **higher is
          better** (0.0 is the empty-pool sentinel).
        - ``"pysindy"``: native PySINDy STLSQ NMSE on the kd term library,
          coefficients preserved, **lower is better**.

        ``EarlyStoppingCallback(mode="min"|"max")`` should be set accordingly.
        """
        return self._require_result().best_score

    @property
    def result_(self) -> ExperimentResult:
        """Full ``ExperimentResult`` from the last fit (post-fit only)."""
        return self._require_result()

    @property
    def algorithm_(self) -> SearchAlgorithm:
        """The fitted ``SearchAlgorithm`` plugin instance (post-fit only).

        Exposed for advanced workflows such as ``VizEngine.render_all``,
        which needs the live plugin to render plugin-specific viz.
        """
        self._check_fitted()
        if self._algorithm is None:
            raise RuntimeError(_FIT_REQUIRED_MSG)
        return self._algorithm

    def __repr__(self) -> str:
        """Friendly representation of the model state."""
        if not self._fitted or self._result is None:
            return (
                f"Model(algorithm={self.algorithm!r}, "
                f"generations={self.generations}, fitted=False)"
            )





        algo_label = self._result.config.get("algorithm", self._result.algorithm_name)
        return (
            f"Model(algorithm={algo_label!r}, fitted=True, "
            f"best_expr={self._result.best_expression!r}, "
            f"best_score={self._result.best_score:.4g})"
        )



    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if ``.fit()`` has not been called."""
        if not self._fitted:
            raise RuntimeError(_FIT_REQUIRED_MSG)

    def _require_result(self) -> ExperimentResult:
        """Return ``self._result`` after asserting fitted state.

        Used by post-fit properties so mypy can narrow ``Optional`` and so the
        ``None`` check also survives ``python -O`` (unlike a bare ``assert``).
        """
        self._check_fitted()
        if self._result is None:
            raise RuntimeError(_FIT_REQUIRED_MSG)
        return self._result

    def _build_config(self) -> SGAConfig:
        """Construct an ``SGAConfig`` from the constructor arguments.

        If an SGA ``config`` override was provided, return a deep copy so
        that later mutations of the user's config object do not affect the
        fit. Otherwise, map the individual SGA facade parameters onto
        ``SGAConfig`` and forward only kwargs that match real ``SGAConfig``
        fields (validated up-front in ``__init__``). SGA is the one algorithm
        whose config is assembled from facade knobs rather than a single
        default factory call.

        Only invoked from the SGA branch (``algorithm='sga'``); DLGA uses
        ``_build_dlga_config``. A non-SGA config override raises here so
        ``Model(algorithm='sga', config=DLGAConfig(...))`` fails fast. See
        ``_resolve_config`` for the shared override/deep-copy/type-check
        mechanics.
        """
        return self._resolve_config(SGAConfig, self._default_sga_config)

    def _default_sga_config(self) -> SGAConfig:
        """Facade default SGAConfig assembled from the individual SGA knobs.

        All kwargs were validated in ``__init__`` to be valid ``SGAConfig``
        fields and to not collide with explicitly-mapped facade params.
        """
        return SGAConfig(
            num=self.population,
            depth=self.depth,
            width=self.width,
            aic_ratio=self.aic_ratio,
            seed=self.seed,
            use_autograd=(self.derivatives == "autograd"),
            **self._extra_kwargs,
        )

    def _build_components(self, dataset: PDEDataset) -> PlatformComponents:
        """Wire up the platform stack for the given dataset (declarative).

        Resolves plugin-declared ``DerivativeReqs`` (via
        ``_resolve_derivative_requirements`` helper — Protocol does not
        support default property implementations, so the helper handles the
        getattr+isinstance check) and delegates assembly to
        ``PlatformBuilder``. The builder preserves the LHS resolve +
        validate + ``dataclasses.replace`` writeback semantics that this
        facade previously hard-coded (see the corresponding tests under
        ``tests/unit/core/platform/test_builder.py``).

        Backward compat: a plugin without a ``derivative_requirements``
        property (or with one returning ``None``) gets the SGA-aligned
        default — identical to the facade's original hard-coded wiring.
        """
        from kd.core.platform.builder import (
            PlatformBuilder,
            _resolve_derivative_requirements,
        )

        reqs = _resolve_derivative_requirements(self._algorithm)
        self._check_dataset_supported(dataset, reqs)
        return PlatformBuilder(dataset, reqs, device=self.device).build()

    def _check_dataset_supported(
        self,
        dataset: PDEDataset,
        reqs: DerivativeReqs,
    ) -> None:
        """Fail loud when the dataset's (topology, LHS order) is unsupported.

        Delegates to ``assert_dataset_supported`` (arch041 step 3b / C-A): the
        dataset's ``topology`` must be in the plugin's ``supported_topologies``,
        and its ``lhs_order`` (the DATA-0 science target) must equal the plugin's
        selected ``reqs.lhs_order``. A mismatch means the chosen algorithm cannot
        honestly fit the dataset, so raise HERE — before the (expensive)
        ``PlatformBuilder.build()`` / surrogate training — rather than silently
        fitting the wrong target ("trusted but wrong", the exact failure this
        seam exists to kill).

        This gate is the facade's FAST-FAIL leg (it raises before the expensive
        build); ``ExperimentRunner.run`` carries the same check as an
        execution-layer backstop so direct ``PlatformBuilder + ExperimentRunner``
        callers fail loud too. The narrow agent entry (``kd.evaluate_terms`` /
        ``validate_terms``) intentionally allows an explicit ``lhs_order``
        override against any dataset and routes through neither.
        """
        from kd.core.platform.requirements import assert_dataset_supported

        assert_dataset_supported(
            dataset.lhs_order, dataset.topology, reqs, self.algorithm
        )

    def _effective_max_iterations(self) -> int:
        """Return the runner loop length this Model will actually drive.

        One-shot plugins (PySR / PySINDy) are idempotent: a second ``propose``
        returns ``[]``. Driving one for ``generations`` rounds would pollute the
        recorder with empty no-op iterations and confuse early-stop, so the kd
        runner loop is pinned to a single iteration. The facade ``generations``
        still reaches PySR via its internal GP ``niterations`` (see
        ``_build_pysr_config``). All other algorithms are genuinely iterative
        and keep ``max_iterations == generations``. The one-shot flag is the
        plugin's own ``FacadeWiringContract.one_shot`` declaration (no
        facade-resident ``algorithm == "pysr"`` special case).

        Single source for ``fit``'s runner AND the verbose progress printer's
        denominator: printing ``Generation 1/500`` for a one-shot run would
        confirm an iteration budget that never ran.
        """
        if _PLUGIN_CLASS_BY_ALGORITHM[self.algorithm].one_shot:
            return 1
        return self.generations

    def _build_callbacks(self) -> list[RunnerCallback]:
        """Return the runner callback list.

        Always starts with any user-provided callbacks (in order). Appends a
        FRESH ``CheckpointCallback`` when ``checkpoint_dir`` is set (fresh
        per fit so ``_last_iteration`` cannot leak across fits), then
        ``_ProgressPrinter`` when ``verbose=True``.
        """
        cbs: list[RunnerCallback] = list(self._user_callbacks or [])
        if self.checkpoint_dir is not None:
            cbs.append(
                CheckpointCallback(
                    directory=self.checkpoint_dir,
                    every_n=self.checkpoint_every,
                    keep_last_n=self.checkpoint_keep_last,
                )
            )
        if self.verbose:
            cbs.append(
                _ProgressPrinter(total_generations=self._effective_max_iterations())
            )
        return cbs









if set(Model._CONFIG_BUILDER_BY_ALGORITHM) != set(_PLUGIN_CLASS_BY_ALGORITHM):
    raise RuntimeError(
        "kd.api registry drift: _CONFIG_BUILDER_BY_ALGORITHM keys "
        f"{sorted(Model._CONFIG_BUILDER_BY_ALGORITHM)} != "
        f"_PLUGIN_CLASS_BY_ALGORITHM keys {sorted(_PLUGIN_CLASS_BY_ALGORITHM)}. "
        "Every registered algorithm needs a config-builder entry (and vice versa)."
    )
