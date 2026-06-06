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
import warnings
from typing import TYPE_CHECKING, Any, Literal

import torch

from kd.search.callbacks import EarlyStoppingCallback, RunnerCallback
from kd.search.discover import DiscoverConfig, DISCOVERPlugin
from kd.search.discover.config import (
    DEFAULT_N_ITERATIONS,
    DEFAULT_STABILITY_QUEUE_CAPACITY,
    DEFAULT_STABILITY_SELECTION,
)
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin
from kd.viz._labels import score_label as _viz_score_label

if TYPE_CHECKING:
    from kd.core.evaluator import EvaluationResult
    from kd.data.schema import PDEDataset
    from kd.search.protocol import SearchAlgorithm
    from kd.search.result import ExperimentResult

__all__ = ["Model"]





_SUPPORTED_ALGORITHMS = ("sga", "dlga", "discover")



_EARLY_STOP_MODE_BY_ALGORITHM: dict[str, Literal["min", "max"]] = {
    "sga": "min",
    "dlga": "min",
    "discover": "max",
}
_DEFAULT_LHS_FIELD = "u"
_DEFAULT_LHS_AXIS = "t"





_FACADE_LHS_ORDER = 1
_PROGRESS_PREFIX = "[kd]"
_FIT_REQUIRED_MSG = "Model has not been fit. Call .fit(dataset) first."






_UNSET: Any = object()




_DEFAULT_POPULATION = 20
_DEFAULT_DEPTH = 4
_DEFAULT_WIDTH = 5
_DEFAULT_AIC_RATIO = 1.0
_DEFAULT_DERIVATIVES = "finite_diff"
_DEFAULT_SEED = 0

_VALID_DERIVATIVES = frozenset({"finite_diff", "autograd"})




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
    experiment end.
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

    SGA + DLGA minimize AIC; discover maximizes a reward. The plugin
    config carries the algorithm name in the ``"algorithm"`` key (set by
    each plugin's ``config`` property). Unknown algorithms fall back to
    ``"Score"`` for consistency with HTML reports.
    """
    return _viz_score_label(algorithm.config.get("algorithm", ""))







class Model:
    """High-level facade for PDE discovery (PySR-style API).

    The facade wraps the ``ExperimentRunner`` + plugin + components stack
    into a single class with sklearn-style ``.fit()`` and trailing-underscore
    post-fit attributes (``best_expr_``, ``best_score_``, ``result_``).

    Limitations:
        Only **first-order LHS** PDE is supported (e.g., ``u_t = f(u, u_x, ...)``).
        Higher-order LHS such as the wave equation ``u_tt = c**2 * u_xx`` is
        NOT supported by this facade — both the LHS target inside SGA and the
        Evaluator are wired with ``order=1`` (see ``_FACADE_LHS_ORDER``). To
        discover a second-order-in-time PDE, reduce it to a first-order
        system manually (introduce ``v = u_t``, then discover ``u_t = v`` and
        ``v_t = ...`` separately).

    Args:
        algorithm: Search algorithm name. Supported: ``"sga"`` (default,
            full facade-parameter coverage), ``"dlga"`` (driven through
            ``config=DLGAConfig(...)`` + ``surrogate_model=`` only), or
            ``"discover"`` (driven through ``config=DiscoverConfig(...)``
            only). For DLGA and discover the individual facade parameters
            below (population/depth/width/aic_ratio/derivatives) are SGA-only.
        generations: Maximum number of search iterations (all algorithms).
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
        seed: Random seed for reproducibility (all algorithms; for DLGA and
            discover it is forwarded through the plugin config when
            ``config=`` is unset, otherwise the config's own seed wins).
        verbose: When True, print per-iteration progress to stdout. The label
            is algorithm-aware (``AIC`` for SGA/DLGA, ``reward`` for discover).
        config: Optional pre-built ``SGAConfig``, ``DLGAConfig``, or
            ``DiscoverConfig``. When provided, it is the single source of
            plugin settings; passing any non-default SGA-related facade
            parameters (``population``, ``depth``, ``width``, ``aic_ratio``,
            ``derivatives``, ``seed``) or extra ``kwargs`` raises
            ``ValueError`` (SGA path) / ``TypeError`` (DLGA + discover paths).
            Only ``algorithm``, ``generations``, ``verbose``, ``callbacks``,
            and (for DLGA) ``surrogate_model`` remain effective on the facade.
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
        **kwargs: Forwarded to ``SGAConfig`` if the field name matches an
            allowed (non-explicitly-mapped) field. Unknown or colliding
            kwargs raise ``TypeError``. SGA-only — DLGA and discover paths
            raise if any kwargs are supplied (drive those algorithms via
            ``config=DLGAConfig(...)`` / ``config=DiscoverConfig(...)``).
    """

    def __init__(
        self,
        algorithm: str = "sga",
        generations: int = 50,
        population: int = _UNSET,
        depth: int = _UNSET,
        width: int = _UNSET,
        aic_ratio: float = _UNSET,
        derivatives: str = _UNSET,
        seed: int = _UNSET,
        verbose: bool = True,
        config: SGAConfig | DLGAConfig | DiscoverConfig | None = None,
        callbacks: list[RunnerCallback] | None = None,
        surrogate_model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:




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
            population=population,
            depth=depth,
            width=width,
            aic_ratio=aic_ratio,
            derivatives=derivatives,
        )




        self._validate_discover_config_facade_compat(algorithm, config)

        self.algorithm = algorithm
        self.generations = generations
        self.population = population_resolved
        self.depth = depth_resolved
        self.width = width_resolved
        self.aic_ratio = aic_ratio_resolved
        self.derivatives = derivatives_resolved
        self.seed = seed_resolved
        self.verbose = verbose





        self._config_override = copy.deepcopy(config) if config is not None else None



        self._user_callbacks: list[RunnerCallback] | None = (
            list(callbacks) if callbacks is not None else None
        )
        self._extra_kwargs: dict[str, Any] = kwargs




        self._surrogate_model = surrogate_model


        self._fitted: bool = False
        self._result: ExperimentResult | None = None
        self._algorithm: SearchAlgorithm | None = None



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
        config: SGAConfig | DLGAConfig | DiscoverConfig | None,
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
                f"'generations', 'verbose', 'callbacks' (and "
                f"'surrogate_model' for DLGA) remain effective on the facade."
            )

    @staticmethod
    def _validate_algorithm_arg_exclusivity(
        algorithm: str,
        *,
        kwargs: dict[str, Any],
        surrogate_model: torch.nn.Module | None,
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

        - ``algorithm='dlga'`` + SGA-only facade params (``population``,
          ``depth``, ``width``, ``aic_ratio``, ``derivatives``) or extra
          kwargs (``lam``, ``p_var``, ``autograd_train_*``, ...) would
          also silently drop. Raise — DLGA users must drive non-default
          settings through ``DLGAConfig`` via ``config=``.
        """
        if algorithm != "dlga" and surrogate_model is not None:
            raise TypeError(
                f"Model(algorithm={algorithm!r}, surrogate_model=...) is not "
                "supported. ``surrogate_model`` is consumed only by "
                "``algorithm='dlga'``; for SGA opt-in autograd derivatives use "
                "``derivatives='autograd'`` instead."
            )
        if algorithm == "dlga":
            sga_only_set = []
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
                    f"Model(algorithm='dlga', ...): the following SGA-only "
                    f"parameters cannot be passed alongside DLGA: "
                    f"{sorted(set(sga_only_set))}. Drive DLGA via "
                    "``config=DLGAConfig(...)`` instead — the facade does not "
                    "expose individual DLGA fields as ``Model(...)`` parameters."
                )
        if algorithm == "discover":
            sga_only_set = []
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
                    f"Model(algorithm='discover', ...): the following SGA-only "
                    f"parameters cannot be passed alongside discover: "
                    f"{sorted(set(sga_only_set))}. Drive discover via "
                    "``config=DiscoverConfig(...)`` instead — the facade does "
                    "not expose individual DISCOVER fields as ``Model(...)`` "
                    "parameters."
                )

    @staticmethod
    def _validate_discover_config_facade_compat(
        algorithm: str,
        config: SGAConfig | DLGAConfig | DiscoverConfig | None,
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
        (``scripts/run_discover.py``); the check belongs in the facade layer,
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
                "standalone MODE2 entry in scripts/run_discover.py, or drop pinn."
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
        pass silently (AUDIT-05); this checks both directions symmetrically.
        """
        if callbacks is None:
            return
        required = _EARLY_STOP_MODE_BY_ALGORITHM.get(algorithm)
        if required is None:
            return
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



    def fit(self, dataset: PDEDataset) -> Model:
        """Run the search and populate post-fit attributes.

        Args:
            dataset: The PDE dataset to discover an equation for.

        Returns:
            ``self`` for sklearn-style chaining.

        Raises:
            NotImplementedError: If ``algorithm`` is not supported.
            ValueError: If the dataset is missing the required LHS field
                or axis.
        """


        self._fitted = False
        self._result = None
        self._algorithm = None

        if self.algorithm not in _SUPPORTED_ALGORITHMS:
            raise NotImplementedError(
                f"Algorithm '{self.algorithm}' is not implemented. "
                f"Supported algorithms: {list(_SUPPORTED_ALGORITHMS)}"
            )




        plugin, batch_size = self._build_plugin()
        self._algorithm = plugin
        components = self._build_components(dataset)

        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=self.generations,
            batch_size=batch_size,
            callbacks=self._build_callbacks(),
        )
        self._result = runner.run(components)
        self._fitted = True
        return self

    def _build_plugin(self) -> tuple[SearchAlgorithm, int]:
        """Construct the search plugin for the configured ``algorithm``.

        Returns:
            ``(plugin, batch_size)`` — the plugin instance and the runner
            batch size. Batch size derives from the plugin config (SGA
            ``num``, DLGA ``pop_size``).
        """
        if self.algorithm == "sga":
            sga_cfg = self._build_config()
            return SGAPlugin(sga_cfg), sga_cfg.num
        if self.algorithm == "dlga":
            dlga_cfg = self._build_dlga_config()
            return (
                DLGAPlugin(dlga_cfg, surrogate_model=self._surrogate_model),
                dlga_cfg.pop_size,
            )
        if self.algorithm == "discover":
            discover_cfg = self._build_discover_config()
            return DISCOVERPlugin(discover_cfg), discover_cfg.batch_size

        raise NotImplementedError(
            f"Algorithm '{self.algorithm}' has no plugin builder."
        )

    def _build_dlga_config(self) -> DLGAConfig:
        """Resolve the DLGAConfig: user override (deep-copied) or facade default.

        Unlike SGA, the facade does not yet expose individual DLGA fields as
        ``Model(...)`` parameters — power users must pass a complete
        ``DLGAConfig`` via ``config=`` for non-default settings. The default
        path (``Model(algorithm='dlga')`` without ``config=``) yields
        ``DLGAConfig()`` which targets Xu 2020 Stage I baseline (sin+5×50,
        SVD null space, pop_size=400).
        """
        if isinstance(self._config_override, DLGAConfig):
            return copy.deepcopy(self._config_override)
        if self._config_override is not None:
            raise TypeError(
                f"Model(algorithm='dlga', config=...) requires a DLGAConfig; "
                f"got {type(self._config_override).__name__}."
            )
        return DLGAConfig(seed=self.seed)

    def _build_discover_config(self) -> DiscoverConfig:
        """Resolve the DiscoverConfig: user override (deep-copied) or facade default.

        Same pattern as ``_build_dlga_config``: the facade does not expose
        individual DISCOVER fields as ``Model(...)`` parameters — users
        must pass a complete ``DiscoverConfig`` via ``config=`` for
        non-default settings. The default path
        (``Model(algorithm='discover', seed=N)`` without ``config=``) yields
        ``DiscoverConfig(seed=self.seed)`` so the facade ``seed=`` parameter
        threads through to the plugin (mirrors SGA's ``_build_config`` which
        also forwards ``self.seed`` to ``SGAConfig``).
        """
        if isinstance(self._config_override, DiscoverConfig):
            return copy.deepcopy(self._config_override)
        if self._config_override is not None:
            raise TypeError(
                f"Model(algorithm='discover', config=...) requires a "
                f"DiscoverConfig; got {type(self._config_override).__name__}."
            )
        return DiscoverConfig(seed=self.seed)

    @property
    def best_expr_(self) -> str:
        """Best discovered expression string (post-fit only)."""
        return self._require_result().best_expression

    @property
    def best_score_(self) -> float:
        """Best score from the last fit (post-fit only).

        Algorithm-specific direction:

        - ``"sga"`` / ``"dlga"``: AIC, **lower is better**.
        - ``"discover"``: reward in roughly ``[0, 1]``, **higher is better**.

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
        fit. Otherwise, forward only kwargs that match real ``SGAConfig``
        fields (validated up-front in ``__init__``).

        Only invoked from the SGA branch (``algorithm='sga'``); DLGA uses
        ``_build_dlga_config``. A non-SGA config override raises here so
        ``Model(algorithm='sga', config=DLGAConfig(...))`` fails fast.
        """
        if self._config_override is not None:
            if not isinstance(self._config_override, SGAConfig):
                raise TypeError(
                    f"Model(algorithm='sga', config=...) requires an SGAConfig; "
                    f"got {type(self._config_override).__name__}."
                )
            return copy.deepcopy(self._config_override)



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
        """Wire up the platform stack for the given dataset (POT-5 declarative).

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
        default — the facade behavior is identical to the pre-POT-5 wiring.
        """
        from kd.core.platform.builder import (
            PlatformBuilder,
            _resolve_derivative_requirements,
        )

        reqs = _resolve_derivative_requirements(self._algorithm)
        return PlatformBuilder(dataset, reqs).build()

    def _build_callbacks(self) -> list[RunnerCallback]:
        """Return the runner callback list.

        Always starts with any user-provided callbacks (in order). Appends
        ``_ProgressPrinter`` when ``verbose=True``.
        """
        cbs: list[RunnerCallback] = list(self._user_callbacks or [])
        if self.verbose:
            cbs.append(_ProgressPrinter(total_generations=self.generations))
        return cbs
