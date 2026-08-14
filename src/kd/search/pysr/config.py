"""Configuration for the PySR plugin.

``PySRConfig`` mirrors the project ``DiscoverConfig`` / ``DLGAConfig`` idiom:
a frozen, fully typed dataclass whose sequence fields are tuples (so an
instance stays hashable / immutable) and whose scalar knobs are JSON-safe
(``asdict`` round-trips through ``json.dumps`` when ``extra_pysr_kwargs`` is
``None``). ``__post_init__`` rejects illegal values with ``ValueError`` and
never mutates fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any




_DEFAULT_TERMS: tuple[str, ...] = ("u", "u_x", "u_xx", "mul(u, u_x)")




_DEFAULT_BINARY_OPERATORS: tuple[str, ...] = ("+", "-", "*", "/")
_DEFAULT_UNARY_OPERATORS: tuple[str, ...] = ("sin", "cos", "exp", "log")







_TYPED_PYSR_FIELDS: tuple[str, ...] = (
    "niterations",
    "population_size",
    "populations",
    "maxsize",
    "binary_operators",
    "unary_operators",
    "random_state",
)


@dataclass(frozen=True)
class PySRConfig:
    """Frozen configuration for a PySR symbolic-regression run.

    Fields:
    - ``terms``: the Theta library -- the candidate term columns (kd funcall
      IR) PySR regresses over. Must be non-empty.
    - ``seed``: random seed forwarded to PySR's ``random_state``. Weak
      reproducibility only: PySR upstream warns that ``random_state``
      without ``deterministic=True`` and serial execution does not pin
      run-to-run results (observed in smoke), so under kd defaults the
      seed makes runs statistically similar, not bit-identical -- and the
      run manifest's ``seed`` inherits this weaker meaning for PySR runs.
      For full determinism pass ``extra_pysr_kwargs={"deterministic":
      True, "parallelism": "serial"}``, at a significant speed cost.
    - ``niterations``: number of PySR *internal* GP iterations (algebraic
      generations inside PySR). This is **not** the kd ``generations`` knob --
      it controls only PySR's own evolutionary loop.
    - ``population_size`` / ``populations`` / ``maxsize``: PySR GP knobs
      (per-population members, number of populations, max expression size).
    - ``binary_operators`` / ``unary_operators``: operator sets exposed to
      PySR. Defaults are the kd-IR-convertible subset; adding ``"^"`` or a
      custom operator may cause the downstream ``from_sympy`` conversion to
      fail hard.
    - ``extra_pysr_kwargs``: a pass-through seam splatted into the underlying
      ``PySRRegressor`` constructor. The caller owns the JSON-safe (and
      PySR-valid) responsibility for whatever it puts here; defaults to
      ``None`` (never a shared mutable ``{}``).
    """

    terms: tuple[str, ...] = _DEFAULT_TERMS
    seed: int = 0
    niterations: int = 40
    population_size: int = 33
    populations: int = 15
    maxsize: int = 20
    binary_operators: tuple[str, ...] = _DEFAULT_BINARY_OPERATORS
    unary_operators: tuple[str, ...] = _DEFAULT_UNARY_OPERATORS
    extra_pysr_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate the configuration, raising ``ValueError`` on illegal input.

        Guards reject only genuinely illegal values; legitimate small ones
        (e.g. a single term, size knobs == 1) are accepted. No field is
        mutated here -- the dataclass is frozen and these checks are
        raise-only.
        """
        if not self.terms:
            raise ValueError("terms must be a non-empty tuple of candidate terms")
        if self.niterations <= 0:
            raise ValueError(f"niterations must be > 0, got {self.niterations}")
        if self.maxsize <= 0:
            raise ValueError(f"maxsize must be > 0, got {self.maxsize}")
        if self.population_size <= 0:
            raise ValueError(f"population_size must be > 0, got {self.population_size}")
        if self.extra_pysr_kwargs:
            collisions = sorted(
                set(self.extra_pysr_kwargs) & set(_TYPED_PYSR_FIELDS)
            )
            if collisions:
                raise ValueError(
                    "extra_pysr_kwargs must not override typed fields "
                    f"{collisions}; set the typed field instead "
                    "(niterations is facade-owned: use Model(generations=...), "
                    "and random_state is Model(seed=...))"
                )
