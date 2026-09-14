
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

import sympy
import torch
from torch import Tensor

import kd
from kd.core.expr import from_sympy, to_sympy





from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs

NMSE_THRESHOLD = 1e-2

COEFF_RTOL = 0.1

JUDGE_MAX_ORDER = 3


@dataclass(frozen=True)
class Estimator:

    provider_kind: Literal["finite_diff", "autograd"] = "finite_diff"
    surrogate_model: torch.nn.Module | None = None
    surrogate_max_order: int | None = None

    def __post_init__(self) -> None:
        if self.provider_kind == "autograd" and self.surrogate_model is None:
            raise ValueError(
                "Estimator(provider_kind='autograd') needs the trained "
                "surrogate_model the arm searched with: without it the platform "
                "builder trains a fresh default network, and judging on a "
                "second network is not judging on the arm's estimator"
            )
        if self.provider_kind == "finite_diff" and self.surrogate_model is not None:
            raise ValueError(
                "Estimator(provider_kind='finite_diff') takes no surrogate_model: "
                "the finite-difference provider never consults one, so passing it "
                "would read as a surrogate judgement that never happened"
            )
        if self.surrogate_max_order is not None:
            if self.provider_kind != "autograd":
                raise ValueError(
                    "Estimator(surrogate_max_order=...) only means something on "
                    f"provider_kind='autograd', got {self.provider_kind!r}: there "
                    "is no surrogate to bound the orders of"
                )
            if self.surrogate_max_order < 1:
                raise ValueError(
                    "surrogate_max_order must be >= 1 (a surrogate that answers "
                    f"no order at all is the finite-difference estimator), got "
                    f"{self.surrogate_max_order}"
                )

    @property
    def name(self) -> str:
        if self.surrogate_max_order is None:
            return self.provider_kind
        return f"autograd-o{self.surrogate_max_order}+finite_diff"


FINITE_DIFF = Estimator()


class JudgeError(ValueError):
    pass






_DERIVATIVE_NAME = re.compile(r"^([A-Za-z][A-Za-z0-9]*?)((?:_[xyzt]+)+)$")


def _normalize_symbol_name(name: str) -> str:
    match = _DERIVATIVE_NAME.match(name)
    if match is None:
        return name
    base, suffixes = match.groups()
    axes = sorted(suffixes.replace("_", ""))
    return f"{base}_{''.join(axes)}"


def _coordinate_substitutions(expr: sympy.Expr) -> sympy.Expr:
    replacements: dict[sympy.Expr, sympy.Expr] = {}
    for symbol in expr.free_symbols:
        name = str(symbol)
        normalized = _normalize_symbol_name(name)
        match = re.fullmatch(r"([xyzt])_(\1+)", normalized)
        if match is not None:
            order = len(match.group(2))
            replacements[symbol] = sympy.Integer(1 if order == 1 else 0)
        elif normalized != name:
            replacements[symbol] = sympy.Symbol(normalized)
    return sympy.expand(expr.subs(replacements)) if replacements else expr


_NEXT_SYMBOL: dict[str, str | int] = {
    "u": "u_x",
    "u_x": "u_xx",
    "u_xx": "u_xxx",
    "u_xxx": "u_xxxx",
    "x": 1,
    "t": 0,
}


def _push_derivative(arg: sympy.Expr) -> sympy.Expr | None:
    total: sympy.Expr = sympy.Integer(0)
    for symbol in arg.free_symbols:
        successor = _NEXT_SYMBOL.get(str(symbol))
        if successor is None:
            return None
        replacement = (
            sympy.Integer(successor)
            if isinstance(successor, int)
            else sympy.Symbol(successor)
        )
        total = total + sympy.diff(arg, symbol) * replacement
    return sympy.expand(total)


def _expand_opaque_derivatives(expr: sympy.Expr) -> sympy.Expr:
    for _ in range(8):
        pending = [
            node
            for node in expr.atoms(sympy.Function)
            if re.fullmatch(r"diff([0-9]*)_x", type(node).__name__)
        ]
        if not pending:
            break
        replacements: dict[sympy.Expr, sympy.Expr] = {}
        for node in pending:
            name = type(node).__name__
            order_text = name[len("diff"): -len("_x")]
            order = int(order_text) if order_text else 1
            value: sympy.Expr | None = sympy.expand(node.args[0])
            for _ in range(order):
                value = None if value is None else _push_derivative(value)
            if value is not None:
                replacements[node] = value
        if not replacements:
            break
        expr = sympy.expand(expr.subs(replacements))
    return expr


def normalize_expression(code: str) -> sympy.Expr:
    parsed = _coordinate_substitutions(sympy.expand(to_sympy(code)))
    return _coordinate_substitutions(_expand_opaque_derivatives(parsed))


def _term_map(expr: sympy.Expr) -> dict[sympy.Expr, float]:
    out: dict[sympy.Expr, float] = {}
    for monomial, coefficient in sympy.expand(expr).as_coefficients_dict().items():
        out[monomial] = float(coefficient)
    return out


def _law_term_map(
    support: list[str], coefficients: list[float | None] | None
) -> dict[sympy.Expr, float]:
    if coefficients is None:
        raise JudgeError(
            "law.coefficients is missing; neither the coefficient axis nor the "
            "reconstruction has anything to stand on"
        )
    if len(coefficients) != len(support):
        raise JudgeError(
            f"law.support has {len(support)} terms and coefficients has "
            f"{len(coefficients)} entries; they do not line up"
        )
    total: sympy.Expr = sympy.Integer(0)
    for term, coefficient in zip(support, coefficients, strict=True):
        if coefficient is None:
            raise JudgeError(
                f"the coefficient of term {term!r} is null; it cannot be judged"
            )
        try:
            total = total + sympy.Float(coefficient) * normalize_expression(term)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            raise JudgeError(f"term {term!r} cannot be parsed: {exc}") from exc
    return _term_map(sympy.expand(total))









from kd import RecoveryVerdict, judge_recovery, span_floor







@dataclass(frozen=True)
class _DatasetJudge:

    dataset_id: str
    lhs_name: str
    truth_coefficients: dict[str, float]
    truth_columns: dict[str, Tensor]
    u_t: Tensor
    executor: Any
    context: Any

    def column(self, monomial: sympy.Expr) -> Tensor:
        if not monomial.free_symbols:




            raise JudgeError(
                f"the constant term {monomial} has no corresponding column in "
                "kd's Theta; it cannot be judged"
            )










        exact = {
            atom: sympy.Float(atom)
            for atom in monomial.atoms(sympy.Rational)
            if not atom.is_Integer
        }
        if exact:
            monomial = monomial.subs(exact)
        try:
            code = from_sympy(monomial)
        except ValueError as exc:
            raise JudgeError(
                f"the monomial {monomial} cannot be written back to kd IR: {exc}"
            ) from exc
        try:






            return cast(Tensor, self.executor.execute(code, self.context).value)
        except Exception as exc:
            raise JudgeError(
                f"term {code!r} cannot be executed on {self.dataset_id}: {exc}"
            ) from exc


def _truth_term_map(spec: kd.DatasetSpec) -> dict[sympy.Expr, float]:
    if "=" not in spec.equation:
        raise JudgeError(
            f"the equation of {spec.id}, {spec.equation!r}, has no equals sign, "
            "so no right-hand side can be taken from it"
        )



    rhs = spec.equation.split("=", 1)[1].strip().replace("^", "**")
    return _term_map(normalize_expression(rhs))


class _MixedProvider:

    def __init__(self, surrogate: Any, finite_diff: Any, max_order: int) -> None:
        self._surrogate = surrogate
        self._finite_diff = finite_diff
        self._max_order = max_order

    def get_derivative(self, field: str, axis: str, order: int) -> Tensor:
        source = self._surrogate if order <= self._max_order else self._finite_diff
        return cast(Tensor, source.get_derivative(field, axis, order))

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        merged = dict.fromkeys(
            [
                *self._surrogate.available_derivatives(),
                *self._finite_diff.available_derivatives(),
            ]
        )
        return list(merged)

    def __getattr__(self, name: str) -> Any:


        return getattr(self._surrogate, name)


class _GridShapedProvider:

    def __init__(self, base: Any, shape: tuple[int, ...]) -> None:
        self._base = base
        self._shape = shape

    def get_derivative(self, field: str, axis: str, order: int) -> Tensor:
        value = self._base.get_derivative(field, axis, order)
        return cast(Tensor, value.detach().reshape(self._shape))

    def __getattr__(self, name: str) -> Any:


        return getattr(self._base, name)


def _grid_shape(dataset: kd.PDEDataset) -> tuple[int, ...]:
    if dataset.fields is None or dataset.lhs_field not in dataset.fields:
        raise JudgeError(
            f"{dataset.name!r} has no field {dataset.lhs_field!r} to take a grid "
            "shape from, so a surrogate's flat answers cannot be put back on it"
        )
    return tuple(dataset.fields[dataset.lhs_field].values.shape)


def _finite_diff_provider(dataset: kd.PDEDataset) -> Any:
    components = PlatformBuilder(
        dataset,
        DerivativeReqs(max_atomic_order=JUDGE_MAX_ORDER, lhs_order=dataset.lhs_order),
    ).build()
    if components.context is None:
        raise JudgeError(
            f"{dataset.name!r} builds no execution context under finite "
            "differences, so the mixed estimator has no upper half"
        )
    return components.context.derivative_provider


def _foundation(
    spec: kd.DatasetSpec,
    dataset: kd.PDEDataset,
    *,
    estimator: Estimator = FINITE_DIFF,
) -> _DatasetJudge:
    dataset_id = spec.id
    components = PlatformBuilder(
        dataset,
        DerivativeReqs(
            max_atomic_order=JUDGE_MAX_ORDER,
            lhs_order=dataset.lhs_order,
            provider_kind=estimator.provider_kind,
            surrogate_model=estimator.surrogate_model,
        ),
    ).build()
    executor, context = components.executor, components.context
    if context is None or spec.lhs is None:





        raise JudgeError(
            f"{dataset_id} has no execution context or no lhs, so the criteria "
            "cannot be built (steady-state and scattered data do not take this "
            "path)"
        )
    if estimator.provider_kind == "autograd":




        provider: Any = context.derivative_provider
        if estimator.surrogate_max_order is not None:
            provider = _MixedProvider(
                provider,
                _finite_diff_provider(dataset),
                estimator.surrogate_max_order,
            )
        context.derivative_provider = _GridShapedProvider(
            provider, _grid_shape(dataset)
        )
    truth = _truth_term_map(spec)
    truth_columns: dict[str, Tensor] = {}
    truth_coefficients: dict[str, float] = {}
    for monomial, coefficient in truth.items():
        label = str(monomial)
        truth_coefficients[label] = coefficient
        truth_columns[label] = executor.execute(from_sympy(monomial), context).value
    return _DatasetJudge(
        dataset_id=dataset_id,
        lhs_name=spec.lhs,
        truth_coefficients=truth_coefficients,
        truth_columns=truth_columns,
        u_t=executor.execute(spec.lhs, context).value,
        executor=executor,
        context=context,
    )


@lru_cache(maxsize=None)
def _dataset_judge(dataset_id: str) -> _DatasetJudge:
    spec = kd.get_dataset(dataset_id)
    return _foundation(spec, spec.loader())


@dataclass(frozen=True)
class LawVerdict:

    dataset: str
    verdict: RecoveryVerdict
    span_floor: float
    span_floor_fit: dict[str, float]
    truth_coefficients: dict[str, float]
    normalized_terms: dict[str, float]
    nmse_threshold: float = NMSE_THRESHOLD
    coeff_rtol: float = COEFF_RTOL
    estimator: str = FINITE_DIFF.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "structure_ok": self.verdict.structure_ok,
            "coefficients_ok": self.verdict.coefficients_ok,
            "nmse_ok": self.verdict.nmse_ok,
            "overall": self.verdict.overall,
            "nmse": self.verdict.nmse,
            "span_nmse": self.verdict.span_nmse,
            "projected_coefficients": self.verdict.projected_coefficients,
            "drop_one_nmse": self.verdict.drop_one_nmse,
            "span_floor": self.span_floor,
            "span_floor_fit": self.span_floor_fit,
            "truth_coefficients": self.truth_coefficients,
            "normalized_terms": self.normalized_terms,
            "nmse_threshold": self.nmse_threshold,
            "coeff_rtol": self.coeff_rtol,
            "estimator": self.estimator,
        }


def _base_for(
    dataset_id: str,
    dataset: kd.PDEDataset | None,
    estimator: Estimator = FINITE_DIFF,
) -> _DatasetJudge:
    if dataset is None and estimator == FINITE_DIFF:
        return _dataset_judge(dataset_id)
    if dataset is None:
        spec = kd.get_dataset(dataset_id)
        return _foundation(spec, spec.loader(), estimator=estimator)
    name = getattr(dataset, "name", None)
    if name != dataset_id:



        raise JudgeError(
            f"the dataset handed in is named {name!r} but the truth asked for is "
            f"{dataset_id!r}: judging one dataset's field against another's truth "
            "would produce a verdict labelled with the wrong dataset"
        )
    return _foundation(kd.get_dataset(dataset_id), dataset, estimator=estimator)


def _verdict(
    base: _DatasetJudge,
    terms: dict[sympy.Expr, float],
    *,
    nmse_threshold: float,
    coeff_rtol: float,
    estimator: Estimator = FINITE_DIFF,
) -> LawVerdict:
    rhs = torch.zeros_like(base.u_t, dtype=torch.float64)
    for monomial, coefficient in terms.items():
        rhs = rhs + coefficient * base.column(monomial).double()
    verdict = judge_recovery(
        rhs,
        base.u_t,
        truth_columns=base.truth_columns,
        truth_coefficients=base.truth_coefficients,
        coeff_rtol=coeff_rtol,
        nmse_threshold=nmse_threshold,
    )
    floor, fit = span_floor(base.u_t, truth_columns=base.truth_columns)
    return LawVerdict(
        dataset=base.dataset_id,
        verdict=verdict,
        span_floor=floor,
        span_floor_fit=fit,
        truth_coefficients=dict(base.truth_coefficients),
        normalized_terms={str(k): v for k, v in terms.items()},
        nmse_threshold=nmse_threshold,
        coeff_rtol=coeff_rtol,
        estimator=estimator.name,
    )


def judge_law(
    dataset_id: str,
    support: list[str],
    coefficients: list[float | None] | None,
    *,
    dataset: kd.PDEDataset | None = None,
    nmse_threshold: float = NMSE_THRESHOLD,
    coeff_rtol: float = COEFF_RTOL,
    estimator: Estimator = FINITE_DIFF,
) -> LawVerdict:
    base = _base_for(dataset_id, dataset, estimator)
    terms = _law_term_map(support, coefficients)
    return _verdict(
        base,
        terms,
        nmse_threshold=nmse_threshold,
        coeff_rtol=coeff_rtol,
        estimator=estimator,
    )


def truth_verdict(
    dataset_id: str,
    *,
    dataset: kd.PDEDataset | None = None,
    nmse_threshold: float = NMSE_THRESHOLD,
    coeff_rtol: float = COEFF_RTOL,
    estimator: Estimator = FINITE_DIFF,
) -> LawVerdict:
    base = _base_for(dataset_id, dataset, estimator)
    spec = kd.get_dataset(dataset_id)
    return _verdict(
        base,
        _truth_term_map(spec),
        nmse_threshold=nmse_threshold,
        coeff_rtol=coeff_rtol,
        estimator=estimator,
    )


def truth_law(dataset_id: str) -> tuple[list[str], list[float]]:
    base = _dataset_judge(dataset_id)
    labels = list(base.truth_coefficients)
    return labels, [base.truth_coefficients[label] for label in labels]


__all__ = [
    "COEFF_RTOL",
    "FINITE_DIFF",
    "JUDGE_MAX_ORDER",
    "NMSE_THRESHOLD",
    "Estimator",
    "JudgeError",
    "LawVerdict",
    "RecoveryVerdict",
    "judge_law",
    "judge_recovery",
    "normalize_expression",
    "span_floor",
    "truth_law",
    "truth_verdict",
]
