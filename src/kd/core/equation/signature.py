
from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kd.core.equation.canonical import canonicalize_expression
from kd.core.equation.projection import active_law
from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.serialize import from_dict
from kd.core.equation.types import (
    Equation,
    Evolution,
    Form,
    Homogeneous,
    LhsSpec,
    Scalar,
)

if TYPE_CHECKING:
    from kd.search.records import EvidenceRecord

LAWSIG_DOMAIN = "kd-lawsig-v1"


@dataclass(frozen=True)
class LawSignature:

    version: str
    structure_key: str
    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    native_form: Form
    native_lhs: LhsSpec | None

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) for value in self.coefficients):
            raise ValueError("LawSignature coefficients must be finite")

    def to_dict(self) -> dict[str, object]:
        native_lhs: dict[str, object] | None = None
        if self.native_lhs is not None:
            native_lhs = {
                "field": self.native_lhs.field,
                "axis": self.native_lhs.axis,
                "order": self.native_lhs.order,
            }
        return {
            "version": self.version,
            "structure_key": self.structure_key,
            "terms": list(self.terms),
            "coefficients": list(self.coefficients),
            "native_form": self.native_form.value,
            "native_lhs": native_lhs,
        }


@dataclass(frozen=True)
class LawAgreement:

    structure: bool
    support: bool
    coefficient: bool | None
    max_abs_delta: float | None


def _call(name: str, *args: ast.expr) -> ast.Call:
    return ast.Call(func=ast.Name(id=name), args=list(args), keywords=[])


def _is_float_one(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value == 1.0
    )


def _rewrite_aliases(node: ast.expr) -> ast.expr:
    if not isinstance(node, ast.Call):
        return node

    args = [_rewrite_aliases(arg) for arg in node.args]
    keywords = [
        ast.keyword(arg=keyword.arg, value=_rewrite_aliases(keyword.value))
        for keyword in node.keywords
    ]
    rewritten = ast.Call(func=node.func, args=args, keywords=keywords)
    if (
        isinstance(rewritten.func, ast.Name)
        and rewritten.func.id == "div"
        and len(rewritten.args) == 2
        and not rewritten.keywords
    ):
        numerator, denominator = rewritten.args



        if _is_float_one(numerator):
            return _call("recip", denominator)
        return _call("mul", numerator, _call("recip", denominator))
    return rewritten


def _parse_term(term_ir: str) -> ast.expr:
    try:
        parsed = ast.parse(term_ir.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid IR syntax: {exc}") from exc
    if not isinstance(parsed, ast.Expression):
        raise ValueError("Term IR must parse as a Python expression")
    return parsed.body


def _canonical_term(term_ir: str, coefficient: float) -> tuple[str, float]:
    node = _rewrite_aliases(_parse_term(term_ir))
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "neg"
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
        coefficient = -coefficient
    return canonicalize_expression(ast.unparse(node)), coefficient


def _scalar_value(coefficient: object) -> float:
    if not isinstance(coefficient, Scalar):
        raise NotImplementedError(
            f"{type(coefficient).__name__} coefficients are reserved"
        )
    return float(coefficient.value)


def _f0_entries(eq: Equation) -> tuple[list[tuple[str, float]], Form, LhsSpec | None]:
    entries = [
        _canonical_term(term_ir, _scalar_value(coefficient))
        for term_ir, coefficient in eq.terms
    ]
    if isinstance(eq, Evolution):




        if len(eq.lhs_spec.axis) != 1:
            raise ValueError(
                f"ambiguous LHS axis {eq.lhs_spec.axis!r}: derivative "
                "labels only injectively encode single-letter axis names"
            )
        lhs_term = canonicalize_expression(render_lhs_label(eq.lhs_spec))
        entries = [(lhs_term, 1.0)] + [
            (term_ir, -coefficient) for term_ir, coefficient in entries
        ]
        return entries, Form.EVOLUTION, eq.lhs_spec
    if isinstance(eq, Homogeneous):
        return entries, Form.HOMOGENEOUS, None
    raise TypeError(f"unsupported equation type: {type(eq).__name__}")


def _sorted_unique(entries: list[tuple[str, float]]) -> list[tuple[str, float]]:
    ordered = sorted(entries, key=lambda entry: entry[0])
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous[0] == current[0]:
            raise ValueError(f"duplicate canonical term: {current[0]}")
    return ordered


def _normalize(coefficients: tuple[float, ...]) -> tuple[float, ...]:
    if any(not math.isfinite(value) for value in coefficients):
        raise ValueError("law coefficients must be finite")
    max_magnitude = max((abs(value) for value in coefficients), default=0.0)
    if max_magnitude == 0.0:
        raise ValueError("law coefficient vector has zero L2 norm")

    scaled = tuple(value / max_magnitude for value in coefficients)
    scaled_norm = math.sqrt(math.fsum(value * value for value in scaled))
    leader = next(
        index
        for index, value in enumerate(coefficients)
        if abs(value) == max_magnitude
    )
    sign = 1.0 if coefficients[leader] > 0.0 else -1.0
    return tuple(sign * value / scaled_norm for value in scaled)


def _structure_key(terms: tuple[str, ...]) -> str:
    payload = LAWSIG_DOMAIN + "\n" + "\n".join(terms)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def law_signature(eq: Equation) -> LawSignature:
    projected = active_law(eq)
    entries, native_form, native_lhs = _f0_entries(projected)
    ordered = _sorted_unique(entries)
    terms = tuple(term for term, _coefficient in ordered)
    coefficients = _normalize(
        tuple(coefficient for _term, coefficient in ordered)
    )
    return LawSignature(
        version=LAWSIG_DOMAIN,
        structure_key=_structure_key(terms),
        terms=terms,
        coefficients=coefficients,
        native_form=native_form,
        native_lhs=native_lhs,
    )


def law_signature_from_evidence(record: EvidenceRecord) -> LawSignature | None:
    equation_payload = record.catalog_fit
    if equation_payload is None:
        return None
    return law_signature(from_dict(equation_payload))


def compare_laws(
    a: LawSignature,
    b: LawSignature,
    *,
    coeff_atol: float = 1e-2,
) -> LawAgreement:
    structure = a.structure_key == b.structure_key
    support = (
        structure
        and a.native_form is b.native_form
        and a.native_lhs == b.native_lhs
    )
    if not structure:
        return LawAgreement(
            structure=False,
            support=False,
            coefficient=None,
            max_abs_delta=None,
        )
    if len(a.coefficients) != len(b.coefficients):
        raise ValueError("matching structures require aligned coefficient vectors")
    pairs = list(zip(a.coefficients, b.coefficients, strict=True))
    direct = max((abs(x - y) for x, y in pairs), default=0.0)
    flipped = max((abs(x + y) for x, y in pairs), default=0.0)
    max_abs_delta = min(direct, flipped)
    return LawAgreement(
        structure=True,
        support=support,
        coefficient=max_abs_delta <= coeff_atol,
        max_abs_delta=max_abs_delta,
    )


__all__ = [
    "LawAgreement",
    "LAWSIG_DOMAIN",
    "LawSignature",
    "compare_laws",
    "law_signature",
    "law_signature_from_evidence",
]
