
from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import sympy

from kd.core.equation.signature import law_term_entry, law_term_key
from kd.core.equation.sketch import (
    Sketch,
    anchor_spelled_by,
    constraint_admits,
    pinned_monomials,
    term_admission,
)
from kd.core.expr.jet import Monomial
from kd.core.expr.sympy_bridge import to_sympy
from kd.core.expr.term_features import (
    ColumnFingerprint,
    TermFeatures,
    analyze_term,
    column_fingerprint,
)
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.search.llm4ed.prompts import _symbol_list

SKETCH_LEVELS: Final = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="fit_enforced",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="exit_checked",
    field_axis_set="generation_enforced",
)



_OPERATOR_FEATURES: Final[dict[str, frozenset[str]]] = {
    "+": frozenset(),
    "-": frozenset(),
    "*": frozenset({"mul"}),
    "/": frozenset({"mul", "recip"}),
    "^2": frozenset({"n2"}),
    "^3": frozenset({"n3"}),
}


_GENERABLE_FEATURES: Final[frozenset[str]] = frozenset(
    {"add", "sub", "mul", "recip", "n2", "n3"}
)


_TERM_HEADS: Final[frozenset[str]] = _GENERABLE_FEATURES | {"div", "neg"}

_OUTSIDE_VOCABULARY: Final[str] = "outside-vocabulary"
_CONSTRAINT_FILTERED: Final[str] = "constraint-filtered"


def _heads_and_leaves(key: str) -> tuple[list[str], list[str]]:
    heads: list[str] = []
    leaves: list[str] = []
    for node in ast.walk(ast.parse(key, mode="eval")):
        if isinstance(node, ast.Call):
            heads.append(node.func.id if isinstance(node.func, ast.Name) else "")
        elif isinstance(node, ast.Name):
            leaves.append(node.id)
    return heads, [leaf for leaf in leaves if leaf not in heads]


def _leaves(key: str) -> frozenset[str]:
    return frozenset(_heads_and_leaves(key)[1])


def _assert_spelled(key: str, operands: Sequence[str], kind: str) -> None:
    spelling = f"spell it over the operands {list(operands)} with the EDL operators"
    heads, leaves = _heads_and_leaves(key)
    for head in heads:
        if head not in _TERM_HEADS:
            raise ValueError(
                f"LLM4ED {kind} term {key!r} uses {head!r}, an operator outside "
                f"the EDL library; {spelling}"
            )
    for leaf in leaves:
        if leaf not in operands:
            raise ValueError(
                f"LLM4ED {kind} term {key!r} uses {leaf!r}, which is not an "
                f"operand of this dataset; {spelling}"
            )


def _assert_one_term(key: str, term_ir: str) -> None:
    if isinstance(sympy.expand(to_sympy(key)), sympy.Add):
        raise ValueError(
            f"LLM4ED anchored term {term_ir!r} expands to a sum, which the "
            "parser splits into separate terms; an anchor must be one additive "
            "term (anchor each summand separately)"
        )


def _display_key(key: str) -> str:
    return str(to_sympy(key)).replace("**", "^")


@dataclass(frozen=True, kw_only=True)
class Llm4edCompiled:

    sketch: Sketch
    gated: bool
    operands: tuple[str, ...]
    operators: str
    pinned: tuple[tuple[str, float], ...]
    pinned_keys: frozenset[str]
    pinned_monomials: frozenset[Monomial]
    anchor_fingerprints: frozenset[ColumnFingerprint]
    hint: str
    report: CompileReport
    dropped: tuple[tuple[str, str], ...]

    def _spelled(self, term_ir: str) -> str | None:
        return anchor_spelled_by(term_ir, self.sketch)

    def refusal(self, term_irs: Sequence[str]) -> str | None:
        if not self.gated:
            return None
        for term_ir in term_irs:
            reason = term_admission(
                term_ir,
                self.sketch,
                pinned_keys=self.pinned_keys,
                pinned_monomials=self.pinned_monomials,
                anchor_fingerprints=self.anchor_fingerprints,
            )
            if reason is not None:
                return reason
        present = {self._spelled(term_ir) for term_ir in term_irs}
        missing = [
            anchor.term_ir
            for anchor in self.sketch.anchored
            if self._spelled(anchor.term_ir) not in present
        ]
        if missing:
            return f"candidate lacks the anchored term(s) {missing!r}"
        return None

    def protected(self, term_irs: Sequence[str]) -> tuple[int, ...]:
        if not self.gated:
            return ()
        return tuple(
            index
            for index, term_ir in enumerate(term_irs)
            if self._spelled(term_ir) is not None
        )


def _closed_compilation(sketch: Sketch, operands: Sequence[str]) -> Llm4edCompiled:
    pins = ", ".join(pin.term_ir for pin in sketch.pinned)
    return Llm4edCompiled(
        sketch=sketch,
        gated=False,
        operands=tuple(operands),
        operators=_symbol_list(list(_OPERATOR_FEATURES)),
        pinned=(),
        pinned_keys=frozenset(),
        pinned_monomials=frozenset(),
        anchor_fingerprints=frozenset(),
        hint="",
        report=CompileReport(
            levels=SKETCH_LEVELS,
            notes=(
                "closed sketch: the operand vocabulary and the target remain "
                "unchanged because the search result is discarded; lift from "
                "pins alone",
                f"closed sketch pinned terms: {pins}",
            ),
        ),
        dropped=(),
    )


def _pin_entries(
    sketch: Sketch, operands: Sequence[str]
) -> tuple[tuple[str, float], ...]:
    entries = []
    for pin in sketch.pinned:
        key, signed_value = law_term_entry(pin.term_ir, pin.value)
        _assert_spelled(key, operands, "pinned")
        entries.append((key, signed_value))
    return tuple(entries)


def _anchor_features(
    sketch: Sketch, operands: Sequence[str]
) -> tuple[tuple[str, TermFeatures], ...]:
    anchors = []
    for anchor in sketch.anchored:
        key = law_term_key(anchor.term_ir)
        _assert_spelled(key, operands, "anchored")
        _assert_one_term(key, anchor.term_ir)
        anchors.append((key, analyze_term(key, sketch.vocabulary)))
    return tuple(anchors)


def _assert_operator_clauses_generable(sketch: Sketch) -> None:
    for hole in sketch.holes:
        clause = hole.constraint.operators
        if not clause:
            continue
        unknown = clause - _GENERABLE_FEATURES
        if unknown:
            raise ValueError(
                f"LLM4ED hole {hole.id!r} operator clause names {sorted(unknown)!r}; "
                f"an LLM4ED term carries only {sorted(_GENERABLE_FEATURES)!r} "
                "(division is written mul+recip); an empty clause means "
                "terminal-only"
            )


def _is_coordinate(name: str) -> bool:
    return name != "u" and not name.startswith("u_")


def _narrow_operands(
    sketch: Sketch,
    operands: Sequence[str],
    anchors: tuple[tuple[str, TermFeatures], ...],
    dropped: list[tuple[str, str]],
) -> tuple[str, ...]:
    anchor_leaves = frozenset().union(*(_leaves(key) for key, _features in anchors))
    axes: set[str] = set()
    for hole in sketch.holes:
        constraint = hole.constraint
        axes.update(
            sketch.vocabulary.coordinates
            if constraint.axes is None
            else constraint.axes
        )
    kept: list[str] = []
    for name in operands:
        try:
            features = analyze_term(name, sketch.vocabulary)
        except ValueError:
            dropped.append((name, _OUTSIDE_VOCABULARY))
            continue
        admitted = (
            name in axes
            if _is_coordinate(name)
            else any(
                constraint_admits(hole.constraint, features) for hole in sketch.holes
            )
        )
        if name in anchor_leaves or admitted:
            kept.append(name)
        else:
            dropped.append((name, _CONSTRAINT_FILTERED))
    if not kept:
        raise ValueError(
            "LLM4ED sketch admits no operand of this dataset "
            f"(operands {list(operands)}); its holes name fields or axes "
            "the dataset does not carry"
        )
    return tuple(kept)


def _assert_required_holes_live(sketch: Sketch, kept: Sequence[str]) -> None:
    for hole in sketch.holes:
        if hole.min_count < 1:
            continue
        if any(
            constraint_admits(hole.constraint, analyze_term(name, sketch.vocabulary))
            for name in kept
        ):
            continue
        raise ValueError(
            f"LLM4ED hole {hole.id!r} admits no operand of this dataset "
            f"(kept operands {list(kept)}); it names fields or axes the dataset "
            "does not carry"
        )


def _narrow_operators(
    sketch: Sketch,
    anchors: tuple[tuple[str, TermFeatures], ...],
    dropped: list[tuple[str, str]],
) -> str:
    admitted: set[str] = set()
    for hole in sketch.holes:
        clause = hole.constraint.operators
        admitted.update(_GENERABLE_FEATURES if clause is None else clause)
    for _key, features in anchors:
        admitted.update(features.operators)
    kept = [symbol for symbol, needs in _OPERATOR_FEATURES.items() if needs <= admitted]
    dropped.extend(
        (symbol, _CONSTRAINT_FILTERED)
        for symbol in _OPERATOR_FEATURES
        if symbol not in kept
    )
    return _symbol_list(kept)


def _hint(
    sketch: Sketch,
    pinned: tuple[tuple[str, float], ...],
    anchors: tuple[tuple[str, TermFeatures], ...],
) -> str:
    lines = ["Constraints on every equation:"]
    if anchors:
        required = ", ".join(_display_key(key) for key, _features in anchors)
        lines.append(f"- Every equation must contain the term(s) {required}.")
    if pinned:
        known = ", ".join(_display_key(key) for key, _value in pinned)
        lines.append(
            f"- The term(s) {known} are already accounted for; do not include "
            "them or any term equal to them."
        )
    if sketch.holes:
        budget = sum(hole.max_count for hole in sketch.holes)
        lines.append(
            "- Besides the required terms, an equation has at most "
            f"{budget} further term(s)."
        )
    return "\n".join(lines) + "\n"


def _notes(
    sketch: Sketch,
    pinned: tuple[tuple[str, float], ...],
    operands: tuple[str, ...],
    operators: str,
    dropped: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    anchors = ", ".join(anchor.term_ir for anchor in sketch.anchored) or "none"
    pins = ", ".join(key for key, _value in pinned) or "none"
    notes = [
        f"pinned terms deducted from the EDL finite-difference target: {pins}",
        f"anchors required in every scored candidate and kept out of STRidge "
        f"thresholding: {anchors}; the exit still requires |coefficient| >= "
        "support_threshold, and the EDL valid_coef gate still drops a fit "
        "whose anchor coefficient is outside [1e-4, 1e4]",
        "the prompt offers the narrowed operand vocabulary, which the parser "
        "enforces, and the narrowed operator list as a hint; before the fit "
        "the gate refuses any term outside the sketch vocabulary, any term on "
        "a pinned law key or sharing a jet monomial with a pin, any term "
        "carrying an anchor's jet monomial without spelling the anchor, and "
        "any term no hole admits unless it spells an anchor (column "
        "fingerprint and jet expansion)",
        f"operands offered: {list(operands)}; operators offered: {operators}",
        "hole counts are exit-checked",
    ]
    notes.extend(f"dropped {name!r}: {reason}" for name, reason in dropped)
    return tuple(notes)


def compile_for_llm4ed(sketch: Sketch, operands: Sequence[str]) -> Llm4edCompiled:
    if not sketch.anchored and not sketch.holes:
        return _closed_compilation(sketch, operands)
    pin_monomials = pinned_monomials(sketch)
    pinned = _pin_entries(sketch, operands)
    anchors = _anchor_features(sketch, operands)
    _assert_operator_clauses_generable(sketch)
    dropped: list[tuple[str, str]] = []
    kept = _narrow_operands(sketch, operands, anchors, dropped)
    _assert_required_holes_live(sketch, kept)
    operators = _narrow_operators(sketch, anchors, dropped)
    frozen_dropped = tuple(dropped)
    return Llm4edCompiled(
        sketch=sketch,
        gated=True,
        operands=kept,
        operators=operators,
        pinned=pinned,
        pinned_keys=frozenset(key for key, _value in pinned),
        pinned_monomials=pin_monomials,
        anchor_fingerprints=frozenset(
            column_fingerprint(features) for _key, features in anchors
        ),
        hint=_hint(sketch, pinned, anchors),
        report=CompileReport(
            levels=SKETCH_LEVELS,
            notes=_notes(sketch, pinned, kept, operators, frozen_dropped),
        ),
        dropped=frozen_dropped,
    )


__all__ = ["SKETCH_LEVELS", "Llm4edCompiled", "compile_for_llm4ed"]
