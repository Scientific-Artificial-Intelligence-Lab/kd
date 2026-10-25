
from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Final

from kd.core.equation.signature import law_term_key
from kd.core.equation.sketch import (
    Sketch,
    TermConstraint,
    anchor_basis,
    constraint_admits,
    pinned_monomials,
    term_admission,
)
from kd.core.expr.jet import Monomial, jet_monomials
from kd.core.expr.term_features import (
    ColumnFingerprint,
    TermFeatures,
    TermVocabulary,
    analyze_term,
    column_fingerprint,
)
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.search.eqgpt.gates import REDUNDANT_COMBOS, missing_variables
from kd.search.eqgpt.ir_map import TOKEN_IR_ATOM, sentence_to_rhs_terms
from kd.search.eqgpt.sampling import OPERATOR_IDS
from kd.search.eqgpt.vocab import DIV_ID, FIRST_TERM_ID, MUL_ID, Vocab

_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)

_OUTSIDE_VOCABULARY = "outside-vocabulary"
_CONSTRAINT_FILTERED = "constraint-filtered"


_OPERATOR_FEATURES: Final[dict[int, frozenset[str]]] = {
    MUL_ID: frozenset({"mul"}),
    DIV_ID: frozenset({"mul", "recip"}),
}


@dataclass(frozen=True, kw_only=True)
class EqGPTCompiled:

    masked_tokens: frozenset[int]
    gate_variables: tuple[str, ...]
    pinned_keys: frozenset[str]
    pinned_monomials: frozenset[Monomial]
    anchor_fingerprints: frozenset[ColumnFingerprint]
    report: CompileReport
    dropped: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _Grammar:

    atoms: dict[int, TermFeatures]
    operators: dict[int, frozenset[str]]
    outside: tuple[int, ...]


@dataclass(frozen=True)
class _Pins:

    keys: frozenset[str]
    monomials: frozenset[Monomial]
    vocabulary: TermVocabulary

    def cover(self, term_ir: str) -> bool:
        return law_term_key(term_ir) in self.keys or bool(
            self.monomials and jet_monomials(term_ir, self.vocabulary) & self.monomials
        )


def _term_axes(features: TermFeatures) -> frozenset[str]:
    axes = set(features.coordinate_dependencies)
    for _field, multiindex in features.derivative_multiindices:
        axes.update(axis for axis, _order in multiindex)
    return frozenset(axes)


def _atom_ir(vocab: Vocab, token: int) -> str:
    return TOKEN_IR_ATOM[vocab.id2word[token]]


def _grammar(sketch: Sketch, vocab: Vocab, base_masked: frozenset[int]) -> _Grammar:
    atoms: dict[int, TermFeatures] = {}
    outside: list[int] = []
    for token in range(FIRST_TERM_ID, vocab.size):
        if token in base_masked:
            continue
        try:
            atoms[token] = analyze_term(_atom_ir(vocab, token), sketch.vocabulary)
        except ValueError:
            outside.append(token)
    operators = {
        token: features
        for token, features in _OPERATOR_FEATURES.items()
        if token not in base_masked
    }
    return _Grammar(atoms=atoms, operators=operators, outside=tuple(outside))


def _assert_operator_clauses_generable(sketch: Sketch, grammar: _Grammar) -> None:
    token_features = [features.operators for features in grammar.atoms.values()]
    token_features.extend(grammar.operators.values())
    generable = frozenset().union(*token_features)
    for hole in sketch.holes:
        clause = hole.constraint.operators
        if not clause:
            continue
        covered = frozenset().union(*(f for f in token_features if f <= clause))
        uncovered = clause - covered
        if uncovered:
            raise ValueError(
                f"EqGPT hole {hole.id!r} operator clause {sorted(clause)!r} names "
                f"{sorted(uncovered)!r}, which no vocabulary token builds without "
                "adding a feature outside the clause (generable features "
                f"{sorted(generable)!r}; division is written mul+recip and / "
                "brings both; an empty clause means terminal-only)"
            )


def _inside(anchor: TermFeatures, features: TermFeatures) -> bool:
    return (
        features.base_fields <= anchor.base_fields
        and features.coordinate_dependencies <= anchor.coordinate_dependencies
        and features.derivative_multiindices <= anchor.derivative_multiindices
        and features.operators <= anchor.operators
    )


def _chain_matches(
    target: TermFeatures, chain: Sequence[TermFeatures], grammar: _Grammar
) -> bool:
    union = (
        frozenset().union(*(f.base_fields for f in chain)),
        frozenset().union(*(f.coordinate_dependencies for f in chain)),
        frozenset().union(*(f.derivative_multiindices for f in chain)),
    )
    if union != (
        target.base_fields,
        target.coordinate_dependencies,
        target.derivative_multiindices,
    ):
        return False
    operators = frozenset().union(*(f.operators for f in chain))
    return any(
        target.operators == operators | joined for joined in grammar.operators.values()
    )


def _chain_term(chain: Sequence[int], operator: int, vocab: Vocab) -> str:
    sentence = [chain[0]]
    for token in chain[1:]:
        sentence.extend((operator, token))
    (term,) = sentence_to_rhs_terms(vocab, sentence, start_len=0)
    return term


def _spelling_length(
    target: TermFeatures, grammar: _Grammar, sketch: Sketch, vocab: Vocab
) -> int | None:
    fingerprint = column_fingerprint(target)
    if any(
        column_fingerprint(features) == fingerprint
        for features in grammar.atoms.values()
    ):
        return 1
    inside = [
        token for token, features in grammar.atoms.items() if _inside(target, features)
    ]
    if not inside or not _chain_matches(
        target, [grammar.atoms[token] for token in inside], grammar
    ):
        return None
    longest = max(2, len(inside), target.power_degree)
    return next(
        (
            size
            for size in range(2, longest + 1)
            for chain in itertools.combinations_with_replacement(inside, size)
            for operator in grammar.operators
            if column_fingerprint(
                analyze_term(
                    law_term_key(_chain_term(chain, operator, vocab)),
                    sketch.vocabulary,
                )
            )
            == fingerprint
        ),
        None,
    )


def _anchor_spellings(
    sketch: Sketch,
    anchors: tuple[TermFeatures, ...],
    grammar: _Grammar,
    vocab: Vocab,
) -> dict[ColumnFingerprint, int]:
    spellings: dict[ColumnFingerprint, int] = {}
    for anchor, features in zip(sketch.anchored, anchors, strict=True):
        length = _spelling_length(features, grammar, sketch, vocab)
        if length is not None:
            spellings[column_fingerprint(features)] = length
            continue
        words = [
            vocab.id2word[token]
            for token, atom in grammar.atoms.items()
            if _inside(features, atom)
        ]
        live = [vocab.id2word[token] for token in grammar.operators]
        raise ValueError(
            f"EqGPT anchor {anchor.term_ir!r} cannot be spelled from the vocabulary "
            f"atoms under this dataset's mask (atoms fitting inside it: {words!r}; "
            f"operator tokens {live!r}); the base mask removes the atoms above "
            "the provider's derivative order, every t-bearing atom, the atoms "
            "on an undeclared axis, the IR-inexpressible tokens and the tokens in "
            "config.masked_tokens"
        )
    return spellings


def _assert_anchors_not_redundant(
    anchor_fingerprints: frozenset[ColumnFingerprint],
    grammar: _Grammar,
    vocab: Vocab,
) -> None:
    words = frozenset(
        vocab.id2word[token]
        for token, features in grammar.atoms.items()
        if column_fingerprint(features) in anchor_fingerprints
    )
    for combo in REDUNDANT_COMBOS:
        if combo <= words:
            raise ValueError(
                f"EqGPT anchors spelled by the tokens {sorted(combo)!r} form a "
                "redundant combination that the reference reward gate zeroes: "
                "every sentence carrying all of them scores 0 and is outranked by "
                "any positive-reward sentence; drop one anchor"
            )


def _assert_capacity(
    sketch: Sketch,
    spellings: dict[ColumnFingerprint, int],
    witness_lengths: dict[str, int],
    max_length: int,
    prefix_len: int,
) -> None:
    capacity = (max_length - prefix_len + 1) // 2
    hole_atoms = sum(
        hole.min_count * witness_lengths[hole.id]
        for hole in sketch.holes
        if hole.min_count >= 1
    )
    anchor_atoms = sum(spellings.values())
    required = anchor_atoms + hole_atoms
    if required > capacity:
        raise ValueError(
            f"EqGPT sketch requires {required} term positions ({anchor_atoms} for "
            f"{len(spellings)} distinct anchor columns, each spelled by its "
            f"shortest atom or * / chain, plus {hole_atoms} for the required holes, "
            "min_count times the atoms of each hole's shortest admitted term "
            f"{witness_lengths!r}), but max_length={max_length} with the "
            f"{prefix_len}-token start prefix fits at most {capacity} term "
            "positions per sentence"
        )


def _fits(constraint: TermConstraint, features: TermFeatures) -> bool:
    return constraint_admits(constraint, features) or (
        not features.base_fields
        and constraint_admits(replace(constraint, fields=None), features)
    )


def _joins(constraint: TermConstraint, grammar: _Grammar) -> bool:
    return any(
        constraint.operators is None or features <= constraint.operators
        for features in grammar.operators.values()
    )


def _atom_kept(
    features: TermFeatures,
    sketch: Sketch,
    anchors: tuple[TermFeatures, ...],
    grammar: _Grammar,
) -> bool:
    for hole in sketch.holes:
        constraint = hole.constraint
        if _fits(constraint, features) and (
            features.base_fields
            or constraint.fields is None
            or _joins(constraint, grammar)
        ):
            return True
    return any(_inside(anchor, features) for anchor in anchors)


def _operator_kept(
    features: frozenset[str], sketch: Sketch, anchors: tuple[TermFeatures, ...]
) -> bool:
    return any(
        hole.constraint.operators is None or features <= hole.constraint.operators
        for hole in sketch.holes
    ) or any(features <= anchor.operators for anchor in anchors)


def _sketch_mask(
    sketch: Sketch, anchors: tuple[TermFeatures, ...], grammar: _Grammar
) -> tuple[dict[int, TermFeatures], frozenset[int], list[int]]:
    kept: dict[int, TermFeatures] = {}
    filtered: list[int] = []
    for token, features in grammar.atoms.items():
        if _atom_kept(features, sketch, anchors, grammar):
            kept[token] = features
        else:
            filtered.append(token)
    kept_operators: set[int] = set()
    for token, joined in grammar.operators.items():
        if _operator_kept(joined, sketch, anchors):
            kept_operators.add(token)
        else:
            filtered.append(token)
    return kept, frozenset(kept_operators), filtered


def _witness_terms(
    sketch: Sketch,
    kept: dict[int, TermFeatures],
    kept_operators: frozenset[int],
    vocab: Vocab,
) -> Iterator[tuple[str, TermFeatures, int]]:
    for token, features in kept.items():
        yield _atom_ir(vocab, token), features, 1
    for left, operator, right in itertools.product(kept, sorted(kept_operators), kept):
        term = _chain_term((left, right), operator, vocab)
        yield term, analyze_term(law_term_key(term), sketch.vocabulary), 2


def _assert_required_holes_live(
    sketch: Sketch,
    kept: dict[int, TermFeatures],
    kept_operators: frozenset[int],
    pins: _Pins,
    vocab: Vocab,
) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for hole in sketch.holes:
        if hole.min_count < 1:
            continue
        length = next(
            (
                length
                for term, features, length in _witness_terms(
                    sketch, kept, kept_operators, vocab
                )
                if constraint_admits(hole.constraint, features) and not pins.cover(term)
            ),
            None,
        )
        if length is not None:
            lengths[hole.id] = length
            continue
        words = [vocab.id2word[token] for token in kept]
        operators = [vocab.id2word[token] for token in sorted(kept_operators)]
        raise ValueError(
            f"EqGPT hole {hole.id!r} admits no generable term: no kept token, and "
            "no two kept tokens joined by a kept * or /, spell a term it admits "
            f"off the pinned columns (kept tokens {words!r}, operator tokens "
            f"{operators!r}); every vocabulary atom is spelled over the field 'u' "
            "and the axes x, y, z, t; widen the hole's clause, admit mul, or drop "
            "the covering pin"
        )
    return lengths


def _assert_term_token_left(
    kept: dict[int, TermFeatures],
    kept_operators: frozenset[int],
    pins: _Pins,
    vocab: Vocab,
) -> None:
    survivors = [
        token
        for token in kept
        if kept_operators or not pins.cover(_atom_ir(vocab, token))
    ]
    if survivors:
        return
    raise ValueError(
        "EqGPT sketch leaves no term token to sample: the base mask plus the "
        "sketch mask cover the vocabulary, or only atoms on a pinned column remain "
        "with * and / masked; widen a hole or drop a clause"
    )


def _assert_operator_position_live(
    base_masked: frozenset[int],
    sketch_masked: frozenset[int],
    start_words: tuple[str, ...],
    max_length: int,
    vocab: Vocab,
) -> None:
    position = len(start_words) + len(start_words) % 2
    if position >= max_length or not set(OPERATOR_IDS) <= base_masked | sketch_masked:
        return
    base = [vocab.id2word[token] for token in OPERATOR_IDS if token in base_masked]
    by_sketch = [
        vocab.id2word[token] for token in OPERATOR_IDS if token in sketch_masked
    ]
    raise ValueError(
        f"EqGPT sketch leaves no operator token at sentence position {position}, "
        f"which sampling reaches within max_length={max_length}: the base mask "
        f"(config.masked_tokens included) removes {base!r} and the sketch mask "
        f"removes {by_sketch!r}; unmask E or + in config.masked_tokens, or admit "
        "mul in a hole's operator clause"
    )


def _axis_carriers(
    gate_variables: tuple[str, ...],
    kept: dict[int, TermFeatures],
    start_words: tuple[str, ...],
    vocab: Vocab,
) -> dict[str, tuple[int, ...]]:
    return {
        axis: tuple(
            token
            for token in kept
            if not missing_variables((vocab.id2word[token],), (axis,))
        )
        for axis in missing_variables(start_words[1:], gate_variables)
    }


def _assert_gate_axes_carried(
    carriers: dict[str, tuple[int, ...]],
    base_carriers: dict[str, tuple[int, ...]],
    kept: dict[int, TermFeatures],
    start_words: tuple[str, ...],
    vocab: Vocab,
) -> None:
    for axis, tokens in carriers.items():
        if tokens:
            continue
        gate = (
            f"the start prefix {list(start_words)!r} does not spell it and no pin "
            "carries it, but the reference declared-variable reward gate zeroes "
            "every sentence that lacks it, so every candidate would score 0"
        )
        if not base_carriers[axis]:
            raise ValueError(
                f"EqGPT base mask leaves no term token carrying axis {axis!r} (it "
                "removes the atoms above the provider's derivative order, every "
                "t-bearing atom, the atoms on an undeclared axis, the "
                f"IR-inexpressible tokens and config.masked_tokens), {gate}; add a "
                f"word spelling {axis!r} to config.start_words"
            )
        words = [vocab.id2word[token] for token in kept]
        raise ValueError(
            f"EqGPT sketch mask keeps no term token carrying axis {axis!r} (kept "
            f"tokens {words!r}), {gate}; admit a term on {axis!r} in a hole or pin "
            "one"
        )


def _field_free_notes(
    carriers: dict[str, tuple[int, ...]],
    kept: dict[int, TermFeatures],
    sketch: Sketch,
    grammar: _Grammar,
    vocab: Vocab,
) -> list[str]:
    notes: list[str] = []
    for axis, tokens in carriers.items():
        if any(kept[token].base_fields for token in tokens):
            continue
        words = [vocab.id2word[token] for token in tokens]
        product_only = any(
            hole.constraint.fields is not None
            and _fits(hole.constraint, kept[token])
            and _joins(hole.constraint, grammar)
            for hole in sketch.holes
            for token in tokens
        )
        clause = (
            ", which a fields clause admits only inside a product"
            if product_only
            else ""
        )
        notes.append(
            f"axis {axis!r} is spelled only by the field-free tokens {words!r}"
            f"{clause}, so a sentence without one of them scores 0"
        )
    return notes


def _gate_note(
    pinned_axes: frozenset[str],
    gate_variables: tuple[str, ...],
    field_free: list[str],
) -> str:
    parts = [
        f"declared-variable gate: axes carried by pins {sorted(pinned_axes)!r} "
        "count as present, so the gate judges the generated sentence on "
        f"{list(gate_variables)!r}",
        *field_free,
    ]
    parts.append("the redundant-combination gate judges the generated sentence alone")
    return "; ".join(parts)


def _boundary_notes(
    sketch: Sketch,
    pinned_axes: frozenset[str],
    gate_variables: tuple[str, ...],
    field_free: list[str],
) -> list[str]:
    notes: list[str] = []
    if sketch.pinned:
        notes.append(
            "pin reservation refuses a term whose expansion into jet monomials "
            "(total derivatives by the chain rule, symbol-free factors stripped "
            "as coefficients, division by or log of an identically zero column "
            "read as the constant kd's protected operators produce) contains a "
            "pinned monomial, so a composite that carries a pinned monomial "
            "beside another one is refused by design"
        )
        notes.append(
            "pin reservation leaves open: the expansion cancels no common factor "
            "between a numerator and a denominator, so a rational respelling of "
            "a pin is admitted, reachable with / and an atom holding a sum of "
            "derivatives as the denominator; terms are judged one at a time, so "
            "several admitted terms can add up to the pinned column; function "
            "identities (sin or cos squared, log of exp) would respell a pin "
            "too, but the vocabulary has no cos or log atom"
        )
    if sketch.pinned or field_free:
        notes.append(_gate_note(pinned_axes, gate_variables, field_free))
    if sketch.anchored:
        notes.append(
            "anchor generability is judged on the token grammar (atoms, *, /), not "
            "on the pretrained model's propensity"
        )
    return notes


def _report(
    sketch: Sketch,
    pinned_axes: frozenset[str],
    gate_variables: tuple[str, ...],
    field_free: list[str],
    dropped: tuple[tuple[str, str], ...],
) -> CompileReport:
    pins = ", ".join(pin.term_ir for pin in sketch.pinned) or "none"
    notes = [
        "anchors and hole counts are exit-checked",
        "the sketch token mask bounds generation; the pre-score gate refuses any "
        "term outside the sketch vocabulary, any term on a pinned law key or "
        "sharing a jet monomial with a pin, any term carrying an anchor's jet "
        "monomial without spelling the anchor, and any term no hole admits "
        "unless it spells an anchor (column fingerprint and jet expansion); a "
        "candidate's later spellings of one anchor are left out of its fit",
        f"pinned terms are owned by lower/lift: {pins}",
    ]
    notes.extend(_boundary_notes(sketch, pinned_axes, gate_variables, field_free))
    notes.extend(f"masked {word!r}: {reason}" for word, reason in dropped)
    return CompileReport(levels=_LEVELS, notes=tuple(notes))


def _closed_compilation(variables: tuple[str, ...]) -> EqGPTCompiled:
    return EqGPTCompiled(
        masked_tokens=frozenset(),
        gate_variables=variables,
        pinned_keys=frozenset(),
        pinned_monomials=frozenset(),
        anchor_fingerprints=frozenset(),
        report=CompileReport(
            levels=_LEVELS,
            notes=(
                "closed sketch: token mask and reward-gate axes remain unchanged "
                "because the search result is discarded; lift from pins alone",
            ),
        ),
        dropped=(),
    )


def _dropped(
    vocab: Vocab, outside: tuple[int, ...], filtered: list[int]
) -> tuple[tuple[str, str], ...]:
    reasons = {token: _OUTSIDE_VOCABULARY for token in outside}
    reasons.update({token: _CONSTRAINT_FILTERED for token in filtered})
    terms = sorted(token for token in reasons if token >= FIRST_TERM_ID)
    operators = [token for token in _OPERATOR_FEATURES if token in reasons]
    return tuple((vocab.id2word[token], reasons[token]) for token in terms + operators)


def _pin_facts(sketch: Sketch) -> tuple[_Pins, frozenset[str]]:
    pins = _Pins(
        keys=frozenset(law_term_key(pin.term_ir) for pin in sketch.pinned),
        monomials=pinned_monomials(sketch),
        vocabulary=sketch.vocabulary,
    )
    pinned_axes = frozenset().union(
        *(
            _term_axes(analyze_term(pin.term_ir, sketch.vocabulary))
            for pin in sketch.pinned
        )
    )
    return pins, pinned_axes


def _base_checks(
    sketch: Sketch,
    anchors: tuple[TermFeatures, ...],
    grammar: _Grammar,
    vocab: Vocab,
) -> dict[ColumnFingerprint, int]:
    _assert_operator_clauses_generable(sketch, grammar)
    spellings = _anchor_spellings(sketch, anchors, grammar, vocab)
    _assert_anchors_not_redundant(frozenset(spellings), grammar, vocab)
    return spellings


def compile_for_eqgpt(
    sketch: Sketch,
    vocab: Vocab,
    *,
    base_masked: frozenset[int],
    variables: tuple[str, ...],
    max_length: int,
    start_words: tuple[str, ...],
) -> EqGPTCompiled:
    if not sketch.anchored and not sketch.holes:
        return _closed_compilation(variables)
    anchors = tuple(
        analyze_term(item.term_ir, sketch.vocabulary) for item in sketch.anchored
    )
    pins, pinned_axes = _pin_facts(sketch)
    gate_variables = tuple(axis for axis in variables if axis not in pinned_axes)
    grammar = _grammar(sketch, vocab, base_masked)
    spellings = _base_checks(sketch, anchors, grammar, vocab)
    kept, kept_operators, filtered = _sketch_mask(sketch, anchors, grammar)
    masked = frozenset((*grammar.outside, *filtered))
    witness_lengths = _assert_required_holes_live(
        sketch, kept, kept_operators, pins, vocab
    )
    _assert_capacity(sketch, spellings, witness_lengths, max_length, len(start_words))
    _assert_term_token_left(kept, kept_operators, pins, vocab)
    _assert_operator_position_live(base_masked, masked, start_words, max_length, vocab)
    carriers = _axis_carriers(gate_variables, kept, start_words, vocab)
    base_carriers = _axis_carriers(gate_variables, grammar.atoms, start_words, vocab)
    _assert_gate_axes_carried(carriers, base_carriers, kept, start_words, vocab)
    dropped = _dropped(vocab, grammar.outside, filtered)
    field_free = _field_free_notes(carriers, kept, sketch, grammar, vocab)
    return EqGPTCompiled(
        masked_tokens=masked,
        gate_variables=gate_variables,
        pinned_keys=pins.keys,
        pinned_monomials=pins.monomials,
        anchor_fingerprints=frozenset(spellings),
        report=_report(sketch, pinned_axes, gate_variables, field_free, dropped),
        dropped=dropped,
    )


def refuse_candidate(
    candidate: str, sketch: Sketch, compiled: EqGPTCompiled
) -> str | None:
    terms = candidate.split(" + ") if candidate else []
    for term in terms:
        reason = term_admission(
            term,
            sketch,
            pinned_keys=compiled.pinned_keys,
            pinned_monomials=compiled.pinned_monomials,
            anchor_fingerprints=compiled.anchor_fingerprints,
        )
        if reason is not None:
            return reason
    return None


def fit_candidate(candidate: str, sketch: Sketch) -> str:
    terms = candidate.split(" + ") if candidate else []
    kept = anchor_basis(terms, sketch)
    if len(kept) == len(terms):
        return candidate
    return " + ".join(terms[index] for index in kept)


__all__ = ["EqGPTCompiled", "compile_for_eqgpt", "fit_candidate", "refuse_candidate"]
