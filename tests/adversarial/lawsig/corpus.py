
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kd.core.equation import Equation, LhsSpec, Scalar, make_evolution, make_homogeneous

COEFF_ATOL = 1e-9

KNOWN_SPLIT_FAMILIES = frozenset({"semantic-equivalence", "associativity"})




REQUIRED_FAMILIES = frozenset(
    {
        "alias",
        "associativity",
        "catalog-vs-law",
        "coefficient-axis",
        "commutativity",
        "cross-form",
        "global-rescale",
        "lhs-identity",
        "nested-const",
        "semantic-equivalence",
        "sign-anchor",
        "term-order",
        "zero-coefficient",
    }
)


class Expected(Enum):
    MERGE_EXACT = "MERGE_EXACT"
    MERGE_STRUCT = "MERGE_STRUCT"
    SPLIT = "SPLIT"
    RAISES = "RAISES"


@dataclass(frozen=True)
class CorpusPair:
    pair_id: str
    family: str
    expected: Expected
    rationale: str
    eq_a: Equation
    eq_b: Equation | None


_UT = LhsSpec(field="u", axis="t", order=1)
_UTT = LhsSpec(field="u", axis="t", order=2)
_UX_LHS = LhsSpec(field="u", axis="x", order=1)
_VT = LhsSpec(field="v", axis="t", order=1)


def _evo(
    terms: list[tuple[str, float]],
    *,
    lhs: LhsSpec = _UT,
    active: tuple[int, ...] | None = None,
) -> Equation:
    return make_evolution(
        lhs,
        [(ir, Scalar(value)) for ir, value in terms],
        active_indices=active,
    )


def _hom(terms: list[tuple[str, float]]) -> Equation:
    return make_homogeneous(tuple((ir, Scalar(value)) for ir, value in terms))


def _pair(
    pair_id: str,
    family: str,
    expected: Expected,
    rationale: str,
    eq_a: Equation,
    eq_b: Equation | None,
) -> CorpusPair:
    return CorpusPair(pair_id, family, expected, rationale, eq_a, eq_b)


_BURGERS_CATALOG: list[tuple[str, float]] = [
    ("u", 0.5),
    ("u_x", -1.0),
    ("u_xx", 0.1),
    ("mul(u, u_x)", -2.0),
]

CORPUS: tuple[CorpusPair, ...] = (

    _pair(
        "alias-div-vs-mul-recip",
        "alias",
        Expected.MERGE_EXACT,
        "binary div and mul-recip are two spellings of one term (D-3 table)",
        _evo([("div(u_x, x)", -1.0)]),
        _evo([("mul(u_x, recip(x))", -1.0)]),
    ),
    _pair(
        "alias-legacy-div-one",
        "alias",
        Expected.MERGE_EXACT,
        "pre-recip legacy div(1.0,x) must unify with recip(x)",
        _evo([("div(1.0, x)", 2.0)]),
        _evo([("recip(x)", 2.0)]),
    ),
    _pair(
        "alias-div-compound-numerator",
        "alias",
        Expected.MERGE_EXACT,
        "div rewrite applies with a compound numerator",
        _evo([("div(mul(u, u_x), x)", 1.5)]),
        _evo([("mul(mul(u, u_x), recip(x))", 1.5)]),
    ),
    _pair(
        "alias-div-composite-denominator",
        "alias",
        Expected.MERGE_EXACT,
        "div rewrite applies with a composite denominator",
        _evo([("div(u, mul(x, x))", 1.0)]),
        _evo([("mul(u, recip(mul(x, x)))", 1.0)]),
    ),
    _pair(
        "alias-neg-top-level",
        "alias",
        Expected.MERGE_EXACT,
        "top-level neg strips into the coefficient sign",
        _evo([("neg(u_xx)", 1.0)]),
        _evo([("u_xx", -1.0)]),
    ),
    _pair(
        "alias-double-neg",
        "alias",
        Expected.MERGE_EXACT,
        "iterated neg stripping: neg(neg(t)) == t",
        _evo([("neg(neg(u_xx))", 0.5)]),
        _evo([("u_xx", 0.5)]),
    ),
    _pair(
        "alias-neg-around-div",
        "alias",
        Expected.MERGE_EXACT,
        "neg outside a div rewrites and strips in one pass",
        _evo([("neg(div(u_x, x))", 1.0)]),
        _evo([("div(u_x, x)", -1.0)]),
    ),

    _pair(
        "commut-mul-swap",
        "commutativity",
        Expected.MERGE_EXACT,
        "canonicalizer sorts commutative mul args",
        _evo([("mul(u, u_x)", -2.0)]),
        _evo([("mul(u_x, u)", -2.0)]),
    ),
    _pair(
        "commut-add-swap",
        "commutativity",
        Expected.MERGE_EXACT,
        "canonicalizer sorts commutative add args",
        _evo([("add(u, u_x)", 1.0)]),
        _evo([("add(u_x, u)", 1.0)]),
    ),
    _pair(
        "commut-nested-same-shape",
        "commutativity",
        Expected.MERGE_EXACT,
        "commutative sort applies at every level of one nesting shape",
        _evo([("mul(u, mul(u_x, u_xx))", 1.0)]),
        _evo([("mul(mul(u_xx, u_x), u)", 1.0)]),
    ),

    _pair(
        "order-evolution-swap",
        "term-order",
        Expected.MERGE_EXACT,
        "term list order is presentation, not identity",
        _evo([("u_x", -1.0), ("u_xx", 0.1)]),
        _evo([("u_xx", 0.1), ("u_x", -1.0)]),
    ),
    _pair(
        "order-homogeneous-rotation",
        "term-order",
        Expected.MERGE_EXACT,
        "homogeneous pivot position is presentation; the F0 set is the law",
        _hom([("u_t", 1.0), ("u_xx", -1.0), ("mul(u, u_x)", 1.0)]),
        _hom([("u_xx", -1.0), ("mul(u, u_x)", 1.0), ("u_t", 1.0)]),
    ),

    _pair(
        "rescale-positive",
        "global-rescale",
        Expected.MERGE_EXACT,
        "homogeneous law is a ray: x2.5 is the same law",
        _hom([("u_t", 1.0), ("u_xx", -3.0)]),
        _hom([("u_t", 2.5), ("u_xx", -7.5)]),
    ),
    _pair(
        "rescale-negation",
        "global-rescale",
        Expected.MERGE_EXACT,
        "global negation is the same law",
        _hom([("u_t", 1.0), ("u_xx", -3.0)]),
        _hom([("u_t", -1.0), ("u_xx", 3.0)]),
    ),
    _pair(
        "rescale-small",
        "global-rescale",
        Expected.MERGE_EXACT,
        "x1e-3 rescale survives normalization",
        _hom([("u_t", 1.0), ("u_xx", -3.0)]),
        _hom([("u_t", 1e-3), ("u_xx", -3e-3)]),
    ),
    _pair(
        "rescale-denormal-range",
        "global-rescale",
        Expected.MERGE_EXACT,
        "1e-200 magnitudes must not underflow the normalization",
        _hom([("u_t", 1e-200), ("u_xx", -3e-200)]),
        _hom([("u_t", 1.0), ("u_xx", -3.0)]),
    ),
    _pair(
        "rescale-non-unit-pivot",
        "global-rescale",
        Expected.MERGE_EXACT,
        "make_homogeneous carries a non-unit pivot; the signature normalizes",
        _hom([("u_t", 2.0), ("u_xx", -6.0)]),
        _hom([("u_t", 1.0), ("u_xx", -3.0)]),
    ),

    _pair(
        "xform-simple",
        "cross-form",
        Expected.MERGE_EXACT,
        "u_t = 3*u_xx and its F0 homogeneous rewrite are one law",
        _evo([("u_xx", 3.0)]),
        _hom([("u_t", 1.0), ("u_xx", -3.0)]),
    ),
    _pair(
        "xform-burgers",
        "cross-form",
        Expected.MERGE_EXACT,
        "multi-term Burgers law across sentence forms",
        _evo([("mul(u, u_x)", -1.0), ("u_xx", 0.1)]),
        _hom([("u_t", 1.0), ("mul(u, u_x)", 1.0), ("u_xx", -0.1)]),
    ),
    _pair(
        "xform-wave-second-order",
        "cross-form",
        Expected.MERGE_EXACT,
        "second-order LHS token (u_tt) unifies through F0",
        _evo([("u_xx", 1.0)], lhs=_UTT),
        _hom([("u_tt", 1.0), ("u_xx", -1.0)]),
    ),
    _pair(
        "xform-pivot-choice",
        "cross-form",
        Expected.MERGE_EXACT,
        "which term the homogeneous producer made pivot is irrelevant",
        _hom([("u_xx", 3.0), ("u_t", -1.0)]),
        _evo([("u_xx", 3.0)]),
    ),




    _pair(
        "catalog-different-support-splits",
        "catalog-vs-law",
        Expected.SPLIT,
        "catalog-vs-law trap: same 4-term catalog, different selected support",
        _evo(_BURGERS_CATALOG, active=(2,)),
        _evo(_BURGERS_CATALOG, active=(1, 3)),
    ),
    _pair(
        "catalog-sparse-vs-dense-splits",
        "catalog-vs-law",
        Expected.SPLIT,
        "picking one column is not the same law as the dense 4-term fit",
        _evo(_BURGERS_CATALOG, active=(2,)),
        _evo(_BURGERS_CATALOG),
    ),
    _pair(
        "catalog-projection-equals-direct-law",
        "catalog-vs-law",
        Expected.MERGE_EXACT,
        "positive control: the pick IS the hand-built law",
        _evo(_BURGERS_CATALOG, active=(2,)),
        _evo([("u_xx", 0.1)]),
    ),
    _pair(
        "catalog-identity-is-irrelevant",
        "catalog-vs-law",
        Expected.MERGE_EXACT,
        "two different catalogs selecting the same active law merge",
        _evo(
            [("u", 0.5), ("u_x", -1.0), ("one", 3.0)],
            active=(0, 1),
        ),
        _evo(
            [("u", 0.5), ("u_x", -1.0), ("u_xx", 9.9)],
            active=(0, 1),
        ),
    ),
    _pair(
        "catalog-subset-support-splits",
        "catalog-vs-law",
        Expected.SPLIT,
        "dropping a term is a different law, however small its coefficient",
        _evo([("u_x", -1.0), ("u_xx", 0.1)]),
        _evo([("u_x", -1.0)]),
    ),

    _pair(
        "lhs-order-splits",
        "lhs-identity",
        Expected.SPLIT,
        "u_t = u_xx and u_tt = u_xx are different laws",
        _evo([("u_xx", 1.0)], lhs=_UT),
        _evo([("u_xx", 1.0)], lhs=_UTT),
    ),
    _pair(
        "lhs-axis-splits",
        "lhs-identity",
        Expected.SPLIT,
        "evolution along t vs along x is a different sentence",
        _evo([("u_xx", 1.0)], lhs=_UT),
        _evo([("u_xx", 1.0)], lhs=_UX_LHS),
    ),
    _pair(
        "lhs-field-splits",
        "lhs-identity",
        Expected.SPLIT,
        "v_t = v_xx is not u_t = v_xx's law",
        _evo([("diff2_x(v)", 1.0)], lhs=_UT),
        _evo([("diff2_x(v)", 1.0)], lhs=_VT),
    ),
    _pair(
        "lhs-multi-letter-axis-raises",
        "lhs-identity",
        Expected.RAISES,
        "axis 'xx' at order 1 renders as u_xx = the axis-x order-2 label; "
        "the ambiguous rendering must refuse to sign (S6/F1)",
        _evo([("u", 1.0)], lhs=LhsSpec(field="u", axis="xx", order=1)),
        None,
    ),

    _pair(
        "anchor-equal-magnitude-negation",
        "sign-anchor",
        Expected.MERGE_EXACT,
        "all-|1| vector: the tie-break must survive global negation",
        _hom([("u_t", 1.0), ("u_xx", -1.0)]),
        _hom([("u_t", -1.0), ("u_xx", 1.0)]),
    ),
    _pair(
        "anchor-near-tie-perturbation",
        "sign-anchor",
        Expected.MERGE_EXACT,
        "1e-12 magnitude perturbation must not flip the anchored sign",
        _hom([("u_t", 1.0), ("u_xx", -(1.0 + 1e-12))]),
        _hom([("u_t", -1.0), ("u_xx", 1.0 + 1e-12)]),
    ),
    _pair(
        "anchor-three-way-tie",
        "sign-anchor",
        Expected.MERGE_EXACT,
        "three equal magnitudes: lexicographic tie-break is deterministic",
        _hom([("u_t", 1.0), ("u_xx", -1.0), ("one", 1.0)]),
        _hom([("u_t", -1.0), ("u_xx", 1.0), ("one", -1.0)]),
    ),

    _pair(
        "coeff-strength-same-support",
        "coefficient-axis",
        Expected.MERGE_STRUCT,
        "u_t = 3*u_xx vs u_t = 6*u_xx: one support, two laws — the split "
        "belongs to the coefficient axis, not the structure key",
        _evo([("u_xx", 3.0)]),
        _evo([("u_xx", 6.0)]),
    ),
    _pair(
        "coeff-single-sign-flip",
        "coefficient-axis",
        Expected.MERGE_STRUCT,
        "flipping ONE coefficient is not a global negation: same structure, "
        "different normalized vector",
        _hom([("u_t", 1.0), ("u_xx", -3.0)]),
        _hom([("u_t", 1.0), ("u_xx", 3.0)]),
    ),

    _pair(
        "zero-coeff-term-is-still-support",
        "zero-coefficient",
        Expected.SPLIT,
        "v1 does no coefficient-threshold pruning: a 0.0-coefficient active "
        "term is still support",
        _evo([("u_xx", 0.1), ("u_x", 0.0)]),
        _evo([("u_xx", 0.1)]),
    ),
    _pair(
        "zero-coeff-normalization-robust",
        "zero-coefficient",
        Expected.MERGE_EXACT,
        "a zero entry among actives must not destabilize normalization",
        _evo([("u_xx", 0.1), ("u_x", 0.0)]),
        _evo([("u_x", 0.0), ("u_xx", 0.1)]),
    ),

    _pair(
        "const-mul-literal",
        "nested-const",
        Expected.RAISES,
        "numeric literal inside a term violates the canonical contract",
        _evo([("mul(2.0, u)", 1.0)]),
        None,
    ),
    _pair(
        "const-add-literal",
        "nested-const",
        Expected.RAISES,
        "additive constant inside a term fails loud, never silently merges",
        _evo([("add(1.0, u)", 1.0)]),
        None,
    ),
    _pair(
        "const-div-denominator",
        "nested-const",
        Expected.RAISES,
        "alias rewrite must not launder div(u, 2.0) into recip(2.0)",
        _evo([("div(u, 2.0)", 1.0)]),
        None,
    ),
    _pair(
        "const-raw-mul-one",
        "nested-const",
        Expected.RAISES,
        "mul(1.0, x) collapses ONLY when produced by the rewrite itself; raw "
        "input keeps the literal and fails",
        _evo([("mul(1.0, u)", 1.0)]),
        None,
    ),

    _pair(
        "sem-product-vs-named-power",
        "semantic-equivalence",
        Expected.MERGE_EXACT,
        "mul(u,u) == n2(u) semantically; v1 does no semantic rewriting "
        "(declared known-split)",
        _evo([("mul(u, u)", 1.0)]),
        _evo([("n2(u)", 1.0)]),
    ),
    _pair(
        "sem-sub-vs-add-neg",
        "semantic-equivalence",
        Expected.MERGE_EXACT,
        "sub(a,b) == add(a, neg(b)) semantically; inner neg is not stripped "
        "(declared known-split)",
        _evo([("sub(u_x, u_xx)", 1.0)]),
        _evo([("add(u_x, neg(u_xx))", 1.0)]),
    ),
    _pair(
        "sem-double-recip",
        "semantic-equivalence",
        Expected.MERGE_EXACT,
        "recip(recip(u)) == u semantically (declared known-split)",
        _evo([("recip(recip(u))", 1.0)]),
        _evo([("u", 1.0)]),
    ),
    _pair(
        "sem-self-sum-vs-scaled",
        "semantic-equivalence",
        Expected.MERGE_EXACT,
        "1*add(u,u) == 2*u semantically (declared known-split)",
        _evo([("add(u, u)", 1.0)]),
        _evo([("u", 2.0)]),
    ),



    _pair(
        "sem-neg-inside-factor",
        "semantic-equivalence",
        Expected.MERGE_EXACT,
        "+1*mul(neg(u), u_x) == -1*mul(u, u_x) semantically; only TOP-LEVEL "
        "neg moves into the coefficient, so the predecessor splits (declared)",
        _evo([("mul(neg(u), u_x)", 1.0)]),
        _evo([("mul(u, u_x)", -1.0)]),
    ),

    _pair(
        "assoc-mul-nesting",
        "associativity",
        Expected.MERGE_EXACT,
        "mul nesting shapes are one product semantically; the canonicalizer "
        "does not flatten (probed 2026-07-18; declared known-split)",
        _evo([("mul(u, mul(u_x, u_xx))", 1.0)]),
        _evo([("mul(mul(u, u_x), u_xx)", 1.0)]),
    ),
)
