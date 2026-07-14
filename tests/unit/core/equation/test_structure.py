
from __future__ import annotations

import pytest

from kd.core.equation import Form, LhsSpec, Scalar, make_evolution
from kd.core.equation.structure import (
    StructureFingerprint,
    TermDiff,
    structure,
    term_diff,
)

LHS_UT = LhsSpec(field="u", axis="t", order=1)
LHS_UTT = LhsSpec(field="u", axis="t", order=2)


def _evolution(term_irs: tuple[str, ...], lhs: LhsSpec = LHS_UT):
    return make_evolution(
        lhs, tuple((term_ir, Scalar(1.0)) for term_ir in term_irs)
    )







class TestStructure:
    @pytest.mark.smoke
    def test_fingerprint_fields(self) -> None:
        eq = _evolution(("diff2_x(u)", "mul(u, diff_x(u))"))
        fp = structure(eq)
        assert isinstance(fp, StructureFingerprint)
        assert fp.form is Form.EVOLUTION
        assert fp.lhs_spec == LHS_UT
        assert fp.terms == frozenset({"diff2_x(u)", "mul(diff_x(u),u)"})

    def test_terms_are_canonicalized(self) -> None:
        fp = structure(_evolution(("add(b, a)",)))
        assert fp.terms == frozenset({"add(a,b)"})

    def test_coefficient_independent(self) -> None:
        lhs = LHS_UT
        eq_a = make_evolution(lhs, (("u", Scalar(0.5)), ("diff_x(u)", Scalar(-3.0))))
        eq_b = make_evolution(lhs, (("u", Scalar(99.0)), ("diff_x(u)", Scalar(0.01))))
        assert structure(eq_a) == structure(eq_b)

    def test_term_order_independent(self) -> None:
        assert structure(_evolution(("u", "diff_x(u)"))) == structure(
            _evolution(("diff_x(u)", "u"))
        )

    def test_duplicate_canonical_terms_collapse(self) -> None:
        fp = structure(_evolution(("add(a, b)", "add(b, a)")))
        assert fp.terms == frozenset({"add(a,b)"})

    def test_lhs_included_in_fingerprint(self) -> None:
        eq_ut = _evolution(("diff2_x(u)",), lhs=LHS_UT)
        eq_utt = _evolution(("diff2_x(u)",), lhs=LHS_UTT)
        assert structure(eq_ut) != structure(eq_utt)

    def test_fingerprint_hashable(self) -> None:
        fp = structure(_evolution(("u",)))
        assert fp in {fp}

    def test_fingerprint_frozen(self) -> None:
        fp = structure(_evolution(("u",)))
        with pytest.raises(AttributeError):
            fp.terms = frozenset()

    def test_non_canonicalizable_term_names_index_and_term(self) -> None:
        eq = _evolution(("u", "mul(0.5, diff_x(u))"))
        with pytest.raises(ValueError) as excinfo:
            structure(eq)
        message = str(excinfo.value)
        assert "index 1" in message
        assert "mul(0.5, diff_x(u))" in message







class TestTermDiff:
    @pytest.mark.smoke
    def test_identical_structure_empty_diff(self) -> None:
        old = _evolution(("mul(u, diff_x(u))", "diff2_x(u)"))
        new = _evolution(("diff2_x(u)", "mul(diff_x(u),u)"))
        d = term_diff(old, new)
        assert isinstance(d, TermDiff)
        assert d.added == frozenset()
        assert d.removed == frozenset()
        assert d.common == frozenset({"mul(diff_x(u),u)", "diff2_x(u)"})
        assert d.lhs_changed is False
        assert d.form_changed is False

    def test_orientation_old_to_new(self) -> None:
        old = _evolution(("u", "div(u, x)"))
        new = _evolution(("u", "diff2_x(u)"))
        d = term_diff(old, new)
        assert d.added == frozenset({"diff2_x(u)"})
        assert d.removed == frozenset({"div(u,x)"})
        assert d.common == frozenset({"u"})

    def test_antisymmetry(self) -> None:
        a = _evolution(("u", "div(u, x)"))
        b = _evolution(("u", "diff2_x(u)"))
        assert term_diff(a, b).added == term_diff(b, a).removed
        assert term_diff(a, b).removed == term_diff(b, a).added

    def test_lhs_changed_flag(self) -> None:
        d = term_diff(
            _evolution(("diff2_x(u)",), lhs=LHS_UT),
            _evolution(("diff2_x(u)",), lhs=LHS_UTT),
        )
        assert d.added == frozenset()
        assert d.removed == frozenset()
        assert d.lhs_changed is True
        assert d.form_changed is False

    def test_diff_frozen(self) -> None:
        d = term_diff(_evolution(("u",)), _evolution(("u",)))
        with pytest.raises(AttributeError):
            d.added = frozenset()

    def test_non_canonicalizable_term_propagates(self) -> None:
        good = _evolution(("u",))
        bad = _evolution(("mul(2.0, u)",))
        with pytest.raises(ValueError):
            term_diff(good, bad)







def test_package_root_exports() -> None:
    import kd.core.equation as equation_pkg

    for name in ("structure", "term_diff", "StructureFingerprint", "TermDiff"):
        assert name in equation_pkg.__all__
        assert getattr(equation_pkg, name) is not None
