
from __future__ import annotations

import pytest
import sympy
from sympy import Float, Symbol

from kd.core.expr.sympy_bridge import to_sympy


from kd.search.pysr.convert import (
    build_feature_names,
    pysr_sympy_to_kd_terms,
)

pytestmark = pytest.mark.unit


def _is_valid_identifier(name: str) -> bool:
    return name.isidentifier() and name[0].isalpha() and name.isalnum()


def _kd_equivalent_to(kd_ir: str, expected_sympy: sympy.Expr) -> bool:
    return bool(sympy.expand(to_sympy(kd_ir) - expected_sympy) == 0)







class TestBuildFeatureNamesBasic:

    @pytest.mark.smoke
    def test_returns_one_name_per_term(self) -> None:
        names = build_feature_names(["u", "u_x", "u_xx"])
        assert len(names) == 3

    def test_empty_terms_returns_empty(self) -> None:
        assert build_feature_names([]) == []

    def test_names_are_unique(self) -> None:
        names = build_feature_names(["u", "u_x", "u_xx", "mul(u, u_x)"])
        assert len(set(names)) == len(names)

    def test_names_are_valid_identifiers(self) -> None:
        names = build_feature_names(["u", "u_x", "n2(u_x)", "mul(u, u_x)"])
        for name in names:
            assert _is_valid_identifier(name), f"{name!r} is not a valid identifier"

    def test_default_prefix_is_c(self) -> None:
        names = build_feature_names(["u", "u_x"])
        assert names == ["c0", "c1"]







class TestBuildFeatureNamesCollisions:

    def test_no_overlap_with_term_free_symbols(self) -> None:
        terms = ["u", "u_x", "mul(u, u_x)", "n2(u_x)"]
        names = build_feature_names(terms)
        term_symbols: set[str] = set()
        for t in terms:
            term_symbols |= {s.name for s in to_sympy(t).free_symbols}
        assert set(names).isdisjoint(term_symbols)

    def test_no_overlap_with_reserved(self) -> None:
        reserved = ["c0", "c1", "x", "t"]
        names = build_feature_names(["u", "u_x"], reserved=reserved)
        assert set(names).isdisjoint(set(reserved))

    def test_column_symbol_named_c0_forces_prefix_upgrade(self) -> None:
        names = build_feature_names(["c0"])
        assert len(names) == 1
        assert names[0] != "c0"

        term_symbols = {s.name for s in to_sympy("c0").free_symbols}
        assert names[0] not in term_symbols

    def test_collision_with_multiple_c_prefixed_terms(self) -> None:
        terms = ["c0", "c1", "c2"]
        names = build_feature_names(terms)
        assert len(names) == 3
        assert len(set(names)) == 3
        term_symbols: set[str] = set()
        for t in terms:
            term_symbols |= {s.name for s in to_sympy(t).free_symbols}
        assert set(names).isdisjoint(term_symbols)

    def test_diff_term_symbol_name_avoided(self) -> None:
        terms = ["diff_x(u)", "u"]
        names = build_feature_names(terms)
        term_symbols: set[str] = set()
        for t in terms:
            term_symbols |= {s.name for s in to_sympy(t).free_symbols}
        assert set(names).isdisjoint(term_symbols)







class TestPysrSympyToKdTermsBasic:

    @pytest.mark.smoke
    def test_linear_two_terms(self) -> None:
        terms = ["u", "u_x"]
        feats = ["c0", "c1"]
        pysr_expr = Symbol("c0") + Symbol("c1")
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 2
        recovered = {tuple(sorted(to_sympy(r).free_symbols, key=str)) for r in result}
        u = (sympy.Symbol("u"),)
        u_x = (sympy.Symbol("u_x"),)
        assert u in recovered
        assert u_x in recovered

    def test_numeric_coefficient_is_stripped(self) -> None:
        terms = ["u"]
        feats = ["c0"]
        pysr_expr = Float(2.5) * Symbol("c0")
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 1
        assert _kd_equivalent_to(result[0], sympy.Symbol("u"))

    def test_product_of_features(self) -> None:
        terms = ["u", "u_x"]
        feats = ["c0", "c1"]
        pysr_expr = Symbol("c0") * Symbol("c1")
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 1
        expected = sympy.Symbol("u") * sympy.Symbol("u_x")
        assert _kd_equivalent_to(result[0], expected)

    def test_square_of_feature(self) -> None:
        terms = ["u_x"]
        feats = ["c0"]
        pysr_expr = Symbol("c0") ** 2
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 1
        expected = sympy.Symbol("u_x") ** 2
        assert _kd_equivalent_to(result[0], expected)

    def test_compound_term_squared(self) -> None:
        terms = ["mul(u, u_x)"]
        feats = ["c0"]
        pysr_expr = Symbol("c0") ** 2
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 1
        expected = (sympy.Symbol("u") * sympy.Symbol("u_x")) ** 2
        assert _kd_equivalent_to(result[0], expected)

    def test_unary_function_of_feature(self) -> None:
        terms = ["u"]
        feats = ["c0"]
        pysr_expr = sympy.sin(Symbol("c0"))
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 1
        expected = sympy.sin(sympy.Symbol("u"))
        assert _kd_equivalent_to(result[0], expected)

    def test_pure_constant_yields_no_terms(self) -> None:
        terms = ["u"]
        feats = ["c0"]
        pysr_expr = Float(3.0)
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert result == []

    def test_structural_plus_trailing_constant_drops_constant(self) -> None:
        terms = ["u"]
        feats = ["c0"]
        pysr_expr = Float(2.5) * Symbol("c0") + Float(0.3)
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 1
        assert _kd_equivalent_to(result[0], sympy.Symbol("u"))

        assert all(to_sympy(r) != sympy.Integer(1) for r in result)

    def test_two_features_plus_constant_keeps_only_structural(self) -> None:
        terms = ["u", "u_x"]
        feats = ["c0", "c1"]
        pysr_expr = Symbol("c0") + Symbol("c1") + Float(0.5)
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 2
        recovered = {tuple(sorted(to_sympy(r).free_symbols, key=str)) for r in result}
        assert (sympy.Symbol("u"),) in recovered
        assert (sympy.Symbol("u_x"),) in recovered
        assert all(to_sympy(r) != sympy.Integer(1) for r in result)







class TestPysrSympyToKdTermsCombined:

    def test_mixed_linear_and_nonlinear(self) -> None:
        terms = ["u", "u_x", "u_xx"]
        feats = ["c0", "c1", "c2"]
        pysr_expr = Float(-1.0) * Symbol("c0") * Symbol("c1") + Float(0.1) * Symbol(
            "c2"
        )
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 2
        u_ux = sympy.Symbol("u") * sympy.Symbol("u_x")
        u_xx = sympy.Symbol("u_xx")
        matched_product = any(_kd_equivalent_to(r, u_ux) for r in result)
        matched_linear = any(_kd_equivalent_to(r, u_xx) for r in result)
        assert matched_product
        assert matched_linear

    def test_duplicate_structural_terms_deduplicated(self) -> None:
        terms = ["u"]
        feats = ["c0"]
        pysr_expr = Float(2.0) * Symbol("c0") + Float(3.0) * Symbol("c0")
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)



        assert len(result) == len({to_sympy(r) for r in result})
        assert any(_kd_equivalent_to(r, sympy.Symbol("u")) for r in result)

    def test_result_terms_are_coefficient_free(self) -> None:
        terms = ["u", "u_x"]
        feats = ["c0", "c1"]
        pysr_expr = Float(7.0) * Symbol("c0") + Float(-2.0) * Symbol("c1")
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        for r in result:
            coeff, _rest = to_sympy(r).as_coeff_Mul()
            assert coeff == sympy.Integer(1), f"{r!r} still carries coefficient {coeff}"







class TestPysrSympyToKdTermsFailures:

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            pysr_sympy_to_kd_terms(Symbol("c0"), ["u", "u_x"], ["c0"])

    def test_length_mismatch_too_many_features_raises(self) -> None:
        with pytest.raises(ValueError):
            pysr_sympy_to_kd_terms(Symbol("c0"), ["u"], ["c0", "c1"])

    def test_residual_feature_symbol_raises(self) -> None:
        with pytest.raises(ValueError):
            pysr_sympy_to_kd_terms(
                Symbol("c0") + Symbol("c2"),
                ["u", "u_x"],
                ["c0", "c1"],
            )

    def test_sqrt_of_feature_raises(self) -> None:
        with pytest.raises(ValueError):
            pysr_sympy_to_kd_terms(sympy.sqrt(Symbol("c0")), ["u"], ["c0"])

    def test_abs_of_feature_raises(self) -> None:
        with pytest.raises(ValueError):
            pysr_sympy_to_kd_terms(sympy.Abs(Symbol("c0")), ["u"], ["c0"])

    def test_rational_power_raises(self) -> None:
        with pytest.raises(ValueError):
            pysr_sympy_to_kd_terms(Symbol("c0") ** sympy.Rational(1, 2), ["u"], ["c0"])

    def test_unknown_function_raises(self) -> None:
        weird = sympy.Function("weird")
        with pytest.raises(ValueError):
            pysr_sympy_to_kd_terms(weird(Symbol("c0")), ["u"], ["c0"])







class TestPysrSympyToKdTermsRoundTrip:

    def test_identity_substitution_recovers_terms(self) -> None:
        terms = ["u", "u_x", "u_xx"]
        feats = build_feature_names(terms)
        pysr_expr = sum((Symbol(f) for f in feats), sympy.Integer(0))
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 3
        for original in terms:
            expected = to_sympy(original)
            assert any(_kd_equivalent_to(r, expected) for r in result), (
                f"term {original!r} not recovered in {result!r}"
            )

    def test_build_feature_names_integrates_with_subs(self) -> None:
        terms = ["c0", "u_x"]
        feats = build_feature_names(terms)
        pysr_expr = Symbol(feats[0]) * Symbol(feats[1])
        result = pysr_sympy_to_kd_terms(pysr_expr, terms, feats)
        assert len(result) == 1
        expected = sympy.Symbol("c0") * sympy.Symbol("u_x")
        assert _kd_equivalent_to(result[0], expected)
