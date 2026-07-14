
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from kd.core.equation.canonical import canonicalize_expression
from kd.search.llm4ed.columns import evaluate_ir_column
from kd.search.llm4ed.parse import parse_equation
from kd.search.llm4ed.score import score_equation

FloatArray = npt.NDArray[np.float64]

_OPERANDS = ("x", "u_x", "u_xx", "u_xxx", "u")


def _features(n: int = 64, seed: int = 5) -> dict[str, FloatArray]:
    rng = np.random.default_rng(seed)

    return {
        "u": rng.standard_normal(n).astype(np.float64),
        "u_x": rng.standard_normal(n).astype(np.float64),
        "u_xx": rng.standard_normal(n).astype(np.float64),
        "u_xxx": rng.standard_normal(n).astype(np.float64),
        "x": (rng.standard_normal(n) + 3.0).astype(np.float64),
    }


class TestParseBoundaryFields:
    def test_base_ir_is_coefficient_free_and_canonical(self) -> None:
        parsed = parse_equation("0.5*u_xx + u*u_x + u^2", _OPERANDS)
        by_str = {t.term_str: t for t in parsed.terms}


        coef_term = by_str["0.5*u_xx"]
        assert coef_term.coeff == pytest.approx(0.5)
        assert canonicalize_expression(coef_term.base_ir) == "u_xx"

        assert coef_term.ir == "mul(0.5, u_xx)"


        assert by_str["u*u_x"].coeff == pytest.approx(1.0)
        assert canonicalize_expression(by_str["u*u_x"].base_ir) == "mul(u,u_x)"
        assert by_str["u^2"].coeff == pytest.approx(1.0)
        assert canonicalize_expression(by_str["u^2"].base_ir) == "n2(u)"

    def test_integer_expand_factor_is_split(self) -> None:


        (term,) = parse_equation("u*u_x + u_x*u", _OPERANDS).terms
        assert term.coeff == pytest.approx(2.0)
        assert canonicalize_expression(term.base_ir) == "mul(u,u_x)"

    def test_coeff_times_base_reconstructs_whole_term(self) -> None:
        feats = _features()
        for term in parse_equation("0.5*u_xx + 2*u*u_x + u^3", _OPERANDS).terms:
            whole = evaluate_ir_column(term.ir, feats)
            base = evaluate_ir_column(term.base_ir, feats)
            np.testing.assert_allclose(term.coeff * base, whole, rtol=1e-12)


class TestScoreBoundaryAlignment:
    def test_term_irs_canonicalize_and_stay_aligned(self) -> None:
        feats = _features()
        lhs = (2.0 * feats["u"] + 3.0 * feats["u_xx"]).reshape(-1, 1)
        score = score_equation("u + u_xx + u*u_x", lhs, feats, operands=_OPERANDS)

        assert score.valid and score.coefficients is not None
        n = len(score.term_strs)
        assert len(score.term_irs) == n
        assert len(score.term_coeffs) == n
        assert score.coefficients.reshape(-1).shape[0] == n
        for ir in score.term_irs:

            canonicalize_expression(ir)

    def test_boundary_folds_numeric_coefficient(self) -> None:
        feats = _features()
        lhs = (4.0 * feats["u_xx"]).reshape(-1, 1)


        score = score_equation("0.5*u_xx", lhs, feats, operands=_OPERANDS)

        assert score.valid and score.coefficients is not None
        assert score.term_coeffs == (pytest.approx(0.5),)
        assert canonicalize_expression(score.term_irs[0]) == "u_xx"
        raw = score.coefficients.reshape(-1)
        effective = raw * np.asarray(score.term_coeffs, dtype=np.float64)
        np.testing.assert_allclose(effective, [4.0], rtol=1e-6)


class TestReciprocalNowCanonicalizes:
    def test_pure_reciprocal_base_canonicalizes_via_recip(self) -> None:



        (term,) = parse_equation("u_x/x", _OPERANDS).terms
        assert term.coeff == pytest.approx(1.0)
        assert term.base_ir == "mul(u_x, recip(x))"
        assert canonicalize_expression(term.base_ir) == "mul(recip(x),u_x)"


class TestResidualException:
    def test_unexpandable_denominator_retains_internal_literals(self) -> None:




        (term,) = parse_equation("u_x/(u+1)", _OPERANDS).terms
        assert term.coeff == pytest.approx(1.0)
        assert term.base_ir == "mul(u_x, recip(add(1.0, u)))"
        with pytest.raises(ValueError, match="Numeric constants"):
            canonicalize_expression(term.base_ir)


class TestDedupOverwriteMapping:
    def test_survivor_maps_to_its_own_base_ir(self) -> None:





        n = 8
        ones = np.ones(n, dtype=np.float64)
        alt = np.array([1.0, -1.0] * (n // 2), dtype=np.float64)
        feats: dict[str, FloatArray] = {
            "u": ones,
            "u_x": alt,
            "u_xx": np.zeros(n, dtype=np.float64),
            "u_xxx": np.zeros(n, dtype=np.float64),
            "x": ones,
        }


        lhs = np.linspace(1.0, 2.0, n, dtype=np.float64).reshape(-1, 1)
        score = score_equation("u + u*u_x", lhs, feats, operands=_OPERANDS)

        assert score.valid and score.coefficients is not None
        assert score.term_strs == ("u",)
        assert score.term_irs == ("u",)
        assert score.term_coeffs == (pytest.approx(1.0),)
        assert score.coefficients.reshape(-1).shape[0] == 1
