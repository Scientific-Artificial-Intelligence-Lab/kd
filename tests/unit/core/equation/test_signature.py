
from __future__ import annotations

import dataclasses
import json
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kd.core.equation import Form, LhsSpec, Scalar, make_evolution, make_homogeneous
from kd.core.equation.signature import (
    LAWSIG_DOMAIN,
    LawSignature,
    compare_laws,
    law_signature,
    law_signature_from_evidence,
)
from kd.search.records import EvidenceRecord

_LHS = LhsSpec(field="u", axis="t", order=1)


def _evolution(terms: list[tuple[str, float]], *, lhs: LhsSpec = _LHS, active=None):
    return make_evolution(
        lhs,
        [(ir, Scalar(value)) for ir, value in terms],
        active_indices=active,
    )


def _homogeneous(terms: list[tuple[str, float]], *, active=None):
    return make_homogeneous(
        tuple((ir, Scalar(value)) for ir, value in terms),
        active_indices=active,
    )







class TestSignatureIdentity:
    @pytest.mark.unit
    def test_domain_constant(self) -> None:
        assert LAWSIG_DOMAIN == "kd-lawsig-v1"

    @pytest.mark.unit
    def test_signature_shape(self) -> None:
        sig = law_signature(_evolution([("u_xx", 3.0)]))

        assert isinstance(sig, LawSignature)
        assert sig.version == LAWSIG_DOMAIN
        assert len(sig.structure_key) == 16
        assert int(sig.structure_key, 16) >= 0
        assert sig.terms == tuple(sorted(sig.terms))
        assert len(sig.terms) == len(sig.coefficients)
        assert sig.native_form is Form.EVOLUTION
        assert sig.native_lhs == _LHS

    @pytest.mark.unit
    def test_coefficients_unit_l2_and_sign_anchored(self) -> None:
        sig = law_signature(_evolution([("u_xx", 3.0)]))

        norm = math.sqrt(sum(c * c for c in sig.coefficients))
        assert norm == pytest.approx(1.0)
        anchor = max(sig.coefficients, key=abs)
        assert anchor > 0

    @pytest.mark.unit
    def test_homogeneous_native_metadata(self) -> None:
        sig = law_signature(_homogeneous([("u_xx", 1.0), ("one", -2.0)]))

        assert sig.native_form is Form.HOMOGENEOUS
        assert sig.native_lhs is None

    @pytest.mark.unit
    def test_to_dict_json_safe_and_faithful(self) -> None:
        sig = law_signature(_evolution([("u_xx", 3.0), ("u_x", -1.0)]))

        payload = json.dumps(sig.to_dict(), allow_nan=False)
        decoded = json.loads(payload)
        assert decoded["version"] == LAWSIG_DOMAIN
        assert decoded["structure_key"] == sig.structure_key
        assert tuple(decoded["terms"]) == sig.terms







class TestActiveLawKeying:
    @pytest.mark.unit
    def test_same_catalog_different_support_split(self) -> None:
        catalog = [("u", 0.5), ("u_x", -1.0), ("u_xx", 0.1), ("mul(u, u_x)", -2.0)]
        a = _evolution(catalog, active=(2,))
        b = _evolution(catalog, active=(1, 3))

        assert law_signature(a).structure_key != law_signature(b).structure_key

    @pytest.mark.unit
    def test_sparse_pick_equals_hand_built_law(self) -> None:
        catalog = [("u", 0.5), ("u_x", -1.0), ("u_xx", 0.1), ("mul(u, u_x)", -2.0)]
        picked = law_signature(_evolution(catalog, active=(2,)))
        direct = law_signature(_evolution([("u_xx", 0.1)]))

        assert picked.structure_key == direct.structure_key
        assert picked.coefficients == pytest.approx(direct.coefficients)







class TestCrossFormUnification:
    @pytest.mark.unit
    def test_evolution_matches_homogeneous_f0_rewrite(self) -> None:
        evo = law_signature(_evolution([("u_xx", 3.0)]))
        hom = law_signature(_homogeneous([("u_t", 1.0), ("u_xx", -3.0)]))

        assert evo.structure_key == hom.structure_key
        assert evo.coefficients == pytest.approx(hom.coefficients)

    @pytest.mark.unit
    def test_different_lhs_order_splits(self) -> None:
        first = law_signature(_evolution([("u_xx", 1.0)], lhs=_LHS))
        second = law_signature(
            _evolution([("u_xx", 1.0)], lhs=LhsSpec(field="u", axis="t", order=2))
        )

        assert first.structure_key != second.structure_key

    @pytest.mark.unit
    def test_f0_duplicate_canonical_term_raises(self) -> None:
        eq = _evolution([("u_t", 0.5), ("u_xx", 1.0)])

        with pytest.raises(ValueError, match="duplicate|collid"):
            law_signature(eq)







class TestAliasUnification:
    @pytest.mark.unit
    def test_div_and_mul_recip_spellings_merge(self) -> None:
        a = law_signature(_evolution([("div(u_x, x)", -1.0)]))
        b = law_signature(_evolution([("mul(u_x, recip(x))", -1.0)]))

        assert a.structure_key == b.structure_key

    @pytest.mark.unit
    def test_legacy_div_one_unifies_with_recip(self) -> None:
        legacy = law_signature(_evolution([("div(1.0, x)", 2.0)]))
        modern = law_signature(_evolution([("recip(x)", 2.0)]))

        assert legacy.structure_key == modern.structure_key

    @pytest.mark.unit
    def test_top_level_neg_moves_sign_into_coefficient(self) -> None:
        neg_term = law_signature(_evolution([("neg(u_xx)", 1.0)]))
        plain = law_signature(_evolution([("u_xx", -1.0)]))

        assert neg_term.structure_key == plain.structure_key
        assert neg_term.coefficients == pytest.approx(plain.coefficients)

    @pytest.mark.unit
    def test_non_canonicalizable_term_raises_never_drops(self) -> None:
        eq = _evolution([("mul(2.0, u)", 1.0), ("u_xx", 0.5)])

        with pytest.raises(ValueError):
            law_signature(eq)







class TestNormalizationInvariance:
    @pytest.mark.unit
    def test_term_order_is_irrelevant(self) -> None:
        a = law_signature(_evolution([("u_x", -1.0), ("u_xx", 0.1)]))
        b = law_signature(_evolution([("u_xx", 0.1), ("u_x", -1.0)]))

        assert a.structure_key == b.structure_key
        assert a.coefficients == pytest.approx(b.coefficients)

    @pytest.mark.unit
    @given(
        scale=st.floats(
            min_value=1e-3,
            max_value=1e3,
            allow_nan=False,
            allow_infinity=False,
        ),
        negate=st.booleans(),
    )
    def test_global_rescale_invariance(self, scale: float, negate: bool) -> None:
        factor = -scale if negate else scale
        base = law_signature(_homogeneous([("u_t", 1.0), ("u_xx", -3.0)]))
        scaled = law_signature(
            _homogeneous([("u_t", 1.0 * factor), ("u_xx", -3.0 * factor)])
        )

        assert scaled.structure_key == base.structure_key
        assert scaled.coefficients == pytest.approx(base.coefficients)







def _record(expression: str, equation_payload: dict | None) -> EvidenceRecord:
    return EvidenceRecord(
        instrument="sga",
        dataset_name="burgers",
        dataset_cache_fingerprint="abc",
        seed=0,
        is_valid=True,
        expression=expression,
        score_kind="NMSE",
        score_direction="min",
        headline_coefficient_source="native",
        catalog_fit=equation_payload,
    )


class TestSignatureFromEvidence:
    @pytest.mark.unit
    def test_expression_is_never_part_of_identity(self) -> None:
        from kd.core.equation import to_dict

        payload = to_dict(_evolution([("u_xx", 3.0)]))
        a = law_signature_from_evidence(_record("u_t = 3*u_xx", payload))
        b = law_signature_from_evidence(_record("whatever display", payload))

        assert a is not None and b is not None
        assert a.structure_key == b.structure_key
        assert a.coefficients == pytest.approx(b.coefficients)

    @pytest.mark.unit
    def test_missing_equation_payload_yields_none(self) -> None:
        assert law_signature_from_evidence(_record("u_t = u_xx", None)) is None







class TestCompareLaws:
    @pytest.mark.unit
    def test_same_law_different_spelling_structure_true(self) -> None:
        a = law_signature(_evolution([("div(u_x, x)", -1.0)]))
        b = law_signature(_evolution([("mul(u_x, recip(x))", -1.0)]))

        agreement = compare_laws(a, b)

        assert agreement.structure is True
        assert agreement.support is True
        assert agreement.coefficient is True
        assert not hasattr(agreement, "empirical")

    @pytest.mark.unit
    def test_cross_form_structure_true_support_false(self) -> None:
        evo = law_signature(_evolution([("u_xx", 3.0)]))
        hom = law_signature(_homogeneous([("u_t", 1.0), ("u_xx", -3.0)]))

        agreement = compare_laws(evo, hom)

        assert agreement.structure is True
        assert agreement.support is False

    @pytest.mark.unit
    def test_coefficient_axis_none_when_structure_differs(self) -> None:
        a = law_signature(_evolution([("u_xx", 1.0)]))
        b = law_signature(_evolution([("u_x", 1.0)]))

        agreement = compare_laws(a, b)

        assert agreement.structure is False
        assert agreement.coefficient is None
        assert agreement.max_abs_delta is None

    @pytest.mark.unit
    def test_coefficient_axis_tolerance(self) -> None:
        a = law_signature(_evolution([("u_xx", 3.0)]))
        near = law_signature(_evolution([("u_xx", 3.001)]))
        far = law_signature(_evolution([("u_xx", 30.0)]))

        near_agreement = compare_laws(a, near)
        far_agreement = compare_laws(a, far)

        assert near_agreement.coefficient is True
        assert near_agreement.max_abs_delta is not None
        assert near_agreement.max_abs_delta < 1e-2
        assert far_agreement.structure is True
        assert far_agreement.coefficient is False







class TestLhsAxisInjectivity:

    @pytest.mark.unit
    def test_multi_letter_axis_refuses_to_sign(self) -> None:
        ambiguous = _evolution(
            [("u", 1.0)], lhs=LhsSpec(field="u", axis="xx", order=1)
        )

        with pytest.raises(ValueError, match="axis"):
            law_signature(ambiguous)

    @pytest.mark.unit
    def test_single_letter_collision_partner_still_signs(self) -> None:
        legal = _evolution(
            [("u", 1.0)], lhs=LhsSpec(field="u", axis="x", order=2)
        )

        assert "u_xx" in law_signature(legal).terms


class TestCoefficientCompareModuloSign:

    @pytest.mark.unit
    def test_pure_global_sign_flip_is_not_a_coefficient_split(self) -> None:
        base = law_signature(_evolution([("u_xx", 2.0), ("u_x", -1.0)]))
        flipped = dataclasses.replace(
            base,
            coefficients=tuple(-value for value in base.coefficients),
        )

        agreement = compare_laws(base, flipped)

        assert agreement.coefficient is True
        assert agreement.max_abs_delta == 0.0
