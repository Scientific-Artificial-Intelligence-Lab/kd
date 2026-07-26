
from __future__ import annotations

import pytest

from kd.core.equation import (
    EquationAttrs,
    Evolution,
    Homogeneous,
    LhsSpec,
    Scalar,
    active_law,
    make_evolution,
    make_homogeneous,
)

_LHS = LhsSpec(field="u", axis="t", order=1)


def _evolution_4term(active_indices: tuple[int, ...] | None) -> Evolution:
    return make_evolution(
        _LHS,
        [
            ("u", Scalar(0.5)),
            ("u_x", Scalar(-1.0)),
            ("diff2_x(u)", Scalar(0.1)),
            ("mul(u, u_x)", Scalar(-2.0)),
        ],
        active_indices=active_indices,
    )


class TestActiveLawEvolution:
    @pytest.mark.unit
    def test_projects_to_selected_terms_only(self) -> None:
        eq = _evolution_4term((2,))

        law = active_law(eq)

        assert isinstance(law, Evolution)
        assert law.terms == (("diff2_x(u)", Scalar(0.1)),)
        assert law.active_indices is None
        assert law.lhs_spec == _LHS

    @pytest.mark.unit
    def test_multi_term_projection_preserves_pairs(self) -> None:
        eq = _evolution_4term((1, 3))

        law = active_law(eq)

        assert law.terms == (("u_x", Scalar(-1.0)), ("mul(u, u_x)", Scalar(-2.0)))

    @pytest.mark.unit
    def test_dense_fit_is_identity(self) -> None:
        eq = _evolution_4term(None)

        assert active_law(eq) == eq

    @pytest.mark.unit
    def test_projection_is_idempotent(self) -> None:
        law = active_law(_evolution_4term((0, 2)))

        assert active_law(law) == law


class TestActiveLawHomogeneous:
    @pytest.mark.unit
    def test_projects_keeping_pivot(self) -> None:
        eq = make_homogeneous(
            (
                ("diff2_x(u)", Scalar(1.0)),
                ("diff2_y(u)", Scalar(0.0)),
                ("one", Scalar(-3.0)),
            ),
            active_indices=(0, 2),
        )

        law = active_law(eq)

        assert isinstance(law, Homogeneous)
        assert law.terms == (("diff2_x(u)", Scalar(1.0)), ("one", Scalar(-3.0)))
        assert law.active_indices is None

    @pytest.mark.unit
    def test_missing_pivot_raises(self) -> None:
        eq = make_homogeneous(
            (
                ("diff2_x(u)", Scalar(1.0)),
                ("diff2_y(u)", Scalar(0.5)),
                ("one", Scalar(-3.0)),
            ),
            active_indices=(1, 2),
        )

        with pytest.raises(ValueError, match="pivot"):
            active_law(eq)

    @pytest.mark.unit
    def test_dense_fit_is_identity(self) -> None:
        eq = make_homogeneous((("diff2_x(u)", Scalar(1.0)), ("one", Scalar(2.0))))

        assert active_law(eq) == eq


class TestActiveLawCorruptPicklist:

    @pytest.mark.unit
    def test_out_of_range_index_raises(self) -> None:
        eq = Evolution(
            lhs_spec=_LHS,
            terms=(("u_x", Scalar(1.0)),),
            attrs=EquationAttrs(),
            active_indices=(5,),
        )

        with pytest.raises(ValueError, match="range"):
            active_law(eq)

    @pytest.mark.unit
    def test_duplicate_index_raises(self) -> None:
        eq = Evolution(
            lhs_spec=_LHS,
            terms=(("u", Scalar(1.0)), ("u_x", Scalar(2.0))),
            attrs=EquationAttrs(),
            active_indices=(1, 1),
        )

        with pytest.raises(ValueError, match="duplicate"):
            active_law(eq)

    @pytest.mark.unit
    def test_empty_picklist_raises(self) -> None:

        eq = Evolution(
            lhs_spec=_LHS,
            terms=(("u_x", Scalar(1.0)),),
            attrs=EquationAttrs(),
            active_indices=(),
        )

        with pytest.raises(ValueError, match="empty"):
            active_law(eq)
