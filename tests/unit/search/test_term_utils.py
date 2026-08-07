
from __future__ import annotations

import pytest

from kd.search.term_utils import fold_add, infer_max_atomic_order


@pytest.mark.parametrize(
    ("terms", "expected"),
    [
        pytest.param(["u"], "u", id="single"),
        pytest.param(
            ["u", "u_x", "u_xx"],
            "add(u, add(u_x, u_xx))",
            id="right-folded-multiple",
        ),
    ],
)
def test_fold_add_preserves_term_order(terms: list[str], expected: str) -> None:
    assert fold_add(terms) == expected


def test_fold_add_rejects_empty_term_list() -> None:
    with pytest.raises(ValueError, match="at least one term"):
        fold_add([])


@pytest.mark.parametrize(
    ("terms", "expected"),
    [
        pytest.param(["u", "mul(u,v)"], 0, id="derivative-free"),
        pytest.param(["u_x", "mul(u,u_xx)"], 2, id="simple"),
        pytest.param(
            ["add(u_xxx,mul(v_xy,u))", "w_yy"],
            3,
            id="maximum-over-all-symbols",
        ),
    ],
)
def test_infer_max_atomic_order(terms: list[str], expected: int) -> None:
    assert infer_max_atomic_order(terms) == expected
