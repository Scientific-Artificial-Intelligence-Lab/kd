
from __future__ import annotations

from dataclasses import dataclass

import pytest




pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from kd.core.equation.canonical import canonicalize_expression
from kd.core.expr.canonicalizer import canonicalize_code
from kd.core.expr.registry import FunctionRegistry
from kd.search.discover.utils.canonicalize import (
    canonicalize_expression as discover_alias,
)







SGA_POPULATION: list[tuple[str, str, str]] = [
    ("sga-plain", "mul(u, diff_x(u))", "mul(diff_x(u),u)"),
    ("sga-plain-swapped", "mul(diff_x(u), u)", "mul(diff_x(u),u)"),
    ("sga-nested-mul", "mul(mul(u, u), diff_x(u))", "mul(diff_x(u),mul(u,u))"),
    ("sga-add-nested", "add(mul(u, u), diff2_x(u))", "add(diff2_x(u),mul(u,u))"),
    ("sga-div", "div(diff_x(u), u)", "div(diff_x(u),u)"),
    (
        "sga-burgers",
        "sub(diff2_x(u), mul(diff_x(u), u))",
        "sub(diff2_x(u),mul(diff_x(u),u))",
    ),
    (
        "sga-burgers-swapped",
        "sub(diff2_x(u), mul(u, diff_x(u)))",
        "sub(diff2_x(u),mul(diff_x(u),u))",
    ),
    (
        "sga-deep",
        "add(mul(diff_x(u), mul(u, u)), div(u, diff2_x(u)))",
        "add(div(u,diff2_x(u)),mul(diff_x(u),mul(u,u)))",
    ),
]



DISCOVER_POPULATION: list[tuple[str, str, str]] = [
    (
        "disc-golden-b42",
        "sub(mul(diff_x(u),u),diff2_x(u))",
        "sub(mul(diff_x(u),u),diff2_x(u))",
    ),
    (
        "disc-golden-b123",
        "add(diff2_x(u),mul(diff_x(u),u))",
        "add(diff2_x(u),mul(diff_x(u),u))",
    ),
    (
        "disc-golden-c42",
        "add(add(n3(u),u),diff2_x(u))",
        "add(add(n3(u),u),diff2_x(u))",
    ),
    (
        "disc-golden-c123",
        "add(div(mul(sub(diff2_x(u),u),u),u),n3(u))",
        "add(div(mul(sub(diff2_x(u),u),u),u),n3(u))",
    ),
    (
        "disc-golden-c777",
        "sub(u,sub(mul(n2(u),u),diff2_x(u)))",
        "sub(u,sub(mul(n2(u),u),diff2_x(u)))",
    ),
    ("disc-neg-term", "neg(diff2_x(u))", "neg(diff2_x(u))"),
    ("disc-term-n2", "mul(u, n2(u))", "mul(n2(u),u)"),
]

VALID_CORPUS: list[tuple[str, str, str]] = SGA_POPULATION + DISCOVER_POPULATION



REJECTED_CORPUS: list[tuple[str, str, str]] = [
    ("reject-const-mul", "mul(2, u)", "constants"),
    ("reject-const-add", "add(u, 1)", "constants"),
    ("reject-const-float", "mul(0.5, diff_x(u))", "constants"),
    ("reject-kwarg", "add(a, b=1)", "Keyword arguments"),
    ("reject-attr-callee", "np.add(b, a)", "bare token names"),
]







@dataclass(frozen=True)
class Divergence:

    axis: str
    case_id: str
    expression: str
    retired_core_output: str
    survivor_output: str | None
    survivor_error: str | None = None


DIVERGENCE_RECORD: tuple[Divergence, ...] = (


    Divergence(
        axis="sort-key",
        case_id="ax1-name-call",
        expression="add(c, mul(a, b))",
        retired_core_output="add(mul(a, b), c)",
        survivor_output="add(c,mul(a,b))",
    ),
    Divergence(
        axis="sort-key",
        case_id="ax1-nested",
        expression="add(add(c, b), a)",
        retired_core_output="add(add(b, c), a)",
        survivor_output="add(a,add(b,c))",
    ),



    Divergence(
        axis="constants",
        case_id="ax2-const-add",
        expression="add(u, 1)",
        retired_core_output="add(1, u)",
        survivor_output=None,
        survivor_error="constants",
    ),
    Divergence(
        axis="constants",
        case_id="ax2-const-float",
        expression="mul(0.5, diff_x(u))",
        retired_core_output="mul(diff_x(u), 0.5)",
        survivor_output=None,
        survivor_error="constants",
    ),




    Divergence(
        axis="commutative-set(spacing-only)",
        case_id="ax3-sub",
        expression="sub(b, a)",
        retired_core_output="sub(b, a)",
        survivor_output="sub(b,a)",
    ),

    Divergence(
        axis="output-shape",
        case_id="ax4-spacing",
        expression="add(b,a)",
        retired_core_output="add(a, b)",
        survivor_output="add(a,b)",
    ),




    Divergence(
        axis="kwargs",
        case_id="ax5-kwarg",
        expression="add(a, b=1)",
        retired_core_output="add(a)",
        survivor_output=None,
        survivor_error="Keyword arguments",
    ),


    Divergence(
        axis="callee-form",
        case_id="ax6-attr-callee",
        expression="np.add(b, a)",
        retired_core_output="np.add(b, a)",
        survivor_output=None,
        survivor_error="bare token names",
    ),
)







@pytest.mark.parametrize(
    ("case_id", "expression", "expected"),
    VALID_CORPUS,
    ids=[c[0] for c in VALID_CORPUS],
)
def test_survivor_canonical_form(case_id: str, expression: str, expected: str) -> None:
    assert canonicalize_expression(expression) == expected


def test_discover_alias_is_the_survivor() -> None:
    assert discover_alias is canonicalize_expression


@pytest.mark.parametrize(
    ("case_id", "expression", "expected"),
    VALID_CORPUS,
    ids=[c[0] for c in VALID_CORPUS],
)
def test_all_paths_agree_byte_identically(
    case_id: str, expression: str, expected: str
) -> None:
    survivor = canonicalize_expression(expression)
    via_discover = discover_alias(expression)
    via_shim = canonicalize_code(expression, FunctionRegistry.create_default())
    assert survivor == via_discover == via_shim == expected


@pytest.mark.parametrize(
    ("case_id", "expression", "match"),
    REJECTED_CORPUS,
    ids=[c[0] for c in REJECTED_CORPUS],
)
def test_all_paths_reject_identically(
    case_id: str, expression: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        canonicalize_expression(expression)
    with pytest.raises(ValueError, match=match):
        discover_alias(expression)
    with pytest.raises(ValueError, match=match):
        canonicalize_code(expression, FunctionRegistry.create_default())


def test_shim_ignores_registry_argument() -> None:
    empty = FunctionRegistry()
    default = FunctionRegistry.create_default()
    for _, expression, expected in VALID_CORPUS:
        assert canonicalize_code(expression, empty) == expected
        assert canonicalize_code(expression, default) == expected







@pytest.mark.parametrize(
    "record", DIVERGENCE_RECORD, ids=[d.case_id for d in DIVERGENCE_RECORD]
)
def test_divergence_record_survivor_side_is_live(record: Divergence) -> None:
    if record.survivor_output is not None:
        assert canonicalize_expression(record.expression) == record.survivor_output
    else:
        assert record.survivor_error is not None
        with pytest.raises(ValueError, match=record.survivor_error):
            canonicalize_expression(record.expression)


@pytest.mark.parametrize(
    "record", DIVERGENCE_RECORD, ids=[d.case_id for d in DIVERGENCE_RECORD]
)
def test_divergence_record_registers_real_divergence(record: Divergence) -> None:
    assert record.retired_core_output
    assert record.retired_core_output != record.survivor_output


def test_default_registry_commutative_set_matches_survivor_hardcode() -> None:
    reg = FunctionRegistry.create_default()
    commutative = {name for name in reg.list_names() if reg.is_commutative(name)}
    assert commutative == {"add", "mul"}
