
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "discover" / "v1ship_audit.py"



_SIGN_FLIPPED_AC2D = "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))"
_CORRECT_AC2D = "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_v1ship_audit_module",
        _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    return _load_script_module()







def _ac2d_payload(expression: str) -> dict[str, Any]:
    return {"mode1_run": {"best_expression": expression}}


@pytest.mark.unit
def test_structural_ok_false_on_sign_flipped_laplacian(audit: ModuleType) -> None:
    assert audit._check_structural_ok_ac2d(_ac2d_payload(_SIGN_FLIPPED_AC2D)) is False


@pytest.mark.unit
def test_structural_ok_true_on_correct_form(audit: ModuleType) -> None:
    assert audit._check_structural_ok_ac2d(_ac2d_payload(_CORRECT_AC2D)) is True


@pytest.mark.unit
def test_structural_ok_logs_reason_on_rejection(
    audit: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="v1ship_audit"):
        audit._check_structural_ok_ac2d(_ac2d_payload(_SIGN_FLIPPED_AC2D))
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "rejection must be logged at WARNING"
    assert "diff2_x" in records[0].getMessage()


@pytest.mark.unit
def test_structural_ok_none_without_expression(audit: ModuleType) -> None:
    assert audit._check_structural_ok_ac2d({}) is None
    assert audit._check_structural_ok_ac2d({"mode1_run": {}}) is None







@pytest.mark.unit
def test_canonicalize_term_merges_commutative_spellings(audit: ModuleType) -> None:
    assert (
        audit._canonicalize_term("mul(u, diff_x(u))")[0]
        == audit._canonicalize_term("mul(diff_x(u), u)")[0]
    )
    assert (
        audit._canonicalize_term("mul(u,diff_x(u))")[0]
        == audit._canonicalize_term("mul(diff_x(u), u)")[0]
    )
    assert (
        audit._canonicalize_term("mul(diff2_x(u), u)")[0]
        == audit._canonicalize_term("mul(u, diff2_x(u))")[0]
    )


@pytest.mark.unit
def test_term_aliases_table_is_gone(audit: ModuleType) -> None:
    assert not hasattr(audit, "TERM_ALIASES")


@pytest.mark.unit
def test_canonicalize_term_keeps_sign_channel(audit: ModuleType) -> None:
    assert audit._canonicalize_term("neg(u)") == ("u", -1)
    assert audit._canonicalize_term("neg(neg(u))") == ("u", 1)
    assert audit._canonicalize_term("u") == ("u", 1)


@pytest.mark.unit
def test_canonicalize_term_strips_inner_padding(audit: ModuleType) -> None:
    assert audit._canonicalize_term("neg( neg(u) )") == ("u", 1)
    assert audit._canonicalize_term(" neg(u) ") == ("u", -1)


@pytest.mark.unit
def test_canonicalize_term_warns_on_numeric_factor(
    audit: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="v1ship_audit"):
        key, sign = audit._canonicalize_term("mul(-2.0, u)")
    assert (key, sign) == ("u", 1)
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "numeric-factor strip must be disclosed"
    assert "not trustworthy" in records[0].getMessage()


@pytest.mark.unit
def test_canonicalize_term_no_warning_for_clean_terms(
    audit: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="v1ship_audit"):
        audit._canonicalize_term("neg(mul(u, diff_x(u)))")
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.unit
def test_reduce_terms_merges_commutative_duplicates(audit: ModuleType) -> None:
    reduced = audit._reduce_terms(
        ["mul(u,diff_x(u))", "neg(mul(diff_x(u), u))"],
        [1.0, 0.25],
    )
    assert list(reduced) == [audit._canonicalize_term("mul(diff_x(u), u)")[0]]
    assert reduced[audit._canonicalize_term("mul(diff_x(u), u)")[0]] == pytest.approx(
        0.75
    )


@pytest.mark.unit
def test_audit_one_keys_both_sides_of_the_gt_comparison(audit: ModuleType) -> None:
    payload = {
        "result": {
            "best_terms": ["diff2_x(u)", "mul(u,diff_x(u))"],
            "best_coefficients": [0.1, -1.0],
        }
    }
    block = audit._audit_one(payload, "burgers")
    assert block["term_set_match_gt"] is True
    assert block["reported_equation_matches_gt"] is True
