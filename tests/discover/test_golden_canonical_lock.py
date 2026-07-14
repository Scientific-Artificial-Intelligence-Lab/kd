
from __future__ import annotations

from pathlib import Path

import pytest

from kd.core.equation.canonical import canonicalize_expression
from kd.search.discover.golden.constants import PROJECT_ROOT
from kd.search.discover.golden.fixture_io import load_fixture

GOLDEN_DIR: Path = PROJECT_ROOT / "refs" / "baseline" / "golden"


class _Lock:

    __slots__ = ("expression_canonical", "term_canonicals")

    def __init__(
        self,
        expression_canonical: str,
        term_canonicals: dict[str, str],
    ) -> None:
        self.expression_canonical = expression_canonical
        self.term_canonicals = term_canonicals


_LOCK: dict[str, _Lock] = {
    "burgers_mode1_seed42": _Lock(
        expression_canonical="sub(mul(diff_x(u),u),diff2_x(u))",
        term_canonicals={
            "mul(diff_x(u), u)": "mul(diff_x(u),u)",
            "neg(diff2_x(u))": "neg(diff2_x(u))",
        },
    ),
    "burgers_mode1_seed123": _Lock(
        expression_canonical="add(diff2_x(u),mul(diff_x(u),u))",
        term_canonicals={
            "diff2_x(u)": "diff2_x(u)",
            "mul(diff_x(u), u)": "mul(diff_x(u),u)",
        },
    ),
    "burgers_mode1_seed777": _Lock(
        expression_canonical="sub(mul(diff_x(u),u),diff2_x(u))",
        term_canonicals={
            "mul(u, diff_x(u))": "mul(diff_x(u),u)",
            "neg(diff2_x(u))": "neg(diff2_x(u))",
        },
    ),
    "chafee_mode1_seed42": _Lock(
        expression_canonical="add(add(n3(u),u),diff2_x(u))",
        term_canonicals={
            "diff2_x(u)": "diff2_x(u)",
            "n3(u)": "n3(u)",
            "u": "u",
        },
    ),
    "chafee_mode1_seed123": _Lock(
        expression_canonical="add(div(mul(sub(diff2_x(u),u),u),u),n3(u))",
        term_canonicals={
            "div(mul(u, sub(diff2_x(u), u)), u)": "div(mul(sub(diff2_x(u),u),u),u)",
            "n3(u)": "n3(u)",
        },
    ),
    "chafee_mode1_seed777": _Lock(
        expression_canonical="sub(u,sub(mul(n2(u),u),diff2_x(u)))",
        term_canonicals={
            "diff2_x(u)": "diff2_x(u)",
            "neg(mul(u, n2(u)))": "neg(mul(n2(u),u))",
            "u": "u",
        },
    ),
}


def _discover_fixtures() -> list[Path]:
    if not GOLDEN_DIR.exists():
        return []
    return sorted(GOLDEN_DIR.glob("*.json"))


_FIXTURES = _discover_fixtures()

_FIXTURE_PARAMS = (
    _FIXTURES
    if _FIXTURES
    else [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason="No golden fixtures present at refs/baseline/golden/",
            ),
        )
    ]
)
_FIXTURE_IDS = [p.stem for p in _FIXTURES] or ["no-fixtures"]


@pytest.mark.parametrize("fixture_path", _FIXTURE_PARAMS, ids=_FIXTURE_IDS)
def test_raw_expression_recanonicalizes_byte_identically(
    fixture_path: Path,
) -> None:
    result = load_fixture(fixture_path)["result"]
    assert (
        canonicalize_expression(result["expression_raw"])
        == result["expression_canonical"]
    )


@pytest.mark.parametrize("fixture_path", _FIXTURE_PARAMS, ids=_FIXTURE_IDS)
def test_archived_canonical_is_fixed_point(fixture_path: Path) -> None:
    result = load_fixture(fixture_path)["result"]
    canonical = result["expression_canonical"]
    assert canonicalize_expression(canonical) == canonical


@pytest.mark.parametrize("fixture_path", _FIXTURE_PARAMS, ids=_FIXTURE_IDS)
def test_fixture_matches_embedded_lock(fixture_path: Path) -> None:
    payload = load_fixture(fixture_path)
    fixture_name = str(payload["fixture_id"])
    assert fixture_name in _LOCK, (
        f"{fixture_path.name} has no lock entry; extend _LOCK via the golden "
        "re-baseline discipline (ledger + user confirmation)."
    )
    lock = _LOCK[fixture_name]
    result = payload["result"]
    assert result["expression_canonical"] == lock.expression_canonical
    archived_terms = set(result["term_set_sorted"]) | set(result["coefs_by_term"])
    assert archived_terms == set(lock.term_canonicals)


@pytest.mark.parametrize("fixture_path", _FIXTURE_PARAMS, ids=_FIXTURE_IDS)
def test_term_strings_canonicalize_to_locked_values(fixture_path: Path) -> None:
    fixture_name = str(load_fixture(fixture_path)["fixture_id"])
    assert fixture_name in _LOCK, (
        f"{fixture_path.name} has no lock entry; extend _LOCK via the golden "
        "re-baseline discipline (ledger + user confirmation)."
    )
    lock = _LOCK[fixture_name]
    for term, expected in lock.term_canonicals.items():
        assert canonicalize_expression(term) == expected


@pytest.mark.skipif(
    not GOLDEN_DIR.exists(),
    reason="requires refs/baseline/golden fixtures (not shipped in the public tree)",
)
def test_lock_covers_exactly_the_fixture_set() -> None:
    fixture_ids = {p.stem for p in _FIXTURES}
    assert fixture_ids == set(_LOCK), (
        f"lock/fixture mismatch: fixtures={sorted(fixture_ids)} "
        f"lock={sorted(_LOCK)}"
    )
