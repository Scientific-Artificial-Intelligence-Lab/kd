
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kd.search.discover.golden import (
    GoldenRunResult,
    fixture_id,
    load_fixture,
    run_golden,
)
from kd.search.discover.golden.constants import resolve_project_path
from kd.search.discover.paths import GOLDEN_BASELINE_DIR, PROJECT_ROOT
from kd.search.discover.utils.canonicalize import canonicalize_expression

GOLDEN_DIR: Path = GOLDEN_BASELINE_DIR


COEF_ABS_TOL: float = 1e-4
REWARD_ABS_TOL: float = 1e-5
METRIC_REL_TOL: float = 1e-4












_REQUIRED_HASHSEED: str = "0"


@pytest.fixture(scope="module")
def _enforce_pythonhashseed() -> None:
    actual = os.environ.get("PYTHONHASHSEED")
    if actual != _REQUIRED_HASHSEED:
        pytest.fail(
            "PYTHONHASHSEED must be set BEFORE Python starts; cannot be "
            "patched at runtime. Re-invoke pytest as:\n"
            f" PYTHONHASHSEED={_REQUIRED_HASHSEED} uv run pytest "
            "-m golden --timeout=3600\n"
            f" (current value: {actual!r})",
            pytrace=False,
        )


@dataclass(frozen=True, slots=True)
class ExpectedGolden:

    pde: str
    mode: str
    seed: int
    expression_canonical: str
    term_set_sorted: tuple[str, ...]
    coefs_by_term: dict[str, float]
    reward: float
    mse: float
    nmse: float
    n_iterations_to_best: int











EXPECTED_GOLDENS: dict[str, ExpectedGolden] = {
    "burgers_mode1_seed42": ExpectedGolden(
        pde="burgers",
        mode="mode1",
        seed=42,
        expression_canonical="sub(mul(diff_x(u),u),diff2_x(u))",
        term_set_sorted=("mul(diff_x(u), u)", "neg(diff2_x(u))"),
        coefs_by_term={
            "mul(diff_x(u), u)": -1.0000022081692508,
            "neg(diff2_x(u))": -0.10018764586954143,
        },
        reward=0.9767343997955322,
        mse=7.631660410276479e-08,
        nmse=1.1178109675300483e-05,
        n_iterations_to_best=87,
    ),
    "burgers_mode1_seed123": ExpectedGolden(
        pde="burgers",
        mode="mode1",
        seed=123,
        expression_canonical="add(diff2_x(u),mul(diff_x(u),u))",
        term_set_sorted=("diff2_x(u)", "mul(diff_x(u), u)"),
        coefs_by_term={
            "diff2_x(u)": 0.10018764586954143,
            "mul(diff_x(u), u)": -1.0000022081692508,
        },
        reward=0.9767343997955322,
        mse=7.631660410276479e-08,
        nmse=1.1178109675300483e-05,
        n_iterations_to_best=30,
    ),
    "burgers_mode1_seed777": ExpectedGolden(
        pde="burgers",
        mode="mode1",
        seed=777,
        expression_canonical="sub(mul(diff_x(u),u),diff2_x(u))",
        term_set_sorted=("mul(u, diff_x(u))", "neg(diff2_x(u))"),
        coefs_by_term={
            "mul(u, diff_x(u))": -1.0000022081692508,
            "neg(diff2_x(u))": -0.10018764586954143,
        },
        reward=0.9767343997955322,
        mse=7.631660410276479e-08,
        nmse=1.1178109675300483e-05,
        n_iterations_to_best=43,
    ),
    "chafee_mode1_seed42": ExpectedGolden(
        pde="chafee",
        mode="mode1",
        seed=42,
        expression_canonical="add(add(n3(u),u),diff2_x(u))",
        term_set_sorted=("diff2_x(u)", "n3(u)", "u"),
        coefs_by_term={
            "diff2_x(u)": 0.9923814610979355,
            "n3(u)": 0.992582704028159,
            "u": -0.9791696135885402,
        },
        reward=0.9548244476318359,
        mse=0.008009744104737317,
        nmse=0.00025260529631462804,
        n_iterations_to_best=22,
    ),
    "chafee_mode1_seed123": ExpectedGolden(
        pde="chafee",
        mode="mode1",
        seed=123,
        expression_canonical="add(div(mul(sub(diff2_x(u),u),u),u),n3(u))",
        term_set_sorted=(
            "div(mul(u, sub(diff2_x(u), u)), u)",
            "n3(u)",
        ),
        coefs_by_term={
            "div(mul(u, sub(diff2_x(u), u)), u)": 0.989974435865382,
            "n3(u)": 0.9932017556230645,
        },
        reward=0.9645625352859497,
        mse=0.008122094938209494,
        nmse=0.0002561485325540541,
        n_iterations_to_best=144,
    ),
    "chafee_mode1_seed777": ExpectedGolden(
        pde="chafee",
        mode="mode1",
        seed=777,
        expression_canonical="sub(u,sub(mul(n2(u),u),diff2_x(u)))",
        term_set_sorted=("diff2_x(u)", "neg(mul(u, n2(u)))", "u"),
        coefs_by_term={
            "diff2_x(u)": 0.9923814610979529,
            "neg(mul(u, n2(u)))": -0.9925827040281552,
            "u": -0.9791696135885081,
        },
        reward=0.9548244476318359,
        mse=0.008009744104737307,
        nmse=0.0002526052963146277,
        n_iterations_to_best=20,
    ),
}


def _discover_fixtures() -> list[Path]:
    if not GOLDEN_DIR.exists():
        return []
    return sorted(GOLDEN_DIR.glob("*.json"))


_FIXTURES = _discover_fixtures()


def _id_for_fixture(path: Path) -> str:
    return path.stem


@pytest.mark.golden
@pytest.mark.slow
@pytest.mark.usefixtures("_enforce_pythonhashseed")
@pytest.mark.parametrize(
    "fixture_path",


    _FIXTURES
    if _FIXTURES
    else [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason="No golden fixtures present at refs/baseline/golden/",
            ),
        )
    ],
    ids=[_id_for_fixture(p) for p in _FIXTURES] or ["no-fixtures"],
)
def test_golden_baseline_matches_hardcoded_oracle(fixture_path: Path) -> None:
    payload = load_fixture(fixture_path)
    fixture_name = str(payload["fixture_id"])
    assert fixture_name in EXPECTED_GOLDENS, (
        f"{fixture_path.name} has no hard-coded oracle; add it to "
        "EXPECTED_GOLDENS or remove the fixture."
    )

    expected = EXPECTED_GOLDENS[fixture_name]
    _assert_fixture_metadata(payload, expected)
    _assert_serialized_result_matches(payload["result"], expected, fixture_path)

    config = payload["config"]
    data_path = _resolve_optional_data_path(config.get("data_path"))
    actual_result, _actual_config = run_golden(
        pde=config["pde"],
        mode=config["mode"],
        seed=config["seed"],
        data_path=data_path,
    )



    _assert_result_matches(
        actual_result,
        expected,
        fixture_path,
        canonicalize_terms=True,
    )


@pytest.mark.skipif(
    not GOLDEN_DIR.exists(),
    reason="requires refs/baseline/golden fixtures (not shipped in the public tree)",
)
def test_golden_fixture_set_matches_oracle_set() -> None:
    fixture_ids = {_id_for_fixture(path) for path in _FIXTURES}
    assert fixture_ids == set(EXPECTED_GOLDENS), (
        f"fixture/oracle mismatch: fixtures={sorted(fixture_ids)} "
        f"oracles={sorted(EXPECTED_GOLDENS)}"
    )


@pytest.mark.skipif(
    not GOLDEN_DIR.exists(),



    reason="requires the private tree (scripts/ and refs/ are not exported)",
)
def test_fixture_diff_scope_paths_all_exist() -> None:
    for pathspec in _FIXTURE_DIFF_SCOPE:
        bare = pathspec.removeprefix(":(exclude)")
        matched = subprocess.check_output(
            ["git", "ls-files", "--", bare],
            cwd=PROJECT_ROOT,
            text=True,
        )
        assert matched.strip(), (
            f"_FIXTURE_DIFF_SCOPE entry {pathspec!r} matches no tracked file; "
            "it contributes nothing to the staleness check"
        )


def _assert_fixture_metadata(
    payload: dict[str, Any],
    expected: ExpectedGolden,
) -> None:
    config = payload["config"]
    expected_id = fixture_id(
        pde=expected.pde,
        mode=expected.mode,
        seed=expected.seed,
    )
    assert payload["fixture_id"] == expected_id
    ok, reason = _fixture_commit_is_current(str(payload["commit"]))
    assert ok, f"Stale fixture {payload['fixture_id']}: {reason}"
    assert config["pde"] == expected.pde
    assert config["mode"] == expected.mode
    assert config["seed"] == expected.seed
    assert "status" not in payload, "placeholder fixtures are out of scope"


def _assert_serialized_result_matches(
    serialized: dict[str, Any],
    expected: ExpectedGolden,
    fixture_path: Path,
) -> None:
    artifact = GoldenRunResult(
        expression_canonical=serialized["expression_canonical"],
        expression_raw=serialized["expression_raw"],
        term_set_sorted=list(serialized["term_set_sorted"]),
        coefs_by_term=dict(serialized["coefs_by_term"]),
        reward=float(serialized["reward"]),
        mse=float(serialized["mse"]),
        nmse=float(serialized["nmse"]),
        n_iterations_to_best=int(serialized["n_iterations_to_best"]),
        wall_time_seconds=float(serialized["wall_time_seconds"]),
    )
    _assert_result_matches(
        artifact,
        expected,
        fixture_path,
        canonicalize_terms=False,
    )


def _assert_result_matches(
    actual: GoldenRunResult,
    expected: ExpectedGolden,
    fixture_path: Path,
    *,
    canonicalize_terms: bool,
) -> None:
    diffs: list[str] = []
    if actual.expression_canonical != expected.expression_canonical:
        diffs.append(
            f"expression_canonical: "
            f"expected={expected.expression_canonical!r} "
            f"got={actual.expression_canonical!r}",
        )
    if canonicalize_terms:
        expected_terms_cmp = sorted(
            canonicalize_expression(t) for t in expected.term_set_sorted
        )
        actual_terms_cmp = sorted(
            canonicalize_expression(t) for t in actual.term_set_sorted
        )
        terms_label = "term_set_sorted (canonicalized)"
    else:
        expected_terms_cmp = sorted(expected.term_set_sorted)
        actual_terms_cmp = sorted(actual.term_set_sorted)
        terms_label = "term_set_sorted (raw)"
    if actual_terms_cmp != expected_terms_cmp:
        diffs.append(
            f"{terms_label}: expected={expected_terms_cmp} got={actual_terms_cmp}",
        )
    if actual.n_iterations_to_best != expected.n_iterations_to_best:
        diffs.append(
            f"n_iterations_to_best: expected={expected.n_iterations_to_best} "
            f"got={actual.n_iterations_to_best}",
        )
    diffs.extend(
        _check_coefs_by_term(
            actual.coefs_by_term,
            expected.coefs_by_term,
            canonicalize_terms=canonicalize_terms,
        )
    )
    diffs.extend(_check_scalar_metrics(actual, expected))

    if diffs:
        header = f"Golden baseline mismatch in {fixture_path.name}:"
        body = "\n".join(f" - {d}" for d in diffs)
        raise AssertionError(f"{header}\n{body}")


def _canonicalize_coef_keys(coefs: dict[str, float]) -> dict[str, float]:
    return {canonicalize_expression(term): val for term, val in coefs.items()}


def _check_coefs_by_term(
    actual: dict[str, float],
    expected: dict[str, float],
    *,
    canonicalize_terms: bool,
) -> list[str]:
    if canonicalize_terms:
        actual_cmp = _canonicalize_coef_keys(actual)
        expected_cmp = _canonicalize_coef_keys(expected)
        keys_label = "coefs_by_term keys (canonicalized)"
    else:
        actual_cmp = dict(actual)
        expected_cmp = dict(expected)
        keys_label = "coefs_by_term keys (raw)"
    diffs: list[str] = []
    if set(actual_cmp) != set(expected_cmp):
        diffs.append(
            f"{keys_label}: expected={sorted(expected_cmp)} got={sorted(actual_cmp)}",
        )
        return diffs
    for term, exp_coef in expected_cmp.items():
        got_coef = actual_cmp[term]
        if abs(got_coef - exp_coef) >= COEF_ABS_TOL:
            diffs.append(
                f"coefs_by_term[{term!r}]: expected={exp_coef} got={got_coef} "
                f"|diff|={abs(got_coef - exp_coef):.3e} "
                f"(tol={COEF_ABS_TOL:.0e})",
            )
    return diffs


def _check_scalar_metrics(
    actual: GoldenRunResult,
    expected: ExpectedGolden,
) -> list[str]:
    diffs: list[str] = []
    if abs(actual.reward - expected.reward) >= REWARD_ABS_TOL:
        diffs.append(
            f"reward: expected={expected.reward} got={actual.reward} "
            f"|diff|={abs(actual.reward - expected.reward):.3e} "
            f"(tol={REWARD_ABS_TOL:.0e})",
        )
    diffs.extend(_check_relative("mse", actual.mse, expected.mse, METRIC_REL_TOL))
    diffs.extend(_check_relative("nmse", actual.nmse, expected.nmse, METRIC_REL_TOL))
    return diffs


def _check_relative(
    name: str,
    actual: float,
    expected: float,
    rel_tol: float,
) -> list[str]:
    denom = abs(expected) if abs(expected) > 0 else 1.0
    rel = abs(actual - expected) / denom
    if rel >= rel_tol:
        return [
            f"{name}: expected={expected} got={actual} "
            f"|rel_diff|={rel:.3e} (tol={rel_tol:.0e})",
        ]
    return []


def _resolve_optional_data_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return resolve_project_path(Path(value))






















_FIXTURE_DIFF_SCOPE: tuple[str, ...] = (
    "src/kd/search/discover/",
    "src/kd/core/",
    "src/kd/data/",
    "scripts/discover/generate_golden_baseline.py",
    ":(exclude)src/kd/search/discover/viz.py",
)


def _fixture_commit_is_current(fixture_commit: str) -> tuple[bool, str]:
    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", fixture_commit, "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    if ancestor_check.returncode != 0:
        return (
            False,
            f"fixture commit {fixture_commit[:12]} is not an ancestor of HEAD",
        )
    diff_args = [
        "git",
        "diff",
        "--name-only",
        f"{fixture_commit}..HEAD",
        "--",
        *_FIXTURE_DIFF_SCOPE,
    ]
    diff = (
        subprocess.check_output(
            diff_args,
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )
    if diff:
        affected = diff.replace("\n", "\n ")
        return (
            False,
            (
                f"code/test changes between fixture commit "
                f"{fixture_commit[:12]} and HEAD; affected files:\n "
                f"{affected}"
            ),
        )
    return True, ""
