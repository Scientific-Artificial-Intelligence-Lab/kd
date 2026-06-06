
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"


_FAST_TIMEOUT_SEC = 120



_SLOW_TIMEOUT_SEC = 300


_SENTINELS = ("Discovered", "[kd] Done.")


def _run_example(
    script_name: str, timeout_sec: int
) -> subprocess.CompletedProcess[str]:
    script_path = _EXAMPLES_DIR / script_name
    assert script_path.exists(), f"Missing example script: {script_path}"
    return subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=_REPO_ROOT,
    )


def _assert_sentinel(stdout: str) -> None:
    assert any(s in stdout for s in _SENTINELS), (
        f"No sentinel found in example stdout. Expected one of {_SENTINELS}.\n"
        f"--- stdout ---\n{stdout}"
    )


@pytest.mark.smoke
def test_example_01_quickstart() -> None:
    result = _run_example("01_quickstart.py", _FAST_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)


@pytest.mark.smoke
def test_example_02_your_data() -> None:
    result = _run_example("02_your_data.py", _FAST_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)


@pytest.mark.smoke
@pytest.mark.slow
def test_example_03_visualize() -> None:
    result = _run_example("03_visualize.py", _SLOW_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)
    assert "Report" in result.stdout, (
        "Example 03 did not report an HTML report path.\n"
        f"--- stdout ---\n{result.stdout}"
    )


@pytest.mark.smoke
@pytest.mark.slow
def test_example_04_noisy_data() -> None:
    result = _run_example("04_noisy_data.py", _SLOW_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)


@pytest.mark.smoke
def test_example_05_save_load() -> None:
    result = _run_example("05_save_load.py", _FAST_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)
    assert "Round-trip OK" in result.stdout, (
        f"Example 05 did not report round-trip status.\n--- stdout ---\n{result.stdout}"
    )
