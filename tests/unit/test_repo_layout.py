
from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_README_PATH = _PROJECT_ROOT / "README.md"


@pytest.mark.smoke
def test_root_readme_exists() -> None:
    assert _README_PATH.exists(), (
        f'pyproject.toml\'s `readme = "README.md"` points to a missing '
        f"file. Expected {_README_PATH} to exist (the public export "
        f"pipeline replaces this with scripts/public-assets/README.md "
        f"for release; this stub is for private-repo `pip install .`)."
    )


@pytest.mark.smoke
def test_root_readme_non_empty() -> None:
    text = _README_PATH.read_text(encoding="utf-8")
    assert text.strip(), "README.md must not be empty"
    assert "kd" in text.lower(), (
        f"README.md must mention the project (look for 'kd'); got: {text[:200]!r}"
    )
