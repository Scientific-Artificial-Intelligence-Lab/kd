
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import matplotlib

DEFAULT_STYLE: dict[str, Any] = {
    "font.size": 12,
    "figure.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


@contextmanager
def style_context(
    extra_style: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    original = matplotlib.rcParams.copy()
    try:
        combined: dict[str, Any] = dict(DEFAULT_STYLE)
        if extra_style:
            combined.update(extra_style)
        matplotlib.rcParams.update(combined)
        yield
    finally:
        matplotlib.rcParams.update(original)
