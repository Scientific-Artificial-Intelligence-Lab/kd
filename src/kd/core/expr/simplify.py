
from __future__ import annotations

import sympy

from kd.core.expr.sympy_bridge import _sympy_to_ir, to_sympy


def expand_linear_diffs(code: str, *, strict: bool = True) -> str:
    if not code.strip():
        return code
    try:
        return _sympy_to_ir(sympy.expand(to_sympy(code, strict=True)))
    except Exception:
        if strict:
            raise
        return code


__all__ = ["expand_linear_diffs"]
