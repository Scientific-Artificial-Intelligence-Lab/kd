
from __future__ import annotations

from typing import Any

import sympy
from sympy.printing.pretty.pretty_symbology import (
    pretty_symbol,
)
from sympy.printing.pretty.stringpict import (
    prettyForm,
)

_DISPLAY_CLASS_CACHE: dict[str, Any] = {}


def _pow_wrapped(body: str, exp: str | None) -> str:
    return body if exp is None else rf"\left({body} \right)^{{{exp}}}"


def diff_display_class(name: str, axis: str, order: int) -> Any:
    cached = _DISPLAY_CLASS_CACHE.get(name)
    if cached is not None:
        return cached
    suffix = axis * order
    cls = sympy.Function(name)

    def _latex(self: Any, printer: Any, exp: str | None = None) -> str:
        inner = printer._print(self.args[0])
        return _pow_wrapped(rf"\left({inner} \right)_{{{suffix}}}", exp)

    def _sympystr(self: Any, printer: Any) -> str:

        return f"({printer._print(self.args[0])})_{suffix}"

    def _pretty(self: Any, printer: Any) -> Any:


        pform = prettyForm(*printer._print(self.args[0]).parens())
        pform = prettyForm(*pform.right(pretty_symbol(f"Z_{suffix}")[1:]))

        pform.binding = prettyForm.POW
        return pform

    cls._latex = _latex
    cls._sympystr = _sympystr
    cls._pretty = _pretty
    _DISPLAY_CLASS_CACHE[name] = cls
    return cls


def lap_display_class() -> Any:
    cls = sympy.Function("lap")

    def _latex(self: Any, printer: Any, exp: str | None = None) -> str:
        inner = printer._print(self.args[0])
        return _pow_wrapped(rf"\nabla^{{2}}\left({inner} \right)", exp)

    def _sympystr(self: Any, printer: Any) -> str:

        return f"∇²({printer._print(self.args[0])})"

    def _pretty(self: Any, printer: Any) -> Any:
        pform = prettyForm(*printer._print(self.args[0]).parens())
        pform = prettyForm(*pform.left("∇²"))
        pform.binding = prettyForm.POW
        return pform

    cls._latex = _latex
    cls._sympystr = _sympystr
    cls._pretty = _pretty
    return cls


__all__ = ["diff_display_class", "lap_display_class"]
