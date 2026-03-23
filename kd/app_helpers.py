"""Gradio app helper functions — no gradio dependency.

Pure-Python helpers for dataset preview, operator validation, and model
selection. Extracted from ``app.py`` so they can be tested without
importing gradio.
"""

from __future__ import annotations

import re
from typing import Optional

from kd.dataset._registry import PDE_REGISTRY


# ── Constants ─────────────────────────────────────────────────────

# Display name -> registry key
MODEL_KEYS = {
    "KD_SGA": "sga",
    "KD_DLGA": "dlga",
    "KD_DSCV": "discover",
    "KD_DSCV_SPR": "discover_spr",
    "KD_EqGPT": "eqgpt",
}

DSCV_BINARY_DEFAULT = "add, mul, diff, diff2"
DSCV_UNARY_DEFAULT = "n2"


# ── Parsing ───────────────────────────────────────────────────────

def _parse_ops(text: Optional[str]) -> list[str]:
    """Parse comma-separated operator string into list."""
    if not text:
        return []
    return [op.strip() for op in text.split(",") if op.strip()]


# ── New helpers (stubs) ───────────────────────────────────────────

def _parse_sym_true_operators(sym_true: Optional[str]) -> set[str]:
    """Extract operator names from a prefix-notation sym_true string.

    Parameters
    ----------
    sym_true : str or None
        Comma-separated prefix expression, e.g.
        ``"add,mul,u1,diff,u1,x1,diff2,u1,x1"``.
        ``None`` or empty string yields an empty set.

    Returns
    -------
    set[str]
        Set of operator tokens found in the expression.
        Operand tokens like ``u1``, ``x1`` are excluded.
    """
    if not sym_true or not sym_true.strip():
        return set()

    _OPERAND_RE = re.compile(r"^[utx]\d+$")
    tokens = [t.strip() for t in sym_true.split(",") if t.strip()]
    operators: set[str] = set()
    for token in tokens:
        # Skip operands: variable-like tokens (u1, x1, x2) and numeric constants
        if _OPERAND_RE.match(token):
            continue
        try:
            float(token)
            continue  # numeric constant
        except ValueError:
            pass
        operators.add(token)
    return operators


def _format_dataset_info(name: str) -> str:
    """Return a Markdown-formatted summary of a registered dataset.

    The summary includes shape, domain, ground-truth equation, and
    recommended operators (extracted from *sym_true*).

    Parameters
    ----------
    name : str
        Dataset key in ``PDE_REGISTRY``.

    Returns
    -------
    str
        Markdown string suitable for display in a ``gr.Markdown`` component.

    Raises
    ------
    KeyError or ValueError
        If *name* is not in the registry.
    """
    if name not in PDE_REGISTRY:
        raise KeyError(
            f"Unknown dataset: {name!r}. "
            f"Available: {list(PDE_REGISTRY.keys())}"
        )
    info = PDE_REGISTRY[name]
    lines: list[str] = [f"### {name}"]

    # Shape
    shape = info.get("shape")
    if shape is not None:
        lines.append(f"- **Shape**: {shape[0]} x {shape[1]}")

    # Domain
    domain = info.get("domain")
    if domain is not None:
        parts = [f"{k}: [{v[0]}, {v[1]}]" for k, v in domain.items()]
        lines.append(f"- **Domain**: {', '.join(parts)}")

    # Ground-truth equation
    sym_true = info.get("sym_true")
    if sym_true is not None:
        lines.append(f"- **Equation**: `{sym_true}`")
        ops = _parse_sym_true_operators(sym_true)
        if ops:
            sorted_ops = sorted(ops)
            lines.append(f"- **Recommended operators**: {', '.join(sorted_ops)}")
    else:
        lines.append("- **Equation**: _(unknown)_")

    return "\n".join(lines)


def validate_operators(
    ops: list[str], valid_set: set[str]
) -> tuple[bool, list[str]]:
    """Check whether all operator names are in the valid set.

    Parameters
    ----------
    ops : list[str]
        Operator names to validate (e.g. from ``_parse_ops``).
    valid_set : set[str]
        Set of known-valid operator names.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_valid, invalid_names)`` where *is_valid* is ``True`` when
        every name is recognized, and *invalid_names* lists the bad ones
        in input order.
    """
    invalid: list[str] = [op for op in ops if op not in valid_set]
    return (len(invalid) == 0, invalid)


def get_compatible_models(dataset_name: Optional[str]) -> list[str]:
    """Return model display names compatible with a given dataset.

    EqGPT is always included but should be annotated to indicate it
    uses its own built-in data rather than the selected dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset key in ``PDE_REGISTRY``.

    Returns
    -------
    list[str]
        Model display names. EqGPT entry carries an annotation.
    """
    if not dataset_name:
        return []
    info = PDE_REGISTRY.get(dataset_name, {})
    models_map = info.get("models", {})
    result = [
        display for display, key in MODEL_KEYS.items()
        if key != "eqgpt" and models_map.get(key, False)
    ]
    # EqGPT uses its own wave_breaking data, always available
    result.append("KD_EqGPT (built-in data)")
    return result
