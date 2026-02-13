"""Helpers for extracting and formatting SGA equation details."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from .pde import evaluate_mse

_DEFAULT_TOL = 1e-6


@dataclass
class SGAEquationTerm:
    """Structured representation of a single term in the discovered PDE."""

    label: str
    source: Literal['default', 'generated']
    coefficient: float
    tree: object | None = None  # Tree instances are only used for richer rendering.


@dataclass
class SGAEquationDetails:
    """Structured representation of the full SGA-discovered equation."""

    lhs: str
    terms: Sequence[SGAEquationTerm]
    predicted_rhs: np.ndarray | None = None

    def nonzero_terms(self) -> Iterable[SGAEquationTerm]:
        """Return terms whose coefficients are non-zero."""

        return (term for term in self.terms if term.coefficient != 0.0)


def _infer_lhs_label(config) -> str:
    raw = getattr(config, 'left_side', None)
    if not raw:
        return 'u_t'
    if isinstance(raw, str) and '=' in raw:
        return raw.split('=', 1)[1].strip() or 'u_t'
    return 'u_t'


def extract_equation_details(pde, context, *, coefficient_tol: float = _DEFAULT_TOL) -> SGAEquationDetails:
    """Extract structured equation details from the best PDE candidate."""

    if pde is None:
        raise ValueError('PDE candidate must not be None.')
    if context is None:
        raise ValueError('ProblemContext must not be None.')

    elements_copy = copy.deepcopy(getattr(pde, 'elements', []))
    terms, coefficients, feature_matrix = evaluate_mse(
        elements_copy,
        context,
        True,
        return_matrix=True,
    )

    coeff_array = np.asarray(coefficients).reshape(-1)
    term_list: list[SGAEquationTerm] = []
    predicted_rhs: np.ndarray | None = None

    if feature_matrix is not None and feature_matrix.size and coeff_array.size:
        try:
            weights = np.asarray(coefficients).reshape(-1, 1)
            prediction = feature_matrix.dot(weights)
            prediction = prediction.reshape(context.ut.shape)
            predicted_rhs = np.real_if_close(prediction)
        except Exception:
            predicted_rhs = None

    default_count = getattr(context, 'num_default', 0)
    default_names = list(getattr(context, 'default_names', []))

    for idx in range(min(default_count, coeff_array.size)):
        coeff = float(coeff_array[idx])
        if abs(coeff) < coefficient_tol:
            continue
        label = default_names[idx] if idx < len(default_names) else f'default_{idx}'
        term_list.append(
            SGAEquationTerm(
                label=label,
                source='default',
                coefficient=coeff,
                tree=None,
            )
        )

    offset = default_count
    for local_idx, term in enumerate(terms):
        coeff_idx = offset + local_idx
        if coeff_idx >= coeff_array.size:
            break
        coeff = float(coeff_array[coeff_idx])
        if abs(coeff) < coefficient_tol:
            continue
        label = getattr(term, 'inorder', None) or getattr(term, 'preorder', None) or f'term_{local_idx}'
        term_list.append(
            SGAEquationTerm(
                label=label,
                source='generated',
                coefficient=coeff,
                tree=term,
            )
        )

    lhs_label = _infer_lhs_label(getattr(context, 'config', None))
    return SGAEquationDetails(lhs=lhs_label, terms=tuple(term_list), predicted_rhs=predicted_rhs)


_VALID_NOTATIONS = ("subscript", "leibniz")


def sga_equation_to_latex(
    details: SGAEquationDetails,
    *,
    include_coefficients: bool = True,
    notation: str = "subscript",
) -> str:
    """Convert structured equation details into a LaTeX string."""

    if details is None:
        raise ValueError('SGA equation details must not be None.')
    if notation not in _VALID_NOTATIONS:
        raise ValueError(f"notation must be one of {_VALID_NOTATIONS}, got {notation!r}")

    lhs = _format_lhs(details.lhs)
    terms = list(details.nonzero_terms())
    if not terms:
        return f"{lhs} = 0"

    formatted_terms = []
    for index, term in enumerate(terms):
        sign, body = _format_term(term, include_coefficients=include_coefficients, notation=notation)
        if index == 0:
            formatted_terms.append(body if sign == '+' else f"- {body}")
        else:
            formatted_terms.append(f"- {body}" if sign == '-' else f"+ {body}")
    rhs = ' '.join(formatted_terms)
    return f"{lhs} = {rhs}"


def sga_equation_structure(details: SGAEquationDetails, *, notation: str = "subscript") -> str:
    """Return a LaTeX string showing equation structure without coefficients."""

    return sga_equation_to_latex(details, include_coefficients=False, notation=notation)


def _format_lhs(lhs: str) -> str:
    raw = (lhs or 'u_t').strip()
    if not raw:
        raw = 'u_t'
    return _sanitize_label(raw)


def _format_term(term: SGAEquationTerm, *, include_coefficients: bool, notation: str = "subscript") -> tuple[str, str]:
    coeff = term.coefficient
    magnitude = abs(coeff)
    sign = '-' if coeff < 0 else '+'

    expr = _render_term_expression(term, notation=notation)
    if not expr:
        expr = '1'

    if include_coefficients:
        needs_coeff = (magnitude > 1.0 + _DEFAULT_TOL) or (magnitude < 1.0 - _DEFAULT_TOL) or _expression_is_scalar(expr)
        if not needs_coeff and magnitude != 0.0:
            body = expr
        elif expr in {'', '1'}:
            body = f"{magnitude:.6g}"
        else:
            body = f"{magnitude:.6g}\\,{expr}"
    else:
        body = expr if expr not in {'', '1'} else '1'

    return sign, body


def _expression_is_scalar(expr: str) -> bool:
    return expr in {'1', '0'}


def _render_term_expression(term: SGAEquationTerm, *, notation: str = "subscript") -> str:
    if term.tree is not None and hasattr(term.tree, 'tree'):
        try:
            return _render_tree(term.tree, notation=notation)
        except Exception:
            pass
    return _sanitize_label(term.label)


def _sanitize_label(label: str) -> str:
    cleaned = (label or '').strip()
    if not cleaned:
        return '1'
    leaf = _render_leaf(cleaned)
    if leaf != cleaned:
        return leaf
    replacements = {
        '*': '\\cdot ',
        '^2': '^{2}',
        '^3': '^{3}',
    }
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = cleaned.replace(' ', '')
    return cleaned


def _render_tree(tree, *, notation: str = "subscript") -> str:
    nodes = getattr(tree, 'tree', None)
    if not nodes:
        return '1'

    def helper(depth: int, idx: int) -> str:
        node = nodes[depth][idx]
        name = node.name
        if node.child_num == 0 or node.child_st is None:
            return _render_leaf(name)

        child_expressions = []
        next_depth = depth + 1
        for i in range(node.child_num):
            child_idx = (node.child_st or 0) + i
            if next_depth >= len(nodes) or child_idx >= len(nodes[next_depth]):
                child_expressions.append('1')
            else:
                child_expressions.append(helper(next_depth, child_idx))
        children = child_expressions
        return _combine_node(name, children, notation=notation)

    return helper(0, 0)


def _render_leaf(name: str) -> str:
    mapping = {
        'u': 'u',
        'x': 'x',
        't': 't',
        'ux': 'u_{x}',
        'u_x': 'u_{x}',
        'uxx': 'u_{xx}',
        'uxxx': 'u_{xxx}',
        'ut': 'u_{t}',
        '0': '0',
        # Common Greek letters for parameter fields.
        'alpha': '\\alpha',
        'beta': '\\beta',
        'gamma': '\\gamma',
        'delta': '\\delta',
        'epsilon': '\\epsilon',
        'kappa': '\\kappa',
        'lambda': '\\lambda',
        'mu': '\\mu',
        'nu': '\\nu',
        'rho': '\\rho',
        'sigma': '\\sigma',
    }
    return mapping.get(name, name)


def _combine_node(name: str, children: Sequence[str], *, notation: str = "subscript") -> str:
    if name == '+':
        return f"({children[0]} + {children[1]})"
    if name == '-':
        return f"({children[0]} - {children[1]})"
    if name == '*':
        return f"{children[0]} \\cdot {children[1]}"
    if name == '/':
        return f"\\frac{{{children[0]}}}{{{children[1]}}}"
    if name == '^2':
        base = _ensure_group(children[0])
        return f"{base}^{{2}}"
    if name == '^3':
        base = _ensure_group(children[0])
        return f"{base}^{{3}}"
    if name == 'd':
        var = _render_derivative_variable(children)
        operand = children[0]
        if notation == "subscript":
            return f"{operand}_{{{var}}}"
        return f"\\frac{{\\partial {operand}}}{{\\partial {var}}}"
    if name == 'd^2':
        var = _render_derivative_variable(children)
        operand = children[0]
        if notation == "subscript":
            return f"{operand}_{{{var}{var}}}"
        return f"\\frac{{\\partial^{{2}} {operand}}}{{\\partial {var}^{{2}}}}"
    if not children:
        return name
    return f"{name}({', '.join(children)})"


def _render_derivative_variable(children: Sequence[str]) -> str:
    if len(children) < 2:
        return 'x'
    return children[1]


def _ensure_group(expr: str) -> str:
    if expr.startswith('(') and expr.endswith(')'):
        return expr
    if expr.startswith('\\frac'):
        return f"({expr})"
    if any(ch in expr for ch in [' ', '+', '-']):
        return f"({expr})"
    return expr


__all__ = [
    'SGAEquationTerm',
    'SGAEquationDetails',
    'extract_equation_details',
    'sga_equation_to_latex',
    'sga_equation_structure',
]
