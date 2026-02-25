import logging
from typing import Optional

import sympy

logger = logging.getLogger(__name__)
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.latex import LatexPrinter

from kd.viz.dlga_eq2latex import _format_full_latex_term

# FIXME: This list is static; ideally it should be derived from Program.library
# or injected as a parameter.
DEEPRL_SYMBOLS_FOR_SYMPY = {
    name: sympy.Symbol(name)
    for name in ['u1', 'x1', 'x2', 'x3', 'c', 'p1', 'p2', 'p3']
}

# ... (imports, DEEPRL_SYMBOLS_FOR_SYMPY, DEBUG_RENDERER_MODE) ...

DEBUG_RENDERER_MODE = False  # Whether to enable verbose debug logging.

_SYMBOL_DISPLAY = {
    "u1": "u", "x1": "x", "x2": "y", "x3": "z",
    "p1": "p_{1}", "p2": "p_{2}", "p3": "p_{3}",
}

# Function-based symbols for Leibniz mode (produces \partial instead of d)
_x1, _x2, _x3 = sympy.symbols('x1 x2 x3')
_LEIBNIZ_SYMBOLS_FOR_SYMPY = {
    'u1': sympy.Function('u1')(_x1, _x2, _x3),
    'x1': _x1, 'x2': _x2, 'x3': _x3,
    'c': sympy.Symbol('c'),
    'p1': sympy.Symbol('p1'), 'p2': sympy.Symbol('p2'), 'p3': sympy.Symbol('p3'),
}


class _SubscriptLatexPrinter(LatexPrinter):
    """Render Derivative(u, x, x) as u_{xx} instead of Leibniz notation."""

    def _print_Derivative(self, expr):
        func = expr.args[0]
        if isinstance(func, sympy.Symbol):
            func_name = _SYMBOL_DISPLAY.get(func.name, func.name)
        elif hasattr(func, 'func') and hasattr(func.func, 'name'):
            # Applied function like u1(x1, x2): func.func has .name
            func_name = _SYMBOL_DISPLAY.get(func.func.name, func.func.name)
        else:
            # Compound expression (Pow, Mul, Add, etc.): render via printer
            func_name = "(" + self._print(func) + ")"

        subscript_parts = []
        for var, count in expr.variable_count:
            var_name = _SYMBOL_DISPLAY.get(var.name, var.name)
            subscript_parts.append(var_name * count)

        subscript = "".join(subscript_parts)
        return f"{func_name}_{{{subscript}}}"

    def _print_Symbol(self, expr):
        name = expr.name
        return _SYMBOL_DISPLAY.get(name, super()._print_Symbol(expr))


# ---------------------------------------------------------------------------
# First-order compound derivative expansion
# ---------------------------------------------------------------------------

# Symbols that represent dependent variables (functions of x1, x2, x3)
_FUNC_SYMBOL_NAMES = frozenset({'u1'})
_VAR_SYMBOL_NAMES = frozenset({'x1', 'x2', 'x3'})


def _is_first_order_compound_derivative(expr: sympy.Basic) -> bool:
    """Check if *expr* is a first-order Derivative of a compound expression."""
    if not isinstance(expr, sympy.Derivative):
        return False
    total_order = sum(c for _, c in expr.variable_count)
    if total_order != 1:
        return False
    func = expr.args[0]
    from sympy.core.function import AppliedUndef
    return not isinstance(func, (sympy.Symbol, AppliedUndef))


def _expand_first_order_compound_derivatives(
    expr: sympy.Basic,
) -> tuple:
    """Expand first-order derivatives of compound expressions.

    Uses SymPy ``doit()`` after temporarily promoting Symbols to Functions
    so that the chain rule is applied.

    Returns ``(numeric_factor, expanded_expr)``.
    Only first-order compound derivatives are expanded; everything else
    is returned unchanged with factor 1.
    """
    if not _is_first_order_compound_derivative(expr):
        return sympy.Integer(1), expr

    # Symbol → Function substitution
    sym_to_fn: dict = {}
    var_syms = [sympy.Symbol(v) for v in sorted(_VAR_SYMBOL_NAMES)]
    for name in _FUNC_SYMBOL_NAMES:
        sym = sympy.Symbol(name)
        fn = sympy.Function(name)(*var_syms)
        sym_to_fn[sym] = fn

    expr_fn = expr
    for sym, fn in sym_to_fn.items():
        expr_fn = expr_fn.subs(sym, fn)

    expanded = expr_fn.doit()

    # Function → Symbol substitution (reverse)
    for sym, fn in sym_to_fn.items():
        # Replace derivatives first (higher order → lower order)
        for vs in var_syms:
            for order in range(4, 0, -1):
                expanded = expanded.subs(
                    sympy.Derivative(fn, (vs, order)),
                    sympy.Derivative(sym, *([vs] * order)),
                )
        expanded = expanded.subs(fn, sym)

    # Extract numeric factor: 2*u1*Derivative(u1, x1) → (2, u1*Derivative(...))
    coeff, rest = expanded.as_coeff_Mul()
    return coeff, rest


# ---------------------------------------------------------------------------
# Node → LaTeX (with optional expansion)
# ---------------------------------------------------------------------------

def _discover_term_node_to_latex_expanded(
    term_node_obj,
    local_sympy_symbols=None,
    *,
    notation: str = "subscript",
) -> tuple:
    """Like ``_discover_term_node_to_latex`` but also expands first-order
    compound derivatives, returning ``(numeric_factor, latex_string)``.
    """
    if local_sympy_symbols is None:
        if notation == "leibniz":
            local_sympy_symbols = _LEIBNIZ_SYMBOLS_FOR_SYMPY
        else:
            local_sympy_symbols = DEEPRL_SYMBOLS_FOR_SYMPY

    sympy_expr_str = term_node_obj.to_sympy_string()
    parsed = parse_expr(
        sympy_expr_str,
        local_dict=local_sympy_symbols,
        transformations='all',
    )

    factor, expanded = _expand_first_order_compound_derivatives(parsed)

    if notation == "subscript":
        latex_output = _SubscriptLatexPrinter().doprint(expanded)
    else:
        latex_output = sympy.latex(expanded)

    return factor, latex_output


# Node -> LaTeX
def _discover_term_node_to_latex(term_node_obj, local_sympy_symbols=None, *, notation="subscript"):
    if local_sympy_symbols is None:
        if notation == "leibniz":
            local_sympy_symbols = _LEIBNIZ_SYMBOLS_FOR_SYMPY
        else:
            local_sympy_symbols = DEEPRL_SYMBOLS_FOR_SYMPY
    try:
        if not hasattr(term_node_obj, 'to_sympy_string'):
            raise AttributeError("Node object does not implement to_sympy_string().")

        sympy_expr_str = term_node_obj.to_sympy_string()

        if DEBUG_RENDERER_MODE:
             logger.debug("Node '%r'.to_sympy_string() -> '%s'", term_node_obj, sympy_expr_str)

        parsed_sympy_expr = parse_expr(
            sympy_expr_str,
            local_dict=local_sympy_symbols,
            transformations='all',  # allow a broad set of SymPy-compatible transformations
        )
        if DEBUG_RENDERER_MODE:
            logger.debug("Parsed SymPy expression: %s", parsed_sympy_expr)

        if notation == "subscript":
            latex_output = _SubscriptLatexPrinter().doprint(parsed_sympy_expr)
        else:
            latex_output = sympy.latex(parsed_sympy_expr)
        return latex_output
    except Exception as e:
        import traceback
        error_repr = repr(term_node_obj) if term_node_obj else "None"
        sympy_str_val = sympy_expr_str if 'sympy_expr_str' in locals() else "<sympy string not generated>"
        logger.error(
            "_discover_term_node_to_latex failed for node '%s' "
            "(SymPy str: '%s'): %s", error_repr, sympy_str_val, e,
        )
        if DEBUG_RENDERER_MODE:
             logger.debug("Traceback:", exc_info=True)
        return f"\\text{{Error converting node: {error_repr}}}"

_VALID_NOTATIONS = ("subscript", "leibniz")

def discover_program_to_latex(program_object, # lhs_name_str,
                            # 可选参数，如果需要覆盖模块级的默认值
                            custom_lhs_latex_map=None,
                            custom_deeprl_symbols=None,
                            lhs_axis: Optional[str] = None,
                            notation: str = "subscript",
                            expand_derivatives: bool = True):
    """
    Convert a Program object to a full LaTeX equation string.

    Parameters
    ----------
    lhs_axis : str, optional
        Axis name for the LHS derivative (e.g. 't', 'x', 'y').
        Defaults to 't' → ``u_t``.
    """
    if notation not in _VALID_NOTATIONS:
        raise ValueError(f"notation must be one of {_VALID_NOTATIONS}, got {notation!r}")

    # 1. 处理 LHS
    if lhs_axis is not None:
        lhs_latex = f"u_{{{lhs_axis}}}"
    else:
        lhs_latex = "u_t"

    # 2. Validate program_object and required attributes.
    if not (program_object and \
            hasattr(program_object, 'w') and \
            hasattr(program_object, 'STRidge') and \
            hasattr(program_object.STRidge, 'terms') and \
            program_object.w is not None and \
            program_object.STRidge.terms is not None and \
            len(program_object.w) == len(program_object.STRidge.terms)):
        if DEBUG_RENDERER_MODE:
            w_status = str(getattr(program_object, 'w', '<w attribute not found>'))
            terms_status = str(
                getattr(
                    getattr(program_object, 'STRidge', None),
                    'terms',
                    '<STRidge.terms not found>',
                )
            )
            logger.warning("program_object is invalid or missing w/STRidge.terms.")
            logger.warning("  program_object: %s", program_object)
            logger.warning("  w: %s", w_status)
            logger.warning("  STRidge.terms: %s", terms_status)
            if hasattr(program_object, 'w') and hasattr(program_object.STRidge, 'terms') and \
               program_object.w is not None and program_object.STRidge.terms is not None:
                 logger.warning("  len(w)=%d, len(STRidge.terms)=%d", len(program_object.w), len(program_object.STRidge.terms))
        return f"${lhs_latex} = 0 \\; (\\text{{Error: Invalid program structure}})$"

    coefficients = program_object.w
    term_nodes = program_object.STRidge.terms
    
    if not term_nodes:
        return f"${lhs_latex} = 0$"

    # 3. 构建 RHS
    rhs_latex_full_terms = []
    processed_terms_count = 0  # used to keep track of first RHS term

    if custom_deeprl_symbols:
        current_sympy_symbols = custom_deeprl_symbols
    elif notation == "leibniz":
        current_sympy_symbols = _LEIBNIZ_SYMBOLS_FOR_SYMPY
    else:
        current_sympy_symbols = DEEPRL_SYMBOLS_FOR_SYMPY

    for i, coeff_val_from_w in enumerate(coefficients):
        # program.w 可能是一个普通列表或1D NumPy数组
        coeff_val = float(coeff_val_from_w)
        term_node = term_nodes[i]

        # Get base term LaTeX for this node.
        if expand_derivatives:
            factor, base_term_latex = _discover_term_node_to_latex_expanded(
                term_node,
                local_sympy_symbols=current_sympy_symbols,
                notation=notation,
            )
            coeff_val *= float(factor)
        else:
            base_term_latex = _discover_term_node_to_latex(
                term_node,
                local_sympy_symbols=current_sympy_symbols,
                notation=notation,
            )

        if DEBUG_RENDERER_MODE and "\\text{Error" in base_term_latex:
            logger.warning(
                "base term conversion failed; node=%r, latex=%s",
                term_node, base_term_latex,
            )

        is_first = (processed_terms_count == 0)
        
        full_term_latex = _format_full_latex_term(coeff_val, base_term_latex, is_first)
        
        rhs_latex_full_terms.append(full_term_latex)
        processed_terms_count += 1
            
    final_rhs_latex = " ".join(rhs_latex_full_terms)
    if not final_rhs_latex or processed_terms_count == 0:
        final_rhs_latex = "0"
    
    final_rhs_latex = final_rhs_latex.replace("  ", " ")

    return f"${lhs_latex} = {final_rhs_latex}$"
