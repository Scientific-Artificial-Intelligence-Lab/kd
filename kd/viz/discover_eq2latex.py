from typing import Optional

import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.latex import LatexPrinter

from kd.viz.dlga_eq2latex import _format_full_latex_term

# FIXME: This list is static; ideally it should be derived from Program.library
# or injected as a parameter.
DEEPRL_SYMBOLS_FOR_SYMPY = {
    name: sympy.Symbol(name) for name in ['u1', 'x1', 'x2', 'x3', 'c'] # 'c' 代表可能的常数符号
}

# ... (imports, DEEPRL_SYMBOLS_FOR_SYMPY, DEBUG_RENDERER_MODE) ...

DEBUG_RENDERER_MODE = False  # Whether to enable verbose debug logging.

_SYMBOL_DISPLAY = {"u1": "u", "x1": "x", "x2": "y", "x3": "z"}

# Function-based symbols for Leibniz mode (produces \partial instead of d)
_x1, _x2, _x3 = sympy.symbols('x1 x2 x3')
_LEIBNIZ_SYMBOLS_FOR_SYMPY = {
    'u1': sympy.Function('u1')(_x1, _x2, _x3),
    'x1': _x1, 'x2': _x2, 'x3': _x3,
    'c': sympy.Symbol('c'),
}


class _SubscriptLatexPrinter(LatexPrinter):
    """Render Derivative(u, x, x) as u_{xx} instead of Leibniz notation."""

    def _print_Derivative(self, expr):
        func = expr.args[0]
        if isinstance(func, sympy.Symbol):
            func_name = _SYMBOL_DISPLAY.get(func.name, func.name)
        else:
            func_name = _SYMBOL_DISPLAY.get(func.func.name, func.func.name)

        subscript_parts = []
        for var, count in expr.variable_count:
            var_name = _SYMBOL_DISPLAY.get(var.name, var.name)
            subscript_parts.append(var_name * count)

        subscript = "".join(subscript_parts)
        return f"{func_name}_{{{subscript}}}"

    def _print_Symbol(self, expr):
        name = expr.name
        return _SYMBOL_DISPLAY.get(name, super()._print_Symbol(expr))


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
             print(f"[discover_eq2latex INFO] Node '{repr(term_node_obj)}'.to_sympy_string() -> '{sympy_expr_str}'")

        parsed_sympy_expr = parse_expr(
            sympy_expr_str,
            local_dict=local_sympy_symbols,
            transformations='all',  # allow a broad set of SymPy-compatible transformations
        )
        if DEBUG_RENDERER_MODE:
            print(f"[discover_eq2latex INFO] Parsed SymPy expression: {parsed_sympy_expr}")

        if notation == "subscript":
            latex_output = _SubscriptLatexPrinter().doprint(parsed_sympy_expr)
        else:
            latex_output = sympy.latex(parsed_sympy_expr)
        return latex_output
    except Exception as e:
        import traceback
        error_repr = repr(term_node_obj) if term_node_obj else "None"
        sympy_str_val = sympy_expr_str if 'sympy_expr_str' in locals() else "<sympy string not generated>"
        print(
            "[discover_eq2latex ERROR] "
            f"_discover_term_node_to_latex failed for node '{error_repr}' "
            f"(SymPy str: '{sympy_str_val}'): {e}"
        )
        if DEBUG_RENDERER_MODE:
             traceback.print_exc()
        return f"\\text{{Error converting node: {error_repr}}}"

_VALID_NOTATIONS = ("subscript", "leibniz")

def discover_program_to_latex(program_object, # lhs_name_str,
                            # 可选参数，如果需要覆盖模块级的默认值
                            custom_lhs_latex_map=None,
                            custom_deeprl_symbols=None,
                            lhs_axis: Optional[str] = None,
                            notation: str = "subscript"):
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
            print("[discover_eq2latex WARNING] program_object is invalid or missing w/STRidge.terms.")
            print(f"  program_object: {program_object}")
            print(f"  w: {w_status}")
            print(f"  STRidge.terms: {terms_status}")
            if hasattr(program_object, 'w') and hasattr(program_object.STRidge, 'terms') and \
               program_object.w is not None and program_object.STRidge.terms is not None:
                 print(f"  len(w)={len(program_object.w)}, len(STRidge.terms)={len(program_object.STRidge.terms)}")
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
        base_term_latex = _discover_term_node_to_latex(
            term_node,
            local_sympy_symbols=current_sympy_symbols,
            notation=notation,
        )

        if DEBUG_RENDERER_MODE and "\\text{Error" in base_term_latex:
            print(
                "[discover_eq2latex WARNING] base term conversion failed; "
                f"node={repr(term_node)}, latex={base_term_latex}"
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
