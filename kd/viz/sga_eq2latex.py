"""
SGA equation to LaTeX converter for the KD framework.

This module provides functionality to convert SGA-PDE discovered equations
into LaTeX format, compatible with the KD visualization system.
"""

import re
import sympy
from sympy.parsing.sympy_parser import parse_expr


def sga_eq2latex(equation_obj, lhs_name="u_t"):
    """
    Convert SGA equation object to LaTeX format.
    
    Parameters
    ----------
    equation_obj : object
        SGA equation object with concise_visualize() method.
    lhs_name : str, default="u_t"
        Left-hand side name for the equation.
        
    Returns
    -------
    latex_str : str
        LaTeX representation of the equation.
    """
    if equation_obj is None:
        return f"${lhs_name} = 0$"
    
    try:
        # Get the string representation from SGA
        eq_str = equation_obj.concise_visualize()
        
        # Convert SGA notation to LaTeX-friendly format
        latex_rhs = _convert_sga_to_latex(eq_str)
        
        return f"${lhs_name} = {latex_rhs}$"
        
    except Exception as e:
        print(f"[sga_eq2latex] Error converting equation: {e}")
        return f"${lhs_name} = \\text{{Error: {str(e)}}}$"


def _convert_sga_to_latex(sga_str):
    """
    Convert SGA string notation to LaTeX format.
    
    Parameters
    ----------
    sga_str : str
        SGA equation string (e.g., "-0.9849u + 0.7137( ^3 u)")
        
    Returns
    -------
    latex_str : str
        LaTeX-formatted string.
    """
    if not sga_str:
        return "0"
    
    # Start with the original string
    latex_str = sga_str
    
    # Handle derivative notations
    # Convert ( ^n u) to u^{(n)} for higher derivatives
    latex_str = re.sub(r'\(\s*\^\s*(\d+)\s+u\s*\)', r'u^{(\1)}', latex_str)
    
    # Convert ux to u_x, uxx to u_{xx}, etc.
    latex_str = re.sub(r'u([x]+)', lambda m: f'u_{{{m.group(1)}}}', latex_str)
    latex_str = re.sub(r'u([t]+)', lambda m: f'u_{{{m.group(1)}}}', latex_str)
    
    # Handle mixed derivatives like uxt -> u_{xt}
    latex_str = re.sub(r'u([xt]+)', lambda m: f'u_{{{m.group(1)}}}', latex_str)
    
    # Convert d x, d^2 x, etc. to \partial_x, \partial_{xx}, etc.
    latex_str = re.sub(r'd\s*\^\s*(\d+)\s*x', r'\\partial_{' + 'x' * 2 + r'}', latex_str)
    latex_str = re.sub(r'd\s*x', r'\\partial_x', latex_str)
    latex_str = re.sub(r'd\s*t', r'\\partial_t', latex_str)
    
    # Handle multiplication signs
    latex_str = re.sub(r'\*', r' \\cdot ', latex_str)
    
    # Handle division
    latex_str = re.sub(r'/', r' / ', latex_str)
    
    # Clean up extra spaces
    latex_str = re.sub(r'\s+', ' ', latex_str).strip()
    
    return latex_str


def sga_equation_to_sympy(equation_obj):
    """
    Convert SGA equation to SymPy expression for advanced processing.
    
    Parameters
    ----------
    equation_obj : object
        SGA equation object.
        
    Returns
    -------
    sympy_expr : sympy.Expr or None
        SymPy expression, or None if conversion fails.
    """
    if equation_obj is None:
        return None
        
    try:
        eq_str = equation_obj.concise_visualize()
        
        # Define symbols for SymPy
        u, x, t = sympy.symbols('u x t')
        ux, ut, uxx, uxt, utt = sympy.symbols('u_x u_t u_xx u_xt u_tt')
        
        # Create a mapping for SGA notation to SymPy symbols
        symbol_map = {
            'u': u,
            'x': x, 
            't': t,
            'ux': ux,
            'ut': ut,
            'uxx': uxx,
            'uxt': uxt,
            'utt': utt,
        }
        
        # Simple preprocessing for SymPy parsing
        sympy_str = eq_str
        
        # Replace SGA derivative notation
        sympy_str = re.sub(r'\(\s*\^\s*2\s+u\s*\)', 'uxx', sympy_str)
        sympy_str = re.sub(r'\(\s*\^\s*3\s+u\s*\)', 'uxxx', sympy_str)
        
        # Parse with SymPy
        expr = parse_expr(sympy_str, local_dict=symbol_map, transformations='all')
        return expr
        
    except Exception as e:
        print(f"[sga_equation_to_sympy] Error: {e}")
        return None


# Integration with existing KD viz system
def format_sga_latex_term(coeff_val, base_term_latex, is_first_term=False):
    """
    Format a single term with coefficient for LaTeX display.
    
    This function is compatible with the existing KD viz formatting system.
    
    Parameters
    ----------
    coeff_val : float
        Coefficient value.
    base_term_latex : str
        LaTeX string for the base term.
    is_first_term : bool, default=False
        Whether this is the first term in the equation.
        
    Returns
    -------
    formatted_term : str
        Formatted LaTeX term.
    """
    # Handle coefficient formatting
    if abs(coeff_val) < 1e-10:
        return ""
    
    # Format coefficient
    if abs(coeff_val - 1.0) < 1e-10:
        coeff_str = ""
    elif abs(coeff_val + 1.0) < 1e-10:
        coeff_str = "-"
    else:
        coeff_str = f"{coeff_val:.4g}"
    
    # Handle signs
    if is_first_term:
        if coeff_val < 0:
            sign = "-"
            coeff_str = coeff_str.lstrip('-')
        else:
            sign = ""
    else:
        if coeff_val < 0:
            sign = " - "
            coeff_str = coeff_str.lstrip('-')
        else:
            sign = " + "
    
    # Combine coefficient and term
    if coeff_str == "":
        if base_term_latex == "1":
            return f"{sign}1"
        else:
            return f"{sign}{base_term_latex}"
    elif coeff_str == "1":
        if base_term_latex == "1":
            return f"{sign}1"
        else:
            return f"{sign}{base_term_latex}"
    else:
        if base_term_latex == "1":
            return f"{sign}{coeff_str}"
        else:
            return f"{sign}{coeff_str} {base_term_latex}"
