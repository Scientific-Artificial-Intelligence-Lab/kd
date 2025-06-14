import sympy
from sympy.parsing.sympy_parser import parse_expr

# FIXME: 这里可能有问题
# 理想情况下，这个列表应该从 Program.library 动态获取或作为参数传入
DEEPRL_SYMBOLS_FOR_SYMPY = {
    name: sympy.Symbol(name) for name in ['u1', 'x1', 'x2', 'x3', 'c'] # 'c' 代表可能的常数符号
}

# ... (imports, DEEPRL_SYMBOLS_FOR_SYMPY, DEBUG_RENDERER_MODE) ...

DEBUG_RENDERER_MODE = True # 是否启用调试模式

# Node -> LaTeX
def _discover_term_node_to_latex(term_node_obj, local_sympy_symbols=None):
    if local_sympy_symbols is None:
        local_sympy_symbols = DEEPRL_SYMBOLS_FOR_SYMPY
    try:
        if not hasattr(term_node_obj, 'to_sympy_string'): # 检查方法是否存在
            raise AttributeError("Node 对象没有 to_sympy_string 方法")

        sympy_expr_str = term_node_obj.to_sympy_string()

        if DEBUG_RENDERER_MODE:
             print(f"[discover_eq2latex INFO] Node '{repr(term_node_obj)}'.to_sympy_string() -> '{sympy_expr_str}'")

        parsed_sympy_expr = parse_expr(sympy_expr_str, 
                                       local_dict=local_sympy_symbols, 
                                       transformations='all') # transformations='all' 可能有助于解析更广泛的 SymPy 兼容字符串
        if DEBUG_RENDERER_MODE:
            print(f"[discover_eq2latex INFO] 解析后的SymPy表达式: {parsed_sympy_expr}")

        latex_output = sympy.latex(parsed_sympy_expr)
        return latex_output
    except Exception as e: # ... (错误处理) ...
        import traceback # 导入 traceback 以便打印更详细的错误信息
        error_repr = repr(term_node_obj) if term_node_obj else "None"
        sympy_str_val = sympy_expr_str if 'sympy_expr_str' in locals() else "未成功生成SymPy字符串"
        print(f"[ERROR] _discover_term_node_to_latex 中转换节点 '{error_repr}' (SymPy str: '{sympy_str_val}') 失败: {e}")
        if DEBUG_RENDERER_MODE:
             traceback.print_exc()
        return f"\\text{{Error converting node: {error_repr}}}"
    
from kd.viz.dlga_eq2latex import _format_full_latex_term

def discover_program_to_latex(program_object, # lhs_name_str,
                            # 可选参数，如果需要覆盖模块级的默认值
                            custom_lhs_latex_map=None, 
                            custom_deeprl_symbols=None):
    """
    将 Program 对象转换为完整的 LaTeX 方程字符串。
    """
    # 1. 处理 LHS
    # current_lhs_map = custom_lhs_latex_map if custom_lhs_latex_map else DEFAULT_LATEX_STYLE_MAP
    # lhs_latex = current_lhs_map.get(lhs_name_str, lhs_name_str)
    lhs_latex = "u_t"

    # 2. 检查 program_object 是否有效且包含所需属性
    if not (program_object and \
            hasattr(program_object, 'w') and \
            hasattr(program_object, 'STRidge') and \
            hasattr(program_object.STRidge, 'terms') and \
            program_object.w is not None and \
            program_object.STRidge.terms is not None and \
            len(program_object.w) == len(program_object.STRidge.terms)):
        if DEBUG_RENDERER_MODE:
            w_status = str(getattr(program_object, 'w', '未找到w属性'))
            terms_status = str(getattr(getattr(program_object, 'STRidge', None), 'terms', '未找到STRidge.terms'))
            print(f"[渲染器警告] discover_program_to_latex: program_object 无效或缺少 w/STRidge.terms。")
            print(f"  program_object: {program_object}")
            print(f"  w: {w_status}")
            print(f"  STRidge.terms: {terms_status}")
            if hasattr(program_object, 'w') and hasattr(program_object.STRidge, 'terms') and \
               program_object.w is not None and program_object.STRidge.terms is not None:
                 print(f"  len(w)={len(program_object.w)}, len(STRidge.terms)={len(program_object.STRidge.terms)}")
        return f"${lhs_latex} = 0 \\; (\\text{{Error: Invalid program structure}})$"

    coefficients = program_object.w
    term_nodes = program_object.STRidge.terms
    
    if not term_nodes: # 如果没有基础项
        return f"${lhs_latex} = 0$"

    # 3. 构建 RHS
    rhs_latex_full_terms = []
    processed_terms_count = 0 # 用于正确传递 is_first_rhs_term 给 _format_full_latex_term

    current_sympy_symbols = custom_deeprl_symbols if custom_deeprl_symbols else DEEPRL_SYMBOLS_FOR_SYMPY

    for i, coeff_val_from_w in enumerate(coefficients):
        # program.w 可能是一个普通列表或1D NumPy数组
        coeff_val = float(coeff_val_from_w) # 确保是 Python 浮点数
        term_node = term_nodes[i]
        
        # (可选) 跳过系数绝对值非常小的项
        # if np.isclose(coeff_val, 0.0, atol=1e-7): 
        #     if DEBUG_RENDERER_MODE:
        #         print(f"[渲染器 INFO] 跳过项 {i+1} (Node: {repr(term_node)})，因其系数 {coeff_val:.3e} 接近零。")
        #     continue

        # 获取基础项的 LaTeX
        base_term_latex = _discover_term_node_to_latex(
            term_node,
            local_sympy_symbols=current_sympy_symbols
        )

        # 如果基础项转换出错，base_term_latex 会包含错误信息 我们依然尝试格式化它，以便错误能显示在最终方程中
        if DEBUG_RENDERER_MODE and "\\text{Error" in base_term_latex:
            print(f"[渲染器警告] 基础项 {repr(term_node)} 转换为LaTeX时出错，内容为: {base_term_latex}")

        is_first = (processed_terms_count == 0)
        
        # 使用通用的 _format_full_latex_term 来组合系数和基础项LaTeX
        full_term_latex = _format_full_latex_term(coeff_val, base_term_latex, is_first)
        
        rhs_latex_full_terms.append(full_term_latex)
        processed_terms_count += 1
            
    final_rhs_latex = " ".join(rhs_latex_full_terms)
    if not final_rhs_latex or processed_terms_count == 0 : # 如果所有项都被跳过或列表为空
        final_rhs_latex = "0"
    
    final_rhs_latex = final_rhs_latex.replace("  ", " ") # 清理多余空格

    return f"${lhs_latex} = {final_rhs_latex}$"