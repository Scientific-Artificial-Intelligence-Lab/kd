# dlga_eq2latex.py

import numpy as np # coefficients 会是 numpy 数组
from collections import Counter # 用于帮助计算重复基因，以实现幂次
import matplotlib.pyplot as plt
import os


# 这个列表定义了 DLGA 内部如何通过索引来查找项的名称。
DLGA_INTERNAL_TERM_NAMES = ["u", "ux", "uxx", "uxxx", "ut", "utt"]

# 默认的 LaTeX 风格映射，可以按需修改或通过 config 传入
DEFAULT_LATEX_STYLE_MAP = {
    "u": "u",
    "ux": "u_x",
    "uxx": "u_{xx}",
    "uxxx": "u_{xxx}",
    "ut": "u_t",    # 更标准的 LaTeX 可以是 r"\frac{\partial u}{\partial t}"
    "utt": "u_{tt}", # r"\frac{\partial^2 u}{\partial t^2}"
}


def _module_to_latex_base_term(module_indices, dlga_term_names, latex_style_map):
    """
    将单个模块（基因索引列表）转换为 LaTeX 基础项字符串（不含系数，但含幂次）。
    例如：[0, 1, 1] -> "u u_x^2"
    """
    if not module_indices: # TODO 检查在dlga中 空基因 是否被定义
        return "1" # 空模块通常代表常数项的基础部分 (乘以系数后即为常数)

    gene_counts = Counter(module_indices)
    latex_parts = []

    # 按基因索引排序以确保一致的项顺序 optional 例如，对于模块 [1, 0] (ux, u)，我们总是得到 "u u_x" 而不是有时 "u_x u"。
    for gene_idx in sorted(gene_counts.keys()):
        count = gene_counts[gene_idx]
        term_name_dlga = dlga_term_names[gene_idx] if 0 <= gene_idx < len(dlga_term_names) else f"未知G({gene_idx})"
        term_latex_base = latex_style_map.get(term_name_dlga, term_name_dlga)

        if count > 1: #    如果基因出现次数大于1，则添加 LaTeX 的幂次表示 "^{count}"
            latex_parts.append(f"{term_latex_base}^{{{count}}}")
        else:
            latex_parts.append(term_latex_base)
    
    return " ".join(latex_parts)
    # 示例 (latex_parts = ["u", "u_x^{2}", "u_{xxx}^{2}"]):
    #       返回 "u u_x^{2} u_{xxx}^{2}"


def _format_full_latex_term(coeff_val, base_term_latex, is_first_rhs_term):
    """
    格式化单个完整的 LaTeX 项，包括系数和符号。
    coeff_val: 浮点数系数
    base_term_latex: 模块转换的基础 LaTeX 项，如 "u u_x^2"
    is_first_rhs_term: 布尔值，指示这是否是 RHS 的第一个项
    """
    # 格式化系数字符串 保留4位小数 去掉符号
    coeff_abs_str = f"{abs(coeff_val):.4f}" # 取绝对值，符号单独处理

    # 处理系数为 0.0 的情况，确保显示为 "0.0000" 而非 "-0.0000"
    if np.isclose(coeff_val, 0.0):
        sign = "" # 对于0，我们不在数字前加符号，后续逻辑决定是否加 "+"
        coeff_display = "0.0000"
    elif coeff_val > 0:
        sign = "+"
        coeff_display = coeff_abs_str
    else: # coeff_val < 0
        sign = "-"
        coeff_display = coeff_abs_str

    # 简化系数为 1 或 -1 的情况
    if np.isclose(abs(coeff_val), 1.0) and base_term_latex != "1": # 如果基础项不是纯常数"1"
        formatted_term = base_term_latex
    elif base_term_latex == "1": # 如果基础项是"1"（纯常数项）最终项就是系数本身
        formatted_term = coeff_display
    else: # 一般情况：系数 + 空格 + 基础项 例如 coeff=0.5, base="u_x" => formatted_term="0.5000 u_x"
        formatted_term = f"{coeff_display} {base_term_latex}"

    # 根据是否为首项和符号添加前导运算符
    if is_first_rhs_term:
        if sign == "-":
            return f"- {formatted_term}"
        else: # 正数或零，首项不加 "+"
            return formatted_term
    else: # 非首项
        if sign == "-":
            return f"- {formatted_term}" # coeff=-0.5, base="u_x", is_first=False => "- 0.5000 u_x"
        else: # 正数或零，后续项加 "+"
            return f"+ {formatted_term}"


def chromosome_to_latex(chromosome, coefficients, lhs_name_str,
                        dlga_term_names=DLGA_INTERNAL_TERM_NAMES,
                        latex_style_map=DEFAULT_LATEX_STYLE_MAP):
    """
    主函数：将 DLGA 染色体、系数和LHS名称转换为完整的 LaTeX 方程字符串。
    """
    # 1. LHS to LaTeX: 示例: lhs_name_str = "ut", latex_style_map["ut"] = "u_t" => lhs_latex = "u_t"
    lhs_latex = latex_style_map.get(lhs_name_str, lhs_name_str)

    # 2. 处理空染色体或空系数的情况
    if not chromosome or coefficients is None or coefficients.size == 0:
        return f"${lhs_latex} = 0$"

    rhs_latex_full_terms = []
    processed_terms_count = 0 # 用于追踪是否是第一个有效RHS项

    # 3. 遍历染色体中的每个模块（项）及其对应的系数:
    for i, module_indices in enumerate(chromosome):
        if i >= len(coefficients): # 安全检查，确保系数存在
            print(f"[警告] 模块索引 {i} 超出系数数组边界。")
            continue
            
        # 获取当前项的系数 coefficients 是一个 (N, 1) 的 NumPy 数组 coefficients[i] 是一个包含单个元素的小数组 (如 [-0.0019])，
        # 用 coefficients[i, 0] 来获取那个浮点数值
        coeff_val = coefficients[i, 0]

        # [optional] 跳过系数绝对值非常小的项
        # if np.isclose(coeff_val, 0.0, atol=1e-7): 
        #     continue 
        # 我们目前包含所有项

        # 4. 用辅助函数获取当前的基础 LaTeX 项 (不含系数，但有幂次):
        base_term = _module_to_latex_base_term(module_indices, dlga_term_names, latex_style_map)
        
        # 5. 用辅助函数格式化完整的 LaTeX 项 含系数和符号:
        is_first = (processed_terms_count == 0)
        full_term_latex = _format_full_latex_term(coeff_val, base_term, is_first)
        
        # 更新结果
        rhs_latex_full_terms.append(full_term_latex)
        processed_terms_count += 1
    
    # 6. 用空格组合所有 RHS 项:
    final_rhs_latex = " ".join(rhs_latex_full_terms)
    if not final_rhs_latex: # 如果所有项都被跳过了（例如都接近于0）
        final_rhs_latex = "0"
    
    # 再次清理一下，例如 "u_t = + 0.5 u_x" 中的 "+ " (如果首项为正且_format_full_latex_term返回了前导+)
    # _format_full_latex_term 设计上首项为正时不带 "+", 所以这里可能不需要了
    # " ".join() 应该不会引入额外的双空格 但保留这个替换通常无害
    final_rhs_latex = final_rhs_latex.replace("  ", " ")

    return f"${lhs_latex} = {final_rhs_latex}$"



def dlga_eq2latex(chromosome, coefficients, lhs_name_str) -> str:
    """
    Receives and prints the core equation data passed from DLGA.
    This is the first step, used to verify the data flow.
    """
    print("\n[Equation Renderer INFO] Successfully received data from DLGA:")
    print(f"  Original chromosome (list of modules, a module is a list of gene indices):")
    for i, module in enumerate(chromosome):
        print(f"    Term {i+1} - Module (gene indices): {module}")
        # Further explain the meaning of the module
        term_parts = [DLGA_INTERNAL_TERM_NAMES[idx] if 0 <= idx < len(DLGA_INTERNAL_TERM_NAMES) else f"Unknown Gene({idx})" for idx in module]
        print(f"      Meaning (product of terms): {' * '.join(term_parts) if term_parts else 'Base constant term'}")

    print(f"  Coefficients (NumPy array, corresponding to each module/term in the chromosome):")
    for i, coeff_row in enumerate(coefficients): # coefficients is an (N, 1) array
        print(f"    Term {i+1} - Coefficient: {coeff_row[0]:.4f}")
        
    print(f"  Left-hand side term name (string): {lhs_name_str}")
    print(f"  (Note: Gene indices in the chromosome map to the following list: {DLGA_INTERNAL_TERM_NAMES})")
    print("--- Equation Renderer data reception end ---\n")

    # Generate LaTeX string
    dlga_latex_str = chromosome_to_latex(
        chromosome=chromosome,
        coefficients=coefficients,
        lhs_name_str=lhs_name_str
        # dlga_term_names and latex_style_map will use the function's default values
    )

    print(f"[Equation Renderer INFO]: LaTeX: {dlga_latex_str}")

    return dlga_latex_str