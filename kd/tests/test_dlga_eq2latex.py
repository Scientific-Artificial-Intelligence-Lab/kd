import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import load_kdv_equation
from kd.model.dlga import DLGA


from kd.viz.dlga_eq2latex import (
    _module_to_latex_base_term,
    _format_full_latex_term,
    chromosome_to_latex,
    DLGA_INTERNAL_TERM_NAMES,  # 导入模块中定义的常量
    DEFAULT_LATEX_STYLE_MAP
)

from kd.viz.equation_renderer import render_latex_to_image

TERMS_FOR_TESTING = DLGA_INTERNAL_TERM_NAMES
LATEX_MAP_FOR_TESTING = DEFAULT_LATEX_STYLE_MAP

TEST_CASES_MODULE_TO_BASE_TERM = [
    (([], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "1", "空模块"),
    (([0], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "u", "单个基因 u"),
    (([1], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "u_x", "单个基因 ux"),
    (([0, 1], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "u u_x", "多个唯一基因 u*ux (排序后)"),
    (([1, 0], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "u u_x", "多个唯一基因 ux*u (排序后)"),
    (([1, 1], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "u_x^{2}", "基因 ux 平方"),
    (([0, 1, 1], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "u u_x^{2}", "基因 u * ux^2 (排序后)"),
    (([0, 1, 1, 3, 3], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "u u_x^{2} u_{xxx}^{2}", "复杂项 u*ux^2*uxxx^2 (排序后)"),
    (([3, 1, 0, 1, 3], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "u u_x^{2} u_{xxx}^{2}", "复杂项乱序输入 (排序后)"),
    (([99], TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), "未知G(99)", "未知基因索引"),
]

def run_tests_for_module_to_latex_base_term():
    """测试 _module_to_latex_base_term 函数"""
    print("\n--- 开始测试: _module_to_latex_base_term ---")
    passed_count = 0
    failed_count = 0

    for i, (test_inputs, expected_output, description) in enumerate(TEST_CASES_MODULE_TO_BASE_TERM):
        module_input, dlga_terms, latex_map = test_inputs # 解包输入参数
        test_id = f"M2B_#{i+1} ({description})"
        try:
            result = _module_to_latex_base_term(module_input, dlga_terms, latex_map)
            assert result == expected_output
            print(f"  [通过] {test_id}: 输入 {module_input} => '{result}'")
            passed_count += 1
        except AssertionError:
            print(f"  [失败] {test_id}: 输入 {module_input}")
            print(f"    预期: '{expected_output}'")
            print(f"    实际: '{result}'")
            failed_count += 1
        except Exception as e:
            print(f"  [错误] {test_id}: 输入 {module_input} 时发生意外: {e}")
            failed_count += 1
            
    print(f"--- 测试 _module_to_latex_base_term 结束: {passed_count} 通过, {failed_count} 失败 ---")
    return failed_count == 0

# 测试用例 for _format_full_latex_term
# 格式: ((coeff_val, base_term_latex, is_first_rhs_term), expected_output, description)
TEST_CASES_FORMAT_FULL_TERM = [
    ((0.5, "u_x", True), "0.5000 u_x", "正系数, 首项"),
    ((0.5, "u_x", False), "+ 0.5000 u_x", "正系数, 非首项"),
    ((-0.5, "u_x", True), "- 0.5000 u_x", "负系数, 首项"),
    ((-0.5, "u_x", False), "- 0.5000 u_x", "负系数, 非首项"),
    ((1.0, "u_x", True), "u_x", "系数1, 首项"),
    ((1.0, "u_x", False), "+ u_x", "系数1, 非首项"),
    ((-1.0, "u_x", True), "- u_x", "系数-1, 首项"),
    ((-1.0, "u_x", False), "- u_x", "系数-1, 非首项"),
    ((0.0, "u_x", True), "0.0000 u_x", "系数0, 首项"),
    ((0.0, "u_x", False), "+ 0.0000 u_x", "系数0, 非首项"),
    ((-0.00001, "u_x", True), "- 0.0000 u_x", "接近0的负数(-0.00001), 首项"),
    ((-0.00001, "u_x", False), "- 0.0000 u_x", "接近0的负数(-0.00001), 非首项"),
    ((1.23456, "u_x", True), "1.2346 u_x", "系数四舍五入"),
    ((2.5, "1", True), "2.5000", "常数项, 首项"),
    ((2.5, "1", False), "+ 2.5000", "常数项, 非首项"),
    ((-1.0, "1", True), "- 1.0000", "系数-1的常数项, 首项"),
    ((-1.0, "1", False), "- 1.0000", "系数-1的常数项, 非首项"),
]

def run_tests_for_format_full_latex_term():
    """测试 _format_full_latex_term 函数"""
    print("\n--- 开始测试: _format_full_latex_term ---")
    passed_count = 0
    failed_count = 0

    for i, (test_inputs, expected_output, description) in enumerate(TEST_CASES_FORMAT_FULL_TERM):
        coeff, base, is_first = test_inputs # 解包输入参数
        test_id = f"FFT_#{i+1} ({description})"
        try:
            result = _format_full_latex_term(coeff, base, is_first)
            assert result == expected_output
            print(f"  [通过] {test_id}: 输入 ({coeff:.5f}, '{base}', {is_first}) => '{result}'")
            passed_count += 1
        except AssertionError:
            print(f"  [失败] {test_id}: 输入 ({coeff:.5f}, '{base}', {is_first})")
            print(f"    预期: '{expected_output}'")
            print(f"    实际: '{result}'")
            failed_count += 1
        except Exception as e:
            print(f"  [错误] {test_id}: 输入 ({coeff:.5f}, '{base}', {is_first}) 时发生意外: {e}")
            failed_count += 1

    print(f"--- 测试 _format_full_latex_term 结束: {passed_count} 通过, {failed_count} 失败 ---")
    return failed_count == 0

CUSTOM_LATEX_MAP_FOR_TESTING = { # 用于测试不同的显示风格
    "u": "U", "ux": r"U_x", "uxx": r"U_{xx}", "uxxx": r"U_{xxx}",
    "ut": r"\frac{\partial U}{\partial t}", "utt": r"\frac{\partial^2 U}{\partial t^2}"
}

dlga_output_chromosome_data = [[3], [1, 2], [0, 1], [1], [1, 3], [0, 1, 1, 3, 3]]
dlga_output_coefficients_data = np.array([
    [-1.90280071e-03], [-9.76468872e-04], [-9.55166272e-01],
    [1.26532346e-01], [-5.66218238e-06], [-1.47671842e-11]
])
expected_dlga_output_latex_data = "$u_t = - 0.0019 u_{xxx} - 0.0010 u_x u_{xx} - 0.9552 u u_x + 0.1265 u_x - 0.0000 u_x u_{xxx} + 0.0000 u u_x^{2} u_{xxx}^{2}$"

# 测试用例 for chromosome_to_latex
# 格式: ((chromosome, coefficients, lhs_name, dlga_terms, latex_map), expected_output, description)
TEST_CASES_CHROMOSOME_TO_LATEX = [
    (([], np.array([]), "ut", TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), 
        "$u_t = 0$", "空染色体"),
    (([[1]], np.array([[0.5]]), "ut", TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), 
        "$u_t = 0.5000 u_x$", "单项正系数"),
    (([[1]], np.array([[1.0]]), "ut", TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), 
        "$u_t = u_x$", "单项系数1"),
    (([[1]], np.array([[-1.0]]), "ut", TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), 
        "$u_t = - u_x$", "单项系数-1"),
    (([[0],[1]], np.array([[-2.0],[0.5]]), "ut", TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), 
        "$u_t = - 2.0000 u + 0.5000 u_x$", "多项混合符号"),
    (([[0,1],[3]], np.array([[-1.0],[0.53218]]), "ut", TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING), 
        "$u_t = - u u_x + 0.5322 u_{xxx}$", "复杂项与系数-1和四舍五入"),
    (([[1]], np.array([[0.75]]), "ut", TERMS_FOR_TESTING, CUSTOM_LATEX_MAP_FOR_TESTING), # 使用自定义map
        r"$\frac{\partial U}{\partial t} = 0.7500 U_x$", "自定义LaTeX map"),
    ((dlga_output_chromosome_data, dlga_output_coefficients_data, "ut", TERMS_FOR_TESTING, LATEX_MAP_FOR_TESTING),
        expected_dlga_output_latex_data, "实际DLGA输出数据"),
]

def run_tests_for_chromosome_to_latex():
    """测试 chromosome_to_latex 主函数"""
    print("\n--- 开始测试: chromosome_to_latex (主函数) ---")
    passed_count = 0
    failed_count = 0
    
    for i, (test_inputs, expected_output, description) in enumerate(TEST_CASES_CHROMOSOME_TO_LATEX):
        chrom, coeffs, lhs, terms, lmap = test_inputs # 解包输入参数
        test_id = f"C2L_#{i+1} ({description})"
        try:
            result = chromosome_to_latex(chrom, coeffs, lhs, terms, lmap)
            # 对于来自实际DLGA输出的复杂情况，进行特殊打印以帮助调试
            if description == "实际DLGA输出数据":
                print(f"    详细信息 ({test_id}):")
                print(f"      输入染色体: {chrom}")
                print(f"      输入系数 (shape {coeffs.shape}): \n{coeffs}")
                print(f"      输入LHS: {lhs}")
                print(f"      生成 LaTeX: {result}")
                print(f"      预期 LaTeX: {expected_output}")


            assert result == expected_output
            print(f"  [通过] {test_id}: LHS '{lhs}', Chrom(len {len(chrom)}) => '{result}'")
            passed_count += 1
        except AssertionError:
            print(f"  [失败] {test_id}: LHS '{lhs}', Chrom(len {len(chrom)})")
            if description != "实际DLGA输出数据": # 复杂数据已在上面打印
                print(f"    预期: '{expected_output}'")
                print(f"    实际: '{result}'")
            failed_count += 1
        except Exception as e:
            print(f"  [错误] {test_id}: LHS '{lhs}', Chrom(len {len(chrom)}) 时发生意外: {e}")
            failed_count += 1
            
    print(f"--- 测试 chromosome_to_latex 结束: {passed_count} 通过, {failed_count} 失败 ---")
    return failed_count == 0

def run_test_dlga_eq2latex():
    # load KdV equation data
    kdv_data = load_kdv_equation()
    x, t, u = kdv_data.x, kdv_data.t, kdv_data.usol
    X_train, y_train = kdv_data.sample(n_samples=1000)
    # init and train model
    model = DLGA(epi=0.01, input_dim=2, max_iter=3000)  # 2D input: (x,t)
    model.fit(X_train, y_train)
    
    render_latex_to_image(model.eq_latex, output_path=".")

    pass

if __name__ == "__main__":
    
    print("="*50)

    all_tests_passed_summary = {}

    all_tests_passed_summary["module_to_base_term"] = run_tests_for_module_to_latex_base_term()
    all_tests_passed_summary["format_full_term"] = run_tests_for_format_full_latex_term()
    all_tests_passed_summary["chromosome_to_latex"] = run_tests_for_chromosome_to_latex()

    print("\n" + "="*20 + " 测试总结 " + "="*20)
    final_result_ok = True
    for test_suite_name, passed_suite in all_tests_passed_summary.items():
        status_msg = "通过" if passed_suite else "存在失败"
        print(f"'{test_suite_name}': {status_msg}")
        if not passed_suite:
            final_result_ok = False
    
    print("="*50)
    if final_result_ok:
        print("所有选定测试均已通过！")
    else:
        print("部分测试失败")

    print("\n运行实际的 DLGA 方程到 LaTeX 测试:")
    run_test_dlga_eq2latex()