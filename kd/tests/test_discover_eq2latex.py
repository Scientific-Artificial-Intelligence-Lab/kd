import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

import scipy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='numpy.*')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.*')
from kd.model import DeepRL

model = DeepRL(
    n_samples_per_batch = 500, # Number of generated traversals by agent per batch
    binary_operators = ['add',"mul", "diff","diff2"],
    unary_operators = ['n2'],
)

np.random.seed(42)
def prepare_data():
    
    data = scipy.io.loadmat('./kd/data_file/burgers2.mat')
    t = np.real(data['t'].flatten()[:,None])
    x = np.real(data['x'].flatten()[:,None])
    Exact = np.real(data['usol']).T  # t first
    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0) 

    x_len = len(x)
    total_num = X_star.shape[0]
    sample_num = int(total_num*0.1)
    print(f"random sample number: {sample_num} ")
    ID = np.random.choice(total_num, sample_num, replace = False)
    X_u_meas = X_star[ID,:]
    u_meas = u_star[ID,:]
    return X_u_meas,u_meas, lb,ub

x,y,lb,ub = prepare_data()
model.import_inner_data(dataset='Burgers', data_type='regular')
step_output = model.train(n_epochs=200)
print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")
# model.plot(fig_type ='tree').view()
# model.plot(fig_type='evolution')

# # =====================================================================
# # 新增：测试 DeepRL 单个基础项的 LaTeX 转换 (使用 Node.to_sympy_string())
# # =====================================================================
# print("\n" + "="*40)
# print("开始测试 DeepRL 单个基础项的 LaTeX 转换")
# print("预期 Node 对象拥有 .to_sympy_string() 方法")
# print("="*40)

# # 1. 导入必要的 SymPy 和我们自定义的函数
# # import sympy # 如果 _discover_term_node_to_latex 中已导入，这里可选
# # from sympy.parsing.sympy_parser import parse_expr # 同上

# # 从您新建的渲染器模块导入测试目标函数和所需常量
# # 确保 kd.viz.discover_eq2latex 在 sys.path 中
# try:
#     from kd.viz.discover_eq2latex import _discover_term_node_to_latex, DEEPRL_SYMBOLS_FOR_SYMPY
#     renderer_imported_successfully = True
# except ImportError as e:
#     print(f"[测试脚本错误] 无法导入 discover_eq2latex 模块或其内容: {e}")
#     print("请确保该文件存在于 kd/viz/ 目录下，并且包含了 _discover_term_node_to_latex 和 DEEPRL_SYMBOLS_FOR_SYMPY。")
#     renderer_imported_successfully = False

# if renderer_imported_successfully:
#     # 2. 从训练结果中获取 Program 对象
#     # 'program' 是 DeepRL 输出中常见的键名，存储了 Program 对象实例
#     if 'program' in step_output and step_output['program'] is not None:
#         best_program_object = step_output['program']
        
#         # 3. 检查 STRidge.terms 是否存在且不为空
#         if hasattr(best_program_object, 'STRidge') and \
#            hasattr(best_program_object.STRidge, 'terms') and \
#            best_program_object.STRidge.terms:
            
#             print(f"\nProgram 对象中 STRidge 识别出的基础项数量: {len(best_program_object.STRidge.terms)}")
#             print(f"将尝试为每个基础项生成 LaTeX：")

#             # 4. 遍历并转换每个基础项 (Node 对象)
#             for i, term_node in enumerate(best_program_object.STRidge.terms):
#                 term_repr_original = repr(term_node) # 例如 "diff2(u1,x1)"
#                 print(f"\n--- 正在转换基础项 #{i+1}: {term_repr_original} (类型: {type(term_node)}) ---")
                
#                 # 确保 Node 对象有 to_sympy_string 方法 (您已添加到 stridge.Node 类中)
#                 if not hasattr(term_node, 'to_sympy_string'):
#                     print(f"    [错误] Node 对象 '{term_repr_original}' 没有 to_sympy_string 方法。")
#                     print(f"    请检查您的 kd/model/discover/stridge.py 文件中的 Node 类定义。")
#                     continue

#                 # 调用转换函数
#                 latex_for_term = _discover_term_node_to_latex(
#                     term_node, # 直接传递 Node 对象
#                     local_sympy_symbols=DEEPRL_SYMBOLS_FOR_SYMPY # 使用渲染器中定义的符号表
#                 )
                
#                 print(f"    基础项 '{term_repr_original}' 的 LaTeX 输出为: {latex_for_term}")

#                 # 附注：在这一步，我们主要关注转换是否成功以及大致的输出。
#                 # 精确的断言可以在确认几个典型案例的正确输出后再添加。
#                 # 例如，如果 term_repr_original == "diff2(u1,x1)":
#                 #    expected_latex = "\\frac{\\partial^{2} u_{1}}{\\partial x_{1}^{2}}" # 或 SymPy 的其他等效输出
#                 #    print(f"    (参考此项的预期 LaTeX: {expected_latex} )")
#                 #    # assert latex_for_term == expected_latex 
#                 # elif term_repr_original == "diff(n2(u1),x1)":
#                 #    expected_latex = "\\frac{\\partial}{\\partial x_{1}} u_{1}^{2}"
#                 #    print(f"    (参考此项的预期 LaTeX: {expected_latex} )")
#                 #    # assert latex_for_term == expected_latex

#             # 进一步：我们可以尝试转换 `step_output['expression']` 中的所有项
#             if hasattr(best_program_object, 'w') and len(best_program_object.w) == len(best_program_object.STRidge.terms):
#                 print("\n--- 尝试组合系数与转换后的基础项 (概念验证) ---")
#                 full_equation_parts = []
#                 # 我们需要复用或重新实现 _format_full_latex_term
#                 # 这里暂时简单拼接，主要看基础项转换是否成功
#                 for i, coeff_val in enumerate(best_program_object.w):
#                     term_node = best_program_object.STRidge.terms[i]
#                     base_latex = _discover_term_node_to_latex(term_node, DEEPRL_SYMBOLS_FOR_SYMPY)
                    
#                     # 简化的系数处理和拼接，没有 _format_full_latex_term 的精细逻辑
#                     term_str = f"{coeff_val:.4f} \\cdot ({base_latex})" 
#                     if i > 0 and coeff_val >= 0:
#                         full_equation_parts.append(f"+ {term_str}")
#                     else:
#                         full_equation_parts.append(term_str)
                
#                 # 假设 LHS 是 u_t
#                 lhs_latex_test = "u_t" # 从渲染器获取ut的LaTeX
#                 print(f"概念验证 - 完整方程 (简化版): ${lhs_latex_test} = {' '.join(full_equation_parts)}$")

#         else:
#             print("[测试脚本错误] 未在 best_program_object 中找到 STRidge.terms 列表，或者该列表为空。")
#             if hasattr(best_program_object, 'STRidge'):
#                 print(f"  best_program_object.STRidge.terms 内容: {getattr(best_program_object.STRidge, 'terms', '不存在或访问出错')}")

#     else:
#         print("[测试脚本错误] 'program' 对象未在 DeepRL 的训练输出 (step_output) 中找到，或其值为 None。")

# print("\n" + "="*40)
# print("DeepRL 单个基础项 LaTeX 转换测试尝试结束。")
# print("="*40)


# --- 新增：测试完整的 DeepRL Program 对象到 LaTeX 的转换 ---
from kd.viz.discover_eq2latex import discover_program_to_latex # 导入新函数

print("\n" + "="*40)
print("开始测试 DeepRL 完整 Program 对象的 LaTeX 转换")
print("="*40)

if 'program' in step_output and step_output['program'] is not None:
    best_program_object = step_output['program']
    # 假设 DeepRL 的 LHS 通常是 "ut" 或 "u1_t" 等，这里我们用 "ut"
    # 您可以根据您的 DeepRL task 配置来确定正确的 LHS
    # 例如，如果您的变量是 u1，那么LHS可能是 "u1_t"
    # 从您的 DeepRL 案例看，主要变量是 u1，所以LHS用 "ut" 可能不完全匹配，
    # 最好从您的模型或任务配置中确定确切的LHS名称。
    # 为简单起见，我们先用 "ut"。
    lhs_for_deeprl = "ut" 
    # 如果您的 DeepRL 模型输出的变量是 u1，那么 LHS 可能是 "u1_t"
    # lhs_for_deeprl = "u1_t" # 这是一个更可能的LHS，如果您的演化目标是du1/dt
    # 或者，如果您的step_output['expression']包含LHS，可以从中解析
    # 例如，如果 expression 是 "ut = ...", lhs_for_deeprl = "ut"

    if hasattr(best_program_object, 'str_expression'):
         print(f"要转换的 Program 的原始字符串表达式: {best_program_object.str_expression}")
    else:
         print(f"Program 对象没有 str_expression 属性。")

    print(f"LHS 将设置为: {lhs_for_deeprl} (请根据实际情况调整)")

    # 调用主转换函数
    full_equation_latex = discover_program_to_latex(
        best_program_object,
        lhs_for_deeprl
        # custom_lhs_latex_map 和 custom_sympy_symbols 可以按需传入
    )

    print("\nDeepRL Program 转换后的最终 LaTeX 方程为:")
    print(full_equation_latex)

    # 您可以根据您的具体输出，在这里写一个预期的 LaTeX 字符串进行断言
    # 例如，对于 "0.0984 * diff2(u1,x1) + -0.5002 * diff(n2(u1),x1)"
    # 预期 (假设LHS是u_t, 并且您的 _format_full_latex_term 按我们讨论的方式工作):
    # expected_final_latex = "$u_t = 0.0984 \\frac{d^{2}}{d x_{1}^{2}} u_{1} - 0.5002 \\frac{d}{d x_{1}} u_{1}^{2}$"
    # print(f"\n参考预期 LaTeX: {expected_final_latex}")
    # assert full_equation_latex == expected_final_latex, "完整方程转换与预期不符"

else:
    print("[测试脚本错误] 'program' 对象未在 DeepRL 的训练输出 (step_output) 中找到，或其值为 None。")

print("\n" + "="*40)
print("DeepRL 完整 Program LaTeX 转换测试结束。")
print("="*40)


from kd.viz.equation_renderer import render_latex_to_image

render_latex_to_image(full_equation_latex)