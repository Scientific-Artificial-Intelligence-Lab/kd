import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)


from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA
from kd.viz import configure, render_equation, render, VizRequest

# 1. 通过统一入口加载数据集
pde_dataset = load_pde('chafee-infante')

# 2. 创建并配置参数 (所有在 SGA config 中定义的参数都已被兼容)
model = KD_SGA(sga_run=2, depth=3)

# 3. 使用新接口直接传入 PDEDataset
model.fit_dataset(pde_dataset)

# 4. 查看结果（字符串与 LaTeX）
print(f"The discovered equation is: {model.best_pde_}")
latex_with_coeff = model.equation_latex()
latex_structure = model.equation_latex(include_coefficients=False)
print(f"LaTeX form (with coefficients): {latex_with_coeff}")
print(f"LaTeX form (structure only): {latex_structure}")

# 5. 使用统一 façade 生成方程图
configure(save_dir="artifacts/sga_viz")
eq_result = render_equation(model)
if eq_result.paths:
    print("Equation figures:")
    for path in eq_result.paths:
        print(f"  - {path}")
else:
    print("Equation figures were not generated.")
print(f"Equation metadata: {eq_result.metadata}")

# 6. 调用 façade 的 SGA field comparison 意图
field_request = VizRequest(kind='field_comparison', target=model, options={})
field_result = render(field_request)
if field_result.paths:
    print("Field comparison figures:")
    for path in field_result.paths:
        print(f"  - {path}")
else:
    print("Field comparison figure was not generated.")

field_summary = field_result.metadata.get('summary', {})
print(f"Field comparison summary: {field_summary}")

# 7. 展示时间切片（u, u_t, u_x, u_{xx}) 对比
time_slice_request = VizRequest(
    kind='time_slices',
    target=model,
    options={'slice_times': [0.0, 0.5, 1.0]},
)
time_slice_result = render(time_slice_request)
if time_slice_result.paths:
    print("Time-slice comparison figures:")
    for path in time_slice_result.paths:
        print(f"  - {path}")
else:
    print("Time-slice comparison figure was not generated.")
time_slice_summary = time_slice_result.metadata.get('time_slices_summary', {})
print(f"Time-slice summary: {time_slice_summary}")

# 8. 保留 legacy 图像输出（headless 已自动处理）
model.plot_results()
