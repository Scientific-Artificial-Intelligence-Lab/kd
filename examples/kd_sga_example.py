import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)


# --- 依赖导入 / Dependency Imports ---
from kd.model import KD_SGA
from kd.dataset import load_chafee_infante_equation, load_pde_dataset
from kd.viz.sga_eq2latex import sga_eq2latex
from kd.viz.equation_renderer import render_latex_to_image


# --- 数据加载 / Data Loading ---
print("📊 Loading Chafee-Infante equation dataset...")

# 使用统一的数据加载函数，支持多种数据格式和数据集
# Use unified data loading functions, supporting multiple data formats and datasets
# chafee_data = load_chafee_infante_equation()

# 也可以使用通用的数据加载函数加载其他格式：
# You can also use the general data loading function for other formats:
# chafee_data = load_pde_dataset(filename="chafee_infante_CI.npy", x_key='x', t_key='t', u_key='usol')
burgers_data = load_pde_dataset(filename="burgers.mat", x_key='x', t_key='t', u_key='usol')

# Legacy 方法：SGA 内置的问题配置系统（不推荐，仅为向后兼容）
# Legacy method: SGA built-in problem configuration system (not recommended, for backward compatibility only)
# model = KD_SGA(problem='chafee-infante', ...)  # 通过 problem 参数自动加载数据
# model.fit(X=None, ...)  # SGA 自动使用内置数据


# --- 模型初始化与配置 / Model Initialization & Configuration ---
print("\n🤖 Initializing KD_SGA model with parameters...")
model = KD_SGA(
    num=20,                                         # 种群大小 / Population size
    depth=4,                                        # 符号树最大深度 / Maximum tree depth  
    width=5,                                        # PDE中最大项数 / Maximum terms in PDE
    p_var=0.5,                                      # 变量生成概率 / Variable generation probability
    p_mute=0.3,                                     # 变异概率 / Mutation probability
    p_cro=0.5,                                      # 交叉概率 / Crossover probability
    data_mode='finite_difference'                   # 导数计算模式 / Derivative calculation mode
)

print("SGA Configuration:")
params = model.get_params()
for key, value in params.items():
    print(f"   {key}: {value}")
print(f"✅ Model initialized: {model}")


# --- 模型训练 / Model Training ---
print("\n🚀 Starting PDE discovery...")
print("This may take a few minutes depending on the number of generations...")

model.fit(
    X=burgers_data,                                  # 使用加载的数据集 / Use loaded dataset
    max_gen=10,                                     # 进化代数 / Evolution generations
    verbose=True                                    # 显示详细进度 / Show detailed progress
)
print("✅ Training completed successfully!")


# --- 结果分析 / Results Analysis ---
print("\n📈 Analyzing discovered equation...")
discovered_eq = model.get_equation_string()
latex_eq = model.get_equation_latex()
aic_score = model.best_aic_
sklearn_score = model.score(X=burgers_data)

print("Results Summary:")
print(f"🔍 Discovered equation: {discovered_eq}")
print(f"📊 AIC score: {aic_score:.6f} (lower is better)")
print(f"📊 Model score: {sklearn_score:.6f}")
# print(f"🎯 True equation: u_t = u_xx - u + u^3") # 这个不能写死吧

if latex_eq: print(f"📝 LaTeX representation: {latex_eq}")


# --- 可视化准备 / Visualization Preparation ---
print("\n🎨 Preparing visualization...")

if model.best_equation_:
    try:
        # 生成SGA专用的LaTeX格式 / Generate SGA-specific LaTeX format
        sga_latex = sga_eq2latex(model.best_equation_, lhs_name="u_t")
        print(f"🔤 SGA LaTeX: {sga_latex}")
        
        # 渲染方程为图片 / Render equation to image
        print("🖼️  Rendering equation to image...")
        render_latex_to_image(sga_latex)
        print("✅ Equation image saved to current directory!")
        
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        print("   This is normal if LaTeX/matplotlib is not properly configured.")
else:
    print("⚠️  No equation available for visualization.")


# --- 模型信息总结 / Model Information Summary ---
print("\nℹ️  Model Information:")
print(f"📋 All parameters: {model.get_params()}")
print(f"🏗️  Model representation: {repr(model)}")
print(f"⚙️  Data processing mode: {model.data_mode}")
print(f"🧬 Population size used: {model.num}")
print(f"🌳 Maximum tree depth: {model.depth}")


# --- 使用建议 / Usage Tips ---
print(f"\n{'='*60}")
print("💡 Usage Tips:")
print("   • Increase 'max_gen' for better results (but slower training)")
print("   • Adjust 'num' (population size) to balance speed vs. exploration")
print("   • Try 'autograd' data_mode for automatic differentiation")
print("   • Use different 'problem' types: 'burgers', 'kdv', 'pde_compound'")
print("   • Lower AIC scores indicate better equation fits")
print(f"{'='*60}")


# --- 可用数据集信息 / Available Datasets Info ---
print("\n📚 Available Predefined Datasets:")
datasets_info = [
    ("chafee-infante", "u_t = u_xx - u + u^3", "Reaction-diffusion equation"),
    ("burgers", "u_t = -u*u_x + 0.1*u_xx", "Nonlinear wave equation"),
    ("kdv", "u_t = -0.0025*u_xxx - u*u_x", "Korteweg-de Vries equation"),
    ("pde_compound", "u_t = u*u_xx + u_x*u_x", "Compound nonlinear PDE"),
    ("pde_divide", "u_t = -u_x/x + 0.25*u_xx", "PDE with division terms")
]

for name, equation, description in datasets_info:
    print(f"   • {name:<15} - {equation:<25} ({description})")

print(f"\n{'='*60}")
print("🎉 KD_SGA Example completed successfully!")
print("   Ready for visualization and further analysis...")
print(f"{'='*60}")
