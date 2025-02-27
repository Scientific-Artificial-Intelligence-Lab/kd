import numpy as np
import torch
import matplotlib.pyplot as plt
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
from kd.plot.scientific.residual import ResidualAnalysis
from kd.plot.scientific.equation import TermsHeatmap, TermsAnalysis

model = DeepRL(
    n_samples_per_batch = 500, # Number of generated traversals by agent per batch
    binary_operators = ['add',"mul", "diff","diff2"],
    unary_operators = ['n2'],
)

np.random.seed(42)
def prepare_data():
    """Prepare data for training and visualization.
    
    Returns:
        tuple: (X_train, y_train, lb, ub, x, t, u_exact)
        - X_train, y_train: Training data
        - lb, ub: Domain bounds
        - x, t: Spatial and temporal coordinates
        - u_exact: Exact solution array (nt x nx)
    """
    data = scipy.io.loadmat('./kd/data_file/burgers2.mat')
    t = np.real(data['t'].flatten()[:,None])
    x = np.real(data['x'].flatten()[:,None])
    Exact = np.real(data['usol']).T  # t first
    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0) 

    # Sample training points
    total_num = X_star.shape[0]
    sample_num = int(total_num*0.1)
    print(f"random sample number: {sample_num} ")
    ID = np.random.choice(total_num, sample_num, replace = False)
    X_u_meas = X_star[ID,:]
    u_meas = u_star[ID,:]
    
    return X_u_meas, u_meas, lb, ub, x.flatten(), t.flatten(), Exact

# Prepare data
X_train, y_train, lb, ub, x, t, u_exact = prepare_data()
model.import_inner_data(dataset='Burgers', data_type='regular')

# 训练模型，并保存最佳程序对象
n_epochs = 100
best_program_obj = None
best_reward_val = -float('inf')
best_expression_str = ""

for epoch in range(n_epochs):
    result = model.searcher.search_one_step(epoch=epoch, verbose=True)
    if result:
        current_reward = result['r']
        current_expression = result['expression']
        print(f"Epoch {epoch+1}: expression = {current_expression}, reward = {current_reward}")
        
        # 记录最佳结果
        if current_reward > best_reward_val:
            best_reward_val = current_reward
            best_program_obj = result['program']  # 保存程序对象
            best_expression_str = current_expression
            print(f"\n【发现更好的表达式】: {best_expression_str}")
            print(f"【奖励值】: {best_reward_val}\n")

print("\n===== 训练完成 =====")
print(f"最佳表达式: {best_expression_str}")
print(f"最佳奖励值: {best_reward_val}")
print(f"最佳程序类型: {type(best_program_obj)}")

# ===== 残差图绘制部分开始 =====
if best_program_obj:
    print("\n===== 开始绘制残差和项分析图 =====")
    # Prepare data for execution
    u_current = u_exact  # Current state
    ut = np.gradient(u_exact, t, axis=0)  # Time derivative
    
    # 使用保存的最佳程序对象执行
    y_hat, left_term, right_term = best_program_obj.execute(u=u_current, x=x, ut=ut)
    
    # 创建和显示残差分析图
    plotter = ResidualAnalysis()
    plotter.plot(u_pred=y_hat,
                u_exact=u_exact,
                x=x, 
                t=t,
                slice_times=[0.25, 0.5, 0.75])
    plt.show()
    
    # 添加45度线比较图
    print(f"绘制45度线图前检查数据形状：")
    print(f"u_exact shape: {u_exact.shape}")
    print(f"u_exact flattened shape: {u_exact.flatten().shape}")
    print(f"y_hat type: {type(y_hat)}, value: {y_hat}")
    
    # 如果y_hat是标量或单元素列表，需要扩展为数组
    if isinstance(y_hat, (int, float)):
        print("y_hat是标量，将其扩展为与u_exact相同形状的数组")
        y_hat_array = np.full_like(u_exact, y_hat)
    elif isinstance(y_hat, list) and len(y_hat) == 1:
        print(f"y_hat是单元素列表: {y_hat[0]}，将其扩展为数组")
        y_hat_array = np.full_like(u_exact, y_hat[0])
    elif isinstance(y_hat, np.ndarray) and y_hat.size == 1:
        print(f"y_hat是单元素数组：{y_hat.item()}，将其扩展为数组")
        y_hat_array = np.full_like(u_exact, y_hat.item())
    elif isinstance(y_hat, list):
        print(f"y_hat是多元素列表，长度为: {len(y_hat)}")
        y_hat_array = np.array(y_hat).reshape(u_exact.shape) if len(y_hat) == u_exact.size else np.array(y_hat)
    else:
        print(f"y_hat是数组，形状为: {np.array(y_hat).shape}")
        y_hat_array = np.array(y_hat)
    
    print(f"处理后的y_hat_array形状: {y_hat_array.shape}")
    
    # 确保y_hat_array是扁平化的以便与u_exact.flatten()比较
    y_hat_flat = y_hat_array.flatten() if hasattr(y_hat_array, 'flatten') else np.array([y_hat_array])
    u_exact_flat = u_exact.flatten()
    
    print(f"扁平化后比较：u_exact_flat shape: {u_exact_flat.shape}, y_hat_flat shape: {y_hat_flat.shape}")
    
    # 如果形状仍然不匹配，使用较小的那个进行截断
    min_size = min(len(u_exact_flat), len(y_hat_flat))
    u_exact_flat = u_exact_flat[:min_size]
    y_hat_flat = y_hat_flat[:min_size]
    
    print(f"最终用于绘图的数据形状：u_exact_flat: {u_exact_flat.shape}, y_hat_flat: {y_hat_flat.shape}")
    
    # 绘制u_exact和y_hat的45度线比较图
    plt.figure(figsize=(8, 8))
    plt.scatter(u_exact_flat, y_hat_flat, alpha=0.5, s=5)
    min_val = min(u_exact_flat.min(), y_hat_flat.min())
    max_val = max(u_exact_flat.max(), y_hat_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想45度线')
    plt.xlabel('真实值 (u_exact)')
    plt.ylabel('预测值 (y_hat)')
    plt.title('预测解与真实解的45度线比较')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 打印数据形状信息
    print("数据形状:")
    print(f"u_exact shape: {u_exact.shape}")
    print(f"ut shape: {ut.shape}")
    print(f"y_hat type: {type(y_hat)}, shape: {np.array(y_hat).shape if isinstance(y_hat, (np.ndarray, list)) else 'scalar'}")
    print(f"right_term type: {type(right_term)}, shape: {np.array(right_term).shape if isinstance(right_term, (np.ndarray, list)) else len(right_term) if isinstance(right_term, list) else 'scalar'}")

    # 确保数据是numpy数组格式
    if isinstance(ut, list):
        ut = np.array(ut)
    
    # 处理right_term
    if isinstance(right_term, list):
        if len(right_term) == 1:  # 如果是单元素列表
            print(f"right_term是单元素列表，值为: {right_term[0]}")
            right_term = np.full_like(u_exact, right_term[0])
        else:
            print(f"right_term是多元素列表，长度为: {len(right_term)}")
            right_term = np.array(right_term).reshape(u_exact.shape)
    elif isinstance(right_term, (int, float)):
        print(f"right_term是标量值: {right_term}")
        right_term = np.full_like(u_exact, right_term)
    elif isinstance(right_term, np.ndarray):
        if right_term.size == 1:
            print(f"right_term是单元素数组，值为: {right_term.item()}")
            right_term = np.full_like(u_exact, right_term.item())
        else:
            print(f"right_term是数组，形状为: {right_term.shape}")
            if right_term.shape != u_exact.shape:
                right_term = right_term.reshape(u_exact.shape)
    
    # 创建左右项字典
    terms = {
        'ut': ut,  # 时间导数项
        'RHS': right_term,  # 右手边项
        'Residual': ut - right_term  # 残差
    }
    
    # 打印处理后的形状
    print("\n处理后的形状:")
    for name, term in terms.items():
        print(f"{name} shape: {term.shape}")
    
    # 绘制左右项云图
    plotter = TermsHeatmap()
    plotter.plot(terms=terms,
                x=x,
                t=t,
                show_colorbar=True,
                normalize=True)
    plt.show()
    
    # 进行更详细的项分析
    analyzer = TermsAnalysis()
    analyzer.plot(terms=terms,
                 x=x,
                 t=t,
                 show_correlations=True,
                 show_statistics=True)
    plt.show()
    
    print(f"运行完成！最佳表达式: {best_expression_str}")
else:
    print("没有找到有效的程序对象，无法生成可视化。")

# Tree
# model.plot(fig_type ='tree')

# Evolution
# model.plot(fig_type='evolution')