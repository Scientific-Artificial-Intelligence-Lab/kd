import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

from kd.model.discover.program import Program


def plot_expression_tree(model, output_dir: str = None):

    graph = model.searcher.plotter.tree_plot(model.searcher.best_p)

    if output_dir is None:
        return graph

    pass


def plot_density(model, epoches = None, output_dir: str = None):

    model.plot(fig_type='density', epoches=epoches)




# def plot_evolution(model, output_dir: str = None):
#     model.plot(fig_type='evolution')




# Set font that supports both English and Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

PLOT_STYLE = {
    'font.size': 12,
    'figure.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.sans-serif': ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial'],
    'axes.unicode_minus': False,
}

# TODO
def plot_evolution(model, figsize=(10, 6)):
    """
    Plot reward changes during training.
    
    Parameters:
        model: Trained KD_DSCV model.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        # Extract reward history from model
        rewards = model.searcher.r_train

        # Calculate metrics
        x = np.arange(1, len(rewards) + 1)
        r_max = [np.max(r) for r in rewards]
        r_avg = [np.mean(r) for r in rewards]
        r_best = np.maximum.accumulate(np.array(r_max))

        # Plot reward curves
        ax.plot(x, r_best, linestyle='-', color='black', linewidth=2, label='Best Reward')
        ax.plot(x, r_max, color='#F47E62', linestyle='-.', linewidth=2, label='Maximum Reward')
        ax.plot(x, r_avg, color='#4F8FBA', linestyle='--', linewidth=2, label='Average Reward')

        ax.set_xlabel('Iteration Count')
        ax.set_ylabel('Reward Value')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', frameon=False)

        plt.show()

        plt.close()



# --- 内部辅助函数 ---
def _finite_difference(y, x, order=1):
    """使用中心差分计算导数。"""
    if len(y.shape) > 1: y = y.flatten()
    if len(x.shape) > 1: x = x.flatten()
    dx = x[1] - x[0]
    if order == 1: 
        return np.gradient(y, dx, edge_order=2)
    elif order == 2: 
        return np.gradient(np.gradient(y, dx, edge_order=2), dx, edge_order=2)
    elif order == 3:
        return np.gradient(np.gradient(np.gradient(y, dx, edge_order=2), dx, edge_order=2), dx, edge_order=2)
    elif order == 4:
        return np.gradient(np.gradient(np.gradient(np.gradient(y, dx, edge_order=2), dx, edge_order=2), dx, edge_order=2), dx, edge_order=2)
    else: raise ValueError("只支持1-4阶导数, 但请求了 {order} 阶")

def _evaluate_term_recursively(node, u_snapshot, x_coords):
    """递归地计算一个符号树(Node)的数值 """
    if not node.children:
        if node.val == 'u1': 
            return u_snapshot
        elif node.val == 'x1': 
            return x_coords
        try: # 尝试将叶子节点的值转换为浮点数 
            return float(node.val)
        except ValueError:
            raise ValueError(f"未知的叶子节点或无法转换为浮点数的常数: {node.val}")
        
    # 递归步骤: 先计算所有子节点的值
    child_values = [_evaluate_term_recursively(child, u_snapshot, x_coords) for child in node.children]
    op_name = node.val.removesuffix('_t') # 统一处理带和不带_t的后缀


    if op_name == 'add': return child_values[0] + child_values[1]
    elif op_name == 'sub': return child_values[0] - child_values[1]
    elif op_name == 'mul': return child_values[0] * child_values[1]
    elif op_name == 'div': return child_values[0] / (child_values[1] + 1e-8)

    elif op_name == 'n2': return child_values[0]**2
    elif op_name == 'n3': return child_values[0]**3
    elif op_name == 'n4': return child_values[0]**4
    elif op_name == 'n5': return child_values[0]**5

    elif op_name == 'inv': return 1.0 / (child_values[0] + 1e-8)
    elif op_name == 'neg': return -child_values[0]

    elif op_name == 'sin': return np.sin(child_values[0])
    elif op_name == 'cos': return np.cos(child_values[0])
    elif op_name == 'tan': return np.tan(child_values[0])

    elif op_name == 'diff': return _finite_difference(child_values[0], x_coords, order=1)
    elif op_name == 'diff2': return _finite_difference(child_values[0], x_coords, order=2)
    elif op_name == 'diff3': return _finite_difference(child_values[0], x_coords, order=3)
    elif op_name == 'diff4': return _finite_difference(child_values[0], x_coords, order=4)
    else: raise ValueError(f"未知的操作: {op_name}")

def _calculate_pde_fields(model, best_program):
    """
    一个私有的核心计算函数，供所有绘图函数调用。
    它负责计算所有必要的场数据并返回一个字典。
    """
    # 1. 获取信息和数据
    data_dict = model.data_class.get_data()
    final_symbolic_terms = best_program.STRidge.terms
    w_best = best_program.w
    u_trimmed = data_dict['u']
    ut_trimmed = data_dict['ut']
    x_axis = data_dict['X'][0].flatten()

    # 2. 计算 Theta 矩阵
    num_space_points, num_timesteps_trimmed = u_trimmed.shape
    Theta_final = np.zeros((u_trimmed.size, len(final_symbolic_terms)))
    for i, term_node in enumerate(final_symbolic_terms):
        term_values_grid = np.zeros_like(u_trimmed)
        for t_idx in range(num_timesteps_trimmed):
            u_snapshot = u_trimmed[:, t_idx]
            term_values_grid[:, t_idx] = _evaluate_term_recursively(term_node, u_snapshot, x_axis)
        Theta_final[:, i] = term_values_grid.flatten()
    
    # 3. 计算 RHS 和残差
    if Theta_final.shape[1] == len(w_best) - 1:
        y_hat_rhs = Theta_final @ w_best[:-1] + w_best[-1]
    else:
        y_hat_rhs = Theta_final @ w_best
    physical_residual = ut_trimmed.flatten() - y_hat_rhs.flatten()
    
    # 4. 构建绘图坐标
    t_axis_trimmed = np.arange(num_timesteps_trimmed)
    X_grid, T_grid = np.meshgrid(x_axis, t_axis_trimmed, indexing='ij')
    coords_for_plot = np.stack([X_grid.flatten(), T_grid.flatten()], axis=1)

    # 5. 将所有计算结果打包成字典返回
    return {
        "residual": physical_residual,
        "coords": coords_for_plot,
        "ut_grid": ut_trimmed,
        "y_hat_grid": y_hat_rhs.reshape(num_space_points, num_timesteps_trimmed),
        "x_axis": x_axis,
        "t_axis": t_axis_trimmed
    }



def plot_pde_residual_analysis(model, best_program, show_plot=True):
    """
    计算并可视化物理残差
    """
    # 调用核心函数获取所有计算结果
    fields = _calculate_pde_fields(model, best_program)

    if show_plot:
        physical_residual = fields["residual"]
        coords_for_plot = fields["coords"]
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sc = plt.scatter(coords_for_plot[:, 1], coords_for_plot[:, 0], c=physical_residual, cmap='coolwarm', s=15, alpha=0.8)
        plt.colorbar(sc, label='Physical Residual ($u_t$ - RHS)')
        plt.xlabel('Time (trimmed, index)')
        plt.ylabel('Space')
        plt.title('Spatiotemporal Distribution of Residuals')
        plt.subplot(1, 2, 2)
        plt.hist(physical_residual, bins=50, density=True, edgecolor='black', alpha=0.7)
        plt.xlabel('Residual Value')
        plt.ylabel('Probability Density')
        plt.title('Residual Distribution')
        plt.tight_layout()
        plt.show()
    else:
        return fields["residual"], fields["coords"]
    

def plot_field_comparison(model, best_program, show_plot=True):
    """
    计算并可视化“预测场”与“真实场”的对比图。
    """
    # 同样调用核心函数获取所有计算结果
    fields = _calculate_pde_fields(model, best_program)

    ut_grid = fields["ut_grid"]
    y_hat_grid = fields["y_hat_grid"]
    x_axis = fields["x_axis"]
    t_axis = fields["t_axis"]

    if show_plot:
        # 计算通用的颜色范围
        vmin = min(ut_grid.min(), y_hat_grid.min())
        vmax = max(ut_grid.max(), y_hat_grid.max())

        # 创建包含两个子图的图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle('Predicted Field vs. True Field Comparison', fontsize=16)

        # 绘制左图：真实场
        ax0 = axes[0]
        im0 = ax0.pcolormesh(t_axis, x_axis, ut_grid, cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
        fig.colorbar(im0, ax=ax0, label='Value')
        ax0.set_title("True Field ($u_t$)", fontsize=14)
        ax0.set_xlabel("Time (trimmed, index)", fontsize=12)
        ax0.set_ylabel("Space", fontsize=12)

        # 绘制右图：预测场
        ax1 = axes[1]
        im1 = ax1.pcolormesh(t_axis, x_axis, y_hat_grid, cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
        fig.colorbar(im1, ax=ax1, label='Value')
        ax1.set_title("Predicted Field (RHS)", fontsize=14)
        ax1.set_xlabel("Time (trimmed, index)", fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        return ut_grid, y_hat_grid, x_axis, t_axis
    

def plot_actual_vs_predicted(model, best_program):
    """
    Plots an "Actual vs. Predicted" scatter plot with a 45-degree reference line.
    
    Args:
        model: The trained KD_DSCV model instance.
        best_program: The final Program object discovered by the model.
    """
    print("Generating 'Actual vs. Predicted' plot...")
    # Call the helper function to get all computed fields
    fields = _calculate_pde_fields(model, best_program)
    
    y_true = fields["ut_grid"]
    y_pred = fields["y_hat_grid"]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, label='Predicted Points')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = 0.1 * (max_val - min_val) if (max_val - min_val) > 0 else 0.1
    plot_limit = (min_val - margin, max_val + margin)
    
    plt.plot(plot_limit, plot_limit, 'r--', label='Perfect Prediction (y=x)')
    
    plt.xlabel('True Values (Ground Truth, $u_t$)')
    plt.ylabel('Predicted Values (RHS)')
    plt.title('Actual vs. Predicted')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(plot_limit)
    plt.ylim(plot_limit)
    plt.show()



# --- 专为 KD_DSCV_Pinn 模型设计的可视化函数 ---

# def _calculate_pinn_fields(model, best_program):
#     """
#     一个专为 KD_DSCV_Pinn 模型设计的私有核心计算函数。
#     """
#     # 1. 获取数据源
#     task = Program.task
#     final_symbolic_terms = best_program.STRidge.terms
#     w_best = best_program.w
    
#     # 2. 从 task 对象中获取数据
#     u_full = task.u[0]
#     ut_full = task.ut
#     x_coords_tensor = task.x[0] # 这是 Tensor
#     t_coords_tensor = task.t     # 这也是 Tensor

#     # 3. 明确地重组坐标
#     x_coords_np = x_coords_tensor.cpu().detach().numpy()
#     t_coords_np = t_coords_tensor.cpu().detach().numpy()
    
#     # 现在可以安全地使用 np.hstack
#     coords_for_plot = np.hstack([x_coords_np, t_coords_np])

#     # 4. 获取缓存的 Theta 矩阵
#     if not hasattr(best_program, 'Theta') or best_program.Theta is None:
#          raise AttributeError("best_program 对象中未找到缓存的 Theta 矩阵。")
#     Theta_final = best_program.Theta
    
#     # 5. 计算 RHS 和残差
#     if Theta_final.shape[1] == len(w_best) - 1:
#         y_hat_rhs = Theta_final @ w_best[:-1] + w_best[-1]
#     else:
#         y_hat_rhs = Theta_final @ w_best
    
#     # ut_full 已经是 NumPy 数组，所以可以直接使用
#     physical_residual = ut_full.flatten() - y_hat_rhs.flatten()
    
#     # 6. 将所有计算结果打包成字典返回
#     return {
#         "residual": physical_residual,
#         "coords": coords_for_plot,
#         "y_true": ut_full.flatten(),
#         "y_pred": y_hat_rhs.flatten(),
#     }


def _calculate_pinn_fields(model, best_program):
    """
    一个专为 KD_DSCV_Pinn 模型设计的、统一的私有核心计算函数。
    """
    # 1. 触发奖励计算以填充缓存
    _ = best_program.r_ridge

    # 2. 从缓存中获取已计算好的值
    y_hat_rhs = best_program.y_hat_rhs
    Theta = best_program.Theta

    if y_hat_rhs is None or Theta is None:
        raise AttributeError("best_program 对象中未找到缓存的 Theta 矩阵或 y_hat_rhs。请确保对 pde_pinn.py 和 program.py 的修改已生效。")

    # 3. 从 Program.task 获取对应的、已被裁剪的数据
    task = Program.task
    ut_full = task.ut
    x_coords_tensor = task.x[0] 
    t_coords_tensor = task.t     

    # 4. 将坐标从 Tensor 转换为 NumPy 数组
    x_coords_np = x_coords_tensor.cpu().detach().numpy()
    t_coords_np = t_coords_tensor.cpu().detach().numpy()
    
    # 5. 计算残差
    physical_residual = ut_full.flatten() - y_hat_rhs.flatten()
    
    # 6. 安全地重组坐标，创建一个 (N, 2) 的数组
    coords_for_plot = np.hstack([x_coords_np, t_coords_np])
    
    # 7. 将所有需要的计算结果打包成一个完整的字典返回
    return {
        "residual": physical_residual,
        "coords": coords_for_plot,
        "coords_x": x_coords_np.flatten(),
        "coords_t": t_coords_np.flatten(),
        "y_true": ut_full.flatten(),
        "y_pred": y_hat_rhs.flatten()
    }

def plot_pinn_residual_analysis(model, best_program):
    """
    计算并可视化 KD_DSCV_Pinn 模型的物理残差。
    这个版本专门处理由 PINN 生成的稀疏/散点元数据。

    Args:
        model: 训练完成的 KD_DSCV_Pinn 模型实例。
        best_program: 模型发现的最佳 Program 对象。
    """
    
    # 调用为 PINN 定制的辅助函数
    fields = _calculate_pinn_fields(model, best_program)

    physical_residual = fields["residual"]
    coords_for_plot = fields["coords"] # 现在这个坐标是正确的 (N, 2) 格式
    
    # 检查维度是否匹配
    if coords_for_plot.shape[0] != len(physical_residual):
        raise ValueError(f"坐标点数量 ({coords_for_plot.shape[0]}) 与残差值数量 ({len(physical_residual)}) 不匹配。")

    plt.figure(figsize=(12, 5))
    
    # 左图: 残差时空分布图
    plt.subplot(1, 2, 1)
    # coords_for_plot[:, 1] 是时间, coords_for_plot[:, 0] 是空间
    sc = plt.scatter(coords_for_plot[:, 1], coords_for_plot[:, 0], c=physical_residual, cmap='coolwarm', s=15, alpha=0.8)
    plt.colorbar(sc, label='Physical Residual ($u_t$ - RHS)')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.title('Spatiotemporal Distribution of Residuals (on meta-data points)')
    
    # 右图: 残差统计直方图
    plt.subplot(1, 2, 2)
    plt.hist(physical_residual, bins=50, density=True, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual Value')
    plt.ylabel('Probability Density')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()


def plot_pinn_actual_vs_predicted(model, best_program):
    """
    为 KD_DSCV_Pinn 模型绘制“真值 vs. 预测值”的45度线图
    """
    print("Generating 'Actual vs. Predicted' plot for PINN model...")
    fields = _calculate_pinn_fields(model, best_program)
    
    y_true = fields["y_true"]
    y_pred = fields["y_pred"]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, label='Predicted Points')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = 0.1 * (max_val - min_val) if (max_val - min_val) > 0 else 0.1
    plot_limit = (min_val - margin, max_val + margin)
    
    plt.plot(plot_limit, plot_limit, 'r--', label='Perfect Prediction (y=x)')
    
    plt.xlabel('True Values (Ground Truth, $u_t$)')
    plt.ylabel('Predicted Values (RHS)')
    plt.title('Actual vs. Predicted (for PINN model)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(plot_limit)
    plt.ylim(plot_limit)
    plt.show()

def plot_pinn_field_comparison(model, best_program):
    """
    为 KD_DSCV_Pinn 模型计算并可视化“预测场”与“真实场”的对比图。
    注意：此函数使用三角剖分来可视化稀疏散点数据
    """
    print("Generating field comparison plot for PINN model...")
    fields = _calculate_pinn_fields(model, best_program)

    x, t, ut, y_hat = fields["coords_x"], fields["coords_t"], fields["y_true"], fields["y_pred"]

    # 使用三角剖分来处理散点数据
    from scipy.interpolate import griddata
    grid_x, grid_t = np.mgrid[min(x):max(x):100j, min(t):max(t):100j]
    
    grid_ut = griddata((x, t), ut, (grid_x, grid_t), method='cubic')
    grid_yhat = griddata((x, t), y_hat, (grid_x, grid_t), method='cubic')

    # 使用通用的颜色范围
    vmin = np.nanmin([grid_ut, grid_yhat])
    vmax = np.nanmax([grid_ut, grid_yhat])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle('Field Comparison (on Sparse Points)', fontsize=16)

    # 绘制左图：真实场
    ax0 = axes[0]
    im0 = ax0.imshow(grid_ut.T, extent=(min(t),max(t),min(x),max(x)), origin='lower', 
                     aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im0, ax=ax0, label='Value')
    ax0.set_title("True Field ($u_t$)")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Space")

    # 绘制右图：预测场
    ax1 = axes[1]
    im1 = ax1.imshow(grid_yhat.T, extent=(min(t),max(t),min(x),max(x)), origin='lower', 
                     aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=ax1, label='Value')
    ax1.set_title("Predicted Field (RHS)")
    ax1.set_xlabel("Time")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()