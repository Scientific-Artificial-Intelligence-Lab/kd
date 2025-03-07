"""DLGA可视化模块 (v0.2)"""

import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
from scipy.interpolate import griddata

# 在模块顶部添加全局配置
PLOT_STYLE = {
    'font.size': 12,
    'figure.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c']),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.h_pad': 0.1,
    'figure.constrained_layout.w_pad': 0.1
}

DEFAULT_CMAP = 'viridis'

def configure_plotting(style: dict = None, cmap: str = None):
    """全局绘图配置函数
    
    参数:
        style: 自定义样式字典
        cmap: 默认颜色映射
    """
    plt.rcParams.update(style or PLOT_STYLE)
    global DEFAULT_CMAP
    if cmap:
        DEFAULT_CMAP = cmap

def plot_training_loss(model, output_dir: str = None):
    """
    绘制训练损失曲线
    
    参数:
        model: 包含train_loss_history的DLGA模型
        output_dir: 输出目录，默认当前目录下的.plot_output
    """
    # 配置样式
    plt.figure(figsize=(10,5))
    plt.plot(model.train_loss_history, 
            color='#1f77b4', 
            linewidth=2)
    plt.xlabel('Training Epoch')
    plt.ylabel('Training Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if output_dir:
        # 创建目录
        viz_dir = Path(output_dir)
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / "training_loss.png", dpi=300)
    else:
        plt.show()
    plt.close()

def plot_validation_loss(model, output_dir: str = None):
    """
    绘制验证损失曲线
    
    参数:
        model: 包含val_loss_history的DLGA模型
        output_dir: 输出目录，默认当前目录下的.plot_output
    """
    plt.figure(figsize=(10,5))
    plt.plot(model.val_loss_history, 
            color='#ff7f0e', 
            linewidth=2,
            alpha=0.8)
    plt.xlabel('Training Epoch')
    plt.ylabel('Validation Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if output_dir:
        viz_dir = Path(output_dir)
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / "validation_loss.png", dpi=300)
    else:
        plt.show()
    plt.close()

def plot_residual_analysis(model, X_train, y_train, u_true, u_pred, output_dir: str = None):
    """
    残差分析可视化
    
    参数:
        model: DLGA模型实例
        X_train: 训练数据坐标
        y_train: 训练数据真实值
        u_true: 完整真实解
        u_pred: 完整预测解
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 4))
    
    # 左图：训练点残差
    plt.subplot(121)
    # 计算训练点预测值
    with torch.no_grad():
        train_pred = model.Net(torch.tensor(X_train, dtype=torch.float32)).numpy().flatten()
    train_residuals = y_train - train_pred
    
    sc = plt.scatter(X_train[:, 1],  # 时间坐标
                    X_train[:, 0],   # 空间坐标
                    c=train_residuals, 
                    cmap='coolwarm', 
                    s=10,
                    edgecolors='w',
                    linewidths=0.5)
    plt.colorbar(sc, label='Residual').outline.set_visible(False)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Space', fontsize=10)
    plt.title('Training Points Residual', pad=10)
    
    # 右图：整体残差分布
    plt.subplot(122)
    residual = u_true - u_pred
    n, bins, patches = plt.hist(residual.flatten(), 
                              bins=50, 
                              density=True,
                              edgecolor='black',
                              linewidth=0.5)
    # 染色以匹配coolwarm配色
    cmap = plt.get_cmap('coolwarm')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        plt.setp(p, 'facecolor', cmap((c - bin_centers[0])/(bin_centers[-1]-bin_centers[0])))
    
    plt.xlabel('Residual Value', fontsize=10)
    plt.ylabel('Probability Density', fontsize=10)
    plt.title('Residual Distribution', pad=10)
    
    if output_dir:
        viz_dir = Path(output_dir)
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_pde_comparison(x, t, u_true, u_pred, output_dir: str = None):
    """PDE解对比可视化"""
    # 应用全局配置
    with plt.style.context(PLOT_STYLE):
        T, X = np.meshgrid(t, x)
        vmin = min(u_true.min(), u_pred.min())
        vmax = max(u_true.max(), u_pred.max())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
        
        # 真实解
        mesh1 = ax1.pcolormesh(T, X, u_true, 
                             shading='gouraud', 
                             cmap=DEFAULT_CMAP,
                             vmin=vmin, vmax=vmax)
        ax1.set(title='Exact Solution', xlabel='Time', ylabel='Space')
        fig.colorbar(mesh1, ax=ax1, label='u(x,t)')
        
        # 预测解
        mesh2 = ax2.pcolormesh(T, X, u_pred,
                             shading='gouraud',
                             cmap=DEFAULT_CMAP,
                             vmin=vmin, vmax=vmax)
        ax2.set(title='Predicted Solution', xlabel='Time')
        fig.colorbar(mesh2, ax=ax2, label='u(x,t)')
        
        if output_dir:
            viz_dir = Path(output_dir)
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / 'pde_comparison.png')
        else:
            plt.show()
        plt.close()

def plot_equation_terms(
    metadata, 
    terms: dict,
    color_var: str = 'u_t',
    equation_name: str = "PDE", 
    output_dir: str = None,
    filename: str = "equation_terms_analysis.png"
):
    """
    通用方程项关系可视化
    
    参数:
        metadata: 包含各方程项值的字典
        terms: 要可视化的方程项字典，格式为 {'x轴项': {'vars': [变量列表], 'label': '显示标签'}, 
                                      'y轴项': {'vars': [变量列表], 'label': '显示标签'}}
        color_var: 用于着色的变量名
        equation_name: 方程名称（用于标题）
        output_dir: 输出目录
        filename: 输出文件名
    
    示例:
        # KdV方程: u_t + 6uu_x + u_xxx = 0
        plot_equation_terms(
            metadata,
            terms={
                'x_term': {'vars': ['u', 'u_x'], 'label': '6uu_x'},
                'y_term': {'vars': ['u_xxx'], 'label': '-u_xxx'}
            },
            equation_name="KdV Equation"
        )
        
        # Burgers方程: u_t + uu_x - νu_xx = 0
        plot_equation_terms(
            metadata,
            terms={
                'x_term': {'vars': ['u', 'u_x'], 'label': 'uu_x'},
                'y_term': {'vars': ['u_xx'], 'label': 'νu_xx'}
            },
            equation_name="Burgers Equation"
        )
    """
    with plt.style.context(PLOT_STYLE):
        # 提取和计算各项
        x_vars = terms.get('x_term', {}).get('vars', [])
        y_vars = terms.get('y_term', {}).get('vars', [])
        
        if not x_vars or not y_vars:
            print("警告: 未指定足够的方程项，无法绘制")
            return
            
        # 动态计算各项乘积
        x_values = np.prod([metadata[key] for key in x_vars], axis=0)
        y_values = np.prod([metadata[key] for key in y_vars], axis=0)
        
        # 获取标签
        x_label = terms.get('x_term', {}).get('label', '-'.join(x_vars))
        y_label = terms.get('y_term', {}).get('label', '-'.join(y_vars))
        
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 子图1: 项关系散点图
        sc = ax1.scatter(
            x_values.flatten(),
            y_values.flatten(),
            c=metadata[color_var].flatten() if color_var in metadata else None,
            cmap=DEFAULT_CMAP,
            alpha=0.6,
            s=10
        )
        ax1.set(xlabel=x_label, ylabel=y_label, 
               title=f'{equation_name} Term Relationship')
        fig.colorbar(sc, ax=ax1, label=color_var)
        
        # 子图2: 时间导数分布
        if color_var in metadata:
            ax2.hist(metadata[color_var].flatten(), bins=50, density=True)
            ax2.set(xlabel=f'{color_var} Value', ylabel='Density',
                   title=f'{color_var} Distribution')
        
        if output_dir:
            viz_dir = Path(output_dir)
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / filename)
        else:
            plt.show()
        plt.close()

def plot_evolution(model, output_dir: str = None):
    """绘制进化过程的可视化图"""
    with plt.style.context(PLOT_STYLE):
        # 检查数据可用性
        if not hasattr(model, 'evolution_history') or not model.evolution_history:
            raise ValueError("没有可用的进化历史数据，请确保已运行evolution()方法")
            
        # 创建图形和子图
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # 提取数据
        generations = range(len(model.evolution_history))
        fitness = np.array([x['fitness'] for x in model.evolution_history])
        complexity = np.array([x['complexity'] for x in model.evolution_history])
        
        # 1. 适应度进化曲线
        ax1.plot(generations, fitness, 
                color='#1f77b4', 
                linewidth=2,
                marker='o',
                markersize=4,
                markeredgecolor='white',
                alpha=0.7)
        ax1.set(xlabel='Generation', 
                ylabel='Best Fitness (Lower is Better)',
                title='Evolution of Best Fitness')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. 方程复杂度变化
        ax2.plot(generations, complexity,
                color='#ff7f0e',
                linewidth=2,
                marker='s',
                markersize=4,
                markeredgecolor='white',
                alpha=0.7)
        ax2.set(xlabel='Generation', 
                ylabel='Equation Complexity',
                title='Evolution of Equation Complexity')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        if output_dir:
            viz_dir = Path(output_dir)
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / "evolution_analysis.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # 打印演化统计信息
        print("\nEvolution Analysis Summary:")
        print(f"Initial fitness: {fitness[0]:.4f}")
        print(f"Final fitness: {fitness[-1]:.4f}")
        print(f"Improvement: {((fitness[0] - fitness[-1])/fitness[0]):.2%}")
        print(f"Initial complexity: {complexity[0]}")
        print(f"Final complexity: {complexity[-1]}")

def plot_optimization_analysis(model, output_dir: str = None):
    """Visualize optimization metrics including weight changes and population diversity."""
    # 创建图形和子图
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    with plt.style.context(PLOT_STYLE):
        # 1. 左图：种群多样性分析
        generations = range(len(model.evolution_history))
        fitness_history = np.array([x['fitness'] for x in model.evolution_history])
        pop_sizes = [x.get('population_size', 0) for x in model.evolution_history]
        unique_modules = [x.get('unique_modules', 0) for x in model.evolution_history]
        
        # 绘制种群多样性趋势 - 使用双y轴
        ax1_twin = ax1.twinx()
        
        l1 = ax1.plot(generations, pop_sizes,
                     label='Population Size', color='#1f77b4',
                     marker='o', markersize=3, alpha=0.7)
        l2 = ax1_twin.plot(generations, unique_modules,
                          label='Unique Modules', color='#ff7f0e',
                          marker='s', markersize=3, alpha=0.7)
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Population Size', color='#1f77b4')
        ax1_twin.set_ylabel('Unique Modules', color='#ff7f0e')
        ax1.set_title('Population Diversity Analysis')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 合并两个y轴的图例
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # 动态设置y轴范围
        pop_min, pop_max = min(pop_sizes), max(pop_sizes)
        mod_min, mod_max = min(unique_modules), max(unique_modules)
        ax1.set_ylim(pop_min - 50, pop_max + 50)
        ax1_twin.set_ylim(mod_min - 5, mod_max + 5)
        
        # 2. 右图：适应度变化分析
        fitness_changes = np.abs(np.diff(fitness_history))
        
        # 计算累积变化并找到主要变化区间
        if len(fitness_changes) > 0:
            cumsum_changes = np.cumsum(fitness_changes)
            total_change = cumsum_changes[-1]
            
            # 找到包含80%变化的区间（改为更合理的阈值）
            threshold = 0.8 * total_change
            significant_idx = np.searchsorted(cumsum_changes, threshold)
            show_gens = min(significant_idx + 5, len(fitness_history) - 1)
            
            # 绘制整体趋势（透明度降低）
            ax2.plot(generations, fitness_history, 
                    color='#2ca02c', linewidth=1,
                    marker='o', markersize=2, alpha=0.3)
            
            # 突出显示主要变化区间
            ax2.plot(generations[:show_gens+1], fitness_history[:show_gens+1],
                    color='#2ca02c', linewidth=2,
                    marker='o', markersize=4, alpha=0.8,
                    label=f'Major changes (first {show_gens} gen)')
            ax2.legend()
        else:
            # 如果没有变化，仅绘制完整曲线
            ax2.plot(generations, fitness_history, 
                    color='#2ca02c', linewidth=2,
                    marker='o', markersize=3)
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness Value')
        ax2.set_title('Fitness Evolution')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        if output_dir:
            viz_dir = Path(output_dir)
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / "optimization_analysis.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # 打印更详细的分析总结（避免索引越界）
        print("\nOptimization Analysis Summary:")
        print(f"Initial fitness: {fitness_history[0]:.4f}")
        print(f"Final fitness: {fitness_history[-1]:.4f}")
        print(f"Major improvements occurred in first {show_gens} generations")
        if show_gens > 0:
            early_improvement = ((fitness_history[0] - fitness_history[show_gens])/fitness_history[0])
            print(f"Improvement in major change period: {early_improvement:.2%}")
        total_improvement = ((fitness_history[0] - fitness_history[-1])/fitness_history[0])
        print(f"Total improvement: {total_improvement:.2%}")
        print(f"Average population size: {np.mean(pop_sizes):.1f}")
        print(f"Average unique modules: {np.mean(unique_modules):.1f}")
        print(f"Diversity ratio: {np.mean(unique_modules)/np.mean(pop_sizes):.2%}")

def plot_time_slices(x, t, u_true, u_pred, slice_times, output_dir: str = None):
    """绘制时间切片对比图
    
    Args:
        x: 空间坐标数组 (nx,)
        t: 时间坐标数组 (nt,)
        u_true: 真实解数组 (nx, nt)
        u_pred: 预测解数组 (nx, nt)
        slice_times: 时间切片列表，值应在0到1之间
    """
    
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(1, len(slice_times), figsize=(15, 5), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
            
        for i, t_slice in enumerate(slice_times):
            # 将相对时间转换为索引
            t_idx = int(t_slice * (len(t) - 1))
            
            # Debug信息
            # print(f"\nPlotting time slice {i+1}:")
            # print(f"Time index: {t_idx} (t = {t[t_idx]:.3f})")
            # print(f"x shape: {x.shape}")
            # print(f"u_true shape: {u_true.shape}")
            # print(f"Slice shape: {u_true[:, t_idx].shape}")
            
            ax = axes[i]
            ax.plot(x, u_true[:, t_idx], 'b-', label='Exact', linewidth=2)
            ax.plot(x, u_pred[:, t_idx], 'r--', label='Prediction', linewidth=2)
            ax.set_xlabel('Space')
            ax.set_title(f't = {t[t_idx]:.2f}')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            if i == 0:
                ax.set_ylabel('u(x,t)')
            if i == len(slice_times) // 2:
                ax.legend()
        
        if output_dir:
            viz_dir = Path(output_dir)
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / "time_slices_comparison.png", dpi=300)
        else:
            plt.show()
        plt.close()

def plot_derivative_relationships(metadata, output_dir: str = None):
    """分析和可视化不同阶导数之间的关系
    
    Args:
        metadata: 包含各导数项的字典，必须包含 'u', 'u_x', 'u_xxx', 'u_t' 等键
        output_dir: 输出目录
    """
    
    with plt.style.context(PLOT_STYLE):
        fig = plt.figure(figsize=(15, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 3)
        
        # 1. u_t vs u_x
        ax1 = fig.add_subplot(gs[0, 0])
        sc1 = ax1.scatter(metadata['u_x'].flatten(), 
                         metadata['u_t'].flatten(),
                         c=metadata['u'].flatten(),  # 按u值着色
                         cmap='viridis',
                         alpha=0.6,
                         s=20)
        ax1.set_xlabel('u_x')
        ax1.set_ylabel('u_t')
        ax1.set_title('First Order Derivatives')
        plt.colorbar(sc1, ax=ax1, label='u')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. Combined Term (u*u_x)
        ax2 = fig.add_subplot(gs[0, 1])
        combined_term = metadata['u'].flatten() * metadata['u_x'].flatten()
        sc2 = ax2.scatter(combined_term,
                         metadata['u_t'].flatten(),
                         c=metadata['u_xxx'].flatten(),  # 按三阶导着色
                         cmap='plasma',
                         alpha=0.6,
                         s=20)
        ax2.set_xlabel('u*u_x')
        ax2.set_ylabel('u_t')
        ax2.set_title('Nonlinear Term vs Time Derivative')
        plt.colorbar(sc2, ax=ax2, label='u_xxx')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 3. Third Order Term
        ax3 = fig.add_subplot(gs[0, 2])
        sc3 = ax3.scatter(metadata['u_xxx'].flatten(),
                         metadata['u_t'].flatten(),
                         c=combined_term,  # 按非线性项着色
                         cmap='coolwarm',
                         alpha=0.6,
                         s=20)
        ax3.set_xlabel('u_xxx')
        ax3.set_ylabel('u_t')
        ax3.set_title('Third Order Term vs Time Derivative')
        plt.colorbar(sc3, ax=ax3, label='u*u_x')
        ax3.grid(True, linestyle='--', alpha=0.5)

        if output_dir:
            viz_dir = Path(output_dir)
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / "derivative_relationships.png", dpi=300)
        else:
            plt.show()
        plt.close()