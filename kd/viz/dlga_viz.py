"""Visualization module for DLGA results (v0.2).

This module provides visualization tools for analyzing and plotting results 
from Deep Learning Genetic Algorithm (DLGA) model runs.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
from scipy.interpolate import griddata

# Global plot configuration
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
    """Configure global plotting settings.

    Args:
        style: Custom style dictionary
        cmap: Default colormap name
    """
    plt.rcParams.update(style or PLOT_STYLE)
    global DEFAULT_CMAP
    if cmap:
        DEFAULT_CMAP = cmap


def plot_training_loss(model, output_dir: str = None):
    """Plot training loss curve.
    
    Args:
        model: DLGA model containing train_loss_history
        output_dir: Output directory path
    """
    # Configure style
    plt.figure(figsize=(10, 5))
    plt.plot(model.train_loss_history,
             color='#1f77b4',
             linewidth=2)
    plt.xlabel('Training Epoch')
    plt.ylabel('Training Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.5)

    if output_dir:
        # Create directory
        viz_dir = Path(output_dir)
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / "training_loss.png", dpi=300)
    else:
        plt.show()
    plt.close()


def plot_validation_loss(model, output_dir: str = None):
    """Plot validation loss curve.
    
    Args:
        model: DLGA model containing val_loss_history
        output_dir: Output directory path
    """
    plt.figure(figsize=(10, 5))
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
    """Visualize residual analysis including training points and overall distribution.
    
    Args:
        model: DLGA model instance
        X_train: Training data coordinates
        y_train: Training data values  
        u_true: True solution array
        u_pred: Predicted solution array
        output_dir: Output directory path
    """
    plt.figure(figsize=(10, 4))

    # Left plot: Training points residuals
    plt.subplot(121)
    # Calculate training point predictions
    with torch.no_grad():
        train_pred = model.Net(torch.tensor(X_train, dtype=torch.float32)).numpy().flatten()
    # train_residuals = y_train - train_pred
    train_residuals = y_train.flatten() - train_pred.flatten()
    
    sc = plt.scatter(X_train[:, 1],  # Time coordinates
                     X_train[:, 0],  # Space coordinates
                     c=train_residuals,
                     cmap='coolwarm',
                     s=10,
                     edgecolors='w',
                     linewidths=0.5)
    plt.colorbar(sc, label='Residual').outline.set_visible(False)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Space', fontsize=10)
    plt.title('Training Points Residual', pad=10)

    # Right plot: Overall residual distribution
    plt.subplot(122)
    residual = u_true - u_pred
    n, bins, patches = plt.hist(residual.flatten(),
                                bins=50,
                                density=True,
                                edgecolor='black',
                                linewidth=0.5)
    # Color patches to match coolwarm colormap
    cmap = plt.get_cmap('coolwarm')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        plt.setp(p, 'facecolor', cmap((c - bin_centers[0]) / (bin_centers[-1] - bin_centers[0])))

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
    """Compare true and predicted PDE solutions.
    
    Args:
        x: Spatial coordinates
        t: Time coordinates
        u_true: True solution array
        u_pred: Predicted solution array
        output_dir: Output directory path
    """

    # Apply global configuration
    with plt.style.context(PLOT_STYLE):

        T, X = np.meshgrid(t, x)
        vmin = min(u_true.min(), u_pred.min())
        vmax = max(u_true.max(), u_pred.max())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

        # True solution
        mesh1 = ax1.pcolormesh(T, X, u_true,
                               shading='gouraud',
                               cmap=DEFAULT_CMAP,
                               vmin=vmin, vmax=vmax)
        ax1.set(title='Exact Solution', xlabel='Time (Exact Solution)', ylabel='Space')
        fig.colorbar(mesh1, ax=ax1, label='u(x,t)')

        # Predicted solution
        mesh2 = ax2.pcolormesh(T, X, u_pred,
                               shading='gouraud',
                               cmap=DEFAULT_CMAP,
                               vmin=vmin, vmax=vmax)
        ax2.set(title='Predicted Solution', xlabel='Time (Predicted Solution)')
        fig.colorbar(mesh2, ax=ax2, label='u(x,t)')



        if output_dir:
            viz_dir = Path(output_dir)
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / 'pde_comparison.png')
        else:
            plt.show()
        plt.close()


def plot_equation_terms(
        model,
        terms: dict,
        color_var: str = 'u_t',
        equation_name: str = "PDE",
        output_dir: str = None,
        filename: str = "equation_terms_analysis.png"
):
    """Visualize relationships between equation terms.
    
    Args:
        model: DLGA model instance
        terms: Dictionary defining terms to visualize
        color_var: Variable name for coloring
        equation_name: Name of equation for title
        output_dir: Output directory path
        filename: Output filename
    """
    metadata = model.metadata
    with plt.style.context(PLOT_STYLE):
        # Extract and calculate terms
        x_vars = terms.get('x_term', {}).get('vars', [])
        y_vars = terms.get('y_term', {}).get('vars', [])

        if not x_vars or not y_vars:
            print("Warning: Insufficient terms specified, cannot plot")
            return

        # 1. 收集本次绘图所需的所有变量名
        required_vars = set(x_vars + y_vars)
        if color_var:
            required_vars.add(color_var)
        
        # 2. 检查这些变量名是否都存在于 metadata 字典中
        missing_vars = [var for var in required_vars if var not in metadata]
        
        # 3. 如果有任何缺失的变量，则打印清晰的错误信息并退出，而不是崩溃
        if missing_vars:
            print(f"警告: 无法绘制该图表，因为以下必需的项在元数据中缺失: {missing_vars}")
            return

        # Dynamically calculate term products
        x_values = np.prod([metadata[key] for key in x_vars], axis=0)
        y_values = np.prod([metadata[key] for key in y_vars], axis=0)

        # Get labels
        x_label = terms.get('x_term', {}).get('label', '-'.join(x_vars))
        y_label = terms.get('y_term', {}).get('label', '-'.join(y_vars))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: Term relationship scatter plot
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

        # Subplot 2: Time derivative distribution
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
    """Visualize the evolution process of genetic algorithm.
    
    Args:
        model: DLGA model with evolution history
        output_dir: Output directory path
    """
    with plt.style.context(PLOT_STYLE):
        # Check data availability
        if not hasattr(model, 'evolution_history') or not model.evolution_history:
            raise ValueError("No evolution history data available, ensure evolution() method has been run")

        # Create figure and subplots
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Extract data
        generations = range(len(model.evolution_history))
        fitness = np.array([x['fitness'] for x in model.evolution_history])
        complexity = np.array([x['complexity'] for x in model.evolution_history])

        # 1. Fitness evolution curve
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

        # 2. Equation complexity evolution
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

        # Print evolution statistics
        print("\nEvolution Analysis Summary:")
        print(f"Initial fitness: {fitness[0]:.4f}")
        print(f"Final fitness: {fitness[-1]:.4f}")
        print(f"Improvement: {((fitness[0] - fitness[-1]) / fitness[0]):.2%}")
        print(f"Initial complexity: {complexity[0]}")
        print(f"Final complexity: {complexity[-1]}")


def plot_optimization_analysis(model, output_dir: str = None):
    """Analyze optimization metrics including population diversity.
    
    Args:
        model: DLGA model with optimization history
        output_dir: Output directory path
    """
    # Create figure and subplots
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    with plt.style.context(PLOT_STYLE):
        # 1. Left plot: Population diversity analysis
        generations = range(len(model.evolution_history))
        fitness_history = np.array([x['fitness'] for x in model.evolution_history])
        pop_sizes = [x.get('population_size', 0) for x in model.evolution_history]
        unique_modules = [x.get('unique_modules', 0) for x in model.evolution_history]

        # Plot population diversity trend - using dual y-axis
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

        # Merge legends of both y-axes
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        # Dynamically set y-axis ranges
        pop_min, pop_max = min(pop_sizes), max(pop_sizes)
        mod_min, mod_max = min(unique_modules), max(unique_modules)
        # ax1.set_ylim(pop_min - 50, pop_max + 50)
        ax1.set_ylim(pop_min * 0.9, pop_max * 1.1) # adjusted for better visibility
        ax1_twin.set_ylim(mod_min - 5, mod_max + 5)

        # 2. Right plot: Fitness evolution analysis
        fitness_changes = np.abs(np.diff(fitness_history))

        # Calculate cumulative changes and find major change interval
        if len(fitness_changes) > 0:
            cumsum_changes = np.cumsum(fitness_changes)
            total_change = cumsum_changes[-1]

            # Find interval containing 80% of changes (adjust threshold as needed)
            threshold = 0.8 * total_change
            significant_idx = np.searchsorted(cumsum_changes, threshold)
            show_gens = min(significant_idx + 5, len(fitness_history) - 1)

            # Plot overall trend (lower opacity)
            ax2.plot(generations, fitness_history,
                     color='#2ca02c', linewidth=1,
                     marker='o', markersize=2, alpha=0.3)

            # Highlight major change interval
            ax2.plot(generations[:show_gens + 1], fitness_history[:show_gens + 1],
                     color='#2ca02c', linewidth=2,
                     marker='o', markersize=4, alpha=0.8,
                     label=f'Major changes (first {show_gens} gen)')
            ax2.legend()
        else:
            # If no changes, plot full curve only
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

        # Print detailed analysis summary (avoid index out of range)
        print("\nOptimization Analysis Summary:")
        print(f"Initial fitness: {fitness_history[0]:.4f}")
        print(f"Final fitness: {fitness_history[-1]:.4f}")
        print(f"Major improvements occurred in first {show_gens} generations")
        if show_gens > 0:
            early_improvement = ((fitness_history[0] - fitness_history[show_gens]) / fitness_history[0])
            print(f"Improvement in major change period: {early_improvement:.2%}")
        total_improvement = ((fitness_history[0] - fitness_history[-1]) / fitness_history[0])
        print(f"Total improvement: {total_improvement:.2%}")
        print(f"Average population size: {np.mean(pop_sizes):.1f}")
        print(f"Average unique modules: {np.mean(unique_modules):.1f}")
        print(f"Diversity ratio: {np.mean(unique_modules) / np.mean(pop_sizes):.2%}")


def plot_time_slices(x, t, u_true, u_pred, slice_times, output_dir: str = None):
    """Plot solution comparisons at different time slices.
    
    Args:
        x: Spatial coordinates
        t: Time coordinates
        u_true: True solution array 
        u_pred: Predicted solution array
        slice_times: List of relative time points for slicing
        output_dir: Output directory path
    """

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(1, len(slice_times), figsize=(15, 5), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, t_slice in enumerate(slice_times):
            # Convert relative time to index
            t_idx = int(t_slice * (len(t) - 1))

            # Debug info
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


# 假设其他 import 保持不变
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 【最终修正】: 函数名恢复为 plot_derivative_relationships
def plot_derivative_relationships(model, top_n_terms: int = 4, output_dir: str = None):
    """
    Dynamically analyzes and visualizes the relationships between the discovered 
    right-hand side (RHS) terms and the left-hand side (LHS) term of the PDE.

    This function automatically parses the best equation found by the model and
    plots the most significant terms, making it a general-purpose diagnostic tool.

    Args:
        model: The trained KD_DLGA model instance containing final results.
        top_n_terms (int): The maximum number of most significant terms to plot.
        output_dir (str): Optional. Directory path to save the plot.
    """
    # --- Step 1: Securely extract results from the trained model ---
    if not hasattr(model, 'Chrom') or not model.Chrom or not model.Chrom[0]:
        print("Warning: Model has not found a valid solution. Cannot generate term relationship plot.")
        return

    best_chrom = model.Chrom[0]
    coefficients = model.coef[0]
    operator_names = model.user_operators
    lhs_name = model.name[0]
    metadata = model.metadata

    if lhs_name not in metadata:
        print(f"Warning: LHS term '{lhs_name}' not found in metadata. Cannot generate plot.")
        return
    
    lhs_values = metadata[lhs_name].flatten()

    # --- Step 2: Parse the chromosome to identify and calculate RHS terms ---
    discovered_terms = []
    for i, module in enumerate(best_chrom):
        if any(gene >= len(operator_names) for gene in module):
            print(f"Warning: Invalid gene found in module {module}. Skipping this term.")
            continue

        term_label = ' * '.join([operator_names[gene] for gene in module])
        term_values = np.prod([metadata[operator_names[gene]] for gene in module], axis=0).flatten()

        discovered_terms.append({
            'label': term_label,
            'values': term_values,
            'coeff': coefficients[i, 0]
        })

    # --- Step 3: Sort terms by coefficient magnitude to focus on the most significant ones ---
    discovered_terms.sort(key=lambda x: abs(x['coeff']), reverse=True)
    terms_to_plot = discovered_terms[:top_n_terms]

    if not terms_to_plot:
        print("Warning: Could not parse any valid terms from the solution to plot.")
        return

    # --- Step 4: Dynamically create subplot grid and generate plots ---
    num_plots = len(terms_to_plot)
    ncols = min(num_plots, 3)
    nrows = int(np.ceil(num_plots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)
    axes = axes.flatten()

    fig.suptitle(f'Relationship of Discovered Terms with {lhs_name}', fontsize=16)

    for i, term_info in enumerate(terms_to_plot):
        ax = axes[i]
        
        ax.scatter(term_info['values'], lhs_values, alpha=0.3, s=15, c=term_info['values'])
        
        # 先准备好标签字符串，再将其放入 f-string，以避免语法错误
        latex_label = term_info['label'].replace(' * ', r' \cdot ')
        ax.set_xlabel(f"Term: ${latex_label}$")

        ax.set_ylabel(lhs_name)
        ax.set_title(f"Discovered Coefficient: {term_info['coeff']:.4f}")
        ax.grid(True, linestyle='--', alpha=0.5)

    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir:
        viz_dir = Path(output_dir)
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / "derivative_relationships.png", dpi=300)
    else:
        plt.show()
    plt.close()


def plot_pde_parity(model, title: str = "Parity Plot for Discovered PDE", output_dir: str = None):
    """
    Generates a parity plot to visually validate the discovered PDE.

    This function calculates the values of the equation's right-hand side (RHS)
    based on the discovered coefficients and terms, and plots them against the 
    true values of the left-hand side (LHS). A perfect match results in points
    lying on the y=x diagonal.

    Args:
        model: The trained KD_DLGA model instance containing final results.
        title (str): The title for the plot.
        output_dir (str): Optional. Directory path to save the plot.
    """
    # Step 1: Securely extract results from the trained model
    if not hasattr(model, 'Chrom') or not model.Chrom or not model.Chrom[0]:
        print("Warning: Model has not found a valid solution. Cannot generate parity plot.")
        return

    best_chrom = model.Chrom[0]
    coefficients = model.coef[0]
    operator_names = model.user_operators
    lhs_name = model.name[0]
    metadata = model.metadata

    if lhs_name not in metadata:
        print(f"Warning: LHS term '{lhs_name}' not found in metadata. Cannot generate plot.")
        return

    # Step 2: Calculate the "true" LHS and "predicted" RHS values
    # The "true" y-values are the numerical values of the LHS term (e.g., u_t)
    y_true_pde = metadata[lhs_name]

    # The "predicted" y-values are the sum of all discovered RHS terms,
    # weighted by their discovered coefficients.
    y_pred_pde = np.zeros_like(y_true_pde)
    for i, module in enumerate(best_chrom):
        # Calculate the numerical values of the current term (e.g., u*u_x)
        term_values = np.prod([metadata[operator_names[gene]] for gene in module], axis=0)
        # Add the weighted term to the total RHS value
        y_pred_pde += coefficients[i, 0] * term_values

    # Step 3: Create the parity plot
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Create a scatter plot of Predicted vs. True values
        ax.scatter(y_pred_pde, y_true_pde, alpha=0.3, s=20, label='Model Prediction vs. True Value')

        # Add a y=x line for reference. Points on this line represent a perfect match.
        perfect_fit_line = np.linspace(min(y_true_pde.min(), y_pred_pde.min()),
                                       max(y_true_pde.max(), y_pred_pde.max()), 100)
        ax.plot(perfect_fit_line, perfect_fit_line, 'r--', linewidth=2, label='Perfect Fit (y=x)')

        ax.set_xlabel('Predicted RHS Values (from discovered equation)')
        ax.set_ylabel(f'True LHS Values ({lhs_name})')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        # Ensure the plot has a 1:1 aspect ratio
        ax.set_aspect('equal', 'box')

        if output_dir:
            viz_dir = Path(output_dir)
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / 'pde_parity_plot.png')
        else:
            plt.show()
        plt.close()