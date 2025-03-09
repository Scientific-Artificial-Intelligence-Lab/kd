"""
Deep Reinforcement Learning Visualization Module

This module provides functions for visualizing the training process and results of deep reinforcement learning models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# Set font that supports both English and Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Global plotting style configuration
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

# Global color map
DEFAULT_CMAP = 'viridis'


def configure_plotting(style=None, cmap=None):
    """
    Configure global plotting style and color map.
    
    Parameters:
        style (dict, optional): Custom style dictionary, will override default style.
        cmap (str, optional): Custom color map name.
    """
    global PLOT_STYLE, DEFAULT_CMAP

    if style:
        PLOT_STYLE.update(style)

    if cmap:
        DEFAULT_CMAP = cmap


def save_figure(fig, output_dir, filename, dpi=None, bbox_inches='tight', **kwargs):
    """
    Save figure to specified directory.
    
    Parameters:
        fig (matplotlib.figure.Figure): Figure object to save.
        output_dir (str): Output directory path.
        filename (str): File name, should include extension (e.g., .png, .pdf).
        dpi (int, optional): Resolution, if None uses the setting in PLOT_STYLE.
        bbox_inches (str, optional): Boundary box setting, default is 'tight'.
        **kwargs: Other parameters passed to plt.savefig.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use global DPI setting (if not specified)
    if dpi is None and 'savefig.dpi' in PLOT_STYLE:
        dpi = PLOT_STYLE['savefig.dpi']

    # Save figure
    fig.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches=bbox_inches, **kwargs)


def plot_training_rewards(model, output_dir=None, figsize=(10, 6)):
    """
    Plot reward changes during training.
    
    Parameters:
        model: Trained DeepRL model.
        output_dir (str, optional): Directory path to save the figure.
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

        # If output directory is provided, save the figure
        if output_dir:
            save_figure(fig, output_dir, 'training_rewards.png')

        return fig


def plot_reward_density(model, epochs=None, output_dir=None, figsize=(10, 6)):
    """
    Plot reward density distribution.
    
    Parameters:
        model: Trained DeepRL model.
        epochs (list, optional): List of specific epochs to plot.
        output_dir (str, optional): Directory path to save the figure.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    from scipy.stats import gaussian_kde
    import numpy as np
    import matplotlib.pyplot as plt

    configure_plotting()

    # Get rewards for all epochs
    r_all = model.searcher.r_history

    # If no epochs specified, select uniformly distributed epochs
    if epochs is None:
        num_epochs = len(r_all)
        if num_epochs <= 5:
            epochs = list(range(num_epochs))
        else:
            epochs = [0, num_epochs // 4, num_epochs // 2, 3 * num_epochs // 4, num_epochs - 1]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

    # Define x-axis range
    x_min, x_max = 0.0, 1.0
    x_grid = np.linspace(x_min, x_max, 1000)

    # Plot density graph
    if len(epochs) > 10:
        # If too many epochs, use color map
        norm = plt.Normalize(1, len(epochs))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])

        for i, epoch in enumerate(epochs):
            if epoch < len(r_all):
                # Filter out infinity and NaN values
                data = r_all[epoch]
                data = data[~np.isinf(data) & ~np.isnan(data)]

                # Ensure data is not empty and has enough unique values
                if len(data) > 5 and len(np.unique(data)) > 1:
                    # Use scipy's gaussian_kde to calculate kernel density estimation
                    try:
                        kde = gaussian_kde(data)
                        density = kde(x_grid)
                        ax.plot(x_grid, density, color=colors[i], alpha=0.7)
                        # Fill area under curve
                        ax.fill_between(x_grid, density, alpha=0.2, color=colors[i])
                    except Exception as e:
                        print(f"Epoch {epoch} KDE calculation failed: {str(e)}")

        cbar = fig.colorbar(sm, ticks=[0, 0.5, 1], ax=ax)
        cbar.ax.set_yticklabels([1, '', len(epochs)])
    else:
        # Only plot specified epochs
        for i, epoch in enumerate(epochs):
            if epoch < len(r_all):
                # Filter out infinity and NaN values
                data = r_all[epoch]
                data = data[~np.isinf(data) & ~np.isnan(data)]

                # Ensure data is not empty and has enough unique values
                if len(data) > 5 and len(np.unique(data)) > 1:
                    try:
                        # Use scipy's gaussian_kde to calculate kernel density estimation
                        kde = gaussian_kde(data)
                        density = kde(x_grid)
                        ax.plot(x_grid, density, color=colors[i], label=f'Epoch={epoch}')
                        # Fill area under curve
                        ax.fill_between(x_grid, density, alpha=0.3, color=colors[i])
                    except Exception as e:
                        print(f"Epoch {epoch} KDE calculation failed: {str(e)}")

        ax.legend(loc='best', frameon=False)

    ax.set_xlabel('Reward Value')
    ax.set_ylabel('Density')
    ax.set_xlim(x_min, x_max)

    # If output directory is provided, save the figure
    if output_dir:
        save_figure(fig, output_dir, 'reward_density.png')

    return fig


def plot_expression_tree(model, output_dir=None):
    """
    Plot the best expression tree.
    
    Parameters:
        model: Trained DeepRL model.
        output_dir (str, optional): Directory path to save the figure.
        
    Returns:
        graphviz.Digraph: Generated graph object, or matplotlib.figure.Figure if Graphviz is not installed.
    """
    # Get the best program
    best_program = model.searcher.best_p

    try:
        # Use existing tree_plot method to create graph
        graph = model.searcher.plotter.tree_plot(best_program)

        # If output directory is provided, save the graph
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            try:
                graph.render(os.path.join(output_dir, 'expression_tree'), format='png', cleanup=True)
            except Exception as e:
                print(f"Warning: Could not render expression tree graph: {str(e)}")
                print("Make sure Graphviz is installed and on your system PATH.")
                print("You can install Graphviz from: https://graphviz.org/download/")
                # Create a simple text file with the expression as a fallback
                if output_dir:
                    with open(os.path.join(output_dir, 'expression_tree.txt'), 'w') as f:
                        f.write(f"Best expression: {best_program.str_expression}")

        return graph
    except Exception as e:
        print(f"Warning: Could not create expression tree graph: {str(e)}")
        print("Make sure Graphviz is installed and on your system PATH.")
        print("You can install Graphviz from: https://graphviz.org/download/")
        
        # Create a simple text file with the expression as a fallback
        if output_dir:
            with open(os.path.join(output_dir, 'expression_tree.txt'), 'w') as f:
                f.write(f"Best expression: {best_program.str_expression}")
        
        # Create a better matplotlib figure as a fallback
        with plt.style.context(PLOT_STYLE):
            # Parse the expression into a more readable format
            expr_str = best_program.str_expression
            
            # Create a figure with appropriate size
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Set the title with the expression
            ax.text(0.5, 0.9, "Best Expression", ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0.5, 0.8, expr_str, ha='center', va='center', fontsize=12, 
                   bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round,pad=0.5'))
            
            # Add reward information
            if hasattr(best_program, 'r_ridge'):
                ax.text(0.5, 0.6, f"Reward: {best_program.r_ridge:.4f}", 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(facecolor='lightgreen', alpha=0.5, boxstyle='round,pad=0.5'))
            
            # Try to visualize the structure if possible
            try:
                # Split the expression into components for a simple tree visualization
                terms = expr_str.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
                unique_terms = sorted(set(terms))
                
                # Create a simple hierarchical layout
                if len(unique_terms) <= 10:  # Only attempt visualization for simpler expressions
                    # Draw a simple tree structure
                    ax.text(0.5, 0.4, "Expression Structure:", ha='center', va='center', fontsize=12)
                    
                    # Draw the unique terms in a circular layout
                    radius = 0.2
                    center_x, center_y = 0.5, 0.2
                    
                    for i, term in enumerate(unique_terms):
                        angle = 2 * np.pi * i / len(unique_terms)
                        x = center_x + radius * np.cos(angle)
                        y = center_y + radius * np.sin(angle)
                        
                        # Count term frequency
                        count = terms.count(term)
                        size = 10 + 2 * count  # Size based on frequency
                        
                        # Draw the term
                        ax.text(x, y, term, ha='center', va='center', fontsize=size,
                               bbox=dict(facecolor='lightyellow', alpha=0.7, boxstyle='round,pad=0.3'))
                        
                        # Draw a line to the center
                        ax.plot([center_x, x], [center_y + 0.1, y], 'k-', alpha=0.3)
            except:
                # If visualization fails, just skip it
                pass
                
            ax.axis('off')
            
            if output_dir:
                save_figure(fig, output_dir, 'expression_tree_fallback.png')
            
            return fig


def plot_best_expressions(model, top_n=5, output_dir=None, figsize=(12, 8)):
    """
    Plot the top N best expressions by reward value.
    
    Parameters:
        model: Trained DeepRL model.
        top_n (int, optional): Number of top expressions to plot.
        output_dir (str, optional): Directory path to save the figure.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        # Get top N expressions from priority queue
        expressions = []
        rewards = []

        # First, try to get the best expression directly from the model
        best_program = model.searcher.best_p
        if best_program is not None:
            expressions.append(best_program.str_expression)
            rewards.append(best_program.r_ridge if hasattr(best_program, 'r_ridge') else 0.0)
            
            # Add a note about the best expression
            print(f"Best expression: {best_program.str_expression}")
            print(f"Reward: {best_program.r_ridge if hasattr(best_program, 'r_ridge') else 'N/A'}")

        # Try to get additional expressions if needed
        if top_n > 1:
            try:
                # Try to get expressions from the priority queue
                for i, (item_id, program) in enumerate(model.searcher.pq.iter_in_order()):
                    # Skip the first one if we already have the best program
                    if i == 0 and len(expressions) > 0:
                        continue
                        
                    if len(expressions) >= top_n:
                        break
                        
                    # Try different ways to extract expression and reward
                    if hasattr(program, 'str_expression') and hasattr(program, 'r_ridge'):
                        expressions.append(program.str_expression)
                        rewards.append(program.r_ridge)
                    elif isinstance(program, dict) and 'str_expression' in program and 'r_ridge' in program:
                        expressions.append(program['str_expression'])
                        rewards.append(program['r_ridge'])
                    else:
                        # Generate placeholder for missing expressions
                        expr_num = len(expressions) + 1
                        expressions.append(f"Expression {expr_num}")
                        rewards.append(0.0)
            except Exception as e:
                print(f"Warning: Error extracting additional expressions: {str(e)}")
                
        # If we still don't have enough expressions, generate placeholders
        while len(expressions) < top_n:
            expr_num = len(expressions) + 1
            expressions.append(f"Expression {expr_num}")
            rewards.append(0.0)
            
        # Make sure we have at least one expression
        if not expressions:
            expressions = ["No expressions available"]
            rewards = [0.0]

        # Plot horizontal bar chart
        y_pos = np.arange(len(expressions))
        ax.barh(y_pos, rewards, align='center', color='skyblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(expressions)
        ax.invert_yaxis()  # Labels read from top to bottom
        ax.set_xlabel('Reward Value')
        ax.set_title('Top Expressions by Reward Value')

        # If output directory is provided, save the figure
        if output_dir:
            save_figure(fig, output_dir, 'best_expressions.png')

        return fig


def plot_simulated_annealing_metrics(model, output_dir=None, figsize=(12, 10)):
    """
    Plot simulated annealing optimization metrics.
    
    Parameters:
        model: Trained DeepRL model.
        output_dir (str, optional): Directory path to save the figure.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object, or None if no simulated annealing data is available.
    """
    try:
        # Check if simulated annealing data is available
        if not hasattr(model.searcher, 'temperature_history'):
            print("No simulated annealing data available")
            return None
            
        # Check if temperature_history exists and is not empty
        if model.searcher.temperature_history is None:
            print("Temperature history is None")
            return None
            
        # Check if it's a list or numpy array and not empty
        if isinstance(model.searcher.temperature_history, (list, tuple)):
            if len(model.searcher.temperature_history) == 0:
                print("Temperature history is empty")
                return None
        elif isinstance(model.searcher.temperature_history, np.ndarray):
            if model.searcher.temperature_history.size == 0:
                print("Temperature history is empty")
                return None
        else:
            print("Temperature history has unknown type")
            return None

        with plt.style.context(PLOT_STYLE):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
            
            # Extract simulated annealing metrics
            temperatures = np.array(model.searcher.temperature_history)
            iterations = np.arange(1, len(temperatures) + 1)
            
            # 检查是否存在接受率历史数据
            acceptance_rates = None
            if hasattr(model.searcher, 'acceptance_rate_history'):
                if model.searcher.acceptance_rate_history is not None:
                    if isinstance(model.searcher.acceptance_rate_history, np.ndarray):
                        if model.searcher.acceptance_rate_history.size > 0:
                            acceptance_rates = model.searcher.acceptance_rate_history
                    elif isinstance(model.searcher.acceptance_rate_history, (list, tuple)):
                        if len(model.searcher.acceptance_rate_history) > 0:
                            acceptance_rates = np.array(model.searcher.acceptance_rate_history)
            
            # 检查是否存在扰动大小历史数据
            perturbation_sizes = None
            if hasattr(model.searcher, 'perturbation_size_history'):
                if model.searcher.perturbation_size_history is not None:
                    if isinstance(model.searcher.perturbation_size_history, np.ndarray):
                        if model.searcher.perturbation_size_history.size > 0:
                            perturbation_sizes = model.searcher.perturbation_size_history
                    elif isinstance(model.searcher.perturbation_size_history, (list, tuple)):
                        if len(model.searcher.perturbation_size_history) > 0:
                            perturbation_sizes = np.array(model.searcher.perturbation_size_history)

            # Plot temperature curve
            ax1.plot(iterations, temperatures, 'r-', linewidth=2)
            ax1.set_ylabel('Temperature')
            ax1.set_title('Simulated Annealing Optimization Metrics')
            ax1.grid(True, linestyle='--', alpha=0.5)

            # If acceptance rate data is available, plot acceptance rate
            if acceptance_rates is not None:
                # 确保 acceptance_rates 是一维数组
                if isinstance(acceptance_rates, (list, tuple)):
                    acceptance_rates = np.array(acceptance_rates)
                if len(acceptance_rates.shape) > 1:
                    acceptance_rates = acceptance_rates.flatten()
                
                # 确保长度匹配
                plot_iterations = iterations
                if len(acceptance_rates) < len(iterations):
                    plot_iterations = iterations[:len(acceptance_rates)]
                elif len(acceptance_rates) > len(iterations):
                    acceptance_rates = acceptance_rates[:len(iterations)]
                
                ax2.plot(plot_iterations, acceptance_rates, 'b-', linewidth=2)
                ax2.set_ylabel('Acceptance Rate')
                ax2.grid(True, linestyle='--', alpha=0.5)
                
                # 添加水平线表示最小和最大接受率阈值（根据记忆中的信息）
                ax2.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Min Acceptance Rate')
                ax2.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Max Acceptance Rate')
                ax2.legend()

            # If perturbation size data is available, plot perturbation size
            if perturbation_sizes is not None:
                # 如果扰动大小是一个列表的列表（分层扰动），则分别绘制每一层
                if isinstance(perturbation_sizes, (list, tuple)) and len(perturbation_sizes) > 0:
                    # Check if the first element is also a list/tuple/array (indicating layered perturbation)
                    if len(perturbation_sizes) > 0 and isinstance(perturbation_sizes[0], (list, tuple, np.ndarray)):
                        # 分层扰动大小
                        perturbation_layers = []
                        for i in range(len(perturbation_sizes[0])):
                            layer_sizes = []
                            for j in range(len(perturbation_sizes)):
                                if i < len(perturbation_sizes[j]):
                                    layer_sizes.append(perturbation_sizes[j][i])
                            perturbation_layers.append(layer_sizes)
                        
                        labels = ['CNN Layer', 'LSTM Layer', 'Linear Layer']
                        colors = ['g', 'b', 'purple']
                        
                        for i, (layer_sizes, label, color) in enumerate(zip(perturbation_layers, labels, colors)):
                            if layer_sizes and len(layer_sizes) > 0:  # 确保不是空列表
                                plot_iter = iterations[:len(layer_sizes)]
                                ax3.plot(plot_iter, layer_sizes, color=color, linewidth=2, label=label)
                        
                        ax3.legend()
                    else:
                        # 单一扰动大小
                        # 确保 perturbation_sizes 是一维数组
                        if isinstance(perturbation_sizes, (list, tuple)):
                            perturbation_sizes = np.array(perturbation_sizes)
                        if len(perturbation_sizes.shape) > 1:
                            perturbation_sizes = perturbation_sizes.flatten()
                        
                        # 确保长度匹配
                        plot_iterations = iterations
                        if len(perturbation_sizes) < len(iterations):
                            plot_iterations = iterations[:len(perturbation_sizes)]
                        elif len(perturbation_sizes) > len(iterations):
                            perturbation_sizes = perturbation_sizes[:len(iterations)]
                        
                        ax3.plot(plot_iterations, perturbation_sizes, 'g-', linewidth=2)
                
                ax3.set_ylabel('Perturbation Size')
                ax3.set_xlabel('Iteration Count')
                ax3.grid(True, linestyle='--', alpha=0.5)

            # 添加初始温度和冷却率的注释（根据记忆中的信息）
            ax1.annotate(f'Initial Temp: 80.0\nCooling Rate: 0.97\nInner Loop: 15', 
                        xy=(0.02, 0.85), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Add reference lines for different layers' perturbation sizes
            ax3.axhline(y=0.002, color='g', linestyle='--', alpha=0.5)
            ax3.axhline(y=0.001, color='b', linestyle='--', alpha=0.5)
            ax3.axhline(y=0.0015, color='purple', linestyle='--', alpha=0.5)
            ax3.text(iterations[-1] * 0.02, 0.002, 'CNN Layer (0.002)', fontsize=8, color='g')
            ax3.text(iterations[-1] * 0.02, 0.001, 'LSTM Layer (0.001)', fontsize=8, color='b')
            ax3.text(iterations[-1] * 0.02, 0.0015, 'Linear Layer (0.0015)', fontsize=8, color='purple')

            plt.tight_layout()
            
            # If output directory is provided, save the figure
            if output_dir:
                save_figure(fig, output_dir, 'simulated_annealing_metrics.png')
                
            return fig
    except Exception as e:
        print(f"Warning: Could not generate simulated annealing metrics plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(models, labels, output_dir=None, figsize=(12, 8)):
    """
    Compare the training performance of multiple models.
    
    Parameters:
        models (list): List of trained DeepRL models.
        labels (list): List of labels corresponding to the models.
        output_dir (str, optional): Directory path to save the figure.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the best reward for each model
        for i, model in enumerate(models):
            rewards = model.searcher.r_train
            x = np.arange(1, len(rewards) + 1)
            r_max = [np.max(r) for r in rewards]
            r_best = np.maximum.accumulate(np.array(r_max))

            ax.plot(x, r_best, linewidth=2, label=labels[i])

        ax.set_xlabel('Iteration Count')
        ax.set_ylabel('Best Reward')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', frameon=False)

        # If output directory is provided, save the figure
        if output_dir:
            save_figure(fig, output_dir, 'model_comparison.png')

        return fig


def plot_all_metrics(model, output_dir=None):
    """
    Generate all available metrics plots.
    
    Parameters:
        model: Trained DeepRL model.
        output_dir (str, optional): Directory path to save the figures.
        
    Returns:
        dict: Dictionary of generated figure objects.
    """
    figures = {}

    # If output directory is not provided, create it
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    figures['training_rewards'] = plot_training_rewards(model, output_dir)
    figures['reward_density'] = plot_reward_density(model, output_dir=output_dir)

    # Try to generate expression tree, but don't fail if it's not possible
    expression_tree = plot_expression_tree(model, output_dir)
    if expression_tree is not None:
        figures['expression_tree'] = expression_tree

    figures['best_expressions'] = plot_best_expressions(model, output_dir=output_dir)

    # If simulated annealing data is available, generate simulated annealing plot
    sa_plot = plot_simulated_annealing_metrics(model, output_dir)
    if sa_plot:
        figures['simulated_annealing'] = sa_plot

    return figures
