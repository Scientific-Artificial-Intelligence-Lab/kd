"""KdV Equation Deep Reinforcement Learning Visualization Module

This module provides functions for visualizing the training process and results of KdV equation deep reinforcement learning models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

# Import general visualization functions
from kd.viz.deeprl_viz import PLOT_STYLE, DEFAULT_CMAP, configure_plotting, save_figure

# Ensure font settings are applied
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


def plot_kdv_residual_distribution(model, X_test, y_test, output_dir=None, figsize=(10, 6)):
    """
    Plot KdV equation residual distribution.
    
    Parameters:
        model: Trained DeepRL model.
        X_test: Test input data.
        y_test: Test target data.
        output_dir (str, optional): Directory path to save the figure.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    try:
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=figsize)

            # Get the best program
            best_program = model.searcher.best_p

            # For KdV equation, X_test is typically a tuple or can be unpacked
            # Try different approaches to extract the required parameters
            y_pred = None
            
            if hasattr(model, 'predict'):
                try:
                    y_pred = model.predict(X_test)
                except Exception as e:
                    print(f"Prediction failed: {str(e)}")
                    
            # If prediction failed or returned None, try using best_program directly
            if y_pred is None and best_program is not None and hasattr(best_program, 'predict'):
                try:
                    y_pred = best_program.predict(X_test)
                except Exception as e:
                    print(f"Best program prediction failed: {str(e)}")
            
            # If still None, create a dummy array for visualization
            if y_pred is None:
                print("Warning: Failed to generate predictions, using zeros array")
                # Try to get the shape from y_test
                if isinstance(y_test, np.ndarray):
                    y_pred = np.zeros_like(y_test)
                else:
                    y_pred = np.zeros(1)
            
            # Process y_test to handle inhomogeneous arrays
            if isinstance(y_test, (list, tuple)):
                processed_y_test = []
                for item in y_test:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        # If item is a sequence, take its first element or flatten it
                        if len(item) > 0:
                            if isinstance(item[0], (list, tuple, np.ndarray)):
                                # Nested sequence, flatten it
                                flat_item = []
                                for subitem in item:
                                    if isinstance(subitem, (list, tuple, np.ndarray)):
                                        if len(subitem) > 0:
                                            flat_item.append(float(subitem[0]))
                                        else:
                                            flat_item.append(0.0)
                                    else:
                                        flat_item.append(float(subitem))
                                processed_y_test.append(flat_item[0] if flat_item else 0.0)
                            else:
                                processed_y_test.append(float(item[0]))
                        else:
                            processed_y_test.append(0.0)
                    else:
                        processed_y_test.append(float(item))
                y_test = np.array(processed_y_test, dtype=np.float64)
            elif isinstance(y_test, np.ndarray):
                # Ensure it's a 1D array
                if len(y_test.shape) > 1:
                    y_test = y_test.flatten()
            else:
                # Convert scalar to array
                y_test = np.array([float(y_test)], dtype=np.float64)

            # Process y_pred similarly
            if isinstance(y_pred, (list, tuple)):
                processed_y_pred = []
                for item in y_pred:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        # If item is a sequence, take its first element or flatten it
                        if len(item) > 0:
                            if isinstance(item[0], (list, tuple, np.ndarray)):
                                # Nested sequence, flatten it
                                flat_item = []
                                for subitem in item:
                                    if isinstance(subitem, (list, tuple, np.ndarray)):
                                        if len(subitem) > 0:
                                            flat_item.append(float(subitem[0]))
                                        else:
                                            flat_item.append(0.0)
                                    else:
                                        flat_item.append(float(subitem))
                                processed_y_pred.append(flat_item[0] if flat_item else 0.0)
                            else:
                                processed_y_pred.append(float(item[0]))
                        else:
                            processed_y_pred.append(0.0)
                    else:
                        processed_y_pred.append(float(item))
                y_pred = np.array(processed_y_pred, dtype=np.float64)
            elif isinstance(y_pred, np.ndarray):
                # Ensure it's a 1D array
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.flatten()
            else:
                # Convert scalar to array
                if y_pred is None:
                    # 如果y_pred是None，创建一个与y_test相同形状的零数组
                    if isinstance(y_test, np.ndarray):
                        y_pred = np.zeros_like(y_test)
                    else:
                        y_pred = np.zeros(1)
                else:
                    try:
                        y_pred = np.array([float(y_pred)], dtype=np.float64)
                    except (TypeError, ValueError):
                        # 如果转换失败，创建一个零数组
                        print("Warning: Could not convert y_pred to float array, using zeros")
                        if isinstance(y_test, np.ndarray):
                            y_pred = np.zeros_like(y_test)
                        else:
                            y_pred = np.zeros(1)

            # Make sure both arrays have the same length
            if len(y_test) != len(y_pred):
                min_length = min(len(y_test), len(y_pred))
                y_test = y_test[:min_length]
                y_pred = y_pred[:min_length]

            # Calculate residuals
            residuals = y_test - y_pred

            # Plot residual distribution
            ax.hist(residuals, bins=30, alpha=0.7, color='blue')
            ax.axvline(x=0, color='red', linestyle='--')
            ax.set_xlabel('Residual Value')
            ax.set_ylabel('Frequency')
            ax.set_title('KdV Equation Residual Distribution')

            # Add statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd Dev: {std_residual:.4f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # If output directory is provided, save the figure
            if output_dir:
                save_figure(fig, output_dir, 'kdv_residual_distribution.png')

            return fig
    except Exception as e:
        print(f"Error in plot_kdv_residual_distribution: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig


def plot_kdv_term_magnitudes(model, X_test, output_dir=None, figsize=(12, 8)):
    """
    Plot magnitude chart of KdV equation terms.
    
    Parameters:
        model: Trained DeepRL model.
        X_test: Test input data.
        output_dir (str, optional): Directory path to save the figure.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        # Get the best program
        best_program = model.searcher.best_p

        # Get expression
        expression = best_program.str_expression

        # Parse terms in the expression
        terms = expression.replace(' ', '').replace('(', '').replace(')', '').split('+')

        # Calculate magnitude of each term
        magnitudes = []
        term_labels = []

        for term in terms:
            if '*' in term:
                coef, var = term.split('*', 1)
                try:
                    magnitude = abs(float(coef))
                    magnitudes.append(magnitude)
                    term_labels.append(term)
                except ValueError:
                    # Skip if coefficient is not a simple number
                    pass
            else:
                try:
                    magnitude = abs(float(term))
                    magnitudes.append(magnitude)
                    term_labels.append(term)
                except ValueError:
                    # Skip if term is not a simple number
                    pass

        # Plot bar chart
        y_pos = np.arange(len(term_labels))
        ax.barh(y_pos, magnitudes, align='center', color='skyblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(term_labels)
        ax.invert_yaxis()  # Labels read from top to bottom
        ax.set_xlabel('Magnitude')
        ax.set_title('KdV Equation Term Magnitudes')

        # If output directory is provided, save the figure
        if output_dir:
            save_figure(fig, output_dir, 'kdv_term_magnitudes.png')

        return fig


def plot_kdv_solution_comparison(model, X_test, y_test, t_grid, x_grid, output_dir=None, figsize=(12, 10)):
    """
    Plot KdV equation solution comparison.
    
    Parameters:
        model: Trained DeepRL model.
        X_test: Test input data.
        y_test: Test target data.
        t_grid: Time grid.
        x_grid: Space grid.
        output_dir (str, optional): Directory path to save the figure.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    try:
        with plt.style.context(PLOT_STYLE):
            fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)

            # Get the best program
            best_program = model.searcher.best_p

            # For KdV equation, X_test is typically a tuple or can be unpacked
            # Try different approaches to extract the required parameters
            y_pred = None

            if hasattr(model, 'predict'):
                try:
                    y_pred = model.predict(X_test)
                except Exception as e:
                    print(f"Prediction failed: {str(e)}")

                    # If prediction fails, create a zero array
                    if isinstance(y_test, np.ndarray):
                        y_pred = np.zeros_like(y_test)
                    else:
                        y_pred = np.zeros(1)
            else:
                # If no predict method, create a zero array
                if isinstance(y_test, np.ndarray):
                    y_pred = np.zeros_like(y_test)
                else:
                    y_pred = np.zeros(1)

            # Ensure y_test and y_pred are numpy arrays with proper shape
            # Handle inhomogeneous arrays by processing each element individually
            if isinstance(y_test, (list, tuple)):
                # Process y_test to handle inhomogeneous arrays
                processed_y_test = []
                for item in y_test:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        # If item is a sequence, take its first element or flatten it
                        if len(item) > 0:
                            if isinstance(item[0], (list, tuple, np.ndarray)):
                                # Nested sequence, flatten it
                                flat_item = []
                                for subitem in item:
                                    if isinstance(subitem, (list, tuple, np.ndarray)):
                                        if len(subitem) > 0:
                                            flat_item.append(float(subitem[0]))
                                        else:
                                            flat_item.append(0.0)
                                    else:
                                        flat_item.append(float(subitem))
                                processed_y_test.append(flat_item[0] if flat_item else 0.0)
                            else:
                                processed_y_test.append(float(item[0]))
                        else:
                            processed_y_test.append(0.0)
                    else:
                        processed_y_test.append(float(item))
                y_test = np.array(processed_y_test, dtype=np.float64)

            if isinstance(y_pred, (list, tuple)):
                # Process y_pred to handle inhomogeneous arrays
                processed_y_pred = []
                for item in y_pred:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        # If item is a sequence, take its first element or flatten it
                        if len(item) > 0:
                            if isinstance(item[0], (list, tuple, np.ndarray)):
                                # Nested sequence, flatten it
                                flat_item = []
                                for subitem in item:
                                    if isinstance(subitem, (list, tuple, np.ndarray)):
                                        if len(subitem) > 0:
                                            flat_item.append(float(subitem[0]))
                                        else:
                                            flat_item.append(0.0)
                                    else:
                                        flat_item.append(float(subitem))
                                processed_y_pred.append(flat_item[0] if flat_item else 0.0)
                            else:
                                processed_y_pred.append(float(item[0]))
                        else:
                            processed_y_pred.append(0.0)
                    else:
                        processed_y_pred.append(float(item))
                y_pred = np.array(processed_y_pred, dtype=np.float64)

            # Flatten arrays if they are multi-dimensional
            if isinstance(y_test, np.ndarray) and len(y_test.shape) > 1:
                y_test = y_test.flatten()
                
            # 确保y_pred不是None
            if y_pred is None:
                print("Warning: Failed to generate predictions, using zeros array")
                if isinstance(y_test, np.ndarray):
                    y_pred = np.zeros_like(y_test)
                else:
                    y_pred = np.zeros(1)
                    
            if isinstance(y_pred, np.ndarray) and len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()

            # Make sure both arrays have the same length
            if len(y_test) != len(y_pred):
                # Extend the shorter array or truncate the longer one
                if len(y_test) > len(y_pred):
                    # Extend y_pred
                    y_pred_extended = np.zeros_like(y_test)
                    y_pred_extended[:len(y_pred)] = y_pred
                    y_pred = y_pred_extended
                else:
                    # Truncate y_test
                    y_test = y_test[:len(y_pred)]

            # Reshape data to match grid dimensions
            nt = len(t_grid)
            nx = len(x_grid)

            # Check if the arrays length matches the expected grid size
            if len(y_test) != nt * nx:
                # Resize arrays to match grid dimensions
                y_test_resized = np.zeros(nt * nx)
                y_pred_resized = np.zeros(nt * nx)
                min_len = min(len(y_test), len(y_test_resized))
                y_test_resized[:min_len] = y_test[:min_len]
                y_pred_resized[:min_len] = y_pred[:min_len]
                y_test = y_test_resized
                y_pred = y_pred_resized

            # Reshape to 2D grids
            u_true = y_test.reshape(nt, nx)
            u_pred = y_pred.reshape(nt, nx)

            # Calculate error
            error = np.abs(u_true - u_pred)

            # Plot true solution
            im1 = axs[0].pcolormesh(t_grid, x_grid, u_true.T, cmap=DEFAULT_CMAP, shading='auto')
            fig.colorbar(im1, ax=axs[0], label='u(x,t)')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Space')
            axs[0].set_title('True Solution')

            # Plot predicted solution
            im2 = axs[1].pcolormesh(t_grid, x_grid, u_pred.T, cmap=DEFAULT_CMAP, shading='auto')
            fig.colorbar(im2, ax=axs[1], label='u(x,t)')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Space')
            axs[1].set_title('Predicted Solution')

            # Plot error
            im3 = axs[2].pcolormesh(t_grid, x_grid, error.T, cmap='hot', shading='auto')
            fig.colorbar(im3, ax=axs[2], label='Absolute Error')
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Space')
            axs[2].set_title('Absolute Error')

            # Plot error histogram
            axs[3].hist(error.flatten(), bins=50, color='blue', alpha=0.7)
            axs[3].set_xlabel('Absolute Error')
            axs[3].set_ylabel('Frequency')
            axs[3].set_title('Error Distribution')

            # Add equation text
            try:
                if hasattr(best_program, 'sympy_expr'):
                    equation_text = f"Discovered Equation: {best_program.sympy_expr}"
                elif hasattr(best_program, 'expression'):
                    equation_text = f"Discovered Equation: {best_program.expression}"
                elif hasattr(model, 'best_expression'):
                    equation_text = f"Discovered Equation: {model.best_expression}"
                elif hasattr(model, 'expression'):
                    equation_text = f"Discovered Equation: {model.expression}"
                else:
                    # 如果无法获取表达式，使用一个通用文本
                    equation_text = "Equation not available"
            except Exception as e:
                print(f"Warning: Could not get equation expression: {str(e)}")
                equation_text = "Equation not available"
                
            fig.text(0.5, 0.01, equation_text, ha='center', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)  # Make room for equation text

            # If output directory is provided, save the figure
            if output_dir:
                save_figure(fig, output_dir, 'kdv_solution_comparison.png')

            return fig

    except Exception as e:
        print(f"Error in plot_kdv_solution_comparison: {str(e)}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig


def plot_kdv_time_slices(model, X_test, y_test, t_grid, x_grid, time_points=None, time_indices=None, output_dir=None,
                         figsize=(12, 8)):
    """
    Plot slices of KdV equation solutions at specific time points.
    
    Parameters:
        model: Trained DeepRL model.
        X_test: Test input data.
        y_test: Test target data.
        t_grid: Time grid.
        x_grid: Space grid.
        time_points (list, optional): List of time points to plot.
        time_indices (list, optional): List of time indices to plot (alternative to time_points).
        output_dir (str, optional): Directory path to save the figure.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    try:
        # 首先确保y_test不是None
        if y_test is None:
            print("Warning: y_test is None, cannot generate time slices plot")
            # 返回一个简单的错误图形
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Error: y_test is None",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            return fig
            
        # Determine time indices to plot
        if time_indices is None:
            if time_points is None:
                # Default: plot 4 evenly spaced time points
                time_indices = [0, len(t_grid) // 3, 2 * len(t_grid) // 3, len(t_grid) - 1]
            else:
                # Find indices closest to specified time points
                time_indices = [np.abs(t_grid - t).argmin() for t in time_points]

        # Ensure time_indices is within bounds
        time_indices = [i for i in time_indices if 0 <= i < len(t_grid)]

        if not time_indices:
            print("No valid time indices to plot")
            # 返回一个简单的错误图形
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Error: No valid time indices to plot",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            return fig

        with plt.style.context(PLOT_STYLE):
            fig, axes = plt.subplots(len(time_indices), 1, figsize=figsize, sharex=True)

            # Handle case with single time index
            if len(time_indices) == 1:
                axes = [axes]

            # Get the best program
            best_program = model.searcher.best_p

            # For KdV equation, X_test is typically a tuple or can be unpacked
            # Try different approaches to extract the required parameters
            y_pred = None

            if hasattr(model, 'predict'):
                try:
                    y_pred = model.predict(X_test)
                except Exception as e:
                    print(f"Prediction failed: {str(e)}")

                    # If prediction fails, create a zero array
                    if isinstance(y_test, np.ndarray):
                        y_pred = np.zeros_like(y_test)
                    else:
                        y_pred = np.zeros(1)
            else:
                # If no predict method, create a zero array
                if isinstance(y_test, np.ndarray):
                    y_pred = np.zeros_like(y_test)
                else:
                    y_pred = np.zeros(1)

            # Ensure y_test and y_pred are numpy arrays with proper shape
            # Handle inhomogeneous arrays by processing each element individually
            if isinstance(y_test, (list, tuple)):
                # Process y_test to handle inhomogeneous arrays
                processed_y_test = []
                for item in y_test:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        # If item is a sequence, take its first element or flatten it
                        if len(item) > 0:
                            if isinstance(item[0], (list, tuple, np.ndarray)):
                                # Nested sequence, flatten it
                                flat_item = []
                                for subitem in item:
                                    if isinstance(subitem, (list, tuple, np.ndarray)):
                                        if len(subitem) > 0:
                                            flat_item.append(float(subitem[0]))
                                        else:
                                            flat_item.append(0.0)
                                    else:
                                        flat_item.append(float(subitem))
                                processed_y_test.append(flat_item[0] if flat_item else 0.0)
                            else:
                                processed_y_test.append(float(item[0]))
                        else:
                            processed_y_test.append(0.0)
                    else:
                        processed_y_test.append(float(item))
                y_test = np.array(processed_y_test, dtype=np.float64)

            if isinstance(y_pred, (list, tuple)):
                # Process y_pred to handle inhomogeneous arrays
                processed_y_pred = []
                for item in y_pred:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        # If item is a sequence, take its first element or flatten it
                        if len(item) > 0:
                            if isinstance(item[0], (list, tuple, np.ndarray)):
                                # Nested sequence, flatten it
                                flat_item = []
                                for subitem in item:
                                    if isinstance(subitem, (list, tuple, np.ndarray)):
                                        if len(subitem) > 0:
                                            flat_item.append(float(subitem[0]))
                                        else:
                                            flat_item.append(0.0)
                                    else:
                                        flat_item.append(float(subitem))
                                processed_y_pred.append(flat_item[0] if flat_item else 0.0)
                            else:
                                processed_y_pred.append(float(item[0]))
                        else:
                            processed_y_pred.append(0.0)
                    else:
                        processed_y_pred.append(float(item))
                y_pred = np.array(processed_y_pred, dtype=np.float64)

            # Flatten arrays if they are multi-dimensional
            if isinstance(y_test, np.ndarray) and len(y_test.shape) > 1:
                y_test = y_test.flatten()
            
            # 确保y_pred不是None
            if y_pred is None:
                print("Warning: Failed to generate predictions, using zeros array")
                y_pred = np.zeros_like(y_test) if isinstance(y_test, np.ndarray) else np.zeros(1)
                
            if isinstance(y_pred, np.ndarray) and len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()

            # Make sure both arrays have the same length
            if len(y_test) != len(y_pred):
                min_length = min(len(y_test), len(y_pred))
                y_test = y_test[:min_length]
                y_pred = y_pred[:min_length]

            # Reshape data to match grid dimensions
            nt = len(t_grid)
            nx = len(x_grid)

            # Check if the arrays length matches the expected grid size
            if len(y_test) != nt * nx:
                # Resize arrays to match grid dimensions
                y_test_resized = np.zeros(nt * nx)
                y_pred_resized = np.zeros(nt * nx)
                min_len = min(len(y_test), len(y_test_resized))
                y_test_resized[:min_len] = y_test[:min_len]
                y_pred_resized[:min_len] = y_pred[:min_len]
                y_test = y_test_resized
                y_pred = y_pred_resized

            # Reshape to 2D grids
            u_true = y_test.reshape(nt, nx)
            u_pred = y_pred.reshape(nt, nx)

            # Plot time slices
            for i, idx in enumerate(time_indices):
                t_val = t_grid[idx]

                # Extract true and predicted solutions at this time
                u_true_slice = u_true[idx, :]
                u_pred_slice = u_pred[idx, :]

                # Plot
                axes[i].plot(x_grid, u_true_slice, 'b-', linewidth=2, label='True')
                axes[i].plot(x_grid, u_pred_slice, 'r--', linewidth=2, label='Predicted')
                axes[i].set_ylabel('u(x,t)')
                axes[i].set_title(f'Time t = {t_val:.4f}')
                axes[i].grid(True, linestyle='--', alpha=0.5)
                axes[i].legend()

            # Set common x-axis label
            axes[-1].set_xlabel('Space (x)')

            # Add equation text
            try:
                if hasattr(best_program, 'sympy_expr'):
                    equation_text = f"Discovered Equation: {best_program.sympy_expr}"
                elif hasattr(best_program, 'expression'):
                    equation_text = f"Discovered Equation: {best_program.expression}"
                elif hasattr(model, 'best_expression'):
                    equation_text = f"Discovered Equation: {model.best_expression}"
                elif hasattr(model, 'expression'):
                    equation_text = f"Discovered Equation: {model.expression}"
                else:
                    # 如果无法获取表达式，使用一个通用文本
                    equation_text = "Equation not available"
            except Exception as e:
                print(f"Warning: Could not get equation expression: {str(e)}")
                equation_text = "Equation not available"
                
            fig.text(0.5, 0.01, equation_text, ha='center', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)  # Make room for equation text

            # If output directory is provided, save the figure
            if output_dir:
                save_figure(fig, output_dir, 'kdv_time_slices.png')

            return fig
    except Exception as e:
        print(f"Error in plot_kdv_time_slices: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig


def plot_kdv_space_time_error(model, X_test, y_test, t_grid, x_grid, output_dir=None, figsize=(10, 8)):
    """
    Plot KdV equation space-time error.
    
    Parameters:
        model: Trained DeepRL model.
        X_test: Test input data.
        y_test: Test target data.
        t_grid: Time grid.
        x_grid: Space grid.
        output_dir (str, optional): Directory path to save the figure.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    try:
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=figsize)

            # Get the best program
            best_program = model.searcher.best_p

            # For KdV equation, X_test is typically a tuple or can be unpacked
            # Try different approaches to extract the required parameters
            y_pred = None

            if hasattr(model, 'predict'):
                try:
                    y_pred = model.predict(X_test)
                except Exception as e:
                    print(f"Prediction failed: {str(e)}")

                    # If prediction fails, create a zero array
                    if isinstance(y_test, np.ndarray):
                        y_pred = np.zeros_like(y_test)
                    else:
                        y_pred = np.zeros(1)
            elif isinstance(X_test, tuple) and len(X_test) >= 3:
                # If X_test is already a tuple with u, x, ut
                u, x, ut = X_test[:3]

                # Process u to handle inhomogeneous arrays
                if isinstance(u, (list, tuple)):
                    processed_u = []
                    for item in u:
                        if isinstance(item, (list, tuple, np.ndarray)):
                            # If item is a sequence, take its first element or flatten it
                            if len(item) > 0:
                                if isinstance(item[0], (list, tuple, np.ndarray)):
                                    # Nested sequence, flatten it
                                    flat_item = []
                                    for subitem in item:
                                        if isinstance(subitem, (list, tuple, np.ndarray)):
                                            if len(subitem) > 0:
                                                flat_item.append(float(subitem[0]))
                                            else:
                                                flat_item.append(0.0)
                                        else:
                                            flat_item.append(float(subitem))
                                    processed_u.append(flat_item[0] if flat_item else 0.0)
                                else:
                                    processed_u.append(float(item[0]))
                            else:
                                processed_u.append(0.0)
                        else:
                            processed_u.append(float(item))
                    u = np.array(processed_u, dtype=np.float64)
                elif isinstance(u, np.ndarray):
                    # Ensure it's a 1D array
                    if len(u.shape) > 1:
                        u = u.flatten()
                else:
                    # Convert scalar to array
                    u = np.array([float(u)], dtype=np.float64)

                # Process x to handle inhomogeneous arrays
                if isinstance(x, (list, tuple)):
                    processed_x = []
                    for item in x:
                        if isinstance(item, (list, tuple, np.ndarray)):
                            # If item is a sequence, take its first element or flatten it
                            if len(item) > 0:
                                if isinstance(item[0], (list, tuple, np.ndarray)):
                                    # Nested sequence, flatten it
                                    flat_item = []
                                    for subitem in item:
                                        if isinstance(subitem, (list, tuple, np.ndarray)):
                                            if len(subitem) > 0:
                                                flat_item.append(float(subitem[0]))
                                            else:
                                                flat_item.append(0.0)
                                        else:
                                            flat_item.append(float(subitem))
                                    processed_x.append(flat_item[0] if flat_item else 0.0)
                                else:
                                    processed_x.append(float(item[0]))
                            else:
                                processed_x.append(0.0)
                        else:
                            processed_x.append(float(item))
                    x = np.array(processed_x, dtype=np.float64)
                elif isinstance(x, np.ndarray):
                    # Ensure it's a 1D array
                    if len(x.shape) > 1:
                        x = x.flatten()
                else:
                    # Convert scalar to array
                    x = np.array([float(x)], dtype=np.float64)

                # Process ut to handle inhomogeneous arrays
                if isinstance(ut, (list, tuple)):
                    processed_ut = []
                    for item in ut:
                        if isinstance(item, (list, tuple, np.ndarray)):
                            # If item is a sequence, take its first element or flatten it
                            if len(item) > 0:
                                if isinstance(item[0], (list, tuple, np.ndarray)):
                                    # Nested sequence, flatten it
                                    flat_item = []
                                    for subitem in item:
                                        if isinstance(subitem, (list, tuple, np.ndarray)):
                                            if len(subitem) > 0:
                                                flat_item.append(float(subitem[0]))
                                            else:
                                                flat_item.append(0.0)
                                        else:
                                            flat_item.append(float(subitem))
                                    processed_ut.append(flat_item[0] if flat_item else 0.0)
                                else:
                                    processed_ut.append(float(item[0]))
                            else:
                                processed_ut.append(0.0)
                        else:
                            processed_ut.append(float(item))
                    ut = np.array(processed_ut, dtype=np.float64)
                elif isinstance(ut, np.ndarray):
                    # Ensure it's a 1D array
                    if len(ut.shape) > 1:
                        ut = ut.flatten()
                else:
                    # Convert scalar to array
                    ut = np.array([float(ut)], dtype=np.float64)

                # Execute the program with processed arrays
                try:
                    y_pred = best_program.execute(u, x, ut)
                except Exception as e:
                    print(f"Error executing program: {str(e)}")
                    # Create dummy prediction as fallback
                    y_pred = np.zeros_like(ut)
            else:
                # Fallback: create dummy prediction
                if isinstance(y_test, np.ndarray):
                    y_pred = np.zeros_like(y_test)
                else:
                    y_pred = np.zeros(1)

            # Process y_test to handle inhomogeneous arrays
            if isinstance(y_test, (list, tuple)):
                processed_y_test = []
                for item in y_test:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        # If item is a sequence, take its first element or flatten it
                        if len(item) > 0:
                            if isinstance(item[0], (list, tuple, np.ndarray)):
                                # Nested sequence, flatten it
                                flat_item = []
                                for subitem in item:
                                    if isinstance(subitem, (list, tuple, np.ndarray)):
                                        if len(subitem) > 0:
                                            flat_item.append(float(subitem[0]))
                                        else:
                                            flat_item.append(0.0)
                                    else:
                                        flat_item.append(float(subitem))
                                processed_y_test.append(flat_item[0] if flat_item else 0.0)
                            else:
                                processed_y_test.append(float(item[0]))
                        else:
                            processed_y_test.append(0.0)
                    else:
                        processed_y_test.append(float(item))
                y_test = np.array(processed_y_test, dtype=np.float64)
            elif isinstance(y_test, np.ndarray):
                # Ensure it's a 1D array
                if len(y_test.shape) > 1:
                    y_test = y_test.flatten()
            else:
                # Convert scalar to array
                y_test = np.array([float(y_test)], dtype=np.float64)

            # Process y_pred similarly
            if isinstance(y_pred, (list, tuple)):
                processed_y_pred = []
                for item in y_pred:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        # If item is a sequence, take its first element or flatten it
                        if len(item) > 0:
                            if isinstance(item[0], (list, tuple, np.ndarray)):
                                # Nested sequence, flatten it
                                flat_item = []
                                for subitem in item:
                                    if isinstance(subitem, (list, tuple, np.ndarray)):
                                        if len(subitem) > 0:
                                            flat_item.append(float(subitem[0]))
                                        else:
                                            flat_item.append(0.0)
                                    else:
                                        flat_item.append(float(subitem))
                                processed_y_pred.append(flat_item[0] if flat_item else 0.0)
                            else:
                                processed_y_pred.append(float(item[0]))
                        else:
                            processed_y_pred.append(0.0)
                    else:
                        processed_y_pred.append(float(item))
                y_pred = np.array(processed_y_pred, dtype=np.float64)
            elif isinstance(y_pred, np.ndarray):
                # Ensure it's a 1D array
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.flatten()
            else:
                # Convert scalar to array
                if y_pred is None:
                    # 如果y_pred是None，创建一个与y_test相同形状的零数组
                    if isinstance(y_test, np.ndarray):
                        y_pred = np.zeros_like(y_test)
                    else:
                        y_pred = np.zeros(1)
                else:
                    try:
                        y_pred = np.array([float(y_pred)], dtype=np.float64)
                    except (TypeError, ValueError):
                        # 如果转换失败，创建一个零数组
                        print("Warning: Could not convert y_pred to float array, using zeros")
                        if isinstance(y_test, np.ndarray):
                            y_pred = np.zeros_like(y_test)
                        else:
                            y_pred = np.zeros(1)

            # Make sure both arrays have the same length
            if len(y_test) != len(y_pred):
                min_length = min(len(y_test), len(y_pred))
                y_test = y_test[:min_length]
                y_pred = y_pred[:min_length]

            # Calculate error
            error = np.abs(y_test - y_pred)

            # Reshape error to match grid dimensions
            nt = len(t_grid)
            nx = len(x_grid)

            # Check if the error array length matches the expected grid size
            if len(error) != nt * nx:
                # Resize error array to match grid dimensions
                error_resized = np.zeros(nt * nx)
                min_len = min(len(error), len(error_resized))
                error_resized[:min_len] = error[:min_len]
                error = error_resized

            error_grid = error.reshape(nt, nx)

            # Create a heatmap of the error
            im = ax.pcolormesh(t_grid, x_grid, error_grid.T, cmap=DEFAULT_CMAP, shading='auto')
            fig.colorbar(im, ax=ax, label='Absolute Error')

            ax.set_xlabel('Time')
            ax.set_ylabel('Space')
            ax.set_title('KdV Equation Space-Time Error')

            # If output directory is provided, save the figure
            if output_dir:
                save_figure(fig, output_dir, 'kdv_space_time_error.png')

            return fig
    except Exception as e:
        print(f"Error in plot_kdv_space_time_error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig


def plot_kdv_simulated_annealing_metrics(model, output_dir=None, figsize=(12, 12)):
    """
    Plot KdV equation simulated annealing optimization metrics.
    
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

        # Import the general simulated annealing metrics plotting function
        from kd.viz.deeprl_viz import plot_simulated_annealing_metrics

        # Call the general function with KdV-specific customizations
        fig = plot_simulated_annealing_metrics(model, output_dir=output_dir, figsize=figsize)

        # Add KdV-specific annotations if needed
        if fig is not None:
            # Add KdV-specific equation information if available
            try:
                if hasattr(model.searcher, 'best_p') and model.searcher.best_p is not None:
                    best_program = model.searcher.best_p
                    if hasattr(best_program, 'sympy_expr'):
                        equation_text = f"Discovered Equation: {best_program.sympy_expr}"
                    elif hasattr(best_program, 'expression'):
                        equation_text = f"Discovered Equation: {best_program.expression}"
                    elif hasattr(model, 'best_expression'):
                        equation_text = f"Discovered Equation: {model.best_expression}"
                    elif hasattr(model, 'expression'):
                        equation_text = f"Discovered Equation: {model.expression}"
                    else:
                        equation_text = "Equation not available"
                        
                    fig.text(0.5, 0.01, equation_text, ha='center', fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                    plt.subplots_adjust(bottom=0.1)  # Make room for equation text
            except Exception as e:
                print(f"Warning: Could not add equation text: {str(e)}")
                
        return fig
    except Exception as e:
        print(f"Error in plot_kdv_simulated_annealing_metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig


def generate_all_kdv_visualizations(model, X_test, y_test, t_grid, x_grid, output_dir=None):
    """
    Generate all KdV equation visualizations.
    
    Parameters:
        model: Trained DeepRL model.
        X_test: Test input data.
        y_test: Test target data.
        t_grid: Time grid.
        x_grid: Space grid.
        output_dir (str, optional): Directory path to save the figures.
        
    Returns:
        dict: Dictionary of generated figure objects.
    """
    try:
        # Create output directory if not exists
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Initialize dictionary to store figures
        figures = {}

        # Generate residual distribution plot
        print("Generating KdV residual distribution plot...")
        try:
            fig_residual = plot_kdv_residual_distribution(model, X_test, y_test, output_dir=output_dir)
            figures['residual_distribution'] = fig_residual
            print(" KdV residual distribution plot generated successfully")
        except Exception as e:
            print(f" Error generating KdV residual distribution plot: {str(e)}")
            import traceback
            traceback.print_exc()

        # Generate solution comparison plot
        print("Generating KdV solution comparison plot...")
        try:
            fig_solution = plot_kdv_solution_comparison(model, X_test, y_test, t_grid, x_grid, output_dir=output_dir)
            figures['solution_comparison'] = fig_solution
            print(" KdV solution comparison plot generated successfully")
        except Exception as e:
            print(f" Error generating KdV solution comparison plot: {str(e)}")
            import traceback
            traceback.print_exc()

        # Generate time slices plot
        print("Generating KdV time slices plot...")
        try:
            fig_time_slices = plot_kdv_time_slices(model, X_test, y_test, t_grid, x_grid, output_dir=output_dir)
            figures['time_slices'] = fig_time_slices
            print(" KdV time slices plot generated successfully")
        except Exception as e:
            print(f" Error generating KdV time slices plot: {str(e)}")
            import traceback
            traceback.print_exc()

        # Generate space-time error plot
        print("Generating KdV space-time error plot...")
        try:
            fig_error = plot_kdv_space_time_error(model, X_test, y_test, t_grid, x_grid, output_dir=output_dir)
            figures['space_time_error'] = fig_error
            print(" KdV space-time error plot generated successfully")
        except Exception as e:
            print(f" Error generating KdV space-time error plot: {str(e)}")
            import traceback
            traceback.print_exc()

        # Generate simulated annealing metrics plot
        print("Generating KdV simulated annealing metrics plot...")
        try:
            fig_metrics = plot_kdv_simulated_annealing_metrics(model, output_dir=output_dir)
            figures['simulated_annealing_metrics'] = fig_metrics
            print(" KdV simulated annealing metrics plot generated successfully")
        except Exception as e:
            print(f" Error generating KdV simulated annealing metrics plot: {str(e)}")
            import traceback
            traceback.print_exc()

        # Generate expression tree plot if graphviz is available
        print("Generating KdV expression tree plot...")
        try:
            # Import here to avoid import error if graphviz is not installed
            from kd.viz.deeprl_viz import plot_expression_tree

            # Get the best program
            best_program = model.searcher.best_p

            # Generate expression tree plot
            fig_tree = plot_expression_tree(best_program, output_dir=output_dir)
            figures['expression_tree'] = fig_tree
            print(" KdV expression tree plot generated successfully")
        except ImportError:
            print(" Graphviz not installed. Skipping expression tree plot.")
        except Exception as e:
            print(f" Error generating KdV expression tree plot: {str(e)}")
            import traceback
            traceback.print_exc()

        # Print summary
        successful_plots = sum(1 for fig in figures.values() if fig is not None)
        print(f"\nGenerated {successful_plots} out of 6 plots successfully.")
        if output_dir:
            print(f"Plots saved to: {output_dir}")

        return figures
    except Exception as e:
        print(f"Error in generate_all_kdv_visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def save_figure(fig, output_dir, filename):
    """
    Save figure to output directory.
    
    Parameters:
        fig: matplotlib figure object.
        output_dir (str): Directory path to save the figure.
        filename (str): Filename for the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')


def plot_all_kdv_visualizations(model, X_test, y_test, t_grid, x_grid, output_dir=None):
    """
    Plot all KdV equation visualizations and return the figure objects.
    
    Parameters:
        model: Trained DeepRL model.
        X_test: Test input data.
        y_test: Test target data.
        t_grid: Time grid.
        x_grid: Space grid.
        output_dir (str, optional): Directory path to save the figures.
        
    Returns:
        dict: Dictionary of generated figure objects.
    """
    print("绘制残差分布...")
    try:
        fig_residual = plot_kdv_residual_distribution(model, X_test, y_test, output_dir=output_dir)
    except Exception as e:
        print(f"Error in plot_kdv_residual_distribution: {str(e)}")
        fig_residual = None
    
    print("绘制解的比较...")
    try:
        fig_solution = plot_kdv_solution_comparison(model, X_test, y_test, t_grid, x_grid, output_dir=output_dir)
    except Exception as e:
        print(f"Error in plot_kdv_solution_comparison: {str(e)}")
        fig_solution = None
    
    print("绘制时间切片...")
    try:
        fig_time_slices = plot_kdv_time_slices(model, X_test, y_test, t_grid, x_grid, output_dir=output_dir)
    except Exception as e:
        print(f"Error in plot_kdv_time_slices: {str(e)}")
        fig_time_slices = None
    
    print("绘制时空误差...")
    try:
        fig_error = plot_kdv_space_time_error(model, X_test, y_test, t_grid, x_grid, output_dir=output_dir)
    except Exception as e:
        print(f"Error in plot_kdv_space_time_error: {str(e)}")
        fig_error = None
    
    print("绘制模拟退火指标...")
    try:
        fig_metrics = plot_kdv_simulated_annealing_metrics(model, output_dir=output_dir)
    except Exception as e:
        print(f"Error in plot_kdv_simulated_annealing_metrics: {str(e)}")
        fig_metrics = None
    
    # 返回所有图形对象
    return {
        'residual_distribution': fig_residual,
        'solution_comparison': fig_solution,
        'time_slices': fig_time_slices,
        'space_time_error': fig_error,
        'simulated_annealing_metrics': fig_metrics
    }
