"""Real-time visualization module for scientific data.

This module provides a flexible and extensible framework for real-time data visualization.
It supports:
    - Multiple plot types (line, scatter, equation)
    - Real-time and non-real-time plotting
    - Dynamic subplot management
    - Method chaining for concise updates
    - LaTeX equation rendering

Example:
    >>> vizr = Vizr("My Plot")
    >>> vizr.add(LinePlot, "data", color="red")
    >>> for x, y in data:
    ...     vizr.update("data", x, y).render()
    >>> vizr.show()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Type, Union
from numbers import Real
import numpy as np
from sympy import latex


class VizrPlot(ABC):
    """Abstract base class for all plot types.
    
    This class defines the interface that all plot types must implement.
    Each plot type should provide methods for creation and updating of plots.
    """

    @abstractmethod
    def create(self, ax) -> Any:
        """Create a new plot on the given axes.
        
        Args:
            ax: Matplotlib axes object to create the plot on.
            
        Returns:
            Any: The created plot object (e.g., Line2D, PathCollection).
        """
        pass

    @abstractmethod
    def update(self, plot_obj: Any, xdata: Any, ydata: Any):
        """Update an existing plot with new data.
        
        Args:
            plot_obj: The plot object to update.
            xdata: New x-axis data.
            ydata: New y-axis data.
        """
        pass


@dataclass
class LinePlot(VizrPlot):
    """Line plot implementation.
    
    Creates and updates line plots with customizable appearance.
    
    Attributes:
        label (str): Label for the plot legend.
        color (str): Line color (default: "blue").
        style (str): Line style (default: "-").
        alpha (float): Line transparency (default: 1.0).
    """

    label: str
    color: str = "blue"
    style: str = "-"
    alpha: float = 1.0

    def create(self, ax):
        """Create a line plot.
        
        Args:
            ax: Matplotlib axes object.
            
        Returns:
            Line2D: The created line plot object.
        """
        kwargs = {
            "label": self.label,
            "color": self.color,
            "linestyle": self.style,
            "alpha": self.alpha,
        }
        return ax.plot([], [], **kwargs)[0]

    def update(self, plot, xdata, ydata):
        """Update line plot data.
        
        Args:
            plot: Line2D object to update.
            xdata: New x coordinates.
            ydata: New y coordinates.
        """
        plot.set_data(xdata, ydata)


@dataclass
class ScatterPlot(VizrPlot):
    """Scatter plot implementation.
    
    Creates and updates scatter plots with customizable markers.
    
    Attributes:
        label (str): Label for the plot legend.
        color (str): Marker color (default: "blue").
        marker_size (float): Size of markers (default: 6.0).
        alpha (float): Marker transparency (default: 1.0).
    """

    label: str
    color: str = "blue"
    marker_size: float = 6.0
    alpha: float = 1.0

    def create(self, ax):
        """Create a scatter plot.
        
        Args:
            ax: Matplotlib axes object.
            
        Returns:
            PathCollection: The created scatter plot object.
        """
        kwargs = {
            "label": self.label,
            "color": self.color,
            "s": self.marker_size,
            "alpha": self.alpha,
        }
        return ax.scatter([], [], **kwargs)

    def update(self, plot, xdata, ydata):
        """Update scatter plot data.
        
        Args:
            plot: PathCollection object to update.
            xdata: New x coordinates.
            ydata: New y coordinates.
        """
        plot.set_offsets(list(zip(xdata, ydata)))


@dataclass
class EquationPlot(VizrPlot):
    """LaTeX equation plot implementation.
    
    Displays mathematical equations using LaTeX rendering.
    
    Attributes:
        label (str): Label for the equation.
        fontsize (float): Font size (default: 14).
        color (str): Text color (default: "black").
        x_pos (float): Relative x position (0-1) (default: 0.5).
        y_pos (float): Relative y position (0-1) (default: 0.5).
    """

    label: str
    fontsize: float = 14
    color: str = "black"
    x_pos: float = 0.5
    y_pos: float = 0.5

    def create(self, ax):
        """Create an equation plot.
        
        Args:
            ax: Matplotlib axes object.
            
        Returns:
            Text: The created text object containing the equation.
        """
        # Clear axis elements
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Create and return the text object
        text = ax.text(
            self.x_pos,
            self.y_pos,
            "$u_t = 0.0984\\frac{\\partial^2 u}{\\partial x^2} -0.5002\\frac{\\partial (uu)}{\\partial x}$",
            fontsize=self.fontsize,
            color=self.color,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            usetex=True,
        )
        return text

    def update(self, plot_obj, xdata, ydata):
        """Update equation plot (no-op).
        
        Equation plots are static and don't need updating.
        
        Args:
            plot_obj: The text object.
            xdata: Ignored.
            ydata: Ignored.
        """
        pass


class Vizr:
    """Real-time visualization manager.
    
    This class manages multiple plots and subplots, providing an interface
    for real-time updates and dynamic subplot management.
    
    Key features:
        - Real-time and non-real-time plotting modes
        - Dynamic subplot addition and management
        - Multiple plot types support
        - Method chaining for updates
        - Auto-close functionality to prevent blocking
    
    Attributes:
        title (str): Figure title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        realtime (bool): Whether to update in real-time.
        auto_close (float): Time in seconds after which to auto-close the window (default: 3.0).
        fig: Matplotlib figure object.
        axes: Array of subplot axes.
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        _plots (list): Internal storage for plot data.
    """

    def __init__(self, title="Vizr", nrows=1, ncols=1, realtime=True, auto_close=3.0, **kwargs):
        """Initialize visualization manager.
        
        Args:
            title (str, optional): Figure title. Defaults to "Vizr".
            nrows (int, optional): Initial number of subplot rows. Defaults to 1.
            ncols (int, optional): Initial number of subplot columns. Defaults to 1.
            realtime (bool, optional): Whether to update in real-time. Defaults to True.
            auto_close (float, optional): Time in seconds after which to auto-close. Defaults to 3.0.
            **kwargs: Additional keyword arguments:
                xlabel: Label for x-axis
                ylabel: Label for y-axis
        """
        self.title = title
        self.xlabel = kwargs.get("xlabel", "")
        self.ylabel = kwargs.get("ylabel", "")
        self.realtime = realtime
        self.auto_close = auto_close
        self.create_figure(nrows, ncols)

    def create_figure(self, nrows, ncols):
        """Create or recreate the figure with new dimensions.
        
        This method handles the creation of new figures and the transfer
        of existing plots when the figure layout changes.
        
        Args:
            nrows (int): Number of subplot rows.
            ncols (int): Number of subplot columns.
        """
        if hasattr(self, "fig"):
            plt.close(self.fig)

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        self.fig.suptitle(self.title)

        if nrows * ncols == 1:
            self.axes = np.array([self.axes])
        self.axes = self.axes.flatten()

        # Preserve existing plot data
        old_plots = getattr(self, "_plots", [])
        self._plots = [{} for _ in range(nrows * ncols)]
        for i in range(min(len(old_plots), len(self._plots))):
            self._plots[i] = old_plots[i]

        self.nrows = nrows
        self.ncols = ncols

    def add_subplot(self) -> int:
        """Add a new subplot to the figure.
        
        This method:
            1. Calculates optimal layout for the new subplot
            2. Creates new figure with updated layout
            3. Transfers existing plots to the new figure
            4. Updates subplot management data structures
        
        Returns:
            int: Index of the new subplot.
        """
        current_plots = len(self._plots)
        new_subplot_id = current_plots

        # Calculate optimal layout
        if current_plots + 1 <= 2:
            new_nrows, new_ncols = 1, current_plots + 1
        elif current_plots + 1 <= 4:
            new_nrows, new_ncols = 2, 2
        else:
            new_nrows = int(np.ceil(np.sqrt(current_plots + 1)))
            new_ncols = int(np.ceil((current_plots + 1) / new_nrows))

        # Save current plot data
        old_plots_data = {}
        for subplot_idx, plots in enumerate(self._plots):
            old_plots_data[subplot_idx] = {
                "plots": plots.copy(),
            }

        # Create new figure
        plt.close(self.fig)
        self.fig, axes = plt.subplots(
            new_nrows, new_ncols, figsize=(6 * new_ncols, 4 * new_nrows)
        )
        self.fig.suptitle(self.title)

        if new_nrows * new_ncols == 1:
            axes = np.array([axes])
        self.axes = axes.flatten()

        # Initialize new plots list
        self._plots = [{} for _ in range(current_plots + 1)]

        # Restore existing plots
        for subplot_idx, data in old_plots_data.items():
            for label, plot_info in data["plots"].items():
                plot_obj = plot_info["handler"].create(self.axes[subplot_idx])
                
                self._plots[subplot_idx][label] = {
                    "handler": plot_info["handler"],
                    "plot": plot_obj,
                    "xdata": plot_info["xdata"].copy(),
                    "ydata": plot_info["ydata"].copy()
                }
                
                plot_info["handler"].update(
                    plot_obj,
                    self._plots[subplot_idx][label]["xdata"],
                    self._plots[subplot_idx][label]["ydata"]
                )

            self.axes[subplot_idx].relim()
            self.axes[subplot_idx].autoscale_view()
            self.axes[subplot_idx].legend()

        # Hide unused subplots
        for i in range(current_plots + 1, new_nrows * new_ncols):
            self.axes[i].set_visible(False)

        self.fig.tight_layout()

        return new_subplot_id

    def add(self, plot_type: Type[VizrPlot], name: str, id: Optional[int] = None, **kwargs):
        """Add a new plot to the visualization.
        
        Creates a new plot of the specified type and adds it to the visualization.
        If no subplot ID is provided, creates a new subplot automatically.
        
        Args:
            plot_type (Type[VizrPlot]): The type of plot to create (LinePlot, ScatterPlot, etc.)
            name (str): Unique identifier for the plot
            id (Optional[int], optional): Subplot ID to add plot to. If not specified, creates new subplot.
            **kwargs: Additional arguments passed to the plot constructor
                color (str): Plot color
                label (str): Legend label
                style (str): Line style (for LinePlot)
                marker_size (float): Marker size (for ScatterPlot)
                alpha (float): Transparency
            
        Returns:
            Vizr: Self for method chaining
            
        Example:
            >>> vizr = Vizr()
            >>> vizr.add(LinePlot, "data1", color="red")  # New subplot
            >>> vizr.add(ScatterPlot, "data2", id=0)  # Add to first subplot
        """
        if not isinstance(plot_type, VizrPlot):
            if isinstance(plot_type, type) and issubclass(plot_type, VizrPlot):
                plot_type = plot_type(label=name, **kwargs)
            else:
                raise TypeError(
                    f"{plot_type} is not a VizrPlot instance or subclass"
                )

        if id is None:
            id = 0
        if id >= len(self._plots):
            raise ValueError(f"Subplot index {id} out of range")
        plot_obj = plot_type.create(self.axes[id])

        if name in self._plots[id]:
            raise ValueError(
                f"Plot with label '{name}' already exists in subplot {id}"
            )
        self._plots[id][name] = {
            "handler": plot_type,
            "plot": plot_obj,
            "xdata": [],
            "ydata": [],
        }
        self.axes[id].legend()
        return id

    def update(self, label: str, x: Real, y: Real, id: int = 0) -> "Vizr":
        """Update a plot with new data.
        
        Args:
            label (str): Label of the plot to update.
            x (Real): New x-coordinate.
            y (Real): New y-coordinate.
            id (int, optional): Subplot index. Defaults to 0.
            
        Returns:
            Vizr: Self for method chaining.
            
        Raises:
            TypeError: If x or y are not real numbers.
            ValueError: If subplot ID or plot label is invalid.
        """
        if not isinstance(x, Real) or not isinstance(y, Real):
            raise TypeError("x and y must be real numbers")

        if id >= len(self._plots):
            raise ValueError(f"Subplot id {id} out of range")

        if label not in self._plots[id]:
            raise ValueError(f"Plot '{label}' not found in subplot {id}")

        plot_data = self._plots[id][label]
        plot_data["xdata"].append(x)
        plot_data["ydata"].append(y)

        plot_data["handler"].update(
            plot_data["plot"], plot_data["xdata"], plot_data["ydata"]
        )

        self.axes[id].relim()
        self.axes[id].autoscale_view()
        
        return self

    def render(self, pause_interval: float = 0.001) -> "Vizr":
        """Render the current state of all plots.
        
        Args:
            pause_interval (float, optional): Time to pause between updates.
                Defaults to 0.001.
                
        Returns:
            Vizr: Self for method chaining.
        """
        if self.realtime:
            plt.pause(pause_interval)
        return self

    def close(self):
        """Close the figure window."""
        plt.close(self.fig)

    def show(self):
        """Display all plots.
        
        For non-realtime mode, this will show the final state of all plots.
        If auto_close is set, the window will automatically close after the specified time.
        """
        if not self.realtime:
            for id in range(len(self._plots)):
                self.axes[id].relim()
                self.axes[id].autoscale_view()
        
        if self.auto_close > 0:
            import threading
            def close_after_timeout():
                import time
                time.sleep(self.auto_close)
                plt.close(self.fig)
                # Ensure the function returns immediately after closing
                plt.close('all')
            
            # Start auto-close timer in background thread
            threading.Thread(target=close_after_timeout, daemon=True).start()
        
        plt.show(block=False)
        # Ensure the function returns immediately after showing
        plt.close('all')
