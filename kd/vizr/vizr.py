from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Type, Union
from numbers import Real
import numpy as np
from sympy import latex


class VizrPlot(ABC):
    """Base class for all plot types in Vizr."""

    @abstractmethod
    def create(self, ax) -> Any:
        """Create the plot on given axes."""
        pass

    @abstractmethod
    def update(self, plot_obj: Any, xdata: Any, ydata: Any):
        """Update plot with new data."""
        pass


@dataclass
class LinePlot(VizrPlot):
    """Line plot implementation."""

    label: str
    color: str = "blue"
    style: str = "-"
    alpha: float = 1.0

    def create(self, ax):
        kwargs = {
            "label": self.label,
            "color": self.color,
            "linestyle": self.style,
            "alpha": self.alpha,
        }
        return ax.plot([], [], **kwargs)[0]  # Return the Line2D object

    def update(self, plot, xdata, ydata):
        plot.set_data(xdata, ydata)  # Update data for the Line2D object


@dataclass
class ScatterPlot(VizrPlot):
    """Scatter plot implementation."""

    label: str
    color: str = "blue"
    marker_size: float = 6.0
    alpha: float = 1.0

    def create(self, ax):
        kwargs = {
            "label": self.label,
            "color": self.color,
            "s": self.marker_size,
            "alpha": self.alpha,
        }
        return ax.scatter([], [], **kwargs)  # Return the PathCollection object

    def update(self, plot, xdata, ydata):
        plot.set_offsets(
            list(zip(xdata, ydata))
        )  # Update data for the PathCollection object


@dataclass
class EquationPlot(VizrPlot):
    """Plot for displaying mathematical equations using LaTeX."""

    label: str
    fontsize: float = 14
    color: str = "black"
    x_pos: float = 0.5
    y_pos: float = 0.5

    def create(self, ax):
        """Create text object for equation."""
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
        """Placeholder update method."""
        pass


class Vizr:
    """Visualization manager for real-time plotting."""

    def __init__(self, title="Vizr", nrows=1, ncols=1, realtime=True, **kwargs):
        self.title = title
        self.xlabel = kwargs.get("xlabel", "")
        self.ylabel = kwargs.get("ylabel", "")
        self.realtime = realtime
        self.create_figure(nrows, ncols)

    def create_figure(self, nrows, ncols):
        """Create or recreate the figure with new dimensions"""
        # Close old figure if it exists
        if hasattr(self, "fig"):
            plt.close(self.fig)

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        self.fig.suptitle(self.title)

        # self.axes.set_xlabel(self.xlabel)
        # self.axes.set_ylabel(self.ylabel)

        # Convert axes to array for consistent indexing
        if nrows * ncols == 1:
            self.axes = np.array([self.axes])
        self.axes = self.axes.flatten()

        # Preserve existing plot data if any
        old_plots = getattr(self, "_plots", [])
        self._plots = [{} for _ in range(nrows * ncols)]
        for i in range(min(len(old_plots), len(self._plots))):
            self._plots[i] = old_plots[i]

        self.nrows = nrows
        self.ncols = ncols

    def add_subplot(self) -> int:
        """Add a new subplot to the figure"""
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

        # 保存当前所有的 plot data
        old_plots_data = {}
        for subplot_idx, plots in enumerate(self._plots):
            old_plots_data[subplot_idx] = {
                "plots": plots.copy(),  # 深拷贝 plot data
                # 不保存axis limits，让坐标轴自动调整
            }

        # Close old figure and create new one
        plt.close(self.fig)
        self.fig, axes = plt.subplots(
            new_nrows, new_ncols, figsize=(6 * new_ncols, 4 * new_nrows)
        )
        self.fig.suptitle(self.title)

        # Convert axes to array for consistent indexing
        if new_nrows * new_ncols == 1:
            axes = np.array([axes])
        self.axes = axes.flatten()

        # Initialize new plots list with correct size
        self._plots = [{} for _ in range(current_plots + 1)]

        # 恢复所有已存在的 plots
        for subplot_idx, data in old_plots_data.items():
            for label, plot_info in data["plots"].items():
                # 重新创建 plot object
                plot_obj = plot_info["handler"].create(self.axes[subplot_idx])
                
                # 保持原有数据结构
                self._plots[subplot_idx][label] = {
                    "handler": plot_info["handler"],
                    "plot": plot_obj,
                    "xdata": plot_info["xdata"].copy(),  # 复制数据
                    "ydata": plot_info["ydata"].copy()   # 复制数据
                }
                
                # 更新 plot 数据
                plot_info["handler"].update(
                    plot_obj,
                    self._plots[subplot_idx][label]["xdata"],
                    self._plots[subplot_idx][label]["ydata"]
                )

            # 重新计算坐标轴范围
            self.axes[subplot_idx].relim()
            self.axes[subplot_idx].autoscale_view()
            self.axes[subplot_idx].legend()

        # Hide unused subplots
        for i in range(current_plots + 1, new_nrows * new_ncols):
            self.axes[i].set_visible(False)

        # Adjust layout
        self.fig.tight_layout()

        return new_subplot_id

    # FIXME: id 的做法有待商榷 需要设计一个自增然后存到 self
    def add(
        self,
        plot_handler: Type[VizrPlot],
        label: str,
        id: Optional[int] = None,
        **kwargs,
    ) -> int:

        if not isinstance(plot_handler, VizrPlot):
            # 如果是 class，则实例化
            if isinstance(plot_handler, type) and issubclass(plot_handler, VizrPlot):
                plot_handler = plot_handler(label=label, **kwargs)
            else:
                raise TypeError(
                    f"{plot_handler} is not a VizrPlot instance or subclass"
                )

        # plot_handler 现在已经是实例，直接使用
        if id is None:
            id = 0
        if id >= len(self._plots):
            raise ValueError(f"Subplot index {id} out of range")
        plot_obj = plot_handler.create(self.axes[id])

        if label in self._plots[id]:
            raise ValueError(
                f"Plot with label '{label}' already exists in subplot {id}"
            )
        self._plots[id][label] = {
            "handler": plot_handler,
            "plot": plot_obj,
            "xdata": [],
            "ydata": [],
        }
        self.axes[id].legend()
        return id

    def update(self, label: str, x: Real, y: Real, id: int = 0) -> "Vizr":

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

    # TODO: batch update

    def render(self, pause_interval: float = 0.001) -> "Vizr":
        if self.realtime:
            plt.pause(pause_interval)
        return self

    def close(self):
        plt.close(self.fig)

    def show(self):
        """Display all plots (for non-realtime mode)"""
        if not self.realtime:
            for id in range(len(self._plots)):
                self.axes[id].relim()
                self.axes[id].autoscale_view()
        
        plt.show()
