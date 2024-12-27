from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union
from numbers import Real
import numpy as np


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
class DefaultPlot(VizrPlot):
    """Built-in plot types (line, scatter)."""

    type: str
    label: str
    color: str = "blue"
    style: str = "-"
    marker_size: float = 6.0
    alpha: float = 1.0

    def create(self, ax):
        kwargs = {"label": self.label, "color": self.color, "alpha": self.alpha}

        if self.type == "line":
            kwargs["linestyle"] = self.style
            return ax.plot([], [], **kwargs)[0]
        elif self.type == "scatter":
            kwargs["s"] = self.marker_size
            return ax.scatter([], [], **kwargs)

        raise ValueError(f"Unsupported plot type: {self.type}")

    def update(self, plot, xdata, ydata):
        if self.type == "line":
            plot.set_data(xdata, ydata)
        elif self.type == "scatter":
            plot.set_offsets(list(zip(xdata, ydata)))


class Vizr:
    """Visualization manager for real-time plotting."""

    def __init__(self, title="Title", nrows=1, ncols=1):
        self.title = title  # Store title for later use
        self.create_figure(nrows, ncols)

    def create_figure(self, nrows, ncols):
        """Create or recreate the figure with new dimensions"""
        # Close old figure if it exists
        if hasattr(self, "fig"):
            plt.close(self.fig)

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        self.fig.suptitle(self.title)

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

    def add_subplot(self):
        """Add a new subplot to the figure and optimize layout"""
        current_plots = len(self._plots)

        # Calculate optimal layout
        if current_plots + 1 <= 2:
            new_nrows, new_ncols = 1, current_plots + 1
        elif current_plots + 1 == 3:
            new_nrows, new_ncols = 2, 2  # 2x2 layout for 3 plots
        elif current_plots + 1 == 4:
            new_nrows, new_ncols = 2, 2
        else:
            # For more than 4 plots, use ceiling of square root
            new_nrows = int(np.ceil(np.sqrt(current_plots + 1)))
            new_ncols = int(np.ceil((current_plots + 1) / new_nrows))

        # Store current plot data
        old_plots_data = {}
        for subplot_idx, plots in enumerate(self._plots):
            old_plots_data[subplot_idx] = {}
            for label, plot_info in plots.items():
                old_plots_data[subplot_idx][label] = {
                    "xdata": plot_info["xdata"].copy(),
                    "ydata": plot_info["ydata"].copy(),
                    "handler": plot_info["handler"],
                }

        self.create_figure(new_nrows, new_ncols)

        # Restore all plots
        for subplot_idx, plots in old_plots_data.items():
            for label, plot_info in plots.items():
                self.add_plot(
                    label, subplot_idx=subplot_idx, plot_handler=plot_info["handler"]
                )
                # Restore data
                for x, y in zip(plot_info["xdata"], plot_info["ydata"]):
                    self.update(label, x, y, subplot_idx=subplot_idx)

        # Hide unused subplots
        for i in range(current_plots + 1, new_nrows * new_ncols):
            self.axes[i].set_visible(False)

        # Adjust layout to make better use of space
        self.fig.tight_layout()

        return current_plots  # Returns the index of the new subplot

    def set_subplot_labels(self, subplot_idx: int, xlabel="X", ylabel="Y", title=""):
        """Set labels for a specific subplot."""
        self.axes[subplot_idx].set_xlabel(xlabel)
        self.axes[subplot_idx].set_ylabel(ylabel)
        self.axes[subplot_idx].set_title(title)
        return self

    def add_plot(
        self,
        label: str,
        subplot_idx: int = 0,
        plot_handler: Optional[VizrPlot] = None,
        plot_type="line",
        **kwargs,
    ) -> "Vizr":
        if subplot_idx >= len(self._plots):
            raise ValueError(f"Subplot index {subplot_idx} out of range")

        if plot_handler is None:
            plot_handler = DefaultPlot(type=plot_type, label=label, **kwargs)

        plot_obj = plot_handler.create(self.axes[subplot_idx])

        self._plots[subplot_idx][label] = {
            "handler": plot_handler,
            "plot": plot_obj,
            "xdata": [],
            "ydata": [],
        }
        self.axes[subplot_idx].legend()
        return self

    def update(self, label: str, x: Real, y: Real, subplot_idx: int = 0) -> "Vizr":
        if not isinstance(x, Real) or not isinstance(y, Real):
            raise TypeError("x and y must be real numbers")

        if subplot_idx >= len(self._plots):
            raise ValueError(f"Subplot index {subplot_idx} out of range")

        if label not in self._plots[subplot_idx]:
            raise ValueError(f"Plot '{label}' not found in subplot {subplot_idx}")

        plot_data = self._plots[subplot_idx][label]
        plot_data["xdata"].append(x)
        plot_data["ydata"].append(y)

        plot_data["handler"].update(
            plot_data["plot"], plot_data["xdata"], plot_data["ydata"]
        )

        self.axes[subplot_idx].relim()
        self.axes[subplot_idx].autoscale_view()
        return self

    def render(self, pause_interval: float = 0.1) -> "Vizr":
        plt.pause(pause_interval)
        return self

    def close(self):
        plt.close(self.fig)


# class CustomVizrPlot(VizrPlot):
#     def create(self, ax):
#         # 自定义绘制逻辑
#         pass

#     def update(self, plot_obj, xdata, ydata):
#         # 自定义更新逻辑
#         pass

# vizr.add_plot("custom", plot_handler=CustomVizrPlot())
