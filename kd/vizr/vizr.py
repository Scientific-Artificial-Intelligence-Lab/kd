from collections import namedtuple
import matplotlib.pyplot as plt

# Define a namedtuple to store the configuration of a single plot
PlotConfig = namedtuple("PlotConfig", ["type", "label", "color", "style"])


class Vizr:
    def __init__(self, title="Training Metrics", xlabel="Epoch", ylabel="Value"):
        """Initialize a new Visualizer instance."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.plots = {}  # Store plot configurations and data

    def add_plot(self, label, plot_type="line", color="blue", style="-"):
        """Add a new plot series to the visualization."""
        if plot_type == "line":
            (plot,) = self.ax.plot([], [], label=label, color=color, linestyle=style)
        elif plot_type == "scatter":
            plot = self.ax.scatter([], [], label=label, color=color)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        self.plots[label] = {
            "config": PlotConfig(type=plot_type, label=label, color=color, style=style),
            "plot": plot,
            "xdata": [],
            "ydata": [],
        }
        self.ax.legend()
        return self

    def update(self, label, x, y):
        """Update data for a specific plot series."""
        if label not in self.plots:
            raise ValueError(f"Plot '{label}' is not registered. Use add_plot() first.")

        plot_data = self.plots[label]
        plot_data["xdata"].append(x)
        plot_data["ydata"].append(y)

        if plot_data["config"].type == "line":
            plot_data["plot"].set_data(plot_data["xdata"], plot_data["ydata"])
        elif plot_data["config"].type == "scatter":
            plot_data["plot"].set_offsets(
                list(zip(plot_data["xdata"], plot_data["ydata"]))
            )

        self.ax.relim()
        self.ax.autoscale_view()
        return self

    def render(self):
        """Refresh the visualization."""
        plt.pause(0.1)
        return self

    def close(self):
        """Close the visualization."""
        plt.close(self.fig)
