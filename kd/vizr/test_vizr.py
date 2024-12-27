import unittest

import numpy as np
import matplotlib.pyplot as plt

from kd.vizr import Vizr


class TestVizr(unittest.TestCase):

    def test_line_plot(self):
        vizr = Vizr("single line plot")
        vizr.add_plot("test_line", plot_type="line", color="cyan", style="-")

        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        for x, y in zip(x_data, y_data):
            vizr.update("test_line", x, y).render(0.01)

        vizr.close()

    def test_double_line_plot(self):

        vizr = Vizr(title="Test Double Line Plot", xlabel="X", ylabel="Y")
        vizr.add_plot("test_line1", plot_type="line", color="red", style="-")
        vizr.add_plot("test_line2", plot_type="line", color="blue", style="-")

        x_data = np.linspace(0, 10, 100)
        y_data1 = np.sin(x_data)
        y_data2 = np.cos(x_data)

        for x, y1, y2 in zip(x_data, y_data1, y_data2):
            vizr.update("test_line1", x, y1).update("test_line2", x, y2).render()

        vizr.close()

    def test_scatter_plot(self):

        vizr = Vizr(title="Test Scatter Plot", xlabel="X", ylabel="Y")
        vizr.add_plot("test_scatter", plot_type="scatter", color="blue")
        x_data = np.random.normal(0, 2, 10)
        y_data = np.random.normal(0, 2, 10)

        # TODO: 坐标范围也需要动态更新 不然越界了就不会显示的 和楼上那个自动变大不一样
        vizr.ax.set_xlim(-10, 10)
        vizr.ax.set_ylim(-10, 10)
        for x, y in zip(x_data, y_data):
            vizr.update("test_scatter", x, y).render()

        plt.show()
        vizr.close()

    def test_multi_subplot(self):
        vizr = Vizr("Multi Subplot", nrows=2, ncols=2)

        vizr.set_subplot_labels(0, title="Sin")
        vizr.set_subplot_labels(1, title="Cos")
        vizr.set_subplot_labels(2, title="Square")
        vizr.set_subplot_labels(3, title="Cubic")

        vizr.add_plot("sin_line", subplot_idx=0, plot_type="line", color="red")
        vizr.add_plot("cos_line", subplot_idx=1, plot_type="line", color="blue")
        vizr.add_plot("square_line", subplot_idx=2, plot_type="line", color="green")
        vizr.add_plot("cubic_line", subplot_idx=3, plot_type="line", color="purple")

        x_data = np.linspace(0, 10, 100)
        for x in x_data:
            vizr.update("sin_line", x, np.sin(x), subplot_idx=0).update(
                "cos_line", x, np.cos(x), subplot_idx=1
            ).update("square_line", x, x**2, subplot_idx=2).update(
                "cubic_line", x, x**3, subplot_idx=3
            ).render(
                0.01
            )

        plt.show()
        vizr.close()

    def test_dynamic_add_plot(self):
        vizr = Vizr(title="Dynamic Adding Plots")
        vizr.add_plot("line1", plot_type="line", color="red")
        vizr.add_plot("line2", plot_type="line", color="blue")

        x_data = np.linspace(0, 10, 100)

        # First plot 50 points with just 2 lines
        for i in range(50):
            x = x_data[i]
            vizr.update("line1", x, np.sin(x)).update("line2", x, np.cos(x)).render(
                0.01
            )

        # Add a third line halfway through
        vizr.add_plot("line3", plot_type="line", color="green")

        # Continue plotting with all 3 lines
        for i in range(50, 100):
            x = x_data[i]
            vizr.update("line1", x, np.sin(x)).update("line2", x, np.cos(x)).update(
                "line3", x, np.sin(2 * x)
            ).render(0.01)

        plt.show()
        vizr.close()

    def test_dynamic_add_subplot(self):
        vizr = Vizr(title="Dynamic Adding Subplots")

        vizr.add_plot("line1", subplot_idx=0, plot_type="line", color="red")
        x_data = np.linspace(0, 10, 100)
        for i in range(50):
            x = x_data[i]
            vizr.update("line1", x, np.sin(x), subplot_idx=0).render(0.01)

        # Add another subplot
        vizr.add_subplot()
        vizr.add_plot("line2", subplot_idx=1, plot_type="line", color="blue")
        for i in range(50, 100):
            x = x_data[i]
            vizr.update("line1", x, np.sin(x), subplot_idx=0).update(
                "line2", x, np.cos(x), subplot_idx=1
            ).render(0.01)

        plt.show()
        vizr.close()


if __name__ == "__main__":
    suite = unittest.TestSuite()

    # suite.addTest(TestVizr("test_line_plot"))
    # suite.addTest(TestVizr("test_double_line_plot"))
    # suite.addTest(TestVizr("test_scatter_plot"))
    # suite.addTest(TestVizr("test_multi_subplot"))
    # suite.addTest(TestVizr("test_dynamic_add_plot"))
    suite.addTest(TestVizr("test_dynamic_add_subplot"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
