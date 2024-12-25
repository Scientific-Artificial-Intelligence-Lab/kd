import unittest
from vizr import Vizr
import numpy as np
import matplotlib.pyplot as plt


class TestVizr(unittest.TestCase):

    def test_line_plot(self):

        vizr = Vizr(title="Test Line Plot", xlabel="X", ylabel="Y")
        vizr.add_plot("test_line", plot_type="line", color="red", style="-")

        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        for x, y in zip(x_data, y_data):
            vizr.update("test_line", x, y)
            vizr.render()

        vizr.close()

    def test_double_line_plot(self):

        vizr = Vizr(title="Test Double Line Plot", xlabel="X", ylabel="Y")
        vizr.add_plot("test_line1", plot_type="line", color="red", style="-")
        vizr.add_plot("test_line2", plot_type="line", color="blue", style="-")

        x_data = np.linspace(0, 10, 100)
        y_data1 = np.sin(x_data)
        y_data2 = np.cos(x_data)

        for x, y1, y2 in zip(x_data, y_data1, y_data2):
            vizr.update("test_line1", x, y1)
            vizr.update("test_line2", x, y2)
            vizr.render()

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
            vizr.update("test_scatter", x, y)
            vizr.render()

        plt.show()
        vizr.close()


if __name__ == "__main__":
    unittest.main()
