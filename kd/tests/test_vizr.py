import unittest

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from kd.vizr.vizr import *


class TestVizr(unittest.TestCase):

    def test_line_plot(self):
        vizr = Vizr("single line plot", realtime=False)
        vizr.add(LinePlot, "test_line")

        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        for x, y in zip(x_data, y_data):
            vizr.update("test_line", x, y)
            vizr.render()

        vizr.show()
        vizr.close()

    def test_double_line_plot(self):

        vizr = Vizr(title="Test Double Line Plot", realtime=False)
        vizr.add(LinePlot, "test_line1", color="red", style="-")
        vizr.add(LinePlot, "test_line2", color="blue", style="-")

        x_data = np.linspace(0, 10, 100)
        y_data1 = np.sin(x_data)
        y_data2 = np.cos(x_data)

        for x, y1, y2 in zip(x_data, y_data1, y_data2):
            vizr.update("test_line1", x, y1).update("test_line2", x, y2).render()

        vizr.show()
        vizr.close()

    def test_scatter_plot(self):

        vizr = Vizr(title="Test Scatter Plot", xlabel="X", ylabel="Y", realtime=True)
        vizr.add(ScatterPlot, "test_scatter")

        x_data = np.random.normal(0, 2, 10)
        y_data = np.random.normal(0, 2, 10)
        vizr.axes[0].set_xlim(-10, 10)
        vizr.axes[0].set_ylim(-10, 10)
        for x, y in zip(x_data, y_data):
            vizr.update("test_scatter", x, y).render()  

        vizr.show()
        vizr.close()

    def test_multi_subplot(self):
        vizr = Vizr("Multi Subplot", nrows=2, ncols=2)

        vizr.add(LinePlot, "sin_line", id=0, color="red")
        vizr.add(LinePlot, "cos_line", id=1, color="blue")
        vizr.add(LinePlot, "square_line", id=2, color="green")
        vizr.add(LinePlot, "cubic_line", id=3, color="purple")

        x_data = np.linspace(0, 10, 100)
        for x in x_data:
            vizr.update("sin_line", x, np.sin(x), id=0).update(
                "cos_line", x, np.cos(x), id=1).update(
                "square_line", x, x**2, id=2).update(
                "cubic_line", x, x**3, id=3).render()

        vizr.show()
        vizr.close()

    def test_dynamic_add_plot(self):
        vizr = Vizr(title="Dynamic Adding Plots")
        vid = vizr.add(LinePlot, "line1", color="red")
        vizr.add(LinePlot, "line2", vid, color="blue")

        x_data = np.linspace(0, 10, 100)

        for i in range(50):
            x = x_data[i]
            vizr.update("line1", x, np.sin(x)).update("line2", x, np.cos(x)).render()

        vizr.add(LinePlot, "line3", color="green")

        for i in range(50, 100):
            x = x_data[i]
            vizr.update("line1", x, np.sin(x)).update("line2", x, np.cos(x)).update("line3", x, np.sin(2 * x)).render()

        vizr.show()

    def test_dynamic_add_subplot(self):
        vizr = Vizr(title="Dynamic Adding Subplots", realtime=False)

        vizr.add(LinePlot, "line1", color="red")
        x_data = np.linspace(0, 10, 100)
        for i in range(50):
            x = i
            vizr.update("line1", x, np.sin(x), id=0).render()

        # Add another subplot
        new_id = vizr.add_subplot()
        vizr.add(LinePlot, "line2", id=new_id, color="blue")

        for i in range(50, 100):
            x = i - 50
            vizr.update("line1", 50 + x, np.sin(x), id=0).update("line2", x, np.cos(x), id=1).render()

        vizr.show()

    def test_equation_plot(self):
        """Test equation plot visualization."""
        vizr = Vizr("Test Equation Plot")
        vizr.add(EquationPlot, "test_equation", fontsize=16)
        plt.show()


if __name__ == "__main__":
    suite = unittest.TestSuite()
    # suite.addTest(TestVizr("test_line_plot"))
    # suite.addTest(TestVizr("test_double_line_plot"))
    # suite.addTest(TestVizr("test_scatter_plot"))
    # suite.addTest(TestVizr("test_dynamic_add_plot"))
    suite.addTest(TestVizr("test_dynamic_add_subplot"))
    # suite.addTest(TestVizr("test_equation_plot"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
