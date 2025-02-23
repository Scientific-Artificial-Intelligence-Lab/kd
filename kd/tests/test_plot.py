"""Test cases for plotting functionality."""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import tempfile

from kd.plot.interface.dlga import DLGAPlotter
from kd.plot.scientific.residual import (
    ResidualHeatmap,
    ResidualTimeslice,
    ResidualHistogram,
    ResidualAnalysis
)
from kd.plot.scientific.comparison import (
    ComparisonScatter,
    ComparisonLine,
    ComparisonAnalysis
)
from kd.plot.scientific.equation import (
    TermsHeatmap,
    TermsAnalysis
)
from kd.plot.scientific.optimization import (
    LossPlot,
    WeightsPlot,
    DiversityPlot,
    OptimizationAnalysis
)


def generate_sample_data(nx=100, nt=100):
    """Generate sample PDE data for testing."""
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    
    # Generate exact solution (simple wave)
    X, T = np.meshgrid(x, t)
    u_exact = np.sin(np.pi * X) * np.cos(2 * np.pi * T)
    
    # Generate predicted solution (with some noise)
    u_pred = u_exact + 0.1 * np.random.randn(nt, nx)
    
    return x, t, u_exact, u_pred


def generate_noisy_data(n_samples=1000, noise_level=0.1):
    """Generate sample data with controlled noise for comparison plots."""
    # Generate true values
    x_true = np.linspace(-2, 2, n_samples)
    y_true = x_true ** 2  # Example: y = x^2
    
    # Add noise
    x_noisy = x_true + noise_level * np.random.randn(n_samples)
    y_noisy = y_true + noise_level * np.random.randn(n_samples)
    
    return x_true, y_true, x_noisy, y_noisy


def generate_equation_terms(nx=50, nt=50):
    """Generate sample equation terms data for testing."""
    x = np.linspace(-5, 5, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    # Generate some example terms (e.g., for Burgers equation)
    u = np.sin(np.pi * X) * np.exp(-T)  # Base solution
    
    # Generate terms
    terms = {
        'u_t': -np.pi * np.sin(np.pi * X) * np.exp(-T),  # Time derivative
        'u_x': np.pi * np.cos(np.pi * X) * np.exp(-T),    # Space derivative
        'u·u_x': u * (np.pi * np.cos(np.pi * X) * np.exp(-T)),  # Nonlinear term
        'u_xx': -np.pi**2 * np.sin(np.pi * X) * np.exp(-T)  # Second derivative
    }
    
    return x, t, terms


def generate_optimization_data(n_epochs=100, n_terms=4):
    """Generate sample optimization process data for testing."""
    epochs = np.arange(n_epochs)
    
    # Generate loss history (decreasing with noise)
    loss_history = 1.0 / (1 + 0.1 * epochs) + 0.1 * np.random.randn(n_epochs)
    
    # Generate weights history (converging with noise)
    weights_history = []
    for i in range(n_terms):
        target_weight = np.random.uniform(-1, 1)
        weights = target_weight + np.exp(-0.05 * epochs) * np.random.randn(n_epochs)
        weights_history.append(weights)
    weights_history = np.array(weights_history)
    
    # Generate diversity history (decreasing with noise)
    diversity_history = 0.5 * np.exp(-0.02 * epochs) + 0.1 * np.random.randn(n_epochs)
    diversity_history = np.abs(diversity_history)  # Ensure positive
    
    # Generate best individual history
    best_individual_history = {
        'fitness': loss_history,
        'weights': weights_history,
        'complexity': 10 + np.random.randint(-2, 3, n_epochs)
    }
    
    return {
        'epochs': epochs,
        'loss_history': loss_history,
        'weights_history': weights_history,
        'diversity_history': diversity_history,
        'best_individual_history': best_individual_history
    }


class TestResidualPlots(unittest.TestCase):
    """Test cases for residual plotting."""
    
    def setUp(self):
        """Set up test data."""
        self.x, self.t, self.u_exact, self.u_pred = generate_sample_data()
        plt.close('all')  # Close any existing plots
        
    def test_residual_heatmap(self):
        """Test residual heatmap plot."""
        plotter = ResidualHeatmap()
        plotter.plot(self.u_pred, self.u_exact, self.x, self.t)
        plotter.save("test_heatmap.png")
        plt.close()
        
    def test_residual_timeslice(self):
        """Test residual time slice plot."""
        plotter = ResidualTimeslice()
        plotter.plot(self.u_pred, self.u_exact, self.x, self.t)
        plotter.save("test_timeslice.png")
        plt.close()
        
    def test_residual_histogram(self):
        """Test residual histogram plot."""
        plotter = ResidualHistogram()
        plotter.plot(self.u_pred, self.u_exact)
        plotter.save("test_histogram.png")
        plt.close()
        
    def test_residual_analysis(self):
        """Test comprehensive residual analysis."""
        plotter = ResidualAnalysis()
        plotter.plot(self.u_pred, self.u_exact, self.x, self.t)
        plotter.save("test_analysis.png")
        plt.close()
        
    def test_dlga_interface(self):
        """Test DLGA plotting interface."""
        plotter = DLGAPlotter()
        
        # Test each plot type
        for plot_type in ['heatmap', 'timeslice', 'histogram', 'all']:
            plotter.plot_residual(
                self.u_pred,
                self.u_exact,
                self.x,
                self.t,
                plot_type=plot_type,
                save_path=f"test_dlga_{plot_type}.png"
            )
            plt.close()
            
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')


class TestPlotStorage(unittest.TestCase):
    """Test cases for plot storage functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
        
    def test_default_storage_path(self):
        """Test default storage path creation and usage."""
        plotter = ResidualHeatmap()
        self.assertTrue(os.path.exists(plotter.default_save_dir))
        self.assertTrue(os.path.isdir(plotter.default_save_dir))
        
    def test_custom_storage_path(self):
        """Test custom storage path configuration."""
        custom_path = os.path.join(self.test_dir, "custom_plots")
        plotter = ResidualHeatmap(save_dir=custom_path)
        self.assertEqual(plotter.save_dir, custom_path)
        self.assertTrue(os.path.exists(custom_path))
        
    def test_file_saving(self):
        """Test file saving with configured paths."""
        # Test with default path
        plotter1 = ResidualHeatmap()
        filename1 = "test_plot1.png"
        plotter1.save_path = os.path.join(plotter1.save_dir, filename1)
        
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 10)
        u_pred = np.random.rand(10, 10)
        u_exact = np.random.rand(10, 10)
        
        plotter1.plot(u_pred, u_exact, x, t)
        plotter1.save()
        self.assertTrue(os.path.exists(plotter1.save_path))
        
        # Test with custom path
        custom_path = os.path.join(self.test_dir, "custom_plots")
        plotter2 = ResidualHeatmap(save_dir=custom_path)
        filename2 = "test_plot2.png"
        plotter2.save_path = os.path.join(custom_path, filename2)
        
        plotter2.plot(u_pred, u_exact, x, t)
        plotter2.save()
        self.assertTrue(os.path.exists(plotter2.save_path))


class TestComparisonPlots(unittest.TestCase):
    """Test cases for comparison plotting."""
    
    def setUp(self):
        """Set up test data."""
        self.x_true, self.y_true, self.x_noisy, self.y_noisy = generate_noisy_data()
        plt.close('all')
    
    def test_comparison_scatter(self):
        """Test scatter plot with noise."""
        plotter = ComparisonScatter(save_path="test_scatter.png")
        plotter.plot(
            self.y_true, 
            self.y_noisy,
            error_bars=True,
            confidence_interval=0.95
        )
        self.assertTrue(os.path.exists(plotter.save_path))
        plt.close()
    
    def test_comparison_line(self):
        """Test 45-degree line plot."""
        plotter = ComparisonLine(save_path="test_line.png")
        plotter.plot(
            self.y_true,
            self.y_noisy,
            show_bounds=True,
            show_statistics=True
        )
        self.assertTrue(os.path.exists(plotter.save_path))
        plt.close()
    
    def test_comparison_analysis(self):
        """Test comprehensive comparison analysis."""
        plotter = ComparisonAnalysis(save_path="test_comparison.png")
        plotter.plot(
            self.y_true,
            self.y_noisy,
            show_error_dist=True,
            show_statistics=True
        )
        self.assertTrue(os.path.exists(plotter.save_path))
        plt.close()
    
    def test_dlga_comparison_interface(self):
        """Test DLGA comparison plotting interface."""
        plotter = DLGAPlotter()
        plotter.plot_comparison(
            self.y_true,
            self.y_noisy,
            plot_type='all',
            save_path="test_dlga_comparison.png"
        )
        plt.close()


class TestEquationPlots(unittest.TestCase):
    """Test cases for equation term plotting."""
    
    def setUp(self):
        """Set up test data."""
        self.x, self.t, self.terms = generate_equation_terms()
        plt.close('all')
    
    def test_terms_heatmap(self):
        """Test equation terms heatmap plot."""
        plotter = TermsHeatmap(save_path="test_terms_heatmap.png")
        plotter.plot(
            self.terms,
            self.x,
            self.t,
            show_colorbar=True,
            normalize=True
        )
        self.assertTrue(os.path.exists(plotter.save_path))
        plt.close()
    
    def test_terms_analysis(self):
        """Test comprehensive terms analysis."""
        plotter = TermsAnalysis(save_path="test_terms_analysis.png")
        plotter.plot(
            self.terms,
            self.x,
            self.t,
            show_correlations=True,
            show_statistics=True
        )
        self.assertTrue(os.path.exists(plotter.save_path))
        plt.close()
    
    def test_dlga_terms_interface(self):
        """Test DLGA terms plotting interface."""
        plotter = DLGAPlotter()
        plotter.plot_terms(
            self.terms,
            self.x,
            self.t,
            plot_type='all',
            save_path="test_dlga_terms.png"
        )
        plt.close()


class TestOptimizationPlots(unittest.TestCase):
    """Test cases for optimization process plotting."""
    
    def setUp(self):
        """Set up test data."""
        self.opt_data = generate_optimization_data()
        plt.close('all')
    
    def test_loss_plot(self):
        """Test loss history plot."""
        plotter = LossPlot(save_path="test_loss.png")
        plotter.plot(
            self.opt_data['epochs'],
            self.opt_data['loss_history'],
            show_trend=True,
            show_best=True
        )
        self.assertTrue(os.path.exists(plotter.save_path))
        plt.close()
    
    def test_weights_plot(self):
        """Test weights evolution plot."""
        plotter = WeightsPlot(save_path="test_weights.png")
        plotter.plot(
            self.opt_data['epochs'],
            self.opt_data['weights_history'],
            show_confidence=True
        )
        self.assertTrue(os.path.exists(plotter.save_path))
        plt.close()
    
    def test_diversity_plot(self):
        """Test population diversity plot."""
        plotter = DiversityPlot(save_path="test_diversity.png")
        plotter.plot(
            self.opt_data['epochs'],
            self.opt_data['diversity_history'],
            show_statistics=True
        )
        self.assertTrue(os.path.exists(plotter.save_path))
        plt.close()
    
    def test_optimization_analysis(self):
        """Test comprehensive optimization analysis."""
        plotter = OptimizationAnalysis(save_path="test_optimization.png")
        plotter.plot(
            self.opt_data['epochs'],
            loss_history=self.opt_data['loss_history'],
            weights_history=self.opt_data['weights_history'],
            diversity_history=self.opt_data['diversity_history'],
            best_individual_history=self.opt_data['best_individual_history']
        )
        self.assertTrue(os.path.exists(plotter.save_path))
        plt.close()
    
    def test_dlga_optimization_interface(self):
        """Test DLGA optimization plotting interface."""
        plotter = DLGAPlotter()
        plotter.plot_optimization(
            self.opt_data['epochs'],
            loss_history=self.opt_data['loss_history'],
            weights_history=self.opt_data['weights_history'],
            diversity_history=self.opt_data['diversity_history'],
            best_individual_history=self.opt_data['best_individual_history'],
            plot_type='all',
            save_path="test_dlga_optimization.png"
        )
        plt.close()


if __name__ == '__main__':
    unittest.main() 