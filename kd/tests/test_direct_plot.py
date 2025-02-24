"""Test cases for direct plot interface."""

import unittest
import numpy as np
import os
import shutil
from pathlib import Path

from kd.plot.interface.direct_plot import (
    plot_residual,
    plot_comparison,
    plot_equation_terms,
    plot_optimization,
    plot_evolution
)

def generate_test_data(nx=50, nt=50):
    """Generate test data for plotting."""
    # Spatial and temporal coordinates
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    
    # Generate solutions
    u_exact = np.zeros((nt, nx))
    for i in range(nt):
        u_exact[i] = np.sin(np.pi * x) * np.cos(np.pi * t[i])
    
    # Add some noise for predicted solution
    u_pred = u_exact + 0.1 * np.random.randn(nt, nx)
    
    return x, t, u_exact, u_pred

def generate_optimization_data(n_epochs=100):
    """Generate optimization process data."""
    epochs = np.arange(n_epochs)
    loss = 1.0 / (1 + 0.1 * epochs) + 0.1 * np.random.randn(n_epochs)
    weights = np.random.randn(n_epochs, 5)  # 5 weights
    diversity = np.exp(-0.02 * epochs) + 0.1 * np.random.randn(n_epochs)
    
    return epochs, loss, weights, diversity

class TestDirectPlot(unittest.TestCase):
    """Test cases for direct plot interface."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test output directory in project root
        self.test_dir = Path(os.getcwd()) / ".plot_output" / "test_output"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate test data
        self.x, self.t, self.u_exact, self.u_pred = generate_test_data()
        self.epochs, self.loss, self.weights, self.diversity = generate_optimization_data()
        
        # Generate equation terms
        self.terms = {
            'u_t': np.random.randn(50, 50),
            'u_x': np.random.randn(50, 50),
            'u_xx': np.random.randn(50, 50)
        }
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_plot_residual(self):
        """Test residual plotting functionality."""
        # Test all plot types
        for plot_type in ['heatmap', 'timeslice', 'histogram', 'all']:
            save_path = self.test_dir / f"residual_{plot_type}.png"
            plot_residual(
                self.u_pred,
                self.u_exact,
                self.x,
                self.t,
                plot_type=plot_type,
                save_path=str(save_path)
            )
            self.assertTrue(save_path.exists())
            
        # Test invalid plot type
        with self.assertRaises(ValueError):
            plot_residual(
                self.u_pred,
                self.u_exact,
                self.x,
                self.t,
                plot_type='invalid'
            )
            
        # Test shape mismatch
        with self.assertRaises(ValueError):
            plot_residual(
                self.u_pred[:-1],  # Wrong shape
                self.u_exact,
                self.x,
                self.t
            )
            
    def test_plot_comparison(self):
        """Test comparison plotting functionality."""
        y_true = self.u_exact.flatten()
        y_pred = self.u_pred.flatten()
        
        # Test all plot types
        for plot_type in ['scatter', 'line', 'all']:
            save_path = self.test_dir / f"comparison_{plot_type}.png"
            plot_comparison(
                y_true,
                y_pred,
                plot_type=plot_type,
                save_path=str(save_path)
            )
            self.assertTrue(save_path.exists())
            
    def test_plot_equation_terms(self):
        """Test equation terms plotting functionality."""
        # Test all plot types
        for plot_type in ['heatmap', 'all']:
            save_path = self.test_dir / f"equation_{plot_type}.png"
            plot_equation_terms(
                self.terms,
                self.x,
                self.t,
                plot_type=plot_type,
                save_path=str(save_path)
            )
            self.assertTrue(save_path.exists())
            
    def test_plot_optimization(self):
        """Test optimization plotting functionality."""
        # Test all plot types
        for plot_type in ['loss', 'weights', 'diversity', 'all']:
            save_path = self.test_dir / f"optimization_{plot_type}.png"
            plot_optimization(
                epochs=self.epochs,
                loss_history=self.loss,
                weights_history=self.weights,
                diversity_history=self.diversity,
                plot_type=plot_type,
                save_path=str(save_path)
            )
            self.assertTrue(save_path.exists())
            
    def test_plot_evolution(self):
        """Test evolution plotting functionality."""
        # Generate mock evolution data
        from kd.tests.test_evolution_basic import generate_mock_data
        evolution_data = generate_mock_data(n_generations=5)
        
        # Test animation mode
        anim = plot_evolution(
            evolution_data,
            mode='animation',
            interval=50
        )
        self.assertIsNotNone(anim)
        
        # Test snapshot mode with different formats
        for fmt in ['mp4', 'gif', 'frames']:
            if fmt == 'frames':
                save_path = self.test_dir / "evolution_frames"
            else:
                save_path = self.test_dir / f"evolution.{fmt}"
                
            output = plot_evolution(
                evolution_data,
                mode='snapshot',
                output_format=fmt,
                save_path=str(save_path),
                desired_duration=1  # Short duration for testing
            )
            self.assertTrue(Path(output).exists())
            
if __name__ == '__main__':
    unittest.main() 