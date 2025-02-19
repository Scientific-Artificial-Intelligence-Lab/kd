"""Test PDE comparison visualization."""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from kd.vizr.pde_comparison import plot_pde_comparison


def generate_sample_data(nx=100, nt=100):
    """Generate sample PDE data for testing."""
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    # Generate exact solution
    u_exact = np.zeros((nt, nx))
    for i in range(nt):
        for j in range(nx):
            if x[j] > 0:
                u_exact[i,j] = np.sin(np.pi * x[j]) * (1 - t[i])
            else:
                u_exact[i,j] = -np.sin(np.pi * x[j]) * (1 - t[i])
    
    # Generate predicted solution (with some noise)
    u_pred = u_exact + 0.01 * np.random.randn(nt, nx)
    
    # Generate training points
    n_train = 100
    # More points at boundaries
    x_boundary = np.random.choice([-1, 1], size=n_train//3)
    t_boundary = np.random.uniform(0, 1, size=n_train//3)
    
    # Random points in interior
    x_interior = np.random.uniform(-1, 1, size=n_train-n_train//3)
    t_interior = np.random.uniform(0, 1, size=n_train-n_train//3)
    
    # Combine boundary and interior points
    x_train = np.concatenate([x_boundary, x_interior])
    t_train = np.concatenate([t_boundary, t_interior])
    X_train = np.column_stack((x_train, t_train))
    
    return x, t, X, T, u_exact, u_pred, X_train


class TestPDEComparison(unittest.TestCase):
    """Test cases for PDE comparison visualization."""
    
    def test_plot_pde_comparison(self):
        """Test PDE comparison plot."""
        # Generate sample data
        x, t, X, T, u_exact, u_pred, X_train = generate_sample_data()
        
        # Create visualization
        fig = plot_pde_comparison(x, t, u_exact, u_pred, X_train)
        
        # Save the figure
        plt.savefig('test_pde_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Basic checks
        self.assertTrue(isinstance(fig, plt.Figure))


if __name__ == '__main__':
    unittest.main() 