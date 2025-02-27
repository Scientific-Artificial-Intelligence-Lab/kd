import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
from pathlib import Path

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.plot.scientific.metadata_plane import MetadataValuePlane

class TestMetadataValuePlane(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create output directory
        self.output_dir = Path(os.getcwd()) / ".plot_output" / "metadata_plane_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {self.output_dir}")
        
        # Create a simple test dataset
        self.x = np.linspace(-5, 5, 50)
        self.y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(self.x, self.y)
        
        # Create a test metadata value matrix
        # Using a simple 2D function: z = sin(x^2 + y^2) / (x^2 + y^2 + 1)
        R = np.sqrt(X**2 + Y**2)
        self.Z = np.sin(R) / (R + 1)
        
        # Create a test optimization trajectory
        t = np.linspace(0, 2*np.pi, 20)
        self.trajectory = np.column_stack([3*np.cos(t), 3*np.sin(t)])
        
        # Create 3D trajectory
        z_traj = np.sin(np.sqrt(self.trajectory[:, 0]**2 + self.trajectory[:, 1]**2)) / \
                (np.sqrt(self.trajectory[:, 0]**2 + self.trajectory[:, 1]**2) + 1)
        self.trajectory_3d = np.column_stack([self.trajectory, z_traj])
        
        # Initialize plotter
        self.plotter = MetadataValuePlane()
    
    def test_input_validation(self):
        """Test input validation"""
        # Test non-numpy array input
        with self.assertRaises(TypeError):
            self.plotter._validate_inputs([1,2,3], self.y, self.Z)
            
        # Test NaN value warning
        z_with_nan = self.Z.copy()
        z_with_nan[0,0] = np.nan
        with warnings.catch_warnings(record=True) as w:
            self.plotter._validate_inputs(self.x, self.y, z_with_nan)
            self.assertEqual(len(w), 1)
            self.assertIn("NaN", str(w[0].message))
    
    def test_contour_plot(self):
        """Test contour plot generation"""
        fig = self.plotter.plot(
            x_values=self.x,
            y_values=self.y,
            z_values=self.Z,
            plot_type='contour',
            title='Test Contour Plot'
        )
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        # Save plot
        save_path = self.output_dir / "test_contour.png"
        fig.savefig(save_path)
        print(f"Contour plot saved to: {save_path}")
        plt.close(fig)
    
    def test_heatmap_plot(self):
        """Test heatmap plot generation"""
        fig = self.plotter.plot(
            x_values=self.x,
            y_values=self.y,
            z_values=self.Z,
            plot_type='heatmap',
            title='Test Heatmap'
        )
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        # Save plot
        save_path = self.output_dir / "test_heatmap.png"
        fig.savefig(save_path)
        print(f"Heatmap saved to: {save_path}")
        plt.close(fig)
    
    def test_surface_plot(self):
        """Test 3D surface plot generation"""
        fig = self.plotter.plot(
            x_values=self.x,
            y_values=self.y,
            z_values=self.Z,
            plot_type='surface',
            title='Test 3D Surface Plot'
        )
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        # Save plot
        save_path = self.output_dir / "test_surface.png"
        fig.savefig(save_path)
        print(f"3D surface plot saved to: {save_path}")
        plt.close(fig)
    
    def test_invalid_plot_type(self):
        """Test invalid plot type handling"""
        with self.assertRaises(ValueError):
            self.plotter.plot(
                x_values=self.x,
                y_values=self.y,
                z_values=self.Z,
                plot_type='invalid_type'
            )
    
    def test_add_trajectory(self):
        """Test trajectory addition to plot"""
        fig = self.plotter.plot(
            x_values=self.x,
            y_values=self.y,
            z_values=self.Z,
            plot_type='contour'
        )
        
        fig = self.plotter.add_trajectory(
            trajectory=self.trajectory,
            label='Optimization Path'
        )
        
        self.assertIsNotNone(fig)
        
        # Save plot
        save_path = self.output_dir / "test_trajectory.png"
        fig.savefig(save_path)
        print(f"Plot with trajectory saved to: {save_path}")
        plt.close(fig)
    
    def test_3d_trajectory(self):
        """Test 3D trajectory addition to plot"""
        # Create base 3D plot
        fig = self.plotter.plot(
            x_values=self.x,
            y_values=self.y,
            z_values=self.Z,
            plot_type='surface'
        )
        
        # Add 3D trajectory
        fig = self.plotter.add_trajectory(
            trajectory=self.trajectory_3d,
            label='3D Optimization Path'
        )
        
        self.assertIsNotNone(fig)
        
        # Save plot
        save_path = self.output_dir / "test_3d_trajectory.png"
        fig.savefig(save_path)
        print(f"3D plot with trajectory saved to: {save_path}")
        plt.close(fig)
    
    def test_real_world_example(self):
        """Test real-world example - Hyperparameter optimization"""
        learning_rates = np.logspace(-4, -1, 20)
        batch_sizes = np.linspace(32, 256, 20)
        X, Y = np.meshgrid(learning_rates, batch_sizes)
        
        # Simulate model performance
        Z = -((np.log10(X) + 2.5)**2 + (Y - 128)**2 / 10000)
        
        fig = self.plotter.plot(
            x_values=learning_rates,
            y_values=batch_sizes,
            z_values=Z,
            x_label='Learning Rate',
            y_label='Batch Size',
            z_label='Validation Accuracy',
            plot_type='contour',
            title='Hyperparameter Optimization Landscape'
        )
        
        # Simulate optimization trajectory
        trajectory = np.array([
            [1e-3, 64],
            [5e-4, 96],
            [1e-4, 128],
            [5e-4, 160]
        ])
        
        self.plotter.add_trajectory(
            trajectory=trajectory,
            label='Optimization Path'
        )
        
        self.assertIsNotNone(fig)
        
        # Save plot
        save_path = self.output_dir / "test_hyperparameter.png"
        fig.savefig(save_path)
        print(f"Hyperparameter optimization plot saved to: {save_path}")
        plt.close(fig)

if __name__ == '__main__':
    unittest.main() 