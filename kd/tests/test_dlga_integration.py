"""Integration tests for DLGA visualization system."""

import unittest
import numpy as np
import torch
import os
from pathlib import Path

from kd.model.dlga import DLGA
from kd.plot.monitors.dlga_monitor import DLGAMonitor
from kd.plot.adapters.dlga_adapter import DLGAAdapter
from kd.plot.hooks import DLGAHook

class TestDLGAIntegration(unittest.TestCase):
    """Integration tests for DLGA visualization system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test data
        self.X = np.random.rand(100, 2)  # 100 samples, 2 features (x,t)
        self.y = np.sin(self.X[:, 0]) + np.cos(self.X[:, 1])  # Simple test function
        
        # Initialize DLGA model
        self.model = DLGA(epi=0.1, input_dim=2)
        
        # Initialize monitoring system
        self.monitor = DLGAMonitor(name="test_monitor")
        self.adapter = DLGAAdapter()
        self.adapter.register_monitor(self.monitor)
        
        # Initialize hook
        self.hook = DLGAHook(self.adapter)
        
    def test_hook_training_start(self):
        """Test hook notification at training start."""
        model_config = {
            'epi': 0.1,
            'input_dim': 2,
            'pop_size': 50
        }
        self.hook.on_training_start(self.model, model_config)
        
        # Verify initial state was captured
        self.assertEqual(len(self.monitor.get_training_history()), 0)
        self.assertEqual(len(self.monitor.get_evolution_history()), 0)
        self.assertEqual(self.monitor.model_config, model_config)
        
    def test_hook_epoch_end(self):
        """Test hook notification at epoch end."""
        # Simulate training progress
        for epoch in range(3):
            metrics = {
                'loss': 0.5 - 0.1 * epoch,
                'val_loss': 0.6 - 0.1 * epoch
            }
            self.hook.on_epoch_end(epoch, metrics)
        
        # Verify training history
        history = self.monitor.get_training_history()
        self.assertEqual(len(history), 3)
        self.assertLess(history[-1]['loss'], history[0]['loss'])
        
    def test_hook_generation_end(self):
        """Test hook notification at generation end."""
        # Simulate evolution progress
        for gen in range(3):
            best_chrom = [[0, 1], [2, 3]]  # Mock chromosome
            best_coef = np.array([0.1, -0.2])
            best_fitness = 0.8 - 0.1 * gen
            best_name = "u_t"
            
            stats = {
                'mean_fitness': best_fitness + 0.1,
                'diversity': 0.5 - 0.1 * gen
            }
            
            self.hook.on_generation_end(
                gen, best_chrom, best_coef, best_fitness, best_name, stats
            )
        
        # Verify evolution history
        history = self.monitor.get_evolution_history()
        self.assertEqual(len(history), 3)
        self.assertLess(history[-1]['best_fitness'], history[0]['best_fitness'])
        self.assertIn('diversity', history[0])
        
    def test_full_integration(self):
        """Test full integration with minimal DLGA run."""
        # Initialize monitoring
        model_config = {'epi': 0.1, 'input_dim': 2}
        self.hook.on_training_start(self.model, model_config)
        
        # Simulate short training run
        for epoch in range(2):
            metrics = {'loss': 0.5, 'val_loss': 0.6}
            self.hook.on_epoch_end(epoch, metrics)
            
        # Simulate short evolution run
        for gen in range(2):
            stats = {'mean_fitness': 0.7, 'diversity': 0.5}
            self.hook.on_generation_end(
                gen,
                best_chrom=[[0, 1]],
                best_coef=np.array([0.1]),
                best_fitness=0.8,
                best_name="u_t",
                stats=stats
            )
            
        # Verify history
        self.assertEqual(len(self.monitor.get_training_history()), 2)
        self.assertEqual(len(self.monitor.get_evolution_history()), 2)
        
if __name__ == '__main__':
    unittest.main() 