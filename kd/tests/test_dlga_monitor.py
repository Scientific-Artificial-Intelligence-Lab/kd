"""Test cases for DLGA monitoring functionality."""

import unittest
import numpy as np

from kd.plot.monitors.dlga_monitor import DLGAMonitor
from kd.plot.adapters.dlga_adapter import DLGAAdapter

class TestDLGAMonitor(unittest.TestCase):
    """Test cases for DLGA monitoring system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = DLGAMonitor()
        self.adapter = DLGAAdapter()
        
    def test_monitor_creation(self):
        """Test basic monitor creation."""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(len(self.monitor.training_history), 0)
        
    def test_adapter_creation(self):
        """Test basic adapter creation."""
        self.assertIsNotNone(self.adapter)
        
    def test_monitor_registration(self):
        """Test monitor registration with adapter."""
        self.adapter.register_monitor(self.monitor)
        self.assertIn(self.monitor, self.adapter.monitors)
        
    def test_training_data_collection(self):
        """Test collection of training data."""
        # Setup
        self.adapter.register_monitor(self.monitor)
        initial_state = {'model_type': 'DLGA', 'input_dim': 10}
        
        # Start training
        self.adapter.notify_training_start(initial_state)
        self.assertEqual(len(self.monitor.training_history), 0)
        
        # Collect epoch data
        metrics = {'loss': 0.5, 'val_loss': 0.6}
        self.adapter.notify_epoch_end(1, metrics)
        
        # Verify data
        self.assertEqual(len(self.monitor.training_history), 1)
        entry = self.monitor.training_history[0]
        self.assertEqual(entry['epoch'], 1)
        self.assertEqual(entry['loss'], 0.5)
        self.assertEqual(entry['val_loss'], 0.6)
        
    def test_evolution_data_collection(self):
        """Test collection of evolution data."""
        # Setup
        self.adapter.register_monitor(self.monitor)
        
        # Collect generation data
        population_data = {
            'best_fitness': 0.8,
            'avg_fitness': 0.6,
            'best_equation': 'x + sin(x)'
        }
        self.adapter.notify_generation_end(1, population_data)
        
        # Verify data
        self.assertEqual(len(self.monitor.evolution_history), 1)
        entry = self.monitor.evolution_history[0]
        self.assertEqual(entry['generation'], 1)
        self.assertEqual(entry['best_fitness'], 0.8)
        self.assertEqual(entry['avg_fitness'], 0.6)
        self.assertEqual(entry['best_equation'], 'x + sin(x)')
        
    def test_multiple_monitors(self):
        """Test multiple monitors with one adapter."""
        monitor2 = DLGAMonitor()
        self.adapter.register_monitor(self.monitor)
        self.adapter.register_monitor(monitor2)
        
        # Notify both monitors
        metrics = {'loss': 0.5}
        self.adapter.notify_epoch_end(1, metrics)
        
        # Verify both monitors received data
        self.assertEqual(len(self.monitor.training_history), 1)
        self.assertEqual(len(monitor2.training_history), 1)
        
if __name__ == '__main__':
    unittest.main() 