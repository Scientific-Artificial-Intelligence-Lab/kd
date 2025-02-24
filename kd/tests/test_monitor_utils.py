"""Test cases for monitoring utility functions."""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from kd.model.dlga import DLGA
from kd.plot.interface.monitor_utils import enable_monitoring, disable_monitoring

class TestMonitorUtils(unittest.TestCase):
    """Test cases for monitoring utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = DLGA(epi=0.1, input_dim=2)
        
    def test_enable_monitoring(self):
        """Test enabling monitoring."""
        # Enable monitoring
        monitor = enable_monitoring(self.model, "test_monitor")
        
        # Verify hook was attached
        self.assertTrue(hasattr(self.model, '_hook'))
        
        # Verify monitor name
        self.assertEqual(monitor.name, "test_monitor")
        
        # Verify monitor is collecting data
        with patch.object(self.model, 'train_NN'):
            X = np.random.rand(100, 2)
            y = np.random.rand(100)
            self.model.fit(X, y)
            
            # Check if data was collected
            self.assertTrue(len(monitor.get_training_history()) > 0)
            
    def test_disable_monitoring(self):
        """Test disabling monitoring."""
        # Enable then disable monitoring
        monitor = enable_monitoring(self.model)
        disable_monitoring(self.model)
        
        # Verify hook was removed
        self.assertFalse(hasattr(self.model, '_hook'))
        
        # Verify model still works
        with patch.object(self.model, 'train_NN'):
            X = np.random.rand(100, 2)
            y = np.random.rand(100)
            self.model.fit(X, y)  # Should not raise error
            
    def test_multiple_monitors(self):
        """Test enabling multiple monitors."""
        # Enable two monitors
        monitor1 = enable_monitoring(self.model, "monitor1")
        monitor2 = enable_monitoring(self.model, "monitor2")
        
        # Verify only the last monitor is active
        self.assertEqual(self.model._hook.adapter.monitors[-1].name, "monitor2")
        
        # Both monitors should receive data
        with patch.object(self.model, 'train_NN'):
            X = np.random.rand(100, 2)
            y = np.random.rand(100)
            self.model.fit(X, y)
            
            self.assertTrue(len(monitor1.get_training_history()) > 0)
            self.assertTrue(len(monitor2.get_training_history()) > 0)
            
    def test_enable_after_disable(self):
        """Test re-enabling monitoring after disable."""
        # Enable, disable, then re-enable
        monitor1 = enable_monitoring(self.model)
        disable_monitoring(self.model)
        monitor2 = enable_monitoring(self.model)
        
        # Verify new monitor works
        with patch.object(self.model, 'train_NN'):
            X = np.random.rand(100, 2)
            y = np.random.rand(100)
            self.model.fit(X, y)
            
            self.assertTrue(len(monitor2.get_training_history()) > 0)
            
    def test_disable_without_monitor(self):
        """Test disabling when no monitor is attached."""
        # Should not raise error
        disable_monitoring(self.model)
        
if __name__ == '__main__':
    unittest.main() 