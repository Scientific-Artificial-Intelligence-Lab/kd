"""Test cases for DeepRL hook injection."""

import unittest
import numpy as np
from unittest.mock import MagicMock

from kd.plot.monitors.deeprl_monitor import DeepRLMonitor
from kd.plot.adapters.deeprl_adapter import DeepRLAdapter
from kd.plot.hooks import DeepRLHook, attach_hook

class TestDeepRLHook(unittest.TestCase):
    """Basic test cases for DeepRL hook."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = DeepRLMonitor(name="test_monitor")
        self.adapter = DeepRLAdapter()
        self.adapter.register_monitor(self.monitor)
        self.hook = DeepRLHook(self.adapter)
        
    def test_hook_creation(self):
        """Test basic hook creation."""
        self.assertIsNotNone(self.hook)
        self.assertEqual(self.hook.adapter, self.adapter)
        
    def test_hook_attachment(self):
        """Test hook attachment to mock model."""
        model = MagicMock()
        attach_hook(model, self.hook)
        self.assertTrue(hasattr(model, 'hook'))
        self.assertEqual(model.hook, self.hook)
        
    def test_training_start_hook(self):
        """Test training start hook with mock model."""
        model = MagicMock()
        attach_hook(model, self.hook)
        
        config = {
            'n_samples_per_batch': 500,
            'binary_operators': ['add', 'mul'],
            'unary_operators': ['n2']
        }
        
        # Simulate training start
        self.hook.on_training_start(model, config)
        
        # Verify monitor received config
        self.assertEqual(self.monitor.model_config, config)
        
if __name__ == '__main__':
    unittest.main() 