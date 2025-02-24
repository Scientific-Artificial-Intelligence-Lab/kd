"""Test cases for DLGA hook injection."""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from kd.model.dlga import DLGA
from kd.plot.monitors.dlga_monitor import DLGAMonitor
from kd.plot.adapters.dlga_adapter import DLGAAdapter
from kd.plot.hooks import DLGAHook, attach_hook, inject_hook

# Create a minimal test class with hook injection
class TestModel:
    def __init__(self):
        self.called = []
        
    @inject_hook('on_test')
    def test_method(self, arg1, arg2):
        self.called.append(('original', arg1, arg2))
        return 42

class TestDLGAHookInjection(unittest.TestCase):
    """Test cases for hook injection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = DLGAMonitor(name="test_monitor")
        self.adapter = DLGAAdapter()
        self.adapter.register_monitor(self.monitor)
        self.hook = DLGAHook(self.adapter)
        
    def test_basic_injection(self):
        """Test basic hook injection."""
        model = TestModel()
        
        # Create mock hook
        hook = MagicMock()
        attach_hook(model, hook)
        
        # Call method
        result = model.test_method(1, 2)
        
        # Verify original method was called
        self.assertEqual(result, 42)
        self.assertEqual(model.called, [('original', 1, 2)])
        
        # Verify hook was called
        hook.on_test.assert_called_once_with(1, 2)
        
    def test_hook_without_method(self):
        """Test hook injection when hook doesn't have the method."""
        model = TestModel()
        
        # Create hook without on_test method
        hook = MagicMock(spec=[])
        attach_hook(model, hook)
        
        # Call should not raise error
        result = model.test_method(1, 2)
        self.assertEqual(result, 42)
        
    def test_dlga_integration(self):
        """Test hook injection with actual DLGA model."""
        # Initialize model
        model = DLGA(epi=0.1, input_dim=2)
        
        # Attach hook
        attach_hook(model, self.hook)
        
        # Mock train_NN to avoid actual training
        with patch.object(model, 'train_NN') as mock_train:
            # Call fit
            X = np.random.rand(100, 2)
            y = np.random.rand(100)
            model.fit(X, y)
            
            # Verify hook received data
            self.assertTrue(len(self.monitor.get_training_history()) > 0)
            
    def test_hook_removal(self):
        """Test hook can be removed."""
        model = TestModel()
        
        # Attach and verify hook
        hook = MagicMock()
        attach_hook(model, hook)
        model.test_method(1, 2)
        self.assertEqual(hook.on_test.call_count, 1)
        
        # Remove hook
        delattr(model, '_hook')
        
        # Verify hook is not called
        model.test_method(1, 2)
        self.assertEqual(hook.on_test.call_count, 1)  # Count didn't increase
        
    def test_multiple_hooks(self):
        """Test multiple decorated methods with same hook."""
        # Create test class with multiple decorated methods
        class MultiModel:
            @inject_hook('on_method1')
            def method1(self):
                return 1
                
            @inject_hook('on_method2')
            def method2(self):
                return 2
        
        model = MultiModel()
        hook = MagicMock()
        attach_hook(model, hook)
        
        # Call methods
        model.method1()
        model.method2()
        
        # Verify both hooks were called
        hook.on_method1.assert_called_once()
        hook.on_method2.assert_called_once()
        
if __name__ == '__main__':
    unittest.main() 