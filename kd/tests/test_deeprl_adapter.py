"""Test cases for DeepRL adapter."""

import unittest
import numpy as np
from unittest.mock import MagicMock

from kd.plot.monitors.deeprl_monitor import DeepRLMonitor
from kd.plot.adapters.deeprl_adapter import DeepRLAdapter

class TestDeepRLAdapter(unittest.TestCase):
    """Test cases for DeepRL adapter."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = DeepRLMonitor(name="test_monitor")
        self.adapter = DeepRLAdapter()
        
    def test_adapter_creation(self):
        """Test basic adapter creation."""
        self.assertIsNotNone(self.adapter)
        self.assertEqual(len(self.adapter.monitors), 0)
        
    def test_monitor_registration(self):
        """Test monitor registration."""
        self.adapter.register_monitor(self.monitor)
        self.assertIn(self.monitor, self.adapter.monitors)
        
        # Register same monitor twice should not duplicate
        self.adapter.register_monitor(self.monitor)
        self.assertEqual(len(self.adapter.monitors), 1)
        
    def test_training_start_notification(self):
        """Test training start notification."""
        self.adapter.register_monitor(self.monitor)
        
        config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'hidden_size': 64
        }
        
        self.adapter.notify_training_start(config)
        self.assertEqual(self.monitor.model_config, config)
        
    def test_training_step_notification(self):
        """Test training step notification."""
        self.adapter.register_monitor(self.monitor)
        
        # Notify multiple training steps
        expected_losses = []
        for step in range(3):
            loss = 0.5 - 0.1 * step
            metrics = {
                'loss': loss,
                'val_loss': 0.6 - 0.1 * step,
                'policy_entropy': 0.1
            }
            expected_losses.append(loss)
            self.adapter.notify_training_step(step, metrics)
            
        # Verify training history
        history = self.monitor.get_training_data()
        self.assertEqual(len(history['steps']), 3)
        np.testing.assert_array_almost_equal(
            history['loss'],
            np.array(expected_losses)
        )
        
    def test_episode_end_notification(self):
        """Test episode end notification."""
        self.adapter.register_monitor(self.monitor)
        
        # Notify multiple episode ends
        for episode in range(3):
            metrics = {
                'reward': 10 + episode,
                'value_loss': 0.3 - 0.1 * episode,
                'policy_loss': 0.2,
                'entropy': 0.1
            }
            self.adapter.notify_episode_end(episode, metrics)
            
        # Verify episode history
        history = self.monitor.get_episode_data()
        self.assertEqual(len(history['episodes']), 3)
        self.assertTrue(np.all(history['rewards'] >= 10))
        self.assertTrue('value_loss' in history)
        
    def test_state_action_notification(self):
        """Test state-action data collection."""
        self.adapter.register_monitor(self.monitor)
        
        state = np.array([1.0, 0.0])
        action = np.array([0.5])
        next_state = np.array([1.1, 0.1])
        
        self.adapter.notify_state_action(
            state=state,
            action=action,
            value=0.8,
            reward=1.0,
            next_state=next_state,
            info={'done': False}
        )
        
        # Verify state history
        history = self.monitor.get_state_data()
        self.assertEqual(len(history['states']), 1)
        np.testing.assert_array_equal(history['states'][0], state)
        np.testing.assert_array_equal(history['actions'][0], action)
        self.assertEqual(history['values'][0], 0.8)
        
if __name__ == '__main__':
    unittest.main() 