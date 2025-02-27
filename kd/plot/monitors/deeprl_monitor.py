"""DeepRL Monitor for collecting training and episode data.

这个 monitor 与 DLGAMonitor 并行工作，专注于 DeepRL 模型的数据收集。
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
from dataclasses import dataclass, field
from ..config import PlotConfig

@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    step: int
    loss: float
    val_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class EpisodeMetrics:
    """Evolution metrics data structure."""
    episode: int
    reward: float
    value_loss: Optional[float] = None
    policy_loss: Optional[float] = None
    entropy: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class StateActionData:
    """State-action data structure."""
    state: np.ndarray
    action: np.ndarray
    value: float
    reward: float
    next_state: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)

class DeepRLMonitor:
    """Monitor for collecting DeepRL training and episode data.
    
    专注于数据收集功能，不包含绘图逻辑。提供结构化的数据接口，
    支持 scientific/ 下的所有可视化组件。
    
    Attributes:
        name: Monitor name for identification
        training_history: List of training metrics
        episode_history: List of episode metrics
        state_history: List of state-action data
        model_config: Model configuration
    """
    
    def __init__(self, name: str = "default"):
        """Initialize monitor.
        
        Args:
            name: Monitor name for identification
        """
        self.name = name
        self.training_history: List[TrainingMetrics] = []
        self.episode_history: List[EpisodeMetrics] = []
        self.state_history: List[StateActionData] = []
        self.model_config: Optional[Dict] = None
        
    def on_training_start(self, model_config: Optional[Dict] = None) -> None:
        """Reset histories when training starts.
        
        Args:
            model_config: Optional model configuration to record
        """
        self.training_history = []
        self.episode_history = []
        self.state_history = []
        if model_config:
            self.model_config = model_config
    
    def on_training_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Update training history at each training step.
        
        Args:
            step: Current training step
            metrics: Dictionary of metric names and values
        """
        training_metrics = TrainingMetrics(
            step=step,
            loss=metrics.pop('loss'),
            val_loss=metrics.pop('val_loss', None),
            metrics=metrics
        )
        self.training_history.append(training_metrics)
    
    def on_episode_end(self, episode: int, metrics: Dict[str, float]) -> None:
        """Update episode history at the end of each episode.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metric names and values
        """
        episode_metrics = EpisodeMetrics(
            episode=episode,
            reward=metrics.pop('reward'),
            value_loss=metrics.pop('value_loss', None),
            policy_loss=metrics.pop('policy_loss', None),
            entropy=metrics.pop('entropy', None),
            metrics=metrics
        )
        self.episode_history.append(episode_metrics)
    
    def on_state_action(self, 
                       state: np.ndarray, 
                       action: np.ndarray,
                       value: float, 
                       reward: float,
                       next_state: Optional[np.ndarray] = None,
                       info: Optional[Dict[str, Any]] = None) -> None:
        """Record state-action data.
        
        Args:
            state: Current state
            action: Action taken
            value: Value estimate
            reward: Reward received
            next_state: Next state (optional)
            info: Additional information (optional)
        """
        state_data = StateActionData(
            state=state,
            action=action,
            value=value,
            reward=reward,
            next_state=next_state,
            info=info or {}
        )
        self.state_history.append(state_data)
    
    def get_training_data(self, 
                         metrics: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get training data in format suitable for plotting.
        
        Args:
            metrics: Optional list of metric names to include
            
        Returns:
            Dictionary with training steps and metric values as numpy arrays
        """
        if not self.training_history:
            return {}
            
        data = {
            'steps': np.array([m.step for m in self.training_history]),
            'loss': np.array([m.loss for m in self.training_history])
        }
        
        # Add validation loss if available
        if all(m.val_loss is not None for m in self.training_history):
            data['val_loss'] = np.array([m.val_loss for m in self.training_history])
            
        # Add custom metrics if requested
        if metrics:
            for metric in metrics:
                if all(metric in m.metrics for m in self.training_history):
                    data[metric] = np.array([m.metrics[metric] 
                                           for m in self.training_history])
                    
        return data
    
    def get_episode_data(self) -> Dict[str, np.ndarray]:
        """Get episode data in format suitable for plotting.
        
        Returns:
            Dictionary with episode data as numpy arrays
        """
        if not self.episode_history:
            return {}
            
        data = {
            'episodes': np.array([m.episode for m in self.episode_history]),
            'rewards': np.array([m.reward for m in self.episode_history])
        }
        
        # Add core metrics if available
        for metric in ['value_loss', 'policy_loss', 'entropy']:
            values = [getattr(m, metric) for m in self.episode_history]
            if all(v is not None for v in values):
                data[metric] = np.array(values)
                
        return data
    
    def get_state_data(self) -> Dict[str, np.ndarray]:
        """Get state-action data in format suitable for plotting.
        
        Returns:
            Dictionary with state, action, value and reward arrays
        """
        if not self.state_history:
            return {}
            
        data = {
            'states': np.array([d.state for d in self.state_history]),
            'actions': np.array([d.action for d in self.state_history]),
            'values': np.array([d.value for d in self.state_history]),
            'rewards': np.array([d.reward for d in self.state_history])
        }
        
        # Add next states if available
        next_states = [d.next_state for d in self.state_history 
                      if d.next_state is not None]
        if next_states:
            data['next_states'] = np.array(next_states)
            
        return data 