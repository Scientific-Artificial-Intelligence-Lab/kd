"""DeepRL adapter for monitoring training process.

这个 adapter 负责连接 DeepRL 模型和 Monitor 系统，实现非侵入式的数据收集。
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..monitors.deeprl_monitor import DeepRLMonitor

class DeepRLAdapter:
    """Adapter for DeepRL model monitoring.
    
    负责:
    1. 注册和管理 monitors
    2. 转发训练事件和数据
    3. 提供非侵入式的数据收集接口
    """
    
    def __init__(self):
        """Initialize adapter."""
        self.monitors: List[DeepRLMonitor] = []
        
    def register_monitor(self, monitor: DeepRLMonitor) -> None:
        """Register a monitor.
        
        Args:
            monitor: Monitor to register
        """
        if monitor not in self.monitors:
            self.monitors.append(monitor)
            
    def notify_training_start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Notify monitors of training start.
        
        Args:
            config: Model configuration
        """
        for monitor in self.monitors:
            monitor.on_training_start(config)
            
    def notify_training_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Notify monitors of training step completion.
        
        Args:
            step: Current training step
            metrics: Dictionary of metric values
        """
        for monitor in self.monitors:
            monitor.on_training_step(step, metrics.copy())
            
    def notify_episode_end(self, episode: int, metrics: Dict[str, float]) -> None:
        """Notify monitors of episode completion.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metric values
        """
        for monitor in self.monitors:
            monitor.on_episode_end(episode, metrics.copy())
            
    def notify_state_action(self,
                          state: np.ndarray,
                          action: np.ndarray,
                          value: float,
                          reward: float,
                          next_state: Optional[np.ndarray] = None,
                          info: Optional[Dict[str, Any]] = None) -> None:
        """Notify monitors of state-action data.
        
        Args:
            state: Current state
            action: Action taken
            value: Value estimate
            reward: Reward received
            next_state: Next state (optional)
            info: Additional information (optional)
        """
        for monitor in self.monitors:
            monitor.on_state_action(
                state=state,
                action=action,
                value=value,
                reward=reward,
                next_state=next_state,
                info=info
            ) 