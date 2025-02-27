"""Hook system for model monitoring.

这个 hook system 与现有的 vizr 系统并行工作，不影响原有功能。
"""

from typing import Dict, Any, Optional, Callable
import numpy as np
from functools import wraps
from .adapters.dlga_adapter import DLGAAdapter
from .adapters.deeprl_adapter import DeepRLAdapter

def inject_hook(hook_method: str):
    """Decorator to inject hook calls into model methods.
    
    这个 decorator 用于在不修改原有代码的情况下注入 hook 调用。
    
    Args:
        hook_method: Name of the hook method to call
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get result from original method
            result = func(*args, **kwargs)
            
            # If model has hook, call corresponding hook method
            if len(args) > 0 and hasattr(args[0], 'hook'):
                hook_func = getattr(args[0].hook, hook_method, None)
                if hook_func is not None:
                    hook_func(*args[1:], **kwargs)
                    
            return result
        return wrapper
    return decorator

class DLGAHook:
    """Hook for monitoring DLGA model.
    
    这个 hook 通过 adapter 将 DLGA 的训练和进化过程数据转发给 monitor。
    """
    
    def __init__(self, adapter: DLGAAdapter):
        """Initialize hook.
        
        Args:
            adapter: DLGAAdapter instance for data forwarding
        """
        self.adapter = adapter
        
    def on_training_start(self, model, model_config: Optional[Dict] = None) -> None:
        """Called when training starts.
        
        Args:
            model: DLGA model instance
            model_config: Optional model configuration
        """
        if model_config is None:
            model_config = {
                'epi': getattr(model, 'epi', None),
                'input_dim': getattr(model.Net, 'input_dim', None),
                'pop_size': getattr(model, 'pop_size', None),
                'n_generations': getattr(model, 'n_generations', None)
            }
        self.adapter.notify_training_start(model_config)
        
    def on_epoch_end(self, iter: int, loss: float, loss_validate: float) -> None:
        """Called at the end of each training epoch.
        
        Args:
            iter: Current iteration number
            loss: Training loss
            loss_validate: Validation loss
        """
        metrics = {
            'loss': float(loss),
            'val_loss': float(loss_validate)
        }
        self.adapter.notify_epoch_end(iter, metrics)
        
    def on_generation_end(self, generation: int, 
                         best_chrom: list,
                         best_coef: np.ndarray, 
                         best_fitness: float,
                         best_name: str,
                         stats: Optional[Dict] = None) -> None:
        """Called at the end of each evolution generation.
        
        Args:
            generation: Current generation number
            best_chrom: Best chromosome
            best_coef: Best coefficients
            best_fitness: Best fitness value
            best_name: Equation type
            stats: Optional additional statistics
        """
        if stats is None:
            stats = {}
            
        self.adapter.notify_generation_end(
            generation=generation,
            best_chromosome=best_chrom,
            best_fitness=float(best_fitness),
            equation_str=best_name,
            stats=stats
        )

class DeepRLHook:
    """Hook for monitoring DeepRL model.
    
    这个 hook 通过 adapter 将 DeepRL 的训练过程数据转发给 monitor。
    """
    
    def __init__(self, adapter: DeepRLAdapter):
        """Initialize hook.
        
        Args:
            adapter: DeepRLAdapter instance for data forwarding
        """
        self.adapter = adapter
        
    def on_training_start(self, model, model_config: Optional[Dict] = None) -> None:
        """Called when training starts.
        
        Args:
            model: DeepRL model instance
            model_config: Optional model configuration
        """
        if model_config is None:
            model_config = {
                'n_samples_per_batch': getattr(model, 'n_samples_per_batch', None),
                'binary_operators': getattr(model, 'binary_operators', []),
                'unary_operators': getattr(model, 'unary_operators', [])
            }
        self.adapter.notify_training_start(model_config)
        
    def on_training_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Called at each training step.
        
        Args:
            step: Current training step
            metrics: Dictionary of metric values
        """
        self.adapter.notify_training_step(step, metrics)
        
    def on_episode_end(self, episode: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each episode.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metric values
        """
        self.adapter.notify_episode_end(episode, metrics)
        
    def on_state_action(self, state: np.ndarray, action: np.ndarray,
                       value: float, reward: float,
                       next_state: Optional[np.ndarray] = None,
                       info: Optional[Dict] = None) -> None:
        """Called for each state-action pair.
        
        Args:
            state: Current state
            action: Action taken
            value: Value estimate
            reward: Reward received
            next_state: Next state (optional)
            info: Additional information (optional)
        """
        self.adapter.notify_state_action(
            state=state,
            action=action,
            value=value,
            reward=reward,
            next_state=next_state,
            info=info
        )
        
def attach_hook(model: Any, hook: Any) -> None:
    """Attach hook to model.
    
    Args:
        model: Model instance
        hook: Hook instance to attach
    """
    model.hook = hook 