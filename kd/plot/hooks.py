"""Hook system for DLGA monitoring.

这个 hook system 与现有的 vizr 系统并行工作，不影响原有功能。
"""

from typing import Dict, Any, Optional, Callable
import numpy as np
from functools import wraps
from .adapters.dlga_adapter import DLGAAdapter

def inject_hook(hook_method: str):
    """Decorator to inject hook calls into DLGA methods.
    
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
        
def attach_hook(model: Any, hook: DLGAHook) -> None:
    """Attach hook to model.
    
    Args:
        model: DLGA model instance
        hook: Hook instance to attach
    """
    model.hook = hook 