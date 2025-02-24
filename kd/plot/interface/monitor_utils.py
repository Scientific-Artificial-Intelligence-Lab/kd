"""Helper utilities for setting up monitoring.

提供简单的接口来设置和管理监控功能。
"""

from typing import Optional
from ..monitors.dlga_monitor import DLGAMonitor
from ..adapters.dlga_adapter import DLGAAdapter
from ..hooks import DLGAHook, attach_hook, inject_hook

def _inject_dlga_hooks(model):
    """Inject hooks into DLGA model methods."""
    # Inject hooks into training methods
    model.train_NN = inject_hook('on_epoch_end')(model.train_NN)
    model.evolution = inject_hook('on_generation_end')(model.evolution)
    model.fit = inject_hook('on_training_start')(model.fit)

def enable_monitoring(model, monitor_name: str = "default") -> DLGAMonitor:
    """Enable monitoring for a model.
    
    这个函数会创建并配置所需的所有组件(Monitor, Adapter, Hook)，
    并将它们正确连接。使用方式:
    
    ```python
    model = DLGA(...)
    monitor = enable_monitoring(model)
    model.fit(X, y)  # Monitor will automatically collect data
    ```
    
    Args:
        model: Model instance to monitor
        monitor_name: Name for the monitor
        
    Returns:
        Configured monitor instance
    """
    # Create components
    monitor = DLGAMonitor(name=monitor_name)
    adapter = DLGAAdapter()
    adapter.register_monitor(monitor)
    hook = DLGAHook(adapter)
    
    # Inject hooks into model methods
    _inject_dlga_hooks(model)
    
    # Attach hook to model
    attach_hook(model, hook)
    
    return monitor

def disable_monitoring(model) -> None:
    """Disable monitoring for a model.
    
    Args:
        model: Model instance to disable monitoring for
    """
    if hasattr(model, '_hook'):
        delattr(model, '_hook') 