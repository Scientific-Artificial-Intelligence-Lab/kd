"""DLGA Adapter for connecting model with monitoring system."""

from typing import List, Dict, Any
from ..monitors.dlga_monitor import DLGAMonitor

class DLGAAdapter:
    """Adapter for connecting DLGA model with monitoring system."""
    
    def __init__(self):
        """Initialize the adapter."""
        self.monitors: List[DLGAMonitor] = []
        
    def register_monitor(self, monitor: DLGAMonitor) -> None:
        """Register a monitor with this adapter.
        
        Args:
            monitor: Monitor instance to register
        """
        if monitor not in self.monitors:
            self.monitors.append(monitor)
            
    def notify_training_start(self, initial_state: Dict[str, Any]) -> None:
        """Notify all monitors that training has started.
        
        Args:
            initial_state: Initial model state
        """
        for monitor in self.monitors:
            monitor.on_training_start()
            
    def notify_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Notify all monitors that an epoch has ended.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
        """
        for monitor in self.monitors:
            monitor.on_epoch_end(epoch, metrics)
            
    def notify_generation_end(self, generation: int, best_chromosome: Any,
                            best_fitness: float, equation_str: str,
                            stats: Dict[str, Any] = None) -> None:
        """Notify all monitors that a generation has ended.
        
        Args:
            generation: Current generation number
            best_chromosome: Best chromosome in current generation
            best_fitness: Fitness of best chromosome
            equation_str: String representation of best equation
            stats: Optional additional statistics
        """
        for monitor in self.monitors:
            monitor.on_generation_end(generation, best_chromosome, best_fitness, equation_str, stats) 