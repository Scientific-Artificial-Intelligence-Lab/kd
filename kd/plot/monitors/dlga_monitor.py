"""DLGA Monitor for collecting training and evolution data."""

from typing import Dict, List, Any, Optional, Union, TypedDict
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from ..config import PlotConfig

@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    epoch: int
    loss: float
    val_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class EvolutionMetrics:
    """Evolution metrics data structure."""
    generation: int
    best_fitness: float
    mean_fitness: Optional[float] = None
    diversity: Optional[float] = None
    population_size: Optional[int] = None
    elapsed_time: Optional[float] = None

@dataclass
class EquationData:
    """Equation related data structure."""
    equation_str: str
    terms: Dict[str, np.ndarray]
    coefficients: np.ndarray
    complexity: int
    fitness: float

class DLGAMonitor:
    """Monitor for collecting DLGA training and evolution data.
    
    专注于数据收集功能，不包含绘图逻辑。提供结构化的数据接口，
    支持 scientific/ 下的所有可视化组件。
    
    Attributes:
        name: Monitor name for identification
        training_history: List of training metrics
        evolution_history: List of evolution metrics
        equation_history: List of equation data
        model_config: Model configuration
    """
    
    def __init__(self, name: str = "default"):
        """Initialize the monitor.
        
        Args:
            name: Monitor name for identification
        """
        self.name = name
        self.training_history: List[TrainingMetrics] = []
        self.evolution_history: List[EvolutionMetrics] = []
        self.equation_history: List[EquationData] = []
        self.model_config: Optional[Dict] = None
        
    def on_training_start(self, model_config: Optional[Dict] = None):
        """Reset histories when training starts.
        
        Args:
            model_config: Optional model configuration to record
        """
        self.training_history = []
        self.evolution_history = []
        self.equation_history = []
        if model_config:
            self.model_config = model_config
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Update training history at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric names and values
        """
        training_metrics = TrainingMetrics(
            epoch=epoch,
            loss=metrics.pop('loss'),
            val_loss=metrics.pop('val_loss', None),
            metrics=metrics
        )
        self.training_history.append(training_metrics)
    
    def on_generation_end(self, 
                         generation: int,
                         best_chromosome: Any,
                         best_fitness: float,
                         equation_str: str,
                         terms: Optional[Dict[str, np.ndarray]] = None,
                         coefficients: Optional[np.ndarray] = None,
                         stats: Optional[Dict] = None):
        """Update evolution and equation history at generation end.
        
        Args:
            generation: Current generation number
            best_chromosome: Best chromosome in current generation
            best_fitness: Best fitness value
            equation_str: String representation of best equation
            terms: Dictionary of term values
            coefficients: Array of term coefficients
            stats: Additional statistics
        """
        # Update evolution history
        evolution_metrics = EvolutionMetrics(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=stats.get('mean_fitness') if stats else None,
            diversity=stats.get('diversity') if stats else None,
            population_size=stats.get('population_size') if stats else None,
            elapsed_time=stats.get('elapsed_time') if stats else None
        )
        self.evolution_history.append(evolution_metrics)
        
        # Update equation history if terms data is available
        if terms is not None and coefficients is not None:
            equation_data = EquationData(
                equation_str=equation_str,
                terms=terms,
                coefficients=coefficients,
                complexity=len(terms),
                fitness=best_fitness
            )
            self.equation_history.append(equation_data)
    
    def get_training_data(self, 
                         metrics: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get training data in format suitable for plotting.
        
        Args:
            metrics: Optional list of metric names to include
            
        Returns:
            Dictionary with epochs and metric values as numpy arrays
        """
        if not self.training_history:
            return {}
            
        data = {
            'epochs': np.array([m.epoch for m in self.training_history]),
            'loss': np.array([m.loss for m in self.training_history])
        }
        
        if all(m.val_loss is not None for m in self.training_history):
            data['val_loss'] = np.array([m.val_loss for m in self.training_history])
            
        if metrics:
            for metric in metrics:
                if all(metric in m.metrics for m in self.training_history):
                    data[metric] = np.array([m.metrics[metric] for m in self.training_history])
                    
        return data
    
    def get_evolution_data(self) -> Dict[str, np.ndarray]:
        """Get evolution data in format suitable for plotting.
        
        Returns:
            Dictionary with generation data as numpy arrays
        """
        if not self.evolution_history:
            return {}
            
        data = {
            'generations': np.array([int(m.generation) for m in self.evolution_history]),
            'best_fitness': np.array([float(m.best_fitness) for m in self.evolution_history])
        }
        
        # Add optional metrics if available
        optional_metrics = ['mean_fitness', 'diversity', 'population_size', 'elapsed_time']
        for metric in optional_metrics:
            values = [getattr(m, metric) for m in self.evolution_history]
            if all(v is not None for v in values):
                data[metric] = np.array(values)
                
        return data
    
    def get_equation_data(self) -> Dict[str, Any]:
        """Get equation data in format suitable for plotting.
        
        Returns:
            Dictionary with equation analysis data
        """
        if not self.equation_history:
            return {}
            
        return {
            'equations': [eq.equation_str for eq in self.equation_history],
            'terms': [eq.terms for eq in self.equation_history],
            'coefficients': [eq.coefficients for eq in self.equation_history],
            'complexity': [eq.complexity for eq in self.equation_history],
            'fitness': [eq.fitness for eq in self.equation_history]
        } 