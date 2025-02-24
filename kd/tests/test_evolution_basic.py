"""Basic test cases for evolution visualization."""

import unittest
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

from kd.plot.interface.dlga_plotter import DLGAPlotter

@dataclass
class Individual:
    """Mock individual for testing."""
    equation: str
    fitness: float
    complexity: int

@dataclass
class GenerationData:
    """Mock generation data for testing."""
    generation: int
    population: List[Individual]
    best_individual: Individual
    stats: Dict[str, float]

def generate_mock_data(n_generations=10, pop_size=20):
    """Generate mock evolution data for testing."""
    data = []
    for gen in range(n_generations):
        # Generate population
        population = []
        for _ in range(pop_size):
            ind = Individual(
                equation=f"x^2 + {np.random.rand():.2f}*x",
                fitness=np.random.normal(1.0 - 0.1*gen, 0.1),  # Decreasing trend
                complexity=np.random.randint(3, 10)
            )
            population.append(ind)
            
        # Find best individual
        best_ind = min(population, key=lambda x: x.fitness)
        
        # Calculate stats
        fitness_values = [ind.fitness for ind in population]
        stats = {
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'diversity': np.random.random() * (1.0 - 0.05*gen),  # Decreasing trend
            'elapsed_time': gen * 1.5  # Increasing time
        }
        
        data.append(GenerationData(
            generation=gen,
            population=population,
            best_individual=best_ind,
            stats=stats
        ))
        
    return data

class TestEvolutionBasic(unittest.TestCase):
    """Basic test cases for evolution visualization."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_data = generate_mock_data(n_generations=5, pop_size=10)  # 使用小数据集加快测试
        self.plotter = DLGAPlotter()
        
    def test_animation_creation(self):
        """Test animation mode."""
        anim = self.plotter.plot_evolution(
            self.mock_data,
            mode="animation",
            interval=50
        )
        self.assertIsNotNone(anim)
        
    def test_invalid_mode(self):
        """Test invalid mode handling."""
        with self.assertRaises(ValueError):
            self.plotter.plot_evolution(
                self.mock_data,
                mode="invalid"
            )
            
    def test_data_validation(self):
        """Test data validation."""
        # Test invalid data
        invalid_data = None
        with self.assertRaises(ValueError):
            self.plotter.plot_evolution(invalid_data, mode="snapshot", output_path="test.mp4")
            
        # Test empty data
        empty_data = []
        with self.assertRaises(ValueError):
            self.plotter.plot_evolution(empty_data, mode="snapshot", output_path="test.mp4")
            
        # Test missing output_path in snapshot mode
        with self.assertRaises(ValueError):
            self.plotter.plot_evolution(self.mock_data, mode="snapshot")
            
        # Test invalid output format
        with self.assertRaises(ValueError):
            self.plotter.plot_evolution(
                self.mock_data, 
                mode="snapshot",
                output_path="test.invalid",
                output_format="invalid"
            )
            
    def test_figure_properties(self):
        """Test figure customization."""
        custom_figsize = (15, 12)
        custom_dpi = 150
        
        # Create animation with custom properties
        anim = self.plotter.plot_evolution(
            self.mock_data,
            mode="animation",
            figsize=custom_figsize,
            dpi=custom_dpi
        )
        self.assertIsNotNone(anim)

if __name__ == '__main__':
    unittest.main() 