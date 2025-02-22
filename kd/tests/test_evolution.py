"""Test cases for evolution visualization."""

import unittest
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

from kd.plot.interface.dlga import DLGAPlotter
from kd.plot.scientific.evolution import EvolutionAnimation, EvolutionSnapshot

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
    """Generate mock evolution data for testing.
    
    Args:
        n_generations: Number of generations
        pop_size: Population size per generation
        
    Returns:
        List of GenerationData
    """
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

class TestEvolutionVisualization(unittest.TestCase):
    """Test cases for evolution visualization."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.mock_data = generate_mock_data()
        self.plotter = DLGAPlotter()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def test_animation_creation(self):
        """Test animation mode."""
        anim = self.plotter.plot_evolution(
            self.mock_data,
            mode="animation",
            interval=50
        )
        self.assertIsNotNone(anim)
        
    def test_snapshot_mp4(self):
        """Test snapshot mode with MP4 output."""
        output_path = os.path.join(self.test_dir, "test.mp4")
        result_path = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="mp4",
            output_path=output_path,
            desired_duration=5  # 设置期望的视频时长为5秒
        )
        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.getsize(result_path) > 0)
        
        # 验证视频时长
        from moviepy.editor import VideoFileClip
        with VideoFileClip(result_path) as clip:
            # 允许0.5秒的误差
            self.assertAlmostEqual(clip.duration, 5, delta=0.5)
        
    def test_snapshot_frames(self):
        """Test frame sequence output."""
        output_dir = os.path.join(self.test_dir, "frames")
        result_path = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="frames",
            output_path=output_dir
        )
        
        # 验证输出目录存在
        self.assertTrue(os.path.isdir(result_path))
        
        # 验证帧文件
        frames = sorted(Path(result_path).glob("evolution_frame_*.png"))
        self.assertEqual(len(frames), len(self.mock_data))
        
        # 验证文件名格式
        self.assertTrue(all(f.name.startswith("evolution_frame_") for f in frames))
        
        # 验证文件内容
        self.assertTrue(all(os.path.getsize(f) > 0 for f in frames))
        
        # 验证帧序号连续性
        frame_numbers = [int(f.stem.split('_')[-1]) for f in frames]
        self.assertEqual(frame_numbers, list(range(len(self.mock_data))))
        
    def test_snapshot_gif(self):
        """Test GIF output."""
        output_path = os.path.join(self.test_dir, "evolution.gif")
        result_path = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="gif",
            output_path=output_path,
            desired_duration=5  # 设置期望的GIF时长为5秒
        )
        
        self.assertTrue(os.path.exists(result_path))
        
        # 验证是否为有效的GIF文件
        with open(result_path, 'rb') as f:
            self.assertTrue(f.read().startswith(b'GIF'))
            
        # 验证GIF帧数
        import imageio
        with imageio.get_reader(result_path) as reader:
            self.assertEqual(len(reader), len(self.mock_data))
        
    def test_temp_cleanup(self):
        """Test temporary file cleanup."""
        temp_dir = os.path.join(self.test_dir, ".evolution_temp")
        
        # Generate output with cleanup
        self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            temp_dir=temp_dir,
            cleanup_temp=True
        )
        self.assertFalse(os.path.exists(temp_dir))
        
        # Generate output without cleanup
        self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            temp_dir=temp_dir,
            cleanup_temp=False
        )
        self.assertTrue(os.path.exists(temp_dir))
        
    def test_invalid_mode(self):
        """Test invalid mode handling."""
        with self.assertRaises(ValueError):
            self.plotter.plot_evolution(
                self.mock_data,
                mode="invalid"
            )
            
    def test_parallel_processing(self):
        """Test parallel frame generation."""
        output_path = os.path.join(self.test_dir, "test_parallel.mp4")
        
        # Generate large dataset
        large_data = generate_mock_data(n_generations=50, pop_size=100)
        
        # Time parallel processing
        import time
        start_time = time.time()
        
        self.plotter.plot_evolution(
            large_data,
            mode="snapshot",
            output_path=output_path
        )
        
        processing_time = time.time() - start_time
        
        # Basic checks
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(processing_time < 60)  # Should complete within reasonable time
        
    def test_data_validation(self):
        """Test data validation."""
        # Test with empty data
        with self.assertRaises(ValueError):
            self.plotter.plot_evolution([], mode="snapshot")
            
        # Test with invalid data structure
        invalid_data = [{"invalid": "structure"}]
        with self.assertRaises(AttributeError):
            self.plotter.plot_evolution(invalid_data, mode="snapshot")
            
    def test_figure_properties(self):
        """Test figure customization."""
        custom_figsize = (15, 12)
        custom_dpi = 150
        
        # Create snapshot with custom properties
        output_path = os.path.join(self.test_dir, "test_custom.mp4")
        self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_path=output_path,
            figsize=custom_figsize,
            dpi=custom_dpi
        )
        
        # Basic existence check
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main() 