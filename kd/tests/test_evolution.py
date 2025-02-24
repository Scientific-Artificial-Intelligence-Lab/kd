"""Test cases for evolution visualization."""

import unittest
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

from kd.plot.interface.dlga_plotter import DLGAPlotter
from kd.plot.scientific.evolution import EvolutionAnimation, EvolutionSnapshot
from moviepy.editor import VideoFileClip

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
        # 检查基础输出目录
        base_dir = Path("evolution_output")
        
        # 生成带清理的输出
        result1 = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="mp4",
            output_path=os.path.join(self.test_dir, "test1.mp4"),
            cleanup_temp=True
        )
        
        # 验证基础目录存在但任务目录已被清理
        self.assertTrue(base_dir.exists())
        task_dirs1 = list(base_dir.glob("task_*"))
        self.assertEqual(len(task_dirs1), 0)  # 所有任务目录都应该被清理
        
        # 生成不清理的输出
        result2 = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="mp4",
            output_path=os.path.join(self.test_dir, "test2.mp4"),
            cleanup_temp=False
        )
        
        # 验证任务目录保留
        task_dirs2 = list(base_dir.glob("task_*"))
        self.assertEqual(len(task_dirs2), 1)  # 应该有一个任务目录
        self.assertTrue(task_dirs2[0].name.startswith("task_"))
        
        # 验证输出文件存在
        self.assertTrue(os.path.exists(result1))
        self.assertTrue(os.path.exists(result2))
        
        # 清理测试环境
        shutil.rmtree(base_dir, ignore_errors=True)
        
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
        
        # Generate moderate dataset for testing
        large_data = generate_mock_data(n_generations=30, pop_size=50)
        
        # Time parallel processing
        import time
        start_time = time.time()
        
        self.plotter.plot_evolution(
            large_data,
            mode="snapshot",
            output_path=output_path,
            fps=30,  # 增加帧率以减少处理时间
            cleanup_temp=True,  # 确保清理临时文件
            desired_duration=5  # 设置更短的视频时长
        )
        
        processing_time = time.time() - start_time
        
        # Basic checks
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(processing_time < 120)  # 调整为更合理的时间限制
        
        # 检查视频时长
        with VideoFileClip(output_path) as clip:
            self.assertLess(clip.duration, 10)  # 视频时长应该小于10秒
        
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
            
        # Test invalid mode
        with self.assertRaises(ValueError):
            self.plotter.plot_evolution(self.mock_data, mode="invalid")
            
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