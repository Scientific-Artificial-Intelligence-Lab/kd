from kd.plot.interface.dlga_plotter import DLGAPlotter
from kd.tests.test_evolution_basic import generate_mock_data, Individual, GenerationData
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import os
import unittest
import shutil
from pathlib import Path
import imageio.v2 as imageio
from moviepy.editor import VideoFileClip
from PIL import Image

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

def generate_mock_data(n_generations=30, pop_size=20):
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

class TestEvolutionOutput(unittest.TestCase):
    """Test cases for evolution visualization output quality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = "evolution_output/test_cases"
        os.makedirs(self.test_dir, exist_ok=True)
        self.mock_data = generate_mock_data(n_generations=5, pop_size=10)  # 使用小数据集加快测试
        self.plotter = DLGAPlotter()
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_gif_quality(self):
        """Test GIF output quality."""
        gif_path = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="gif",
            output_path=f"{self.test_dir}/evolution.gif",
            desired_duration=2,
            dpi=150  # 测试高质量输出
        )
        
        # 验证文件存在和大小
        self.assertTrue(os.path.exists(gif_path))
        self.assertGreater(os.path.getsize(gif_path), 0)
        
        # 验证GIF质量
        with Image.open(gif_path) as img:
            self.assertEqual(img.format, 'GIF')
            self.assertGreaterEqual(img.size[0], 800)  # 检查分辨率
            
    def test_frames_quality(self):
        """Test frame sequence quality."""
        frames_path = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="frames",
            output_path=f"{self.test_dir}/frames/",
            dpi=150  # 测试高质量输出
        )
        
        # 验证目录存在
        self.assertTrue(os.path.isdir(frames_path))
        
        # 验证帧质量
        frames = sorted(Path(frames_path).glob("evolution_frame_*.png"))
        self.assertTrue(frames)  # 确保有帧文件
        
        # 检查第一帧的质量
        with Image.open(frames[0]) as img:
            self.assertEqual(img.format, 'PNG')
            self.assertGreaterEqual(img.size[0], 800)  # 检查分辨率
            
    def test_mp4_quality(self):
        """Test MP4 output quality."""
        output_path = os.path.join(self.test_dir, "evolution.mp4")
        result_path = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="mp4",
            output_path=output_path,
            desired_duration=2,
            dpi=150  # 测试高质量输出
        )
        
        # 验证文件存在
        self.assertTrue(os.path.exists(result_path))
        
        # 验证视频质量
        with VideoFileClip(result_path) as clip:
            frame = clip.get_frame(0)  # 获取第一帧
            self.assertGreaterEqual(frame.shape[1], 800)  # 检查分辨率
            
    def test_output_consistency(self):
        """Test consistency across different output formats."""
        # 生成所有格式的输出
        outputs = {}
        for fmt in ['mp4', 'gif', 'frames']:
            if fmt == 'frames':
                path = f"{self.test_dir}/{fmt}/"
            else:
                path = f"{self.test_dir}/evolution.{fmt}"
                
            result = self.plotter.plot_evolution(
                self.mock_data,
                mode="snapshot",
                output_format=fmt,
                output_path=path,
                desired_duration=2,
                dpi=150
            )
            outputs[fmt] = result
            
        # 验证所有输出都存在
        for path in outputs.values():
            self.assertTrue(os.path.exists(path))
            
        # 验证帧数一致性
        n_frames = len(self.mock_data)
        
        # 检查frames
        frames = sorted(Path(outputs['frames']).glob("evolution_frame_*.png"))
        self.assertEqual(len(frames), n_frames)
        
        # 检查GIF
        with imageio.get_reader(outputs['gif']) as reader:
            self.assertEqual(len(reader), n_frames)
            
        # 检查视频时长与帧数关系
        with VideoFileClip(outputs['mp4']) as clip:
            self.assertAlmostEqual(clip.duration, 2, delta=0.5)  # 验证时长
            self.assertGreaterEqual(clip.fps, n_frames/clip.duration)  # 验证帧率

if __name__ == '__main__':
    unittest.main() 