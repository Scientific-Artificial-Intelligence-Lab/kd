"""Test cases for evolution visualization performance."""

import unittest
import os
import shutil
import time
import tempfile
from pathlib import Path
from moviepy.editor import VideoFileClip

from kd.plot.interface.dlga_plotter import DLGAPlotter
from kd.tests.test_evolution_basic import generate_mock_data

class TestEvolutionPerformance(unittest.TestCase):
    """Test cases for evolution visualization performance."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.mock_data = generate_mock_data(n_generations=5, pop_size=10)  # 基础测试数据
        self.plotter = DLGAPlotter()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def test_parallel_processing(self):
        """Test parallel frame generation."""
        output_path = os.path.join(self.test_dir, "test_parallel.mp4")
        
        # Generate moderate dataset for testing
        large_data = generate_mock_data(n_generations=20, pop_size=30)  # 适中的数据集大小
        
        # Time parallel processing
        start_time = time.time()
        
        self.plotter.plot_evolution(
            large_data,
            mode="snapshot",
            output_path=output_path,
            fps=30,
            cleanup_temp=True,
            desired_duration=2  # 设置更短的视频时长
        )
        
        processing_time = time.time() - start_time
        
        # Basic checks
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(processing_time < 60)  # 应该在60秒内完成
        
        # 检查视频时长
        with VideoFileClip(output_path) as clip:
            self.assertLess(clip.duration, 5)  # 视频时长应该小于5秒
            
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
        
    def test_memory_usage(self):
        """Test memory usage with large dataset."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 生成较大的数据集
        large_data = generate_mock_data(n_generations=30, pop_size=50)
        
        # 执行处理
        output_path = os.path.join(self.test_dir, "large_test.mp4")
        self.plotter.plot_evolution(
            large_data,
            mode="snapshot",
            output_format="mp4",
            output_path=output_path,
            cleanup_temp=True
        )
        
        # 检查内存使用
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # Convert to MB
        
        # 内存增长不应超过1GB
        self.assertLess(memory_increase, 1024)
        
        # 验证输出
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

if __name__ == '__main__':
    unittest.main() 