"""Test cases for evolution snapshot functionality."""

import unittest
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
from moviepy.editor import VideoFileClip
import imageio.v2 as imageio

from kd.plot.interface.dlga_plotter import DLGAPlotter
from kd.tests.test_evolution_basic import generate_mock_data

class TestEvolutionSnapshot(unittest.TestCase):
    """Test cases for evolution snapshot functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.mock_data = generate_mock_data(n_generations=5, pop_size=10)  # 使用小数据集加快测试
        self.plotter = DLGAPlotter()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def test_snapshot_mp4(self):
        """Test snapshot mode with MP4 output."""
        output_path = os.path.join(self.test_dir, "test.mp4")
        result_path = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="mp4",
            output_path=output_path,
            desired_duration=2  # 设置更短的时长加快测试
        )
        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.getsize(result_path) > 0)
        
        # 验证视频时长
        with VideoFileClip(result_path) as clip:
            self.assertAlmostEqual(clip.duration, 2, delta=0.5)
        
    def test_snapshot_gif(self):
        """Test GIF output."""
        output_path = os.path.join(self.test_dir, "evolution.gif")
        result_path = self.plotter.plot_evolution(
            self.mock_data,
            mode="snapshot",
            output_format="gif",
            output_path=output_path,
            desired_duration=2  # 设置更短的时长加快测试
        )
        
        self.assertTrue(os.path.exists(result_path))
        
        # 验证是否为有效的GIF文件
        with open(result_path, 'rb') as f:
            self.assertTrue(f.read().startswith(b'GIF'))
            
        # 验证GIF帧数
        with imageio.get_reader(result_path) as reader:
            self.assertEqual(len(reader), len(self.mock_data))
            
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

if __name__ == '__main__':
    unittest.main() 