"""
测试 SGA 与 KD 框架的统一数据目录集成
Test SGA integration with KD framework's unified data directory
"""

import unittest
import numpy as np
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from kd.model.kd_sga import KD_SGA
from kd.model.sga.codes import configure


class TestUnifiedDataIntegration(unittest.TestCase):
    """测试统一数据目录集成"""
    
    def setUp(self):
        """设置测试环境"""
        # 保存原始配置
        self.original_problem = configure.problem
        
    def tearDown(self):
        """清理测试环境"""
        # 恢复原始配置
        configure.problem = self.original_problem
        
    def test_data_path_unified(self):
        """测试数据路径已统一到 kd/dataset/data/"""
        # 测试 configure.py 中的数据加载
        configure.problem = 'chafee-infante'
        configure.ensure_data_loaded()
        
        # 验证数据已正确加载
        self.assertIsNotNone(configure.u)
        self.assertIsNotNone(configure.x)
        self.assertIsNotNone(configure.t)
        
        # 验证数据形状
        self.assertEqual(configure.u.shape, (301, 200))
        self.assertEqual(configure.x.shape, (301,))
        self.assertEqual(configure.t.shape, (200,))
        
    def test_kd_sga_with_unified_data(self):
        """测试 KD_SGA 使用统一数据目录"""
        model = KD_SGA(problem='chafee-infante', num=5, depth=2)
        
        # 运行小规模测试
        model.fit(X=None, max_gen=2, verbose=False)
        
        # 验证结果
        self.assertIsNotNone(model.best_equation_)
        self.assertIsNotNone(model.best_aic_)
        self.assertIsInstance(model.best_aic_, float)
        
    def test_multiple_problems_data_loading(self):
        """测试多个问题的数据加载"""
        problems = ['chafee-infante', 'PDE_compound', 'PDE_divide', 'Burgers']
        
        for problem in problems:
            with self.subTest(problem=problem):
                configure.problem = problem
                configure.ensure_data_loaded()
                
                # 验证数据已加载
                self.assertIsNotNone(configure.u, f"Failed to load u for {problem}")
                self.assertIsNotNone(configure.x, f"Failed to load x for {problem}")
                self.assertIsNotNone(configure.t, f"Failed to load t for {problem}")
                
                # 验证数据类型
                self.assertIsInstance(configure.u, np.ndarray)
                self.assertIsInstance(configure.x, np.ndarray)
                self.assertIsInstance(configure.t, np.ndarray)
                
    def test_data_consistency_with_main_directory(self):
        """测试数据与主目录的一致性"""
        # 这个测试验证 SGA 现在使用的数据与 kd/dataset/data/ 中的数据一致
        
        # 加载 chafee-infante 数据
        configure.problem = 'chafee-infante'
        configure.ensure_data_loaded()
        sga_u = configure.u.copy()
        sga_x = configure.x.copy()
        sga_t = configure.t.copy()
        
        # 直接从主数据目录加载
        main_data_dir = os.path.join(os.path.dirname(__file__), '../../../dataset/data')
        main_u = np.load(os.path.join(main_data_dir, 'chafee_infante_CI.npy'))
        main_x = np.load(os.path.join(main_data_dir, 'chafee_infante_x.npy'))
        main_t = np.load(os.path.join(main_data_dir, 'chafee_infante_t.npy'))
        
        # 验证数据一致性
        np.testing.assert_array_equal(sga_u, main_u, "u data inconsistent")
        np.testing.assert_array_equal(sga_x, main_x, "x data inconsistent")
        np.testing.assert_array_equal(sga_t, main_t, "t data inconsistent")
        
    def test_kd_sga_api_compatibility(self):
        """测试 KD_SGA API 兼容性"""
        model = KD_SGA(problem='chafee-infante', num=3, depth=2)
        
        # 测试 fit 方法
        model.fit(X=None, max_gen=1, verbose=False)
        
        # 测试 API 方法
        eq_str = model.get_equation_string()
        self.assertIsInstance(eq_str, str)
        
        latex_str = model.get_equation_latex()
        # latex_str 可能为 None 如果生成失败
        if latex_str is not None:
            self.assertIsInstance(latex_str, str)
            
        # 测试 score 方法
        score = model.score(X=np.random.randn(10, 2))
        self.assertIsInstance(score, float)
        
        # 测试 predict 方法（虽然是占位符实现）
        pred = model.predict(X=np.random.randn(10, 2))
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (10,))


if __name__ == '__main__':
    unittest.main()
