"""
SGA-PDE 集成最终验证测试

这个测试文件验证 SGA-PDE 是否成功集成到 KD 框架中。
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))


class TestSGAIntegrationFinal(unittest.TestCase):
    """SGA-PDE 集成最终验证测试类"""

    def test_01_kd_sga_import(self):
        """测试 KD_SGA 是否可以正常导入"""
        try:
            from kd.model import KD_SGA
            self.assertTrue(True, "KD_SGA 导入成功")
        except ImportError as e:
            self.fail(f"KD_SGA 导入失败: {e}")

    def test_02_model_instantiation(self):
        """测试模型实例化"""
        from kd.model import KD_SGA
        
        try:
            model = KD_SGA(num=3, depth=2, width=2)
            self.assertIsNotNone(model, "模型实例化成功")
        except Exception as e:
            self.fail(f"模型实例化失败: {e}")

    def test_03_parameter_management(self):
        """测试参数管理功能"""
        from kd.model import KD_SGA
        
        model = KD_SGA(num=3, depth=2, width=2)
        
        # 测试 get_params
        params = model.get_params()
        self.assertEqual(params['num'], 3)
        self.assertEqual(params['depth'], 2)
        self.assertEqual(params['width'], 2)
        
        # 测试 set_params
        model.set_params(num=5, depth=3)
        updated_params = model.get_params()
        self.assertEqual(updated_params['num'], 5)
        self.assertEqual(updated_params['depth'], 3)

    def test_04_visualization_import(self):
        """测试可视化模块导入"""
        try:
            from kd.viz import sga_eq2latex
            self.assertTrue(True, "可视化模块导入成功")
        except ImportError as e:
            self.fail(f"可视化模块导入失败: {e}")

    def test_05_workflow_access(self):
        """测试内部工作流程访问"""
        try:
            from kd.model.sga.sga_refactored.workflow import inject_dependencies
            self.assertTrue(True, "内部工作流程可访问")
        except ImportError as e:
            self.fail(f"内部工作流程访问失败: {e}")

    def test_06_directory_structure(self):
        """测试目录结构是否正确"""
        import os
        
        # 检查 sga_refactored 目录结构
        sga_refactored_path = os.path.join(
            os.path.dirname(__file__), '..', 'sga_refactored'
        )
        
        expected_files = [
            'data_utils.py',
            'operators.py', 
            'diagnostics.py',
            'workflow.py'
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(sga_refactored_path, file_name)
            self.assertTrue(
                os.path.exists(file_path), 
                f"文件 {file_name} 应该存在"
            )

    def test_07_api_methods_exist(self):
        """测试 API 方法是否存在"""
        from kd.model import KD_SGA
        
        model = KD_SGA(num=3, depth=2, width=2)
        
        # 检查必要的方法是否存在
        required_methods = [
            'fit', 'predict', 'score',
            'get_params', 'set_params',
            'get_equation_string', 'get_equation_latex'
        ]
        
        for method_name in required_methods:
            self.assertTrue(
                hasattr(model, method_name),
                f"方法 {method_name} 应该存在"
            )
            self.assertTrue(
                callable(getattr(model, method_name)),
                f"方法 {method_name} 应该可调用"
            )

    def test_08_inheritance_check(self):
        """测试继承关系是否正确"""
        from kd.model import KD_SGA
        from kd.model._base import BaseGa
        
        model = KD_SGA(num=3, depth=2, width=2)
        self.assertIsInstance(model, BaseGa, "KD_SGA 应该继承自 BaseGa")


if __name__ == '__main__':
    print("🎉 SGA-PDE 集成最终验证测试")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)
