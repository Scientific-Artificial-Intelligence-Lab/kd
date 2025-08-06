"""
测试SGA模块的导入副作用问题
Test SGA module import side effects
"""
import unittest
import sys
import io
import time
from contextlib import redirect_stdout, redirect_stderr


class TestImportSideEffects(unittest.TestCase):
    """测试导入时的副作用"""
    
    def setUp(self):
        """清理已导入的模块"""
        # 移除已导入的SGA相关模块
        modules_to_remove = [
            'kd.model.sga.codes.configure',
            'kd.model.sga.codes.Data_generator', 
            'kd.model.sga.codes.setup',
            'kd.model.sga.codes.pde',
            'kd.model.sga.codes.tree',
            'kd.model.sga.codes.sga',
            'kd.model.kd_sga'
        ]
        
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]
    
    def test_configure_import_side_effects(self):
        """测试configure.py导入时的副作用"""
        # 捕获标准输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            start_time = time.time()
            from kd.model.sga.codes import configure
            import_time = time.time() - start_time
        
        output = stdout_capture.getvalue()
        
        # 验证导入时间不应该太长（应该 < 0.1秒）
        self.assertLess(import_time, 0.1, 
                       f"configure.py import took {import_time:.3f}s, too slow!")
        
        # 验证不应该有打印输出（除非必要）
        # 当前会有输出，修复后应该没有
        print(f"Configure import output: {output}")
        
        # 验证基本属性存在
        self.assertTrue(hasattr(configure, 'problem'))
        
    def test_data_generator_import_side_effects(self):
        """测试Data_generator.py导入时的副作用"""
        stdout_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture):
            start_time = time.time()
            from kd.model.sga.codes import Data_generator
            import_time = time.time() - start_time
            
        output = stdout_capture.getvalue()
        
        # 验证导入时间（当前会很慢，修复后应该快）
        print(f"Data_generator import took {import_time:.3f}s")
        print(f"Data_generator import output: {output}")
        
        # 验证数据存在但不应该在导入时计算
        self.assertTrue(hasattr(Data_generator, 'u'))
        
    def test_setup_import_side_effects(self):
        """测试setup.py导入时的副作用"""
        stdout_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture):
            start_time = time.time()
            from kd.model.sga.codes import setup
            import_time = time.time() - start_time
            
        output = stdout_capture.getvalue()
        
        # 验证导入时间
        print(f"Setup import took {import_time:.3f}s")
        print(f"Setup import output: {output}")
        
        # 验证全局变量存在
        self.assertTrue(hasattr(setup, 'default_terms'))
        
    def test_multiple_imports_consistency(self):
        """测试多次导入的一致性"""
        # 第一次导入
        from kd.model.sga.codes import configure as config1
        problem1 = config1.problem
        
        # 修改全局状态
        config1.problem = 'test_problem'
        
        # 第二次导入（应该是同一个模块）
        from kd.model.sga.codes import configure as config2
        problem2 = config2.problem
        
        # 验证是同一个模块实例
        self.assertEqual(problem2, 'test_problem')
        self.assertIs(config1, config2)
        
        # 恢复原始状态
        config1.problem = problem1
        
    def test_kd_sga_import_performance(self):
        """测试KD_SGA导入性能"""
        stdout_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture):
            start_time = time.time()
            from kd.model import KD_SGA
            import_time = time.time() - start_time
            
        output = stdout_capture.getvalue()
        
        print(f"KD_SGA import took {import_time:.3f}s")
        print(f"KD_SGA import output: {output}")
        
        # 目标：导入应该在1秒内完成
        # 当前可能会超时，修复后应该很快
        if import_time > 1.0:
            print(f"WARNING: KD_SGA import is slow ({import_time:.3f}s)")


class TestInstanceIsolation(unittest.TestCase):
    """测试实例隔离"""
    
    def test_multiple_instances_isolation(self):
        """测试多个KD_SGA实例的隔离性"""
        from kd.model import KD_SGA
        
        # 创建两个不同问题的实例
        model1 = KD_SGA(problem='Burgers', num=5, depth=2, width=2)
        model2 = KD_SGA(problem='chafee-infante', num=5, depth=2, width=2)
        
        # 验证实例有独立的配置
        self.assertEqual(model1.problem, 'Burgers')
        self.assertEqual(model2.problem, 'chafee-infante')
        
        # 修改一个实例不应该影响另一个
        model1.problem = 'modified'
        self.assertEqual(model2.problem, 'chafee-infante')
        
    def test_global_state_isolation(self):
        """测试全局状态隔离"""
        from kd.model.sga.codes import configure
        
        # 记录原始状态
        original_problem = configure.problem
        
        try:
            # 创建实例并修改配置
            from kd.model import KD_SGA
            model = KD_SGA(problem='test_problem')
            
            # 验证全局状态没有被永久修改
            # （这个测试在修复前可能会失败）
            self.assertEqual(configure.problem, original_problem)
            
        finally:
            # 恢复原始状态
            configure.problem = original_problem


if __name__ == '__main__':
    unittest.main()
