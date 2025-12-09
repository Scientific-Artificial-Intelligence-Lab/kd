"""
ProblemContext class to encapsulate all problem-specific data and computations.
This replaces the global state from setup.py.
"""

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys

from .PDE_find import Diff, Diff2, FiniteDiff
from .config import Net

class ProblemContext:
    """Encapsulates all problem-specific data and computations."""
    
    def __init__(self, config):
        """
        Initialize the problem context with the given configuration.
        
        Args:
            config: Configuration object containing problem parameters
        """
        self.config = config
        self.device = config.device

        self.simple_mode = config.simple_mode
        
        # Initialize data
        self._load_data()

        self._initialize_slicing_indices()
        
        # Initialize operators
        self._init_operators()
        
        # Calculate derivatives
        self._calculate_derivatives()
        
        # Calculate errors
        self._calculate_errors()
        
    def _load_data(self):
        """
        加载数据。其优先级如下：
        1. 优先使用直接从 SolverConfig 传入的数据数组（kd框架模式）。
        2. 若无直接数据，则根据 problem_name 从文件加载。
        3. 加载后，根据 use_metadata 开关决定是否生成并使用高分辨率元数据。
        """
        # 1. 决定数据来源：是直接传入，还是从文件加载
        if self.config.u_data is not None:
            print("\tINFO: Loading data directly from provided arrays (Framework Mode).")
            base_u, base_x, base_t = self.config.u_data, self.config.x_data, self.config.t_data
        else:
            print(f"\tINFO: Loading data from file for problem: '{self.config.problem_name}' (Standalone Mode).")
            base_u, base_x, base_t = self.config.u, self.config.x, self.config.t
        
        # 将基础数据存入 _origin 属性，作为永远的备份
        self.u_origin, self.x_origin, self.t_origin = base_u, base_x, base_t
        self.n_origin, self.m_origin = self.u_origin.shape
        self.dx_origin = self.x_origin[1] - self.x_origin[0] if len(self.x_origin) > 1 else 0
        self.dt_origin = self.t_origin[1] - self.t_origin[0] if len(self.t_origin) > 1 else 0

        # 2. 决定计算用数据：是使用元数据，还是使用基础数据
        if self.config.use_metadata:
            self.u, self.x, self.t, self.x_all = self._generate_metadata()
        else:
            self.u, self.x, self.t, self.x_all = self.u_origin, self.x_origin, self.t_origin, self.x_origin

        # 3. 对最终选定的计算用数据进行通用设置
        self.n, self.m = self.u.shape
        self.dx = self.x[1] - self.x[0] if len(self.x) > 1 else 0
        self.dt = self.t[1] - self.t[0] if len(self.t) > 1 else 0

        # 扩展维度以匹配 u 的形状
        self.x = np.tile(self.x, (self.m, 1)).transpose((1, 0))
        self.x_all = np.tile(self.x_all, (self.m, 1)).transpose((1, 0))
        self.t = np.tile(self.t, (self.n, 1))

        # 同样扩展 _origin 数据的维度，以供后续绘图或比较使用
        self.x_origin = np.tile(self.x_origin, (self.m_origin, 1)).transpose((1, 0))
        self.t_origin = np.tile(self.t_origin, (self.n_origin, 1))

        # 4. 根据开关，选择性地切割数据边缘
        if self.config.delete_edges:
            print("\tINFO: Deleting edges from the data (10% on each side).")
            n_slice_start = int(self.n * 0.1)
            n_slice_end = int(self.n * 0.9)
            
            self.u = self.u[n_slice_start:n_slice_end, :]
            self.x = self.x[n_slice_start:n_slice_end, :] # 注意：x也需要同步切割
            self.t = self.t[n_slice_start:n_slice_end, :] # t也需要同步切割
            
            # 更新切割后的维度
            self.n, self.m = self.u.shape
        
    def _init_operators(self):
        """Initialize operator definitions."""
        # Define zeros array
        self.zeros = np.zeros(self.u.shape)
        
        # Define operators (from setup.py)
        self.ALL = np.array([
            ['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], 
            ['/', 2, self.config.divide], ['d', 2, Diff], ['d^2', 2, Diff2], 
            ['u', 0, self.u], ['x', 0, self.x], ['ux', 0, None],  # ux will be set after calculation
            ['0', 0, self.zeros], ['^2', 1, np.square], ['^3', 1, self._cubic]
        ], dtype=object)
        
        self.OPS = np.array([
            ['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], 
            ['/', 2, self.config.divide], ['d', 2, Diff], ['d^2', 2, Diff2], 
            ['^2', 1, np.square], ['^3', 1, self._cubic]
        ], dtype=object)
        
        self.ROOT = np.array([
            ['*', 2, np.multiply], ['d', 2, Diff], ['d^2', 2, Diff2], 
            ['/', 2, self.config.divide], ['^2', 1, np.square], ['^3', 1, self._cubic]
        ], dtype=object)
        
        self.OP1 = np.array([['^2', 1, np.square], ['^3', 1, self._cubic]], dtype=object)
        
        self.OP2 = np.array([
            ['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], 
            ['/', 2, self.config.divide], ['d', 2, Diff], ['d^2', 2, Diff2]
        ], dtype=object)
        
        self.VARS = np.array([
            ['u', 0, self.u], ['x', 0, self.x], ['0', 0, self.zeros], 
            ['ux', 0, None]  # Will be set after calculation
        ], dtype=object)
        
        self.den = np.array([['x', 0, self.x]], dtype=object)
    
    @staticmethod # 最重要的修复! 一定要加这个... 不然第一个参数是 self 的话会爆炸 :(
    def _cubic(inputs):
        """Cubic function."""
        return np.power(inputs, 3)
        
    def _calculate_derivatives(self):
        """Calculate derivatives using finite differences or autograd."""
        
        if self.config.use_autograd:
            self._calculate_derivatives_autograd()
        else:
            self._calculate_derivatives_difference()
            
        # Update operators with calculated derivatives
        self._update_operators_with_derivatives()
        
    def _calculate_derivatives_difference(self):
        """Calculate derivatives using finite differences."""
        n, m = self.n, self.m
        
        # Calculate ut
        self.ut = np.zeros((n, m))
        for idx in range(n):
            self.ut[idx, :] = FiniteDiff(self.u[idx, :], self.dt)
            
        # Calculate ux, uxx, uxxx
        self.ux = np.zeros((n, m))
        self.uxx = np.zeros((n, m))
        self.uxxx = np.zeros((n, m))
        
        for idx in range(m):
            self.ux[:, idx] = FiniteDiff(self.u[:, idx], self.dx)
        for idx in range(m):
            self.uxx[:, idx] = FiniteDiff(self.ux[:, idx], self.dx)
        for idx in range(m):
            self.uxxx[:, idx] = FiniteDiff(self.uxx[:, idx], self.dx)
            
        # Calculate derivatives for original data
        n_origin, m_origin = self.n_origin, self.m_origin
        
        self.ut_origin = np.zeros((n_origin, m_origin))
        for idx in range(n_origin):
            self.ut_origin[idx, :] = FiniteDiff(self.u_origin[idx, :], self.dt_origin)
            
        self.ux_origin = np.zeros((n_origin, m_origin))
        self.uxx_origin = np.zeros((n_origin, m_origin))
        self.uxxx_origin = np.zeros((n_origin, m_origin))
        
        for idx in range(m_origin):
            self.ux_origin[:, idx] = FiniteDiff(self.u_origin[:, idx], self.dx_origin)
        for idx in range(m_origin):
            self.uxx_origin[:, idx] = FiniteDiff(self.ux_origin[:, idx], self.dx_origin)
        for idx in range(m_origin):
            self.uxxx_origin[:, idx] = FiniteDiff(self.uxx_origin[:, idx], self.dx_origin)
            
    def _calculate_derivatives_autograd(self):
        """Calculate derivatives using autograd."""
        # load model
        model = Net(self.config.num_feature, self.config.hidden_dim, 1)
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.to(self.device)

        # autograd function
        def fun(x_data, t_data, model_instance):
            # Combine x and t data and prepare for autograd
            database = torch.cat((x_data, t_data), 1).float().to(self.device)
            database.requires_grad = True # Set requires_grad for differentiation

            # Get model output
            pinn_output = model_instance(database)
            
            # First-order derivatives
            H_grad = torch.autograd.grad(outputs=pinn_output.sum(), inputs=database, create_graph=True)[0]
            Ht = H_grad[:, 1]
            Hx = H_grad[:, 0]
            
            # Second-order derivatives
            Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, 0]
            
            # Third-order derivatives
            Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, 0]
            
            # Return derivatives as numpy arrays
            return Hx.cpu().data.numpy(), Hxx.cpu().data.numpy(), Hxxx.cpu().data.numpy(), Ht.cpu().data.numpy()

        # --- Calculate derivatives for metadata ---
        x_1d = np.reshape(self.x, (self.n * self.m, 1))
        t_1d = np.reshape(self.t, (self.n * self.m, 1))
        
        # Convert numpy arrays to torch tensors
        x_tensor = torch.from_numpy(x_1d)
        t_tensor = torch.from_numpy(t_1d)

        ux, uxx, uxxx, ut = fun(x_tensor, t_tensor, model)

        # Reshape and store results as instance attributes
        self.ut = np.reshape(ut, (self.n, self.m))
        self.ux = np.reshape(ux, (self.n, self.m))
        self.uxx = np.reshape(uxx, (self.n, self.m))
        self.uxxx = np.reshape(uxxx, (self.n, self.m))

        # --- Calculate derivatives for original data ---
        x_1d_origin = np.reshape(self.x_origin, (self.n_origin * self.m_origin, 1))
        t_1d_origin = np.reshape(self.t_origin, (self.n_origin * self.m_origin, 1))

        # Convert numpy arrays to torch tensors
        x_tensor_origin = torch.from_numpy(x_1d_origin)
        t_tensor_origin = torch.from_numpy(t_1d_origin)

        ux_origin, uxx_origin, uxxx_origin, ut_origin = fun(x_tensor_origin, t_tensor_origin, model)
        
        # Reshape and store results as instance attributes
        self.ut_origin = np.reshape(ut_origin, (self.n_origin, self.m_origin))
        self.ux_origin = np.reshape(ux_origin, (self.n_origin, self.m_origin))
        self.uxx_origin = np.reshape(uxx_origin, (self.n_origin, self.m_origin))
        self.uxxx_origin = np.reshape(uxxx_origin, (self.n_origin, self.m_origin))
        
    def _update_operators_with_derivatives(self):
        """Update operator arrays with calculated derivatives."""
        # Find and update ux in ALL array
        for i, op in enumerate(self.ALL):
            if op[0] == 'ux':
                self.ALL[i][2] = self.ux
                
        # Find and update ux in VARS array
        for i, var in enumerate(self.VARS):
            if var[0] == 'ux':
                self.VARS[i][2] = self.ux


    def _initialize_slicing_indices(self):
        """Calculate and store slicing indices as instance attributes."""
        self.n1 = int(self.n * 0.1)
        self.n2 = int(self.n * 0.9)
        self.m1 = int(self.m * 0)
        self.m2 = int(self.m * 1)
        self.n1_origin = int(self.n_origin * 0.1)
        self.n2_origin = int(self.n_origin * 0.9)
        self.m1_origin = int(self.m_origin * 0)
        self.m2_origin = int(self.m_origin * 1)
    
    def _calculate_errors(self):
        """Calculate errors between left and right sides of the PDE."""
        # Prepare default terms for evaluation
        self.default_u = np.reshape(self.u, (self.u.shape[0]*self.u.shape[1], 1))
        self.default_ux = np.reshape(self.ux, (self.u.shape[0]*self.u.shape[1], 1))
        self.default_uxx = np.reshape(self.uxx, (self.u.shape[0]*self.u.shape[1], 1))
        self.default_u2 = np.reshape(self.u**2, (self.u.shape[0]*self.u.shape[1], 1))
        self.default_u3 = np.reshape(self.u**3, (self.u.shape[0]*self.u.shape[1], 1))
        self.default_terms = np.hstack((self.default_u)).reshape(-1, 1)
        self.default_names = ['u']
        self.num_default = self.default_terms.shape[1]

        # 若无解析模板或显式标记为无真解，则跳过 ground-truth 误差计算
        if (
            not getattr(self.config, "has_ground_truth", False)
            or self.config.right_side is None
            or self.config.left_side is None
            or self.config.right_side_origin is None
            or self.config.left_side_origin is None
        ):
            print("\tINFO: No analytic PDE template; skipping ground-truth error computation.")
            return

        # 计算 right_side, right_side_full, right_side_origin, right_side_full_origin
        # 动态执行 config 的表达式
        local_vars = {
            "u": self.u, "ut": self.ut, "ux": self.ux, "uxx": self.uxx, "uxxx": self.uxxx,
            "u_origin": self.u_origin, "ut_origin": self.ut_origin, "ux_origin": self.ux_origin,
            "uxx_origin": self.uxx_origin, "uxxx_origin": self.uxxx_origin
        }
        # right_side, left_side
        exec(self.config.right_side, {}, local_vars)
        exec(self.config.left_side, {}, local_vars)
        right_side = local_vars.get("right_side")
        left_side = local_vars.get("left_side")
        self.right_side_full = right_side
        self.left_side_full = left_side
        self.right_side = right_side[self.n1:self.n2, self.m1:self.m2]
        self.left_side = left_side[self.n1:self.n2, self.m1:self.m2]

        # right_side_origin, left_side_origin
        exec(self.config.right_side_origin, {}, local_vars)
        exec(self.config.left_side_origin, {}, local_vars)
        right_side_origin = local_vars.get("right_side_origin")
        left_side_origin = local_vars.get("left_side_origin")
        self.right_side_full_origin = right_side_origin
        self.left_side_full_origin = left_side_origin
        self.right_side_origin = right_side_origin[self.n1_origin:self.n2_origin, self.m1_origin:self.m2_origin]
        self.left_side_origin = left_side_origin[self.n1_origin:self.n2_origin, self.m1_origin:self.m2_origin]

        # 计算并打印去除边缘的数据误差
        right = np.reshape(self.right_side, ((self.n2-self.n1)*(self.m2-self.m1), 1))
        left = np.reshape(self.left_side, ((self.n2-self.n1)*(self.m2-self.m1), 1))
        diff = np.linalg.norm(left-right, 2)/((self.n2-self.n1)*(self.m2-self.m1))
        print('\tdata error without edges', diff)

        right_origin = np.reshape(self.right_side_origin, ((self.n2_origin-self.n1_origin)*(self.m2_origin-self.m1_origin), 1))
        left_origin = np.reshape(self.left_side_origin, ((self.n2_origin-self.n1_origin)*(self.m2_origin-self.m1_origin), 1))
        diff_origin = np.linalg.norm(left_origin-right_origin, 2)/((self.n2_origin-self.n1_origin)*(self.m2_origin-self.m1_origin))
        print('\tdata error_origin without edges', diff_origin)

        # 重新计算并打印完整数据的误差
        n1_full, n2_full, m1_full, m2_full = 0, self.n, 0, self.m
        right_full_reshaped = np.reshape(self.right_side_full, ((n2_full-n1_full)*(m2_full-m1_full), 1))
        left_full_reshaped = np.reshape(self.left_side_full, ((n2_full-n1_full)*(m2_full-m1_full), 1))
        diff_full = np.linalg.norm(left_full_reshaped - right_full_reshaped, 2)/((n2_full-n1_full)*(m2_full-m1_full))
        print('\tdata error', diff_full)
        
        n1_origin_full, n2_origin_full, m1_origin_full, m2_origin_full = 0, self.n_origin, 0, self.m_origin
        right_origin_full_reshaped = np.reshape(self.right_side_full_origin, ((n2_origin_full-n1_origin_full)*(m2_origin_full-m1_origin_full), 1))
        left_origin_full_reshaped = np.reshape(self.left_side_full_origin, ((n2_origin_full-n1_origin_full)*(m2_origin_full-m1_origin_full), 1))
        diff_origin_full = np.linalg.norm(left_origin_full_reshaped - right_origin_full_reshaped, 2)/((n2_origin_full-n1_origin_full)*(m2_origin_full-m1_origin_full))
        print('\tdata error_origin', diff_origin_full)

    # 使用预训练的神经网络生成高分辨率数据，从旧的 Data_generator.py 迁移而来。
    def _generate_metadata(self):
        """Generate high-resolution metadata using a pre-trained model."""
        print("INFO: Generating high-resolution metadata from pre-trained model (with normalization)...")
        # Load pre-trained model
        model = Net(self.config.num_feature, self.config.hidden_dim, 1)
        try:
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        except FileNotFoundError:
            print(
                "[SGA Metadata ERROR] Failed to generate metadata because the "
                f"pretrained model was not found: {self.config.model_path}"
            )
            sys.exit(1)
        model.to(self.device)

        # 1. 计算原始数据 u_origin 的均值和标准差，用于后续的反归一化
        Y_raw = self.u_origin.reshape(-1, 1)
        Y_raw_mean = Y_raw.mean()
        Y_raw_std = Y_raw.std()

        # 准备高分辨率网格
        n_fine = self.config.fine_ratio * self.n_origin
        m_fine = self.config.fine_ratio * self.m_origin
        x_new = np.linspace(self.x_origin.min(), self.x_origin.max(), n_fine)
        t_new = np.linspace(self.t_origin.min(), self.t_origin.max(), m_fine)

        # 创建新网格的输入坐标
        X1 = np.repeat(x_new.reshape(-1, 1), m_fine, axis=1)
        X2 = np.repeat(t_new.reshape(1, -1), n_fine, axis=0)
        X_grid = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)

        # 2. 根据开关，对输入坐标进行归一化
        if self.config.normal:
            X_grid_mean = X_grid.mean(axis=0)
            X_grid_std = X_grid.std(axis=0)
            X_normalized = (X_grid - X_grid_mean) / X_grid_std
            X_tensor = torch.from_numpy(X_normalized).float().to(self.device)
        else:
            X_tensor = torch.from_numpy(X_grid).float().to(self.device)

        
        # 使用模型进行预测
        y_pred = model(X_tensor)
        y_pred_numpy = y_pred.cpu().data.numpy().flatten()

        # 3. 根据开关，对模型的输出进行反归一化，使其回到原始数据的尺度
        if self.config.normal:
            result_pred_real = y_pred_numpy * Y_raw_std + Y_raw_mean
        else:
            result_pred_real = y_pred_numpy

        u_new = result_pred_real.reshape(n_fine, m_fine)

        # 返回生成的高分辨率数据
        return u_new, x_new, t_new, x_new

    def get_pde_libs(self):
        """Return the PDE library lists (initially empty)."""
        return [], []
