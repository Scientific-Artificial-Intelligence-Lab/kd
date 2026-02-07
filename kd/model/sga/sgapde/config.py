"""
Configuration classes for the SGA-PDE solver.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
import os


class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(int(n_hidden),n_output)
    def forward(self,x):
        out = torch.sin((self.fc1(x)))
        out = torch.sin((self.fc2(out)))
        out = torch.sin((self.fc2(out)))
        out = torch.sin((self.fc2(out)))
        out = self.predict(out) 
        return out

# god class. need to think about refactor later
@dataclass
class SolverConfig:
    """
    Configuration for the SGA-PDE solver.

    在原始 `sgapde` 库中, 这个类同时承担了两类职责:
    1. 算法/运行配置: SGA 树结构、遗传算子、AIC 惩罚、是否使用 metadata / autograd 等。
    2. 问题/数据配置: 通过 ``problem_name`` 在 `_load_problem_config` 中选择预置的
       Chafee-Infante / Burgers / KdV 三个 benchmark, 并直接从本地文件加载 `u/x/t`,
       以及对应的真 PDE 模板字符串、元数据网络路径等。

    在 KD 集成中:
    - 推荐入口是上层的 `KD_SGA.fit_dataset(PDEDataset, ...)`, 数据通常已经由
      `PDEDataset` + `PDE_REGISTRY` 统一加载, 然后通过 `u_data/x_data/t_data`
      传入本配置对象。
    - 对这条“框架模式”数据流来说, `_load_problem_config` 中的文件加载逻辑基本不会
      被使用, 主要意义是为少数内置 benchmark 提供 ground-truth PDE 模板和 legacy
      可视化/误差评估所需的字段。
    - 因此, 可以把 `_load_problem_config` 理解为 *仅服务三种内置问题的遗留分支*;
      新增/自定义数据集应该只通过 `PDEDataset` + `fit_dataset` 接入, 而不是在这里
      再增加新的 `problem_name` 分支。这一点在 `notes/dataloader/sga_work.md` 中有
      更详细的设计说明。
    """
    
    # Problem specification
    problem_name: str = 'chafee_infante'

    # For external data
    u_data: np.ndarray = None
    x_data: np.ndarray = None
    t_data: np.ndarray = None
    fields_data: Optional[Dict[str, np.ndarray]] = None
    coords_1d: Optional[Dict[str, np.ndarray]] = None
    axis_order: Optional[List[str]] = None
    target_field: str = "u"
    lhs_axis: str = "t"
    # Optional override for the legacy ux/uxx/uxxx axis when no 'x' axis exists.
    primary_spatial_axis: Optional[str] = None
    enforce_uniform_grid: bool = True

    # SGA parameters
    num: int = 20  # Number of PDEs in the pool
    depth: int = 4  # Maximum depth of each PDE term
    width: int = 5  # Maximum number of terms in each PDE
    p_var: float = 0.5  # Probability of node being u/t/x instead of operator
    p_mute: float = 0.3  # Mutation probability for each node
    p_cro: float = 0.5  # Crossover probability between PDEs
    p_rep: float = 1.0  # Probability of replacing a term
    sga_run: int = 100  # Number of generations to run
    
    # Random seed
    seed: int = 0
    
    # Data parameters
    use_metadata: bool = False
    delete_edges: bool = False
    fine_ratio: int = 2
    normal: bool = True # Normalize data if True
    
    # AIC hyperparameter
    aic_ratio: float = 1.0
    
    # Neural network parameters (for metadata)
    hidden_dim: int = 50
    num_feature: int = 2
    max_epoch: int = 100000

    simple_mode: bool = True

    # set to False use difference or True use autograd for derivatives
    use_autograd: bool = False
    
    model_path: str = None

    train_ratio: float = 1.0

    # 标记是否具备解析真解模板，用于误差/legacy 可视化
    has_ground_truth: bool = False

    def __post_init__(self):
        """Post-initialization to set up derived attributes."""
        # Set random seed
        np.random.seed(self.seed)
        
        if self.model_path is None:
            # Only derive the legacy default when the caller did not supply one.
            self.model_path = (
                f"{self.problem_name}_sine_sin_50_3fc2_"
                f"{int(self.max_epoch/1000)}k_Adam.pkl"
            )

        has_registry_payload = self.fields_data is not None or self.coords_1d is not None
        if has_registry_payload:
            self.right_side = None
            self.left_side = None
            self.right_side_origin = None
            self.left_side_origin = None
            self.has_ground_truth = False
            return

        # Load problem-specific configuration
        self._load_problem_config()
        
    def _find_data_file(self, filename):
        if isinstance(filename, (list, tuple, set)):
            candidates = tuple(filename)
        else:
            candidates = (filename,)

        base_dir = os.path.dirname(__file__)
        proj_root = os.path.abspath(os.path.join(base_dir, "../../../.."))
        repo_root = os.path.abspath(os.path.join(base_dir, ".."))

        for candidate in candidates:
            local_path = os.path.join(base_dir, "data", candidate)
            if os.path.exists(local_path):
                return local_path

            repo_data_path = os.path.join(repo_root, "data", candidate)
            if os.path.exists(repo_data_path):
                return repo_data_path

            kd_data_path = os.path.join(proj_root, "kd", "dataset", "data", candidate)
            if os.path.exists(kd_data_path):
                return kd_data_path

            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Cannot find data file: {candidates}")

    def _load_problem_config(self):
        """
        Load problem-specific data and equations.

        说明:
        - 在原始 `sgapde` 使用方式下, 调用者只需要给出 `problem_name`, 这里会直接从
          本地文件中加载对应的 `u/x/t` 数据, 并设置真 PDE 的左右端模板字符串等。
        - 在 KD 集成的主路径中, 数据通常已经通过 `PDEDataset`+注册表加载完毕并以
          `u_data/x_data/t_data` 形式传入; 此时 `ProblemContext._load_data` 会优先
          使用这些 inline 数据, 而不是这里设置的 `self.u/self.x/self.t`。
        - 因此, 这里对 `chafee-infante` / `burgers` / `kdv` 的分支可以视为对原论文
          三个 benchmark 的兼容支持; 不建议在此处为新数据集继续扩展硬编码分支,
          新增数据应通过 KD 的 dataloader/registry 体系进入, 再由上层传入 `u_data`。
        """
        has_inline_data = (
            self.u_data is not None
            and self.x_data is not None
            and self.t_data is not None
        )

        normalized_name = self._normalize_problem_name(self.problem_name)

        if normalized_name == 'chafee-infante':
            if has_inline_data:
                self.u = np.asarray(self.u_data)
                self.x = np.asarray(self.x_data)
                self.t = np.asarray(self.t_data)
            else:
                self.u = np.load(self._find_data_file("chafee_infante_CI.npy"))
                self.x = np.load(self._find_data_file("chafee_infante_x.npy"))
                self.t = np.load(self._find_data_file("chafee_infante_t.npy"))
            self.right_side = 'right_side = - 1.0008*u + 1.0004*u**3'
            self.left_side = 'left_side = ut'
            self.right_side_origin = 'right_side_origin = uxx_origin-u_origin+u_origin**3'
            self.left_side_origin = 'left_side_origin = ut_origin'
            self.has_ground_truth = True
        elif normalized_name == 'burgers':
            import scipy.io as scio

            if has_inline_data:
                self.u = np.asarray(self.u_data)
                self.x = np.asarray(self.x_data)
                self.t = np.asarray(self.t_data)
            else:
                data = scio.loadmat(self._find_data_file(("burgers.mat", "burgers2.mat")))
                self.u = data.get("usol")
                self.x = np.squeeze(data.get("x"))
                self.t = np.squeeze(data.get("t").reshape(1, -1))
            self.right_side = 'right_side = -u*ux+0.1*uxx'
            self.left_side = 'left_side = ut'
            self.right_side_origin = 'right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin'
            self.left_side_origin = 'left_side_origin = ut_origin'
            self.has_ground_truth = True
        elif normalized_name == 'kdv':
            import scipy.io as scio

            if has_inline_data:
                self.u = np.asarray(self.u_data)
                self.x = np.asarray(self.x_data)
                self.t = np.asarray(self.t_data)
            else:
                data = scio.loadmat(self._find_data_file(("KdV.mat", "Kdv.mat", "KdV_equation.mat")))
                self.u = data.get("uu")
                self.x = np.squeeze(data.get("x"))
                self.t = np.squeeze(data.get("tt").reshape(1, -1))
            self.right_side = 'right_side = -0.0025*uxxx-u*ux'
            self.left_side = 'left_side = ut'
            self.right_side_origin = 'right_side_origin = -0.0025*uxxx_origin-u_origin*ux_origin'
            self.left_side_origin = 'left_side_origin = ut_origin'
            self.has_ground_truth = True
        else:
            if has_inline_data:
                # custom 模式：允许任意 u/x/t，跳过解析模板
                self.u = np.asarray(self.u_data)
                self.x = np.asarray(self.x_data)
                self.t = np.asarray(self.t_data)
                self.right_side = None
                self.left_side = None
                self.right_side_origin = None
                self.left_side_origin = None
                self.has_ground_truth = False
                return
            raise ValueError(
                "SGA solver requires a known problem definition to obtain RHS templates. "
                f"Received '{self.problem_name}' without inline data."
            )

    @staticmethod
    def _normalize_problem_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return name
        slug = name.strip().lower().replace('_', '-').replace(' ', '-')
        alias_map = {
            'burgers2': 'burgers',
            'burgers-equation': 'burgers',
            'chafee-infante': 'chafee-infante',
            'chafeeinfante': 'chafee-infante',
            'kdv-equation': 'kdv',
            'kdv_equation': 'kdv',
            'kdv-equation-sine': 'kdv',
        }
        return alias_map.get(slug, slug)
            
    @staticmethod
    def divide(up, down, eta=1e-10):
        """Safe division function."""
        while np.any(down == 0):
            down += eta
        return up / down
        
    def get_device(self):
        """Get the appropriate device for computation."""
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
            
    @property
    def device(self):
        """Property to get device."""
        return self.get_device()
