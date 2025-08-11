"""
Configuration classes for the SGA-PDE solver.
"""

from dataclasses import dataclass
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

@dataclass
class SolverConfig:
    """Configuration for the SGA-PDE solver."""
    
    # Problem specification
    problem_name: str = 'chafee_infante'

    # For external data
    u_data: np.ndarray = None
    x_data: np.ndarray = None
    t_data: np.ndarray = None

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

    def __post_init__(self):
        """Post-initialization to set up derived attributes."""
        # Set random seed
        np.random.seed(self.seed)
        
        self.model_path = (
            f"{self.problem_name}_sine_sin_50_3fc2_"
            f"{int(self.max_epoch/1000)}k_Adam.pkl"
        )

        # Load problem-specific configuration
        self._load_problem_config()
        
    def _find_data_file(self, filename):
        # 1. 当前目录下的 ./data/
        local_path = os.path.join(os.path.dirname(__file__), "data", filename)
        if os.path.exists(local_path):
            return local_path
        # 2. 工程根目录下的 kd/dataset/data/
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        kd_data_path = os.path.join(proj_root, "kd", "dataset", "data", filename)
        if os.path.exists(kd_data_path):
            return kd_data_path
        # 3. 直接用 filename（支持绝对路径）
        if os.path.exists(filename):
            return filename
        raise FileNotFoundError(f"Cannot find data file: {filename}")

    def _load_problem_config(self):
        """Load problem-specific data and equations."""
        if self.problem_name == 'chafee-infante':
            self.u = np.load(self._find_data_file("chafee_infante_CI.npy"))
            self.x = np.load(self._find_data_file("chafee_infante_x.npy"))
            self.t = np.load(self._find_data_file("chafee_infante_t.npy"))
            self.right_side = 'right_side = - 1.0008*u + 1.0004*u**3'
            self.left_side = 'left_side = ut'
            self.right_side_origin = 'right_side_origin = uxx_origin-u_origin+u_origin**3'
            self.left_side_origin = 'left_side_origin = ut_origin'
        elif self.problem_name == 'burgers':
            import scipy.io as scio
            data = scio.loadmat(self._find_data_file("burgers.mat"))
            self.u = data.get("usol")
            self.x = np.squeeze(data.get("x"))
            self.t = np.squeeze(data.get("t").reshape(1, 201))
            self.right_side = 'right_side = -u*ux+0.1*uxx'
            self.left_side = 'left_side = ut'
            self.right_side_origin = 'right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin'
            self.left_side_origin = 'left_side_origin = ut_origin'
        elif self.problem_name == 'kdv':
            import scipy.io as scio
            data = scio.loadmat(self._find_data_file("KdV.mat"))
            self.u = data.get("uu")
            self.x = np.squeeze(data.get("x"))
            self.t = np.squeeze(data.get("tt").reshape(1, 201))
            self.right_side = 'right_side = -0.0025*uxxx-u*ux'
            self.left_side = 'left_side = ut'
            self.right_side_origin = 'right_side_origin = -0.0025*uxxx_origin-u_origin*ux_origin'
            self.left_side_origin = 'left_side_origin = ut_origin'
        else:
            raise ValueError(f"Unknown problem: {self.problem_name}")
            
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
