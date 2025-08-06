import numpy as np
import torch
import scipy.io as scio
import torch
import torch.nn as nn
import os

# === 修复导入副作用 / Fix Import Side Effects ===
# 将全局状态和计算封装到函数中，避免导入时执行
# Encapsulate global state and computations in functions to avoid execution during import

# 默认配置 / Default configuration
_DEFAULT_PROBLEM = 'Burgers'
_DEFAULT_SEED = 0

# 全局状态（向后兼容）/ Global state (backward compatibility)
problem = _DEFAULT_PROBLEM
seed = _DEFAULT_SEED

def get_device():
    """获取可用的计算设备 / Get available compute device"""
    if torch.cuda.is_available():
        print("Hardware: NVIDIA GPU detected. Using CUDA for acceleration.")
        return torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        print("Hardware: Apple Silicon GPU detected. Using Metal (MPS) for acceleration.")
        return torch.device('mps')
    else:
        print("Hardware: No compatible GPU detected. Falling back to CPU (execution will be slow).")
        return torch.device('cpu')

# 延迟初始化设备 / Lazy device initialization
device = None

def ensure_device():
    """确保设备已初始化 / Ensure device is initialized"""
    global device
    if device is None:
        device = get_device()
    return device

###########################################################################################
# Neural network
max_epoch = 100 * 1000
path = problem+'_sine_sin_50_3fc2_'+'%d'%(max_epoch/1000)+'k_Adam.pkl'
hidden_dim = 50

train_ratio = 1 # the ratio of training dataset
num_feature = 2
normal = True

###########################################################################################
# Metadata
fine_ratio = 2 # 通过MetaData加密数据的倍数
use_metadata = False
delete_edges = False

# AIC hyperparameter
aic_ratio = 1  # lower this ratio, less important is the number of elements to AIC value

def print_config():
    """打印配置信息 / Print configuration info"""
    print('use_metadata =', use_metadata)
    print('delete_edges =', delete_edges)
    print(path)
    print('fine_ratio = ', fine_ratio)
###########################################################################################
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

# Data
def divide(up, down, eta=1e-10):
    while np.any(down == 0):
        down += eta
    return up/down

def load_problem_data(problem_name):
    """
    [LEGACY] 加载指定问题的数据 / Load data for specified problem
    
    WARNING: This function is LEGACY and will be deprecated in future versions.
    Use kd.dataset.load_pde_dataset() instead for unified data loading.
    
    This legacy data loading approach is maintained for backward compatibility
    with existing SGA-PDE code, but new code should use the unified KD dataset API.
    """
    # 使用统一的数据目录 / Use unified data directory
    _data_dir = os.path.join(os.path.dirname(__file__), "../../../dataset/data")
    
    if problem_name == 'PDE_divide':
        u = np.load(os.path.join(_data_dir, "PDE_divide.npy")).T
        nx = 100
        nt = 251
        x = np.linspace(1, 2, nx)
        t = np.linspace(0, 1, nt)
        right_side = 'right_side = -config.divide(ux, x) + 0.25*uxx'
        left_side = 'left_side = ut'
        right_side_origin = 'right_side_origin = -config.divide(ux_origin, x_all) + 0.25*uxx_origin'
        left_side_origin = 'left_side_origin = ut_origin'
        
    elif problem_name == 'PDE_compound':
        u = np.load(os.path.join(_data_dir, "PDE_compound.npy")).T
        nx = 100
        nt = 251
        x = np.linspace(1, 2, nx)
        t = np.linspace(0, 0.5, nt)
        right_side = 'right_side = u*uxx + ux*ux'
        left_side = 'left_side = ut'
        right_side_origin = 'right_side_origin = u_origin*uxx_origin + ux_origin*ux_origin'
        left_side_origin = 'left_side_origin = ut_origin'
        
    elif problem_name == 'Burgers':
        data = scio.loadmat(os.path.join(_data_dir, "burgers.mat"))
        u = data.get("usol")
        x = np.squeeze(data.get("x"))
        t = np.squeeze(data.get("t").reshape(1, 201))
        right_side = 'right_side = -u*ux+0.1*uxx'
        left_side = 'left_side = ut'
        right_side_origin = 'right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin'
        left_side_origin = 'left_side_origin = ut_origin'
        
    elif problem_name == 'Kdv':
        data = scio.loadmat(os.path.join(_data_dir, "KdV.mat"))
        u = data.get("uu")
        x = np.squeeze(data.get("x"))
        t = np.squeeze(data.get("tt").reshape(1, 201))
        right_side = 'right_side = -0.0025*uxxx-u*ux'
        left_side = 'left_side = ut'
        right_side_origin = 'right_side_origin = -0.0025*uxxx_origin-u_origin*ux_origin'
        left_side_origin = 'left_side_origin = ut_origin'
        
    elif problem_name == 'chafee-infante':
        u = np.load(os.path.join(_data_dir, "chafee_infante_CI.npy"))
        x = np.load(os.path.join(_data_dir, "chafee_infante_x.npy"))
        t = np.load(os.path.join(_data_dir, "chafee_infante_t.npy"))
        right_side = 'right_side = - 1.0008*u + 1.0004*u**3'
        left_side = 'left_side = ut'
        right_side_origin = 'right_side_origin = uxx_origin-u_origin+u_origin**3'
        left_side_origin = 'left_side_origin = ut_origin'
        
    else:
        raise ValueError(f"Unknown problem: {problem_name}")
    
    return {
        'u': u, 'x': x, 't': t,
        'right_side': right_side, 'left_side': left_side,
        'right_side_origin': right_side_origin, 'left_side_origin': left_side_origin
    }

# 全局变量（向后兼容）/ Global variables (backward compatibility)
# 这些变量将在首次访问时延迟加载 / These variables will be lazy-loaded on first access
u = None
x = None
t = None
right_side = None
left_side = None
right_side_origin = None
left_side_origin = None

def ensure_data_loaded():
    """确保数据已加载 / Ensure data is loaded"""
    global u, x, t, right_side, left_side, right_side_origin, left_side_origin
    if u is None:
        data = load_problem_data(problem)
        u = data['u']
        x = data['x']
        t = data['t']
        right_side = data['right_side']
        left_side = data['left_side']
        right_side_origin = data['right_side_origin']
        left_side_origin = data['left_side_origin']
