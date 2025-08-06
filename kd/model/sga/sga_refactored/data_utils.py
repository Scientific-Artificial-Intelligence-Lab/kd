import numpy as np
import torch
from torch.autograd import Variable

# 从原始项目中导入神经网络的定义
# 这是 autograd 模式所必需的
# 没有 from sga_refactored.xxx import ...，无需修正


def FiniteDiff(u, dx):
    """一阶有限差分，带边界处理。"""
    n = u.size
    ux = np.zeros(n)

    for i in range(1, n - 1):
        ux[i] = (u[i + 1] - u[i - 1]) / (2 * dx)

    ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
    ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
    return ux


def FiniteDiff2(u, dx):
    """二阶有限差分，带边界处理。"""
    n = u.size
    ux = np.zeros(n)

    for i in range(1, n - 1):
        ux[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2

    ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
    ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
    return ux


def Diff(u, dxt, name):
    """
    Here dx is a scalar, name is a str indicating what it is
    在一个2D场上沿指定轴应用一阶差分。
    """
    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        for i in range(m):
            uxt[:, i] = FiniteDiff(u[:, i], dxt)
    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff(u[i, :], dxt)
    else:
        raise NotImplementedError(f"Axis '{name}' not supported for Diff.")
    return uxt


def Diff2(u, dxt, name):
    """在一个2D场上沿指定轴应用二阶差分。"""
    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        for i in range(m):
            uxt[:, i] = FiniteDiff2(u[:, i], dxt)
    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff2(u[i, :], dxt)
    else:
        raise NotImplementedError(f"Axis '{name}' not supported for Diff2.")
    return uxt


def derivatives_autograd(u, t, x, model_path, device):
    """
    使用预训练的PyTorch模型和自动微分来计算导数。
    """
    print("   -> Using autograd mode for derivative calculation.")
    n_x, n_t = u.shape

    # 加载预训练的神经网络模型
    model = Net(num_feature, hidden_dim, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 准备输入数据 (将2D网格展平为1D向量以输入网络)
    x_flat = x.reshape(-1, 1)
    t_flat = t.reshape(-1, 1)
    
    # 将 x 和 t 平铺成网格，再展平
    x_grid = np.tile(x_flat, (1, n_t))
    t_grid = np.tile(t_flat.T, (n_x, 1))
    
    x_tensor = torch.from_numpy(x_grid.flatten()).float().to(device).requires_grad_(True)
    t_tensor = torch.from_numpy(t_grid.flatten()).float().to(device).requires_grad_(True)
    
    database = torch.stack([x_tensor, t_tensor], dim=1)

    # 使用 autograd 计算导数
    u_pred = model(database)
    
    # torch.autograd.grad 需要标量输出，所以我们对 u_pred.sum() 求导
    grad_outputs = torch.ones_like(u_pred)
    grad_u = torch.autograd.grad(u_pred, database, grad_outputs=grad_outputs, create_graph=True)[0]
    u_x = grad_u[:, 0]
    u_t = grad_u[:, 1]
    
    # grad_outputs shape 必须与 outputs 完全一致
    grad_outputs_flat = torch.ones_like(u_x).view(-1)
    grad_u_x = torch.autograd.grad(u_x, database, grad_outputs=grad_outputs_flat, create_graph=True)[0]
    u_xx = grad_u_x[:, 0]
    
    grad_outputs_flat2 = torch.ones_like(u_xx).view(-1)
    grad_u_xx = torch.autograd.grad(u_xx, database, grad_outputs=grad_outputs_flat2, create_graph=True)[0]
    u_xxx = grad_u_xx[:, 0]

    # 将计算结果从Tensor转回Numpy数组，并恢复原始形状
    ut = u_t.detach().cpu().numpy().reshape(n_x, n_t)
    ux = u_x.detach().cpu().numpy().reshape(n_x, n_t)
    uxx = u_xx.detach().cpu().numpy().reshape(n_x, n_t)
    uxxx = u_xxx.detach().cpu().numpy().reshape(n_x, n_t)
    
    return ut, ux, uxx, uxxx

def prepare_workspace(u, t, x, mode='finite_difference', model_path=None, device=None):
    """
    接收原始数据，计算所有导数，并返回一个包含所有工作区数据的字典。
    这个函数取代了原始 setup.py 中的数据处理和导数计算部分。
    """
    n_x, n_t = u.shape
    if n_x != len(x) or n_t != len(t):
        raise ValueError("Shape mismatch: u.shape must be (len(x), len(t)).")

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    if mode == 'finite_difference':
        print("   -> Using finite_difference mode for derivative calculation.")
        # 计算导数
        ut = Diff(u, dt, 't')
        ux = Diff(u, dx, 'x')
        uxx = Diff(ux, dx, 'x') 
        uxxx = Diff(uxx, dx, 'x')

    elif mode == 'autograd':
        if model_path is None or device is None:
            raise ValueError("Parameters 'model_path' and 'device' are required for autograd mode.")
        ut, ux, uxx, uxxx = derivatives_autograd(u, t, x, model_path, device)   

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Choose 'finite_difference' or 'autograd'.")

    workspace = {
        'u': u, 't': t, 'x': x,
        'ut': ut, 'ux': ux, 'uxx': uxx, 'uxxx': uxxx,
        'dx': dx, 'dt': dt
    }
    return workspace
