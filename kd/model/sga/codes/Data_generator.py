import numpy as np
import torch
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import random
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from . import configure as config

# === 修复导入副作用 / Fix Import Side Effects ===
# 将所有计算封装到函数中，避免导入时执行
# Encapsulate all computations in functions to avoid execution during import

# 全局变量（向后兼容）/ Global variables (backward compatibility)
u = None
x = None
t = None
x_all = None
n = None
m = None

def initialize_data_generator():
    """初始化数据生成器 / Initialize data generator"""
    global u, x, t, x_all, n, m
    
    # 确保配置数据已加载
    config.ensure_data_loaded()
    
    fine_ratio = config.fine_ratio
    normal = config.normal
    use_metadata = config.use_metadata
    
    seed = config.seed
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = config.ensure_device()
    
    # 获取原始数据
    u_orig = config.u
    x_orig = config.x
    t_orig = config.t
    n_orig, m_orig = u_orig.shape
    
    Y_raw = pd.DataFrame(u_orig.reshape(-1, 1))
    X1 = np.repeat(x_orig.reshape(-1, 1), m_orig, axis=1)
    X2 = np.repeat(t_orig.reshape(1, -1), n_orig, axis=0)
    X_raw_norm = pd.concat([pd.DataFrame(X1.reshape(-1, 1)), pd.DataFrame(X2.reshape(-1, 1))], axis=1, sort=False)

    # 处理数据
    if use_metadata == True:
        # load model
        hidden_dim = config.hidden_dim
        num_feature = config.num_feature
        model = config.Net(num_feature, hidden_dim, 1).to(device)
        model.load_state_dict(torch.load(config.path, map_location=device))

        # generate new data
        n_fine = fine_ratio * n_orig - 1
        m_fine = fine_ratio * m_orig - 1
        x_new = np.linspace(x_orig.min(), x_orig.max(), n_fine)
        t_new = np.linspace(t_orig.min(), t_orig.max(), m_fine)
        X1 = np.repeat(x_new.reshape(-1, 1), m_fine, axis=1)
        X2 = np.repeat(t_new.reshape(1, -1), n_fine, axis=0)
        X_raw_norm = pd.concat([pd.DataFrame(X1.reshape(-1, 1)), pd.DataFrame(X2.reshape(-1, 1))], axis=1, sort=False)
        if normal == True:
            X = ((X_raw_norm - X_raw_norm.mean()) / (X_raw_norm.std()))
        else:
            X = X_raw_norm
        y_pred = model(Variable(torch.from_numpy(X.values).float()).to(device))
        y_pred = y_pred.cpu().data.numpy().flatten()
        if normal == True:
            result_pred_real = y_pred * Y_raw.std()[0] + Y_raw.mean()[0]
        else:
            result_pred_real = y_pred
        u_new = result_pred_real.reshape(n_fine, m_fine)

        u_new2 = np.zeros([n_orig, m_orig])
        for i in range(n_orig):
            for j in range(m_orig):
                u_new2[i, j] = u_new[i * 2, j * 2]

        diff = (u_orig - u_new2) / u_new2

        print(np.max(diff))
        print(np.min(diff))
        print(np.mean(np.abs(diff)))
        print(np.median(np.abs(diff)))

        plt.figure(figsize=(5, 3))
        mm1 = plt.imshow(np.abs(diff), interpolation='nearest', cmap='Blues', origin='lower', vmax=0.05, vmin=0)
        plt.colorbar().ax.tick_params(labelsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title('Data error', fontsize=15)

        plt.figure(figsize=(5, 3))
        plt.hist(diff.reshape(-1, 1))

        plt.figure(figsize=(5, 3))
        x_index = np.linspace(0, 100, n_orig)
        x_index_fine = np.linspace(0, 100, n_fine)
        plt.plot(x_index, u_orig[:, int(m_orig / 2)])
        plt.plot(x_index_fine, u_new[:, int(m_fine / 2)])

        # 使用新数据
        u = u_new
        x = x_new
        t = t_new
    else:
        # 使用原始数据
        u = u_orig
        x = x_orig
        t = t_orig

    x_all = x
    # 提取指定区间内的MetaData数据
    if config.delete_edges == True:
        n_curr, m_curr = u.shape
        u = u[int(n_curr * 0.1):int(n_curr * 0.9), int(m_curr * 0):int(m_curr * 1)]
        x = x[int(n_curr * 0.1):int(n_curr * 0.9)]
        t = t[int(m_curr * 0):int(m_curr * 1)]

    # 更新全局变量
    n, m = u.shape

def ensure_data_generator_initialized():
    """确保数据生成器已初始化 / Ensure data generator is initialized"""
    global u, x, t, x_all, n, m
    if u is None:
        initialize_data_generator()
