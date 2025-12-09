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

from sgapde.config import SolverConfig, Net

def train_and_save_model(config: SolverConfig):

    print(f"--- Starting NN training for problem: {config.problem_name} ---")

    # --- 功能迁移：随机种子设置 ---
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    print(f"Random seed set to: {config.seed}")
    
    # Load data from configuration.
    # Note: this loading logic allows the script to run standalone.
    try:
        print(f"Loading data for problem: {config.problem_name}")
        if config.problem_name == 'chafee-infante':
            u = np.load(f"./data/chafee_infante_CI.npy")
            x = np.load(f"./data/chafee_infante_x.npy")
            t = np.load(f"./data/chafee_infante_t.npy")
        else:
            import scipy.io as scio
            data = scio.loadmat(f'./data/{config.problem_name}.mat')
            u = data.get("usol") or data.get("uu")
            x = np.squeeze(data.get("x"))
            t = np.squeeze(data.get("t") or data.get("tt"))
    except FileNotFoundError:
        print(
            "[MetaNN ERROR] Training data file not found "
            f"for problem '{config.problem_name}'."
        )
        return

    # 数据准备和归一化
    n, m = u.shape
    Y_raw = pd.DataFrame(u.reshape(-1, 1))
    X1 = np.repeat(x.reshape(-1, 1), m, axis=1)
    X2 = np.repeat(t.reshape(1, -1), n, axis=0)
    X_raw_df = pd.DataFrame(np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1))
    
    X = (X_raw_df - X_raw_df.mean()) / (X_raw_df.std())
    Y = (Y_raw - Y_raw.mean()) / (Y_raw.std())

    # --- 功能迁移：训练/测试集划分 ---
    def split(full_list, shuffle=False, ratio=0.8):
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1: return [], full_list
        if shuffle: random.shuffle(full_list)
        return full_list[:offset], full_list[offset:]

    total_ID = list(range(len(Y)))
    train_index, test_index = split(total_ID, shuffle=True, ratio=config.train_ratio)
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    print(f'Data split: {len(train_index)} for training, {len(test_index)} for testing.')

    # 训练模型
    model = Net(config.num_feature, config.hidden_dim, 1).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    x_train_tensor = torch.from_numpy(x_train.values).float().to(config.device)
    y_train_tensor = torch.from_numpy(y_train.values).float().to(config.device)
    
    print("Starting training...")
    for epoch in range(config.max_epoch):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % (config.max_epoch / 10) == 0:
            print(f"Epoch [{epoch+1}/{config.max_epoch}], Loss: {loss.item():.6f}")

    # 保存模型
    torch.save(model.state_dict(), config.model_path)
    print(f"--- Training finished. Model saved to {config.model_path} ---")

    # --- 功能迁移：模型性能评估 ---
    print("\nEvaluating model performance...")
    model.eval()
    y_train_pred = model(x_train_tensor).cpu().data.numpy()
    
    if len(test_index) > 0:
        x_test_tensor = torch.from_numpy(x_test.values).float().to(config.device)
        y_test_pred = model(x_test_tensor).cpu().data.numpy()
        y_test_numpy = y_test.values
    else:
        y_test_pred = np.array([])
        y_test_numpy = np.array([])

    def R2_score(y_true, y_pred):
        if len(y_true) == 0: return 0
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    rmse_train = np.sqrt(mean_squared_error(y_train.values, y_train_pred))
    r2_train = R2_score(y_train.values, y_train_pred)
    print(f"Training Set  -> RMSE: {rmse_train:.6f}, R-squared: {r2_train:.6f}")

    if len(test_index) > 0:
        rmse_test = np.sqrt(mean_squared_error(y_test_numpy, y_test_pred))
        r2_test = R2_score(y_test_numpy, y_test_pred)
        print(f"Test Set      -> RMSE: {rmse_test:.6f}, R-squared: {r2_test:.6f}")

        # 反归一化，还原到原始数据尺度
        y_test_real = y_test_numpy * Y_raw.std().values[0] + Y_raw.mean().values[0]
        y_pred_real = y_test_pred * Y_raw.std().values[0] + Y_raw.mean().values[0]


if __name__ == "__main__":
    # 为了让脚本可以独立运行，我们需要在这里创建一个配置实例
    # 可以修改这里的参数来训练不同的模型
    config = SolverConfig(
        problem_name='chafee-infante',
        max_epoch=100000,
        train_ratio=0.8 # e.g., 80% for training, 20% for testing
    )
    
    # 运行训练和评估
    train_and_save_model(config)
