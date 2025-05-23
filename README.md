# kd

## 项目简介

本项目是一个用于科学计算与机器学习的开源框架，专注于偏微分方程（PDE）求解、数据处理和模型算法开发。通过模块化设计，提供了灵活的数据加载与处理功能，支持用户使用内置数据集或自定义数据进行实验。同时，项目内置多个模型类，可用于构建深度学习、强化学习以及遗传算法等前沿模型。

## 文件结构

```
kd/
├── base.py                  # BaseEstimator 基类，用于参数管理
├── data.py                  # 数据处理
├── dataset                  # 内置数据集
│   ├── __init__.py
│   └── _bsase.py            # 数据类基本IO方法
├── model                    # 模型类
│   ├── discover             # discover实现
│   ├── __init__.py
│   ├── deeprl.py            # 深度强化学习模型类
│   └── dlga.py              # 遗传算法深度学习模型
├── test                     # 测试模块
├── utils                    # 工具模块
└── viz                      # 可视化模块
```

## 安装依赖

在项目根目录下创建并激活虚拟环境，然后安装所需依赖：

```bash
# 创建虚拟环境
python -m venv env
source env/bin/activate  # Linux/Mac
# env\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用指南

### 模块说明

#### 1. **核心模块**
- **`base.py`：参数管理模块**  
  提供 `BaseEstimator` 类，支持模型参数的递归管理、设置和获取，是所有模型的基础。

#### 2. **数据处理模块**
- **`data.py`：数据处理基类与实现**  
  定义了 `BaseDataHandler` 抽象基类，支持数据加载、清洗和处理流程。  
  - **`RegularData`**：处理内置数据集（如 Burgers、Kdv）。  
  - **`SparseData`**：针对稀疏数据集，提供采样和训练集划分功能。

- **`dataset/`**  
  提供数据 I/O 功能，支持 `.csv` 和 `.mat` 格式数据的加载。

#### 3. **模型模块**
- **`model/`**  
  包含不同类型的模型，如深度强化学习（`deeprl.py`）和遗传算法（`dlga.py`）。  
  模型设计基于模块化结构，便于扩展和定制。

#### 4. **测试与工具模块**
- **`test/`**：提供单元测试，确保模块功能的正确性。  
- **`utils/`**：辅助工具模块，为日志记录、可视化等功能提供支持。

#### 5. **可视化**
- **`vir/`**：提供模块化的绘图接口

### 基类说明

1. **BaseRL**：强化学习算法的基础类，定义了拟合和预测的抽象方法，需要在子类中实现。
    - `fit(X, y)`：拟合模型。
    - `predict(X)`：使用拟合的模型进行预测。
   
2. **DeepRL**：继承自 `BaseRL` 的强化学习具体实现类。包含 `_parameter` 属性用于存储参数配置。

3. **BaseEstimator**：用于参数管理的基类，包含以下主要方法：
    - `get_params()`：获取当前参数。
    - `set_params()`：设置参数。
    - `_get_param_names()`：获取构造方法的参数名。
   
4. **BaseDataHandler**：数据处理的抽象基类，支持从文件和 `DataFrame` 中加载数据，并提供数据清洗方法。
    - `load_data()`：根据提供的文件路径或 `DataFrame` 加载数据。
    - `clean_data()`：对加载的数据进行去重和缺失值处理。
    - `process_data()`：抽象方法，需在子类中实现特定数据处理逻辑。

### 代码示例

### 注释

需要 graphviz 被安装

macOS: `brew install graphviz`

ubuntu: `apt install graphviz`

### 致谢

我们感谢以下GitHub仓库提供了宝贵的代码：
1. https://github.com/menggedu/DISCOVER
2. https://github.com/dso-org/deep-symbolic-optimization
3. https://github.com/sympy/sympy
4. https://github.com/lululxvi/deepxde
5. https://github.com/MilesCranmer/PySR
