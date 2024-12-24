# 项目名称

## 项目简介


- `BaseRL`: 强化学习的基类，包含模型拟合和预测的基础结构。
- `DeepRL`: 继承自 `BaseRL` 的具体实现类，主要用于深度强化学习模型的构建。
- `BaseEstimator`: 基础的估计器类，包含获取和设置参数的基本方法。
- `BaseDataHandler`: 数据处理的基类，支持从文件或数据框加载数据，并提供数据清洗和处理方法。

## 文件结构

```
kd/
├── base.py                  # BaseEstimator 基类，用于参数管理
├── data.py                  # 数据处理
├── dataset                  # 内置数据集
├── model                    # 模型类
│   ├── __init__.py
│   ├── deeprl.py
│   └── dlga.py
├── test
└── utils
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
