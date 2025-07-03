
# kd

## 项目简介

**kd** 是一个面向科学计算与机器学习的开源框架，专注于偏微分方程（PDE）符号发现、数据驱动建模与高质量可视化。框架集成了强化学习、遗传算法、物理信息神经网络（PINN）等多种符号回归方法，支持灵活的数据加载、算子扩展和多维度可视化，适用于科研、工程和教学场景。

本项目的核心目标是为用户提供一站式 PDE 发现与符号建模工具，涵盖数据预处理、模型训练、表达式搜索、方程可视化与物理一致性分析等全流程。通过模块化设计，用户可以方便地扩展算子库、集成自定义数据集，并对发现的方程进行多维度可视化和物理解释。

## 文件结构与模块说明

```
kd/
├── base.py                  # BaseEstimator 基类，参数管理与递归配置
├── data.py                  # 数据处理基类与实现
├── dataset/                 # 内置与自定义数据集加载（如 Burgers, KdV）
├── model/                   # 主要模型与符号回归算法
│   ├── discover/            # 表达式树、算子、搜索等核心实现
│   ├── kd_dlga.py           # 遗传算法符号回归模型
│   ├── kd_dscv.py           # 强化学习符号回归模型
│   ├── kd_dscv_pinn.py      # PINN结合符号回归模型
├── utils/                   # 日志、配置、通用工具
├── viz/                     # 可视化与渲染模块
│   ├── dscv_viz.py          # KD_DSCV相关可视化
│   ├── dlga_viz.py          # KD_DLGA相关可视化
│   ├── equation_renderer.py # LaTeX渲染、表达式树等
│   └── ...                  # 其它可视化工具
├── test/                    # 单元测试
└── examples/                # 教程与示例
```

### 主要模块说明

- **dataset/**  
  内置 Burgers、KdV 等典型PDE数据集加载器，支持 `.csv`、`.mat` 格式和自定义数据导入，便于实验复现和算法对比。

- **model/**  
  包含 KD_DSCV（基于强化学习）、KD_DLGA（基于遗传算法）、KD_DSCV_Pinn（结合PINN）等主力模型，均支持符号回归与PDE发现。核心算法实现位于 `discover/` 子目录，支持灵活的算子库扩展和表达式搜索策略定制。

- **viz/**  
  提供表达式树、训练过程、残差分析等多种可视化接口，支持一键渲染LaTeX公式和高质量图片输出，助力模型结果的物理解释与学术展示。

- **utils/**  
  日志、配置、辅助函数等，提升开发与实验效率。

- **test/**  
  各模块单元测试，保障代码质量和功能正确性。

- **examples/**  
  包含典型 PDE 发现流程的脚本和 Jupyter Notebook，便于快速上手和复现论文结果。

### 基类与核心接口

- **BaseEstimator**：参数管理与模型基类，支持递归参数设置与获取，便于模型调参和自动化实验。
- **BaseDataHandler**：数据处理抽象基类，支持多格式数据加载、清洗与预处理，便于扩展自定义数据集。
- **BaseRL**：强化学习算法基础类，定义了模型拟合与预测的标准接口。
- **KD_DSCV / KD_DLGA / KD_DSCV_Pinn**：分别对应强化学习、遗传算法、PINN结合的符号回归模型，均支持灵活的算子库配置和表达式搜索。

## 安装依赖

在项目根目录下创建并激活虚拟环境，然后安装所需依赖：

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# env\Scripts\activate   # Windows
pip install -r requirements.txt
```

> **注意**：部分可视化功能依赖 [Graphviz](https://graphviz.gitlab.io/)，需提前安装系统依赖。
> 
> macOS: `brew install graphviz`
> 
> Ubuntu: `sudo apt install graphviz`

## 致谢

本项目参考和借鉴了以下优秀开源项目：
- https://github.com/menggedu/DISCOVER
- https://github.com/dso-org/deep-symbolic-optimization
- https://github.com/sympy/sympy
- https://github.com/lululxvi/deepxde
- https://github.com/MilesCranmer/PySR
