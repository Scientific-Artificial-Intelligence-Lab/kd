### **SGA-PDE 集成项目状态摘要 (v2.5)**

**0. 设计与架构说明 (Design & Architecture Overview)**

本项目将原始 SGA-PDE 脚本式代码重构为模块化、可测试、可集成的 Python 包，适配科学发现框架 kd。核心设计如下：

- **目录结构**  
  - `codes/`：原始 SGA-PDE 算法主代码，保持原有实现，便于回归验证。
  - `sga_refactored/`：重构后的高内聚模块，包括数据处理（data_utils.py）、符号库（operators.py）、诊断（diagnostics.py）、主流程（main.py）、适配器类（kd_sga.py）。
  - `tests/`：pytest 单元测试，覆盖所有重构模块，TDD 驱动开发。
- **核心模块说明**  
  - `data_utils.py`：统一数据预处理与导数计算，支持 finite_difference/autograd 两种模式。
  - `operators.py`：构建符号库，消除全局状态依赖，保证 shape 一致。
  - `diagnostics.py`：误差诊断，与 setup.py 完全等价，便于黄金标准回归。
  - `main.py`：重构主流程，负责依赖注入、流程调度。
  - `kd_sga.py`：面向 kd 框架的适配器类，支持参数注册、fit 主流程、依赖注入。
- **典型用法**  
  - 直接运行 `python -m sga_refactored.main` 可快速测试主流程。
  - 在 Python 代码中：  
    ```python
    from kd.model.kd_sga import KD_SGA
    model = KD_SGA(num=10, depth=3)
    best_eq, best_aic = model.fit(max_gen=5)
    ```
  - 单元测试：`pytest tests/`，所有核心模块均有覆盖。
- **设计原则**  
  - 保持原始 SGA 算法不变，所有 shape 适配、依赖注入均在外部完成，便于回归和集成。
  - TDD 驱动，所有重构均有黄金标准和自动化测试保障。

---

**1. 项目目标 (Project Goal)**

* **核心任务**: 将一个独立的、面向脚本的研究性代码项目（SGA-PDE），重构并集成到一个已有的、面向对象的工程框架（`kd`）中。
* **最终产物**: 创建一个符合`kd`框架设计规范的求解器类（`KD_SGA`），该类继承自框架的基类（`BaseGa`），并提供一个简洁、标准的API（`__init__` 用于配置，`fit` 用于执行）。用户应能像使用框架中其他模型一样，轻松地调用SGA-PDE的功能，而无需关心其内部复杂的实现。

**2. KD框架架构背景 (KD Framework Architectural Context)**

`kd`框架是一个为科学发现（Scientific Discovery）设计的、结构化的Python库。其架构深受`scikit-learn`的影响，旨在提供模块化、可复用和易于扩展的模型开发体验。
* **统一的基类 (`base.py`)**: 框架的核心是一个`BaseEstimator`基类，它为所有模型（Estimators）提供了统一的参数管理机制，包括`get_params`和`set_params`方法。这使得所有模型都具有一致的接口，便于进行自动化工作流（如超参数搜索）。
* **抽象接口定义 (`kd_dscv.py`, `_base.py`)**: 框架使用抽象基类（Abstract Base Classes, ABCs）来为特定类型的求解器定义“契约”或标准接口。例如，`BaseRL`或`BaseGa`类规定了所有基于强化学习或遗传算法的求解器都必须实现一个`fit`方法。
* **模块化目录结构**: 项目采用了清晰的、职责分离的目录结构，例如`/model`用于存放求解器逻辑，`/dataset`用于数据处理，`/viz`用于可视化。
* **统一的执行入口**: 所有求解器的核心训练/执行逻辑都被封装在`fit`方法中，为用户提供了一个简单而一致的调用入口。我们的集成工作必须严格遵守这些设计原则。

**3. 已完成的工作 (Completed Work)**

* **项目分析与架构设计**：完成 SGA-PDE 源码与论文分析，明确重构目标与集成策略。
* **核心模块重构**：完成 data_utils.py、operators.py、diagnostics.py、main.py、kd_sga.py 的实现，所有 shape 适配、依赖注入均在外部完成，原始算法零侵入。
* **依赖注入机制**：实现并测试 main.py 的 inject_dependencies，确保 tree.py、pde.py 等全局依赖可灵活注入。
* **KD_SGA 适配器类**：实现并测试 KD_SGA 类，支持参数注册、fit 主流程、依赖注入，接口与 kd 框架兼容。
* **diagnostics 测试**：重构 diagnostics.py 并通过严格单元测试，误差输出与 setup.py 完全一致。
* **finite_difference/autograd 测试**：data_utils、operators、diagnostics 等所有核心模块均已通过两种模式下的黄金标准单元测试。
* **包结构现代化**：codes/ 目录已标准化为 Python 包，便于集成与测试。
* **examples/kd_sga_example.py**：已实现最小可运行示例，能直接调用 KD_SGA 并用 SGA 自带数据跑通主流程。

**4. 未完成的工作 (Pending Tasks)**

* **与 kd 框架的正式集成与可视化兼容**
    - [ ] 在 kd/model/ 下完善 API 暴露与注册，确保 KD_SGA 能被主流程、pipeline、自动调参等机制自动发现和调用。
    - [ ] 集成测试：在 kd 项目主流程中调用 KD_SGA，验证能被 pipeline、自动调参等机制正常发现和使用。
    - [ ] 可视化接口 kd 化：将 SGA 的方程渲染、AIC 曲线等可视化接口与 kd.viz 体系兼容，支持 discover_eq2latex、plotter 等统一 API。
    - [ ] 方程渲染兼容：确保 SGA 发现的方程能被 discover_eq2latex、equation_renderer 等工具正确渲染为 LaTeX/可视化。
    - [ ] 补充 README/示例 notebook，演示 KD_SGA 的典型用法、参数配置和可视化。

---

**5. 当前遇到的核心问题 (Current Blocking Issues)**

* **无。** 所有 shape、依赖注入、autograd/finite_difference 测试均已通过，主流程稳定。

---

### 下一步小步任务建议

1. **API 注册与 discover 集成**：在 kd/model/ 下完善 KD_SGA 的 discover 注册与 API 兼容，确保能被 discover pipeline 自动发现。
2. **可视化接口适配**：将 SGA 的方程渲染、AIC 曲线等可视化接口与 kd.viz/discover_eq2latex 等兼容，支持统一的 discover 可视化 API。
3. **方程渲染兼容**：补充/适配 SGA 方程的 latex 渲染、符号化，确保 discover_eq2latex、equation_renderer 能正确处理 SGA 输出。
4. **文档与 notebook**：补充 README、notebook，演示 KD_SGA 的 discover 集成、参数配置和可视化。

---

**可补充建议**
- 统一 discover 相关模型的 fit/score/plot/latex API，便于 pipeline 自动化和文档生成。
- 增加 discover 任务的自动化测试，确保 SGA 与其它 discover 模型在 discover pipeline 下表现一致。
- 适配 kd 的 logging、结果保存、模型持久化等通用工具，提升工程一致性。
