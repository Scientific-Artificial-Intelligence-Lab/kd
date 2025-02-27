# Plot System Roadmap

## DONE

### Core Infrastructure
- Base System (`BasePlot`, `CompositePlot`)
- Monitor System (数据收集与管理)
- Hook System (非侵入式监控)

### Scientific Components
- Residual Analysis
- Comparison Analysis
- Equation Analysis
- Evolution Visualization

### Interface Layer
- Direct Plot API
- DLGA Model Integration

## DOING

### 1. DeepRL Integration [重要]
- [ ] DeepRL adapter 实现
- [ ] Training metrics 收集
- [ ] Evolution data 适配
- [ ] Visualization components 扩展

### 2. DLGA Integration
- [ ] evolution plot 还有点问题
- [ ] Direct Plot Interface 是否完全实现了?

### 3. vizr 移植 和 删除
- [ ] vizr 移植
- [ ] 删除 vizr 依赖

### 4. Scientific Components
- [ ] 残差图
- [ ] 左右项云图
- [ ] 优化过程图
- [ ] 树结构图
- [ ] 演化图
- [ ] metadata的取值平面
- [ ] 45度线
- [ ] 有噪音的时候的散点图


### 5. Documentation
- [ ] tutorials
- [ ] 重新生成 doc

## TODO

### Plotter Interface
计划中的高级接口，将提供:
- 自动数据预处理和格式转换
- 统一的配置管理
- 更简洁的 API
- 内置最佳实践?

### Model Support
- 新模型 adapters
- Custom model support
- Unified adapter interface

### Research Features
- Publication plots
- Result comparison tools
- Batch processing
- Custom analysis components 

### Performance Enhancement
- [ ] Large dataset 处理优化
- [ ] Memory usage 优化

## Future Work

1. 用神经网络做metadata/data generator，这个对于处理稀疏高噪音数据有帮助。以及在这个data generator中结合PINN实现物理信息嵌入。
2. 自动微分机制计算方程的精准度。
3. 评估标准，从MSE的精准度，到AIC/BIC的简洁性，到PIC的物理性，到RDISCOVER的值域空间一致性。
4. 优化器，从lasso和stridge的系数回归，到遗传算法，到RL，到LLM。
5. 表示方法，从封闭候选集，到基因片段表示方法，到符号数学表示方法。
6. 方程结构，从单一方程到方程组（多智能体优化），再到层级方程组（TLC化学方程）
7. 系数的确定方法，方程项外的直接回归就行，方程项内部的两种处理方法，一种是直接给一些基础运算符（比如1,2,5，可以组合出来常见的系数），另一种是用牛顿迭代来求系数。
8. 基础运算符，比如梯度，积分 等等