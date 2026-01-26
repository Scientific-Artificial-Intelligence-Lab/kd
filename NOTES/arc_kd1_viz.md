# KD1 可视化模块 (kd.viz) 技术文档

> 供 KD2 开发组参考借鉴

---

## 目录

1. [模块概述](#1-模块概述)
2. [架构设计](#2-架构设计)
3. [核心数据结构](#3-核心数据结构)
4. [Facade核心实现](#4-facade核心实现)
5. [适配器系统](#5-适配器系统)
6. [数据契约](#6-数据契约)
7. [样式管理](#7-样式管理)
8. [高级API](#8-高级api)
9. [使用示例](#9-使用示例)
10. [设计模式与最佳实践](#10-设计模式与最佳实践)
11. [KD2借鉴建议](#11-kd2借鉴建议)

---

## 1. 模块概述

### 1.1 模块定位

`kd.viz` 是KD框架的统一可视化层，采用**Facade模式**隐藏多模型(SGA/DSCV/DLGA/PySR)的可视化差异，提供一致的高层API。

### 1.2 核心价值

| 特性 | 描述 |
|------|------|
| **一行调用** | `render_equation(model)` 自动路由到正确适配器 |
| **零异常设计** | 错误通过 `VizResult.warnings` 返回，流程不中断 |
| **可扩展** | 新模型只需实现 Adapter，无需修改核心代码 |
| **样式隔离** | 上下文管理器防止 Matplotlib 全局状态污染 |
| **能力发现** | `list_capabilities(model)` 暴露支持的操作 |

### 1.3 文件结构

```
kd/viz/
├── __init__.py              # 包入口，导出公共API
├── core.py                  # Facade核心：VizRequest/VizResult/render()
├── api.py                   # 高级Helper：plot_*(), render_*()
├── registry.py              # 适配器注册表
├── adapters.py              # 默认适配器注册工厂
├── _style.py                # 样式配置与上下文管理
├── _helpers.py              # 通用工具函数
├── _contracts.py            # 数据契约定义
├── equation_renderer.py     # LaTeX公式渲染
├── dlga_eq2latex.py         # DLGA方程→LaTeX转换
├── discover_eq2latex.py     # DSCV方程→LaTeX转换
├── dlga_viz.py              # DLGA遗留可视化（被适配器包装）
├── dscv_viz.py              # DSCV遗留可视化（被适配器包装）
└── _adapters/               # 模型特定适配器
    ├── __init__.py
    ├── sga.py               # SGA适配器 (465行)
    ├── dscv.py              # DSCV适配器 (547行)
    ├── dlga.py              # DLGA适配器 (708行)
    └── pysr.py              # PySR适配器
```

---

## 2. 架构设计

### 2.1 分层架构图

```
┌───────────────────────────────────────────────────────────┐
│                      用户代码                              │
│   kd.viz.render_equation(model)                           │
│   kd.viz.plot_residuals(model, actual, predicted)         │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│           高级API层 (api.py)                              │
│   plot_training_curve() / plot_residuals() / ...         │
│   将用户调用转换为 VizRequest                              │
└─────────────────────────┬─────────────────────────────────┘
                          │ VizRequest
                          ▼
┌───────────────────────────────────────────────────────────┐
│           Facade核心 (core.py - render())                 │
│   - 解析 VizRequest                                       │
│   - 查询适配器注册表                                       │
│   - 验证能力支持                                          │
│   - 应用样式上下文                                         │
│   - 返回标准化 VizResult                                   │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│           适配器注册表 (registry.py)                       │
│   _ADAPTERS: Dict[Type, VizAdapter]                       │
│   get_adapter() 支持继承链查找                             │
└─────────────────────────┬─────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   SGAAdapter       DSCVAdapter       DLGAAdapter
   (5种能力)        (9种能力)         (10种能力)
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│           标准化输出 (VizResult)                           │
│   - paths: 保存的文件列表                                  │
│   - figure: Matplotlib图形对象                            │
│   - warnings: 错误/警告信息                               │
│   - metadata: 结构化元数据                                │
└───────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
用户调用 → VizRequest → render() → registry.get_adapter()
         → adapter.render() → VizResult → 用户
```

---

## 3. 核心数据结构

### 3.1 VizRequest - 统一请求对象

**文件**: `core.py:24-27`

```python
@dataclass
class VizRequest:
    kind: str                    # 意图：'equation', 'residual', 'training_curve'等
    target: Any                  # 模型实例
    options: Dict[str, Any] = field(default_factory=dict)  # 可选参数
```

**设计要点**:
- `kind` 是操作类型的标识符，决定调用哪个处理函数
- `target` 是模型实例，用于查找对应的适配器
- `options` 是透传给适配器的自定义参数（figsize, dpi, cmap等）

### 3.2 VizResult - 统一返回对象

**文件**: `core.py:30-40`

```python
@dataclass
class VizResult:
    intent: str                           # 完成的意图
    figure: Any = None                    # Matplotlib图形对象（可选）
    paths: List[Path] = field(default_factory=list)    # 保存的文件路径
    warnings: List[str] = field(default_factory=list)  # 错误/警告信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 结果元数据

    @property
    def has_content(self) -> bool:
        """判断是否有实质内容"""
        return bool(self.figure) or bool(self.paths)
```

**设计要点**:
- **零异常**: 错误不抛异常，而是存入 `warnings` 列表
- **丰富元数据**: `metadata` 可存储统计信息、数据契约等
- `has_content` 属性便于快速检查是否成功

### 3.3 VizContext - 执行上下文

**文件**: `core.py:43-54`

```python
@dataclass
class VizContext:
    config: VizConfig           # 全局样式/输出配置
    backend: str                # 活动的绘图后端
    options: Dict[str, Any]     # 请求的自定义选项

    def save_path(self, filename: str) -> Path:
        """生成标准化输出路径"""
        return resolve_output_path(filename, self.config.save_dir)

    @property
    def style(self) -> Dict[str, Any]:
        return self.config.style
```

**设计要点**:
- 封装所有执行时需要的配置信息
- `save_path()` 方法统一处理输出路径

---

## 4. Facade核心实现

### 4.1 render() 函数

**文件**: `core.py:66-97`

这是整个Facade的核心入口：

```python
def render(request: VizRequest, *, backend: str = 'matplotlib') -> VizResult:
    # 1. 从注册表获取适配器
    adapter = registry.get_adapter(request.target)

    # 2. 处理后端配置
    requested_backend = None if backend == 'matplotlib' else backend
    config = get_config()
    if requested_backend and requested_backend != config.backend:
        configure_style(backend=requested_backend)
        config = get_config()

    # 3. 构建执行上下文
    active_backend = requested_backend or config.backend or 'matplotlib'
    ctx = VizContext(config=config, backend=active_backend, options=request.options)

    # 4. 验证适配器存在
    if adapter is None:
        return VizResult(
            intent=request.kind,
            warnings=[f"No visualization adapter registered for {type(request.target).__name__}."],
        )

    # 5. 验证意图支持
    capabilities: Iterable[str] = getattr(adapter, 'capabilities', [])
    if request.kind not in capabilities:
        return VizResult(
            intent=request.kind,
            warnings=[f"Adapter {adapter.__class__.__name__} does not support intent '{request.kind}'."],
            metadata={'capabilities': sorted(capabilities)},
        )

    # 6. 在样式上下文中执行
    with style_context(request.options.get('style')):
        return adapter.render(request, ctx)
```

**关键逻辑**:

1. **适配器查找**: 根据 `request.target` 的类型查找注册的适配器
2. **后端配置**: 支持动态切换 Matplotlib 后端
3. **能力验证**: 检查适配器是否支持请求的意图
4. **样式隔离**: 使用上下文管理器隔离样式变更
5. **统一返回**: 无论成功失败都返回 `VizResult`

### 4.2 list_capabilities() 函数

**文件**: `core.py:100-104`

```python
def list_capabilities(target: Any) -> Iterable[str]:
    """发现模型支持的可视化能力"""
    adapter = registry.get_adapter(target)
    if adapter is None:
        return []
    return tuple(adapter.capabilities)
```

---

## 5. 适配器系统

### 5.1 VizAdapter 协议

**文件**: `registry.py:8-12`

```python
class VizAdapter(Protocol):
    """所有适配器必须实现的接口"""
    capabilities: Iterable[str]  # 支持的意图集合

    def render(self, request, ctx):  # pragma: no cover
        ...
```

### 5.2 注册表实现

**文件**: `registry.py:15-49`

```python
_ADAPTERS: Dict[Type[Any], VizAdapter] = {}  # 全局注册表

def register_adapter(model_cls: Type[Any], adapter: VizAdapter) -> None:
    """注册模型类与适配器的映射"""
    _ADAPTERS[model_cls] = adapter

def unregister_adapter(model_cls: Type[Any]) -> None:
    """注销适配器"""
    _ADAPTERS.pop(model_cls, None)

def clear_registry() -> None:
    """清空注册表（主要用于测试）"""
    _ADAPTERS.clear()

def get_adapter(target: Any) -> Optional[VizAdapter]:
    """智能查找适配器，支持继承链"""
    if target is None:
        return None
    model_cls = target if isinstance(target, type) else type(target)

    # 1. 精确匹配
    adapter = _ADAPTERS.get(model_cls)
    if adapter is not None:
        return adapter

    # 2. 继承链查找
    for registered_cls, registered_adapter in _ADAPTERS.items():
        try:
            if issubclass(model_cls, registered_cls):
                return registered_adapter
        except TypeError:
            continue
    return None

def iter_registered() -> Iterable[tuple[Type[Any], VizAdapter]]:
    """遍历所有注册的适配器"""
    return tuple(_ADAPTERS.items())
```

**设计亮点**:
- 支持继承链查找，子类自动继承父类的适配器
- `clear_registry()` 便于测试隔离
- `iter_registered()` 支持调试和诊断

### 5.3 自动注册机制

**文件**: `adapters.py:15-43`

```python
def register_default_adapters() -> None:
    """启动时自动注册所有内置适配器"""

    # DLGA适配器
    try:
        from kd.model.kd_dlga import KD_DLGA
        register_adapter(KD_DLGA, DLGAVizAdapter())
    except Exception:
        pass  # 优雅降级

    # DSCV适配器
    try:
        from kd.model.kd_dscv import KD_DSCV
        register_adapter(KD_DSCV, DSCVVizAdapter())
    except Exception:
        pass

    # SGA适配器
    try:
        from kd.model.kd_sga import KD_SGA
        register_adapter(KD_SGA, SGAVizAdapter())
    except Exception:
        pass

    # PySR适配器（可选依赖）
    try:
        from kd.model.kd_pysr import KD_PySR
        register_adapter(KD_PySR, PySRVizAdapter())
    except Exception:
        pass
```

**设计亮点**:
- **懒注册**: 只有模型模块可用时才注册
- **优雅降级**: 可选依赖缺失不会导致导入失败

---

## 6. 数据契约

### 6.1 契约设计理念

数据契约是适配器与渲染器之间的标准化数据结构，实现：
- 跨模型统一数据格式
- 类型安全（`__post_init__` 中验证）
- 自文档化

**文件**: `_contracts.py`

### 6.2 ResidualPlotData

```python
@dataclass
class ResidualPlotData:
    actual: np.ndarray                    # 实际值
    predicted: np.ndarray                 # 预测值
    residuals: np.ndarray                 # actual - predicted
    input_coordinates: Optional[np.ndarray] = None  # 可选：样本坐标
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 类型转换与验证
        self.actual = np.asarray(self.actual)
        self.predicted = np.asarray(self.predicted)
        self.residuals = np.asarray(self.residuals)

        if self.actual.shape != self.predicted.shape:
            raise ValueError("'actual' and 'predicted' must have identical shapes")
        if self.actual.shape != self.residuals.shape:
            raise ValueError("'residuals' must match the shape of 'actual'")

    @classmethod
    def from_actual_predicted(cls, actual, predicted, *,
                             input_coordinates=None, metadata=None):
        """便捷工厂方法"""
        actual_arr = np.asarray(actual)
        pred_arr = np.asarray(predicted)
        residuals = actual_arr - pred_arr
        return cls(actual=actual_arr, predicted=pred_arr,
                  residuals=residuals, ...)
```

### 6.3 FieldComparisonData

```python
@dataclass
class FieldComparisonData:
    x_coords: np.ndarray           # 空间坐标 shape=(nx,)
    t_coords: np.ndarray           # 时间坐标 shape=(nt,)
    true_field: np.ndarray         # 真实场 shape=(nx, nt)
    predicted_field: np.ndarray    # 预测场 shape=(nx, nt)
    residual_field: Optional[np.ndarray] = None  # 自动计算
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 验证形状一致性
        expected_shape = (self.x_coords.size, self.t_coords.size)
        if self.true_field.shape != expected_shape:
            raise ValueError("Field data must have shape (len(x_coords), len(t_coords))")

        # 自动计算残差场
        if self.residual_field is None:
            self.residual_field = self.true_field - self.predicted_field
```

### 6.4 其他契约

| 契约类 | 用途 |
|-------|------|
| `OptimizationHistoryData` | 优化历史（steps, objective, complexity） |
| `TimeSliceComparisonData` | 时间切片对比 |
| `TermContribution` | 单个项的贡献（label, values, coefficient） |
| `TermRelationshipData` | 导数项关系数据 |
| `ParityPlotData` | 奇偶图数据 |
| `RewardEvolutionData` | 奖励进化数据（用于RL方法） |

---

## 7. 样式管理

### 7.1 VizConfig

**文件**: `_style.py:12-28`

```python
DEFAULT_STYLE: Dict[str, Any] = {
    'font.size': 12,
    'figure.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}

@dataclass
class VizConfig:
    style: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_STYLE))
    save_dir: Optional[Path] = None
    backend: Optional[str] = None

_CONFIG = VizConfig()  # 全局单例
```

### 7.2 configure() 函数

```python
def configure(*, style=None, save_dir=None, backend=None) -> VizConfig:
    """全局配置入口"""
    if style is not None:
        _CONFIG.style = dict(DEFAULT_STYLE)
        _CONFIG.style.update(style)
    if save_dir is not None:
        _CONFIG.save_dir = Path(save_dir)
    if backend is not None:
        matplotlib.use(backend, force=True)
        _CONFIG.backend = backend
    return _CONFIG
```

### 7.3 style_context() 上下文管理器

```python
@contextmanager
def style_context(extra_style: Optional[Dict[str, Any]] = None):
    """临时应用样式，退出时自动恢复"""
    original = matplotlib.rcParams.copy()  # 备份原始配置
    try:
        combined = dict(_CONFIG.style)
        if extra_style:
            combined.update(extra_style)
        matplotlib.rcParams.update(combined)
        yield
    finally:
        matplotlib.rcParams.update(original)  # 恢复原始配置
```

**解决的问题**:
- Matplotlib 全局 rcParams 被不同模块污染
- 并发场景下的样式冲突
- 单次请求的临时样式覆盖

---

## 8. 高级API

### 8.1 API函数列表

**文件**: `api.py`

| 函数 | 意图 | 用途 |
|-----|------|------|
| `render_equation()` | equation | 渲染发现的方程为LaTeX图像 |
| `plot_training_curve()` | training_curve | 绘制训练损失曲线 |
| `plot_validation_curve()` | validation_curve | 绘制验证损失曲线 |
| `plot_search_evolution()` | search_evolution | 可视化搜索/演化历史 |
| `plot_optimization()` | optimization | 可视化优化历史 |
| `plot_residuals()` | residual | 绘制残差分析 |
| `plot_field_comparison()` | field_comparison | 对比真实与预测的2D场 |
| `plot_time_slices()` | time_slices | 绘制特定时刻的场切片 |
| `plot_derivative_relationships()` | derivative_relationships | 绘制导数项关系 |
| `plot_parity()` | parity | 绘制奇偶图（预测vs实际） |

### 8.2 内部分发机制

```python
def _dispatch(kind: str, model: Any, *, show_info: bool,
             options: Dict[str, Any]) -> VizResult:
    """统一分发函数"""
    request = VizRequest(kind=kind, target=model, options=options)
    result = render(request)
    if show_info:
        _emit_info(kind, result)
    return result

def _emit_info(kind: str, result: VizResult) -> None:
    """输出结果信息"""
    if result.warnings:
        print(f"[kd.viz.{kind}] warning: {'; '.join(result.warnings)}")
        return
    if result.paths:
        joined = ', '.join(str(path) for path in result.paths)
        print(f"[kd.viz.{kind}] saved: {joined}")
    elif result.figure is not None:
        print(f"[kd.viz.{kind}] figure ready")
    else:
        print(f"[kd.viz.{kind}] produced no output")
```

### 8.3 参数透传机制

```python
def plot_residuals(model, *, actual, predicted, coordinates=None,
                  show_info=True, **options) -> VizResult:
    """绘制残差分析"""
    payload = dict(options)
    payload['actual'] = actual
    payload['predicted'] = predicted
    if coordinates is not None:
        payload['coordinates'] = coordinates
    return _dispatch('residual', model, show_info=show_info, options=payload)
```

**设计要点**:
- 必需参数显式声明（actual, predicted）
- 可选参数通过 `**options` 透传
- `show_info` 控制是否打印结果信息

---

## 9. 使用示例

### 9.1 最简用法

```python
from kd.viz import configure, render_equation, plot_residuals

# 1. 配置输出目录
configure(save_dir="artifacts/")

# 2. 一行调用 - 自动路由到正确适配器
result = render_equation(model)

# 3. 检查结果
if result.warnings:
    print(f"Warning: {result.warnings[0]}")
else:
    print(f"Saved to: {result.paths[0]}")
```

### 9.2 高级用法

```python
from kd.viz import VizRequest, render, list_capabilities, configure

# 1. 发现模型能力
caps = list_capabilities(model)
print(f"Capabilities: {', '.join(sorted(caps))}")

# 2. 自定义请求
request = VizRequest(
    kind='residual',
    target=model,
    options={
        'figsize': (12, 6),
        'dpi': 150,
        'cmap': 'coolwarm',
        'bins': 50,
        'actual': actual_data,
        'predicted': predicted_data,
    }
)

# 3. 显式渲染
result = render(request, backend='Agg')

# 4. 处理结果
if result.has_content:
    print(f"Generated {len(result.paths)} file(s)")
    if 'summary' in result.metadata:
        print(f"Mean residual: {result.metadata['summary']['mean']}")
```

### 9.3 完整工作流示例

```python
from pathlib import Path
from kd.dataset import load_pde
from kd.model import KD_DSCV
from kd.viz import VizRequest, configure, list_capabilities, render

# 数据加载与模型训练
dataset = load_pde('burgers')
model = KD_DSCV(binary_operators=["add", "mul", "diff"])
model.fit_dataset(dataset)

# 配置Facade
output_root = Path('artifacts/dscv_viz')
configure(save_dir=output_root)

# 发现能力
caps = list_capabilities(model)
print(f"DSCV capabilities: {', '.join(sorted(caps))}")
# 输出: search_evolution, density, tree, equation, residual, ...

# 按意图调用
for intent in ['search_evolution', 'equation', 'residual', 'parity']:
    result = render(VizRequest(intent, model, options={'output_dir': output_root}))
    if result.warnings:
        print(f"[{intent}] warning: {result.warnings[0]}")
    else:
        print(f"[{intent}] saved: {result.paths[0]}")
```

---

## 10. 设计模式与最佳实践

### 10.1 采用的设计模式

| 模式 | 应用位置 | 价值 |
|------|---------|------|
| **Facade** | `render()` 函数 | 统一入口，隐藏复杂性 |
| **Adapter** | `_adapters/*.py` | 统一不同模型的接口 |
| **Registry** | `registry.py` | 松耦合的插件式架构 |
| **Protocol** | `VizAdapter` | 类型安全的接口定义 |
| **Factory Method** | `from_actual_predicted()` | 便捷的对象创建 |
| **Context Manager** | `style_context()` | 资源/状态的安全管理 |

### 10.2 错误处理策略

```python
# ❌ 传统方式：抛异常
def render_bad(request):
    adapter = get_adapter(request.target)
    if adapter is None:
        raise ValueError("No adapter registered")  # 中断调用方
    return adapter.render(request)

# ✅ Facade方式：返回警告
def render_good(request):
    adapter = get_adapter(request.target)
    if adapter is None:
        return VizResult(
            intent=request.kind,
            warnings=["No adapter registered"]  # 不中断，可继续
        )
    return adapter.render(request)
```

### 10.3 适配器实现模板

```python
class NewModelVizAdapter:
    """新模型的可视化适配器模板"""

    capabilities = {'equation', 'residual', 'custom_plot'}

    def render(self, request: VizRequest, ctx: VizContext) -> VizResult:
        handlers = {
            'equation': self._equation,
            'residual': self._residual,
            'custom_plot': self._custom_plot,
        }

        handler = handlers.get(request.kind)
        if handler is None:
            return VizResult(
                intent=request.kind,
                warnings=[f"Unsupported intent: {request.kind}"]
            )

        return handler(request.target, ctx)

    def _equation(self, model, ctx) -> VizResult:
        # 1. 从模型提取数据
        latex = model.get_equation_latex()

        # 2. 渲染
        path = ctx.save_path('equation.png')
        render_latex_to_image(latex, path)

        # 3. 返回标准结果
        return VizResult(intent='equation', paths=[path],
                        metadata={'latex': latex})
```

### 10.4 各适配器能力矩阵

| Intent | SGA | DSCV | DLGA | PySR |
|--------|-----|------|------|------|
| equation | ✓ | ✓ | ✓ | ✓ |
| residual | ✓ | ✓ | ✓ | ✓ |
| parity | ✓ | ✓ | ✓ | ✓ |
| field_comparison | ✓ | ✓ | ✓ | - |
| time_slices | ✓ | - | ✓ | - |
| training_curve | - | - | ✓ | - |
| validation_curve | - | - | ✓ | - |
| search_evolution | - | ✓ | ✓ | - |
| optimization | - | - | ✓ | - |
| density | - | ✓ | - | - |
| tree | - | ✓ | - | - |
| derivative_relationships | - | - | ✓ | - |
| spr_residual | - | ✓ | - | - |
| spr_field_comparison | - | ✓ | - | - |

---

## 11. KD2借鉴建议

### 11.1 可直接复用的模式

1. **VizRequest/VizResult/VizContext 三件套**
   - 统一的请求-响应模型
   - 零异常设计
   - 丰富的元数据支持

2. **适配器注册表**
   - 支持继承链查找
   - 懒注册机制
   - 运行时动态注册/注销

3. **样式隔离**
   - 上下文管理器模式
   - 全局配置单例
   - 临时覆盖支持

4. **数据契约系统**
   - 标准化数据结构
   - `__post_init__` 验证
   - 工厂方法便捷创建

### 11.2 可改进的方向

1. **异步支持**
   ```python
   async def render_async(request: VizRequest) -> VizResult:
       # 用于大规模批量可视化
       ...
   ```

2. **缓存机制**
   ```python
   @lru_cache(maxsize=100)
   def render_latex_cached(latex: str, **kwargs) -> bytes:
       # 缓存昂贵的LaTeX渲染结果
       ...
   ```

3. **流式输出**
   ```python
   def render_streaming(request) -> Iterator[VizResult]:
       # 用于长时间运行的可视化任务
       for partial_result in process():
           yield partial_result
   ```

4. **更丰富的契约**
   ```python
   @dataclass
   class MultiFieldData:  # 多场数据
       fields: Dict[str, np.ndarray]
       coordinates: Dict[str, np.ndarray]

   @dataclass
   class SparsePointData:  # 稀疏点数据
       points: np.ndarray
       values: np.ndarray
   ```

### 11.3 架构优势总结

| 方面 | KD1实现 | 价值 |
|------|---------|------|
| 单一责任 | Facade编排，Adapter实现 | 职责清晰 |
| 开闭原则 | 新模型加Adapter即可 | 易扩展 |
| 依赖反转 | 用户→Facade→Registry→Adapter | 松耦合 |
| 错误恢复 | warnings而非异常 | 流程不中断 |
| 并发安全 | style_context隔离 | 无污染 |
| 可观测性 | VizResult.metadata | 便于调试 |
| 向后兼容 | 旧模块被Adapter包装 | 平滑迁移 |

---

## 附录：关键文件速查

| 需求 | 查看文件 |
|------|---------|
| Facade核心逻辑 | `kd/viz/core.py` |
| 高级API函数 | `kd/viz/api.py` |
| 适配器注册表 | `kd/viz/registry.py` |
| 样式管理 | `kd/viz/_style.py` |
| 数据契约定义 | `kd/viz/_contracts.py` |
| SGA适配器实现 | `kd/viz/_adapters/sga.py` |
| DSCV适配器实现 | `kd/viz/_adapters/dscv.py` |
| DLGA适配器实现 | `kd/viz/_adapters/dlga.py` |
| LaTeX渲染器 | `kd/viz/equation_renderer.py` |
| 使用示例 | `examples/kd_dscv_viz_api_example.py` |
| 单元测试 | `tests/test_viz_core.py` |
