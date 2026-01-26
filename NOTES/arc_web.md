# kd2 Web 兼容性设计

> **关联文档**：[arc_final.md](arc_final.md) - 主文档

---

## 一、设计目标

为未来 Web 界面做好架构准备：
- Web 端绘图渲染
- Web 端参数输入与配置
- 实时进度推送

---

## 二、核心原则：零 UI 依赖

```
kd2/
├── src/kd2/              # 核心包：绝对不依赖 CLI/Web
│   ├── core/
│   ├── data/
│   ├── plugins/
│   └── experiment/
│
├── src/kd2_cli/          # CLI 入口（可选包）
│   └── main.py
│
└── src/kd2_api/          # API 入口（未来 Web 用）
    ├── routes.py         # REST API
    └── websocket.py      # 实时推送
```

**关键约束**：`kd2/` 核心包中禁止 import：
- `click`, `typer`, `argparse` (CLI 框架)
- `fastapi`, `flask`, `gradio` (Web 框架)
- `matplotlib.pyplot` (直接绘图)

---

## 三、结果可序列化（Pydantic DTO）

所有结果类使用 Pydantic BaseModel，支持 JSON 序列化。

```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class EvaluationResultDTO(BaseModel):
    """评估结果数据传输对象"""
    mse: float
    nmse: float
    r2: float
    complexity: float
    term_count: int
    depth: int
    length: int
    aic: float
    coefficients: List[float]
    is_valid: bool
    invalid_reason: Optional[str] = None

class SearchResultDTO(BaseModel):
    """搜索结果数据传输对象"""
    best_expression: str
    best_latex: str              # LaTeX 渲染用
    evaluation: EvaluationResultDTO
    generation: int
    total_evaluated: int
    runtime_seconds: float
```

**内部结果 → DTO 转换**：

```python
class EvaluationResult:
    """内部结果"""
    mse: float
    # ...

    def to_dto(self) -> EvaluationResultDTO:
        """转换为可序列化 DTO"""
        return EvaluationResultDTO(
            mse=self.mse,
            nmse=self.nmse,
            # ...
        )
```

---

## 四、可视化系统（Facade + Registry）

**设计来源**：借鉴 kd1 的成熟设计，采用 Facade + Adapter + Registry 模式。

### 4.1 核心架构

```
用户调用
    │
    ▼
┌─────────────┐
│   Facade    │  ← 统一入口 render(target, intent)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Registry   │  ← 根据 target 类型查找适配器
└──────┬──────┘
       │
   ┌───┴───┬─────────┐
   ▼       ▼         ▼
 Core    SGA扩展   DISCOVER扩展   ← 适配器层
   │       │         │
   └───────┴─────────┘
           │
           ▼
┌─────────────────────┐
│  VizResult          │  ← 统一返回（双模式）
└─────────────────────┘
```

### 4.2 统一入口 API

```python
def render(target: Any, intent: str, **options) -> VizResult:
    """
    统一可视化入口

    Args:
        target: 要可视化的对象（ResultArchive, SearchResult, etc.）
        intent: 可视化意图（"pareto", "equation", "residual", etc.）
        **options: 可视化选项

    Returns:
        VizResult: 包含 figure + data + warnings
    """
    adapter = registry.get(type(target), intent)
    return adapter.render(target, **options)

def list_capabilities(target: Any) -> List[str]:
    """
    能力发现：查询某个对象支持哪些可视化

    Example:
        >>> list_capabilities(result_archive)
        ['pareto', 'equation', 'top_k']
    """
    return registry.capabilities(type(target))
```

### 4.3 核心数据结构

```python
@dataclass
class VizResult:
    """可视化结果（双模式输出）"""
    ok: bool                                    # 是否成功
    figure: Optional[matplotlib.figure.Figure]  # CLI 用
    data: Dict[str, Any]                        # Web 用 (JSON-serializable)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

class VizAdapter(ABC):
    """可视化适配器协议"""

    @abstractmethod
    def capabilities(self) -> List[str]:
        """返回支持的 intent 列表"""

    @abstractmethod
    def render(self, target: Any, intent: str, **options) -> VizResult:
        """执行渲染"""
```

### 4.4 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **入口模式** | Facade | 统一、简洁、可扩展 |
| **路由机制** | Registry | 与插件系统契合 |
| **输出模式** | 双模式 (Figure + Data) | CLI/Web 兼容 |
| **错误处理** | 零异常 + warnings | 批量场景健壮；可选 strict 模式 |
| **适配器来源** | Core 兜底 + 插件可扩展 | 灵活性 |

### 4.5 内置适配器

| 适配器 | 支持的 intent | 目标类型 |
|--------|--------------|---------|
| `ParetoAdapter` | `pareto` | ResultArchive |
| `EquationAdapter` | `equation`, `latex` | SearchResult, ArchivedResult |
| `ResidualAdapter` | `residual` | EvaluationResult |
| `TreeAdapter` | `tree` | AnalysisIR |

<!--
## 实现参考：kd1 设计细节

### 零异常设计
错误通过返回值的 warnings 字段传递，不抛异常、不中断流程。
适用场景：批量可视化、自动化实验、Web 服务。

可选 strict 模式：
```python
def render(target, intent, strict=False, **options):
    result = adapter.render(target, **options)
    if strict and not result.ok:
        raise VizError(result.error)
    return result
```

### 继承链查找
Registry 支持继承链查找，子类自动继承父类适配器：
```python
class VizRegistry:
    def get(self, target_type, intent):
        # 先查 target_type 自己
        # 找不到则沿 MRO 向上查找
        for cls in target_type.__mro__:
            if (cls, intent) in self._adapters:
                return self._adapters[(cls, intent)]
        return self._default_adapter
```

### 插件扩展可视化
插件可注册自己的适配器，覆盖或扩展 Core：
```python
# SGA 插件注册专用可视化
registry.register(SGAPopulation, "genealogy", SGAGenealogyAdapter())
```

### 样式隔离
使用上下文管理器隔离 matplotlib 样式，防止全局状态污染：
```python
with plt.style.context('seaborn'):
    fig = adapter.render(...)
```

### data-first 原则
- data 是 JSON-serializable，作为主要产物
- figure 是本地渲染加成，不进入 ResultArchive 关键路径
- Web 端可用 data 自行渲染（Plotly/ECharts）
-->

---

## 五、Callback 机制

为实时进度推送预留 callback 接口。

```python
@dataclass
class ProgressUpdate:
    """进度更新"""
    generation: int
    total_generations: int
    best_expression: str
    best_score: float
    evaluated_count: int
    timestamp: float

class ProgressCallback(Protocol):
    """进度回调协议"""

    def on_generation_start(self, generation: int) -> None: ...
    def on_generation_end(self, update: ProgressUpdate) -> None: ...
    def on_new_best(self, update: ProgressUpdate) -> None: ...
    def on_search_complete(self, result: SearchResultDTO) -> None: ...
```

**CLI 实现示例**：

```python
class CLIProgressCallback:
    """CLI 进度条回调"""

    def on_generation_end(self, update: ProgressUpdate):
        print(f"Gen {update.generation}: Best = {update.best_expression}")
```

**未来 WebSocket 实现示例**：

```python
class WebSocketProgressCallback:
    """WebSocket 实时推送回调"""

    def __init__(self, websocket):
        self.ws = websocket

    async def on_generation_end(self, update: ProgressUpdate):
        await self.ws.send_json(update.__dict__)
```

---

## 六、配置 Schema（Pydantic）

使用 Pydantic 定义配置 schema，支持 JSON Schema 导出（Web 表单生成）。

```python
from pydantic import BaseModel, Field

class SGAConfig(BaseModel):
    """SGA 算法配置"""
    population_size: int = Field(30, ge=10, le=500, description="种群大小")
    generations: int = Field(100, ge=10, le=1000, description="迭代代数")
    crossover_rate: float = Field(0.8, ge=0.0, le=1.0, description="交叉率")
    mutation_rate: float = Field(0.2, ge=0.0, le=1.0, description="变异率")

# 导出 JSON Schema（Web 表单可用）
schema = SGAConfig.model_json_schema()
```

---

## 七、目录结构

```
kd2/
├── src/kd2/
│   ├── core/
│   ├── data/
│   ├── plugins/
│   ├── experiment/
│   ├── visualization/
│   │   ├── facade.py        # render(), list_capabilities()
│   │   ├── registry.py      # VizRegistry
│   │   ├── base.py          # VizAdapter ABC, VizResult
│   │   ├── adapters/        # 内置适配器
│   │   │   ├── pareto.py
│   │   │   ├── equation.py
│   │   │   ├── residual.py
│   │   │   └── tree.py
│   │   └── utils.py         # 样式隔离等工具
│   └── dto/                  # 数据传输对象
│       ├── results.py       # EvaluationResultDTO, SearchResultDTO
│       ├── viz.py           # VizResult, ParetoPlotDTO 等
│       └── config.py        # 配置 schema
│
├── src/kd2_cli/              # CLI 包（可选安装）
│   ├── __init__.py
│   ├── main.py              # typer/click 入口
│   └── callbacks.py         # CLI 进度回调
│
└── pyproject.toml           # 可选依赖
```

**pyproject.toml 可选依赖**：

```toml
[project.optional-dependencies]
cli = ["typer", "rich"]
api = ["fastapi", "uvicorn", "websockets"]
all = ["kd2[cli]", "kd2[api]"]
```

---

## 八、延后项

以下内容**不在 MVP 范围内**，待 Web 需求明确后实现：

| 延后项 | 说明 |
|--------|------|
| Web 框架选择 | FastAPI / Gradio / Streamlit 待定 |
| WebSocket 协议 | 具体消息格式待定 |
| 前端实现 | 可独立 repo，React/Vue 待定 |
| 认证鉴权 | 多用户场景再考虑 |

<!--
## 实现参考：Viz Data Contracts

### 基础 DTO（Codex 建议）
```python
@dataclass
class EquationDTO:
    expr: str                 # display string
    expr_canonical: str       # canonical string
    latex: str | None
    coeffs: list[float] | None
    metrics: dict[str, float]

@dataclass
class ParetoPointDTO:
    id: str                   # canonical hash
    objectives: dict[str, float]
    equation: EquationDTO

@dataclass
class ParetoPlotDTO:
    x: str                    # objective name
    y: str
    points: list[ParetoPointDTO]
```

### 大数组处理
残差/场可视化会携带大数组，可选：
- InlineArray：小数组直接存 list
- ArrayRef：只存 {artifact_id, shape, dtype}，真实数据走 artifact

实现时根据实际数据规模选择。
-->
