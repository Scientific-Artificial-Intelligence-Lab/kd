# kd2 插件层设计

> **关联文档**：[arc_final.md](arc_final.md) - 主文档

---

## 一、AlgorithmPlugin 接口

**已确定**：采用 `propose()/update()` 模式，平台统一控制评估。

```python
class AlgorithmPlugin(ABC):
    """算法插件抽象基类"""

    name: str
    version: str
    algorithm_type: AlgorithmType  # RL, GA, HYBRID

    @abstractmethod
    def configure(self, config: Dict) -> None:
        """设置算法参数"""

    @abstractmethod
    def setup(
        self,
        dataset: PDEDataset,
        library: Library,
        constraints: JointPrior,
    ) -> None:
        """初始化搜索环境"""

    @abstractmethod
    def propose(self, k: int) -> List[Candidate]:
        """
        生成 k 个候选表达式
        返回: GenIR 列表
        """

    @abstractmethod
    def update(self, feedback: List[Feedback]) -> None:
        """
        根据评估反馈更新内部状态
        feedback: [(candidate_id, EvaluationResult), ...]
        """

    @abstractmethod
    def get_best(self) -> SearchResult:
        """获取当前最优结果"""

    # Checkpointing 支持
    def get_state(self) -> bytes:
        """序列化内部状态（用于检查点）"""

    def set_state(self, state: bytes) -> None:
        """恢复内部状态"""
```

### 1.1 数据结构

```python
@dataclass
class Candidate:
    id: str
    gen_ir: GenIR

@dataclass
class Feedback:
    candidate_id: str
    result: EvaluationResult
```

---

## 二、ExperimentManager

统一调度 propose/eval/update 循环。

```python
class ExperimentManager:
    """实验管理器"""

    def __init__(
        self,
        plugin: AlgorithmPlugin,
        executor: Executor,
        evaluator: Evaluator,
        cache: ExpressionCache,
        logger: Logger,                        # 协议化，可替换
        result_archive: ResultArchive,
        callback: Optional[ProgressCallback] = None
    ):
        self.plugin = plugin
        self.result_archive = result_archive
        # ...

    def run(self, max_iterations: int, batch_size: int = 100, checkpoint_every: int = 10):
        """
        主循环:
        1. plugin.propose(batch_size) → 候选列表
        2. 批量执行 + 评估（带缓存）
        3. plugin.update(feedback)
        4. result_archive.add(results)
        5. 记录日志
        6. 每 checkpoint_every 代保存检查点
        7. 重复直到收敛或达到 max_iterations
        """

    def save_checkpoint(self, path: Path) -> None:
        """保存检查点"""

    def load_checkpoint(self, path: Path) -> None:
        """从检查点恢复"""
```

<!--
## 实现参考：Logger 协议化

Codex 建议：WandB/Hydra 变成"可选外设"，定义 Logger 协议。

```python
class Logger(Protocol):
    def log_metrics(self, metrics: dict[str, float], step: int): ...
    def log_artifact(self, name: str, artifact: Any): ...
    def close(self): ...

class WandBLogger(Logger): ...
class PrintLogger(Logger): ...  # 简单 print
class NullLogger(Logger): ...   # 什么都不做
```

这样 CLI/Notebook/Web 都可以替换不同的 Logger 实现。
-->

---

## 三、Checkpointing

**已确定**：Phase 3 纳入，插件需实现 get_state/set_state。

```python
@dataclass
class Checkpoint:
    """检查点"""
    generation: int
    plugin_state: bytes          # 插件内部状态
    result_archive: ResultArchive
    rng_state: Dict              # 随机数状态
    config: Dict                 # 实验配置
    timestamp: float
```

**使用示例**：

```python
# 保存检查点
manager.save_checkpoint("checkpoints/exp_001_gen_50.ckpt")

# 从检查点恢复
manager.load_checkpoint("checkpoints/exp_001_gen_50.ckpt")
manager.run(max_iterations=100)  # 从 gen 50 继续
```

<!--
## 实现参考：Checkpoint 细节

### 目录结构（Codex 建议）
```
checkpoint/
├── metadata.json       # 元数据（可读可 diff）
├── plugin_state.pt     # 插件状态
└── archive.jsonl       # 关键结果流（可选）
```

### 原子写入
先写临时目录/文件，再 rename（防止中途崩溃损坏）。

### RNG 状态
可复现性需要覆盖：
- random.getstate()
- np.random.get_state()
- torch.random.get_rng_state()
- torch.cuda.get_rng_state_all()（如用 CUDA）

实现时根据实际需求决定覆盖范围。
-->

---

## 四、ResultArchive

**已确定**：简化版，收集结果 + 后处理提取 Pareto。

```python
@dataclass
class ArchivedResult:
    """档案中的结果"""
    gen_ir: GenIR
    expression: str
    latex: str
    nmse: float
    complexity: float
    aic: float
    coefficients: List[float]
    generation_found: int

class ResultArchive:
    """结果档案"""

    def __init__(self, max_size: int = 500):
        self.results: Dict[str, ArchivedResult] = {}  # hash -> result
        self.max_size = max_size

    def add(self, result: EvaluationResult, gen_ir: GenIR) -> bool:
        """添加结果，自动去重和修剪"""

    def get_pareto_front(self, objectives: List[str] = ["nmse", "complexity"]) -> List[ArchivedResult]:
        """后处理提取 Pareto 前沿"""

    def get_best_by(self, metric: str) -> ArchivedResult:
        """按指定指标获取最优"""

    def to_dataframe(self) -> pd.DataFrame:
        """导出为 DataFrame"""

    def to_dto(self) -> ResultArchiveDTO:
        """可序列化（Web 兼容）"""
```

---

## 五、算法插件实现

### 5.1 SGA 插件（首个实现）

```python
class SGAPlugin(AlgorithmPlugin):
    """
    遗传算法插件

    搜索空间: term forest（若干项相加）
    评估: AIC = 2k + 2ln(MSE)
    """

    def __init__(self):
        self.population: List[Individual] = []
        self.generation: int = 0

    def configure(self, config):
        self.pop_size = config.get("population_size", 30)
        self.n_generations = config.get("generations", 100)
        self.crossover_rate = config.get("crossover_rate", 0.8)
        self.mutation_rate = config.get("mutation_rate", 0.2)

    def propose(self, k: int) -> List[Candidate]:
        """
        生成策略:
        - 选择（锦标赛选择）
        - 交叉（子树交叉）
        - 变异（点变异、子树变异）
        """

    def update(self, feedback: List[Feedback]):
        """
        更新策略:
        - 根据 AIC 分数排序
        - 保留精英个体
        """
```

### 5.2 DISCOVER 插件（PyTorch 重写）

```python
class DISCOVERPlugin(AlgorithmPlugin):
    """
    Deep Symbolic Regression 插件

    搜索: LSTM Controller + risk-seeking policy gradient
    评估: reward = (1 - penalty*complexity) / (1 + sqrt(NMSE))
    """

    def __init__(self):
        self.controller: nn.Module = None  # LSTM
        self.optimizer: torch.optim.Optimizer = None
        self.priority_queue: PriorityQueue = None

    def configure(self, config):
        self.hidden_size = config.get("hidden_size", 64)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.epsilon = config.get("epsilon", 0.05)  # top-ε

    def propose(self, k: int) -> List[Candidate]:
        """用 LSTM 逐 token 采样生成表达式"""

    def update(self, feedback: List[Feedback]):
        """
        Risk-seeking policy gradient:
        - 只用 top-ε% 的样本计算梯度
        """
```

### 5.3 其他插件

- **DLGA**：Deep Learning Genetic Algorithm
- **PySR**：外部库包装

---

## 六、配置系统

**已确定**：Hydra 管理配置。

```yaml
# config/experiment/burgers_sga.yaml
defaults:
  - _self_
  - dataset: burgers
  - algorithm: sga

experiment:
  name: "burgers_discovery"
  max_iterations: 100
  batch_size: 50

algorithm:
  population_size: 30
  generations: 100
  crossover_rate: 0.8
  mutation_rate: 0.2
```

```python
@hydra.main(config_path="config", config_name="experiment")
def main(cfg: DictConfig):
    dataset = load_dataset(cfg.dataset.name)
    plugin = get_plugin(cfg.algorithm.name)
    plugin.configure(cfg.algorithm)
    # ...
```
