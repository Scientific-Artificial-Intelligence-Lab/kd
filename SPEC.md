# kd2 - Symbolic Regression for PDE Discovery

> **Status**: Phase 0 - Planning

## Vision

Unified platform for PDE/ODE discovery via symbolic regression. Compare algorithms (DISCOVER, SGA, DLGA, PySR) on same data with common IR.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Application    CLI  │  Web UI (future)  │  Agent (future)      │
├─────────────────────────────────────────────────────────────────┤
│  Experiment     ExperimentManager  │  ResultStore               │
├─────────────────────────────────────────────────────────────────┤
│  Plugins        DISCOVER │ SGA │ DLGA │ PySR │ Custom           │
├─────────────────────────────────────────────────────────────────┤
│  Core           IR │ Library │ Executor │ Evaluator │ Solver    │
├─────────────────────────────────────────────────────────────────┤
│  Foundation     NumPy │ PyTorch │ SymPy                         │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concepts

**Three-Layer IR**:
- **GenIR**: Prefix token sequence (for RL/GA generation)
- **AnalysisIR**: AST tree (for constraints, visualization)
- **ExecIR**: DAG (for efficient execution, CSE) - Phase 5

**Key Interfaces** (TBD):
```python
class PDEDataset:
    axes: Dict[str, AxisInfo]       # {x, y, t}
    fields: Dict[str, FieldData]    # {u, v}

class AlgorithmPlugin(ABC):
    def setup(dataset, library, evaluator) -> None
    def search(max_iterations) -> SearchResult
```

## Key Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Data type | `torch.Tensor` | End-to-end differentiable |
| First plugin | SGA | Simpler, validates architecture |
| Derivatives | Finite diff + autograd | Flexibility |

## Phases

| Phase | Milestone | Status |
|-------|-----------|--------|
| 1 | Core: IR + Executor + Evaluator | Planned |
| 2 | Constraints + Linear Solvers | - |
| 3 | SGA plugin discovers Burgers | - |
| 4 | DISCOVER plugin | - |
| 5 | ExecIR + Visualization | - |

## Validation

| Equation | Formula |
|----------|---------|
| Burgers | `u_t = -u·u_x + 0.1·u_xx` |
| KdV | `u_t = -6u·u_x - u_xxx` |

## References

- IR & Execution: `ref_libs/DISCOVER/dso/dso/program.py`
- Tree structure: `ref_libs/sga/sgapde/pde.py`
- STRidge: `ref_libs/DISCOVER/dso/dso/stridge.py`
