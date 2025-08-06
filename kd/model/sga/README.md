# SGA-PDE Integration for KD Framework

This directory contains the complete integration of the SGA-PDE (Symbolic Genetic Algorithm for Partial Differential Equations) algorithm into the KD framework.

## Overview

SGA-PDE is a symbolic regression algorithm specifically designed for discovering partial differential equations from data. This integration makes it fully compatible with the KD framework's design patterns and provides a unified, sklearn-style API.

## Directory Structure

```
kd/model/sga/
├── README.md                    # This file
├── todo.md                      # Detailed project status and history
├── codes/                       # Original SGA-PDE implementation
│   ├── __init__.py
│   ├── configure.py
│   ├── Data_generator.py
│   ├── pde.py
│   ├── sga.py
│   ├── tree.py
│   └── ...
├── sga_refactored/             # Refactored modular implementation
│   ├── data_utils.py           # Data preprocessing and derivatives
│   ├── operators.py            # Symbol library construction
│   ├── diagnostics.py          # Error diagnostics
│   ├── main.py                 # Main workflow with dependency injection
│   └── kd_sga.py              # KD framework adapter (legacy)
├── data/                       # Test datasets
├── tests/                      # Unit tests
└── *.pkl                      # Pre-trained models
```

## Quick Start

### Basic Usage

```python
from kd.model import KD_SGA

# Create model
model = KD_SGA(num=20, depth=4, width=5)

# Fit model (discovers PDE from data)
model.fit(u=u_data, t=t_coords, x=x_coords, max_gen=10)

# Get results
equation_str = model.get_equation_string()
latex_eq = model.get_equation_latex()
aic_score = model.best_aic_

print(f"Discovered equation: {equation_str}")
print(f"LaTeX: {latex_eq}")
print(f"AIC score: {aic_score}")
```

### Advanced Usage with sklearn-style API

```python
import numpy as np
from kd.model import KD_SGA

# Create model with custom parameters
model = KD_SGA(
    num=50,           # Population size
    depth=5,          # Max tree depth
    width=6,          # Max PDE terms
    p_mute=0.3,       # Mutation probability
    data_mode='finite_difference'  # Derivative calculation mode
)

# Parameter management (sklearn-compatible)
params = model.get_params()
model.set_params(num=30, p_mute=0.4)

# Fit and evaluate
model.fit(X=None, u=u_data, t=t_coords, x=x_coords, max_gen=20)
score = model.score(X=None)  # Returns negative AIC

# Model representation
print(repr(model))  # KD_SGA(num=30, depth=5, width=6)
```

## Key Features

### 🔧 **Framework Integration**
- **BaseGa inheritance**: Fully compatible with KD framework base classes
- **sklearn-style API**: Standard `fit()`, `predict()`, `score()` methods
- **Parameter management**: Automatic `get_params()` and `set_params()` support
- **Unified imports**: Available via `from kd.model import KD_SGA`

### 🎨 **Visualization Support**
- **LaTeX rendering**: Automatic conversion of discovered equations to LaTeX
- **SymPy integration**: Advanced mathematical expression processing
- **kd.viz compatibility**: Seamless integration with KD visualization system
- **Custom renderer**: Specialized SGA equation formatter

### 🧪 **Robust Implementation**
- **Dependency injection**: Non-invasive integration preserving original algorithm
- **Dual derivative modes**: Support for both finite difference and autograd
- **Comprehensive testing**: Full unit test coverage with golden standards
- **Error handling**: Graceful handling of edge cases and failures

### 📊 **Data Flexibility**
- **Multiple input formats**: Direct u,t,x arrays or sklearn-style X,y
- **Default datasets**: Built-in test data for quick experimentation
- **Shape validation**: Automatic data consistency checking

## Examples

See the `examples/` directory for complete usage examples:

- `examples/kd_sga_example.py` - Basic usage demonstration
- `examples/kd_sga_integration_example.py` - Complete framework integration showcase

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num` | int | 20 | Population size for genetic algorithm |
| `depth` | int | 4 | Maximum depth of symbolic trees |
| `width` | int | 5 | Maximum number of terms in PDE |
| `p_var` | float | 0.5 | Probability of generating variables |
| `p_rep` | float | 1.0 | Replacement probability |
| `p_mute` | float | 0.3 | Mutation probability |
| `p_cro` | float | 0.5 | Crossover probability |
| `data_mode` | str | 'finite_difference' | Derivative calculation mode |

## Methods

### Core Methods
- `fit(X, y=None, u=None, t=None, x=None, max_gen=5, verbose=True)` - Discover PDE from data
- `predict(X)` - Make predictions (placeholder implementation)
- `score(X, y=None)` - Return negative AIC score

### Utility Methods
- `get_equation_string()` - Get string representation of discovered equation
- `get_equation_latex()` - Get LaTeX representation of discovered equation
- `get_params(deep=True)` - Get model parameters
- `set_params(**params)` - Set model parameters

## Integration Status

✅ **Complete Integration Achieved**

- [x] Framework compatibility (BaseGa inheritance)
- [x] API registration (`kd.model.__init__.py`)
- [x] Visualization integration (`kd.viz.sga_eq2latex`)
- [x] Parameter management system
- [x] sklearn-style interface
- [x] Comprehensive testing
- [x] Documentation and examples
- [x] Error handling and robustness

## Technical Details

### Architecture
The integration follows a clean architecture pattern:
1. **Original codes/**: Preserved original implementation for reference
2. **sga_refactored/**: Modular, testable components
3. **kd/model/kd_sga.py**: Unified framework adapter
4. **kd/viz/sga_eq2latex.py**: Specialized visualization support

### Design Principles
- **Non-invasive**: Original SGA algorithm remains unchanged
- **Modular**: Clear separation of concerns
- **Testable**: Comprehensive unit test coverage
- **Compatible**: Full sklearn and KD framework compatibility

## Citation

If you use SGA-PDE in your research, please cite the original paper and this integration work.

## License

This integration follows the same license as the KD framework.
