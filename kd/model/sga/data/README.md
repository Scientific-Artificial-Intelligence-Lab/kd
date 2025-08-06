# SGA Data Directory - LEGACY

⚠️ **WARNING: This directory is LEGACY and will be deprecated in future versions.**

## Migration Notice

This data directory (`kd/model/sga/data/`) contains the original SGA-PDE dataset files and is maintained for backward compatibility only. 

**For new code, please use the unified KD dataset API:**

```python
# OLD (Legacy) - DO NOT USE in new code
from kd.model.sga.codes.configure import load_problem_data
data = load_problem_data('chafee-infante')

# NEW (Recommended) - Use this instead
from kd.dataset import load_pde_dataset
dataset = load_pde_dataset(equation_name='chafee-infante')
```

## Data Unification Status

All data files in this directory have been **unified and moved** to the main KD dataset directory:
- `kd/dataset/data/` - **Primary location** (use this)
- `kd/model/sga/data/` - **Legacy location** (will be deprecated)

## Available Datasets

The following datasets are available through the unified API:

| Dataset | Legacy Name | Unified Name | Description |
|---------|-------------|--------------|-------------|
| Chafee-Infante | `chafee-infante` | `chafee-infante` | u_t = u_xx - u + u^3 |
| Burgers | `Burgers` | `burgers` | u_t = -u*u_x + 0.1*u_xx |
| KdV | `Kdv` | `kdv` | u_t = -u*u_x - u_xxx |
| PDE Divide | `PDE_divide` | `pde_divide` | u_t = -u_x/x + 0.25*u_xx |
| PDE Compound | `PDE_compound` | `pde_compound` | u_t = u*u_xx + u_x*u_x |

## Migration Timeline

- **Current**: Both legacy and unified APIs work
- **Future**: Legacy API will be deprecated and removed
- **Recommendation**: Migrate to unified API immediately

## Files in this Directory

- `burgers.mat` - Burgers equation data
- `chafee_infante_*.npy` - Chafee-Infante equation data
- `KdV.mat` - Korteweg-de Vries equation data  
- `PDE_compound.npy` - PDE compound equation data
- `PDE_divide.npy` - PDE divide equation data

All these files are **duplicates** of the files in `kd/dataset/data/` and will be removed in future versions.
