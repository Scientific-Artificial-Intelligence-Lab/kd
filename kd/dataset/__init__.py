from typing import Optional

from ._base import (PDEDataset,
                    load_burgers_equation,
                    load_kdv_equation,
                    load_pde_dataset,
                    load_mat_file)
from ._registry import (PDE_REGISTRY,
                        get_dataset_info,
                        list_available_datasets,
                        get_dataset_sym_true)
import numpy as np
from importlib import resources


def _tag_dataset(dataset: Optional[PDEDataset], name: str) -> Optional[PDEDataset]:
    """Attach registry metadata (registry_name / aliases) to dataset instances."""
    if dataset is None:
        return None

    setattr(dataset, 'registry_name', name)
    info = PDE_REGISTRY.get(name, {})
    aliases = info.get('aliases', {}) or {}
    setattr(dataset, 'aliases', aliases)
    legacy_name = aliases.get('legacy', getattr(dataset, 'legacy_name', None))
    if legacy_name is None:
        legacy_name = name
    setattr(dataset, 'legacy_name', legacy_name)
    return dataset

def load_pde(name: str, **kwargs) -> PDEDataset:
    """Unified entry point for loading PDE benchmark datasets.

    Args:
        name: Dataset name (e.g. ``"kdv"``, ``"burgers"``,
            ``"chafee-infante"``).
        **kwargs: Additional keyword arguments forwarded to
            :class:`PDEDataset`.

    Returns:
        PDEDataset: The loaded dataset instance.
    """
    info = get_dataset_info(name)
    
    # 处理不同文件格式
    if 'files' in info:  # 多文件情况（如chafee-infante）
        data_path = resources.files("kd.dataset.data")
        u = np.asarray(np.load(data_path / info['files']['u']), dtype=float)
        x = np.asarray(np.load(data_path / info['files']['x']), dtype=float).flatten()
        t = np.asarray(np.load(data_path / info['files']['t']), dtype=float).flatten()

        if u.shape == (len(t), len(x)):
            u = u.T

        dataset = PDEDataset(
            equation_name=name,
            pde_data=None,
            x=x, t=t, usol=u,
            domain=info.get('domain'),
            epi=kwargs.get('epi', 1e-3)
        )
        return _tag_dataset(dataset, name)
    elif info.get('file', '').endswith('.npy'):
        # 处理单个npy文件的特殊情况（如PDE_divide, PDE_compound）
        data_path = resources.files("kd.dataset.data")
        data = np.asarray(np.load(data_path / info['file']), dtype=float)

        # 这些数据集通常直接是u矩阵，需要根据shape和domain生成x, t
        target_shape = info.get('shape')
        if target_shape:
            # 数据文件有时是 (nt, nx)，因此需要在匹配时考虑转置
            if data.shape == tuple(reversed(target_shape)):
                data = data.T
            elif data.shape != target_shape:
                data = data.reshape(target_shape)
        shape = data.shape

        domain = info.get('domain') or {'x': (0.0, 1.0), 't': (0.0, 1.0)}
        x_bounds = domain.get('x', (0.0, 1.0))
        t_bounds = domain.get('t', (0.0, 1.0))

        nx, nt = shape
        x = np.linspace(x_bounds[0], x_bounds[1], nx)
        t = np.linspace(t_bounds[0], t_bounds[1], nt)

        dataset = PDEDataset(
            equation_name=name,
            pde_data=None,
            x=x, t=t, usol=data,
            domain=domain,
            epi=kwargs.get('epi', 1e-3)
        )
        return _tag_dataset(dataset, name)
    else:  # 单mat文件情况
        dataset = load_pde_dataset(
            info['file'],
            equation_name=name,
            domain=info.get('domain'),
            **info.get('keys', {}),
            **kwargs)
        return _tag_dataset(dataset, name)

_REGRESSION_REGISTRY = {
    "tlc_cc_t1": {
        "file": "tlc_cc_Rf_t1.npy",
        "x_cols": [0, 1],
        "y_col": 2,
        "var_names": ["R_F", "r"],
        "target_name": "mean(V_S)",
        "description": "TLC-CC: Rf vs start retention volume (mean)",
        "source": "Nature Comm. 2025:16:832",
    },
    "tlc_cc_t2": {
        "file": "tlc_cc_Rf_t2.npy",
        "x_cols": [0, 1],
        "y_col": 2,
        "var_names": ["R_F", "r"],
        "target_name": "mean(V_E)",
        "description": "TLC-CC: Rf vs end retention volume (mean)",
        "source": "Nature Comm. 2025:16:832",
    },
}


def load_regression(name: str):
    """Load a tabular regression dataset.

    Args:
        name: Dataset name (e.g. ``"tlc_cc_t1"``, ``"tlc_cc_t2"``).

    Returns:
        tuple: ``(X, y, meta)`` where *X* is ``(n_samples, n_features)``,
        *y* is ``(n_samples,)``, and *meta* is a dict with
        ``var_names``, ``target_name``, etc.
    """
    if name not in _REGRESSION_REGISTRY:
        available = ", ".join(sorted(_REGRESSION_REGISTRY))
        raise ValueError(
            f"Unknown regression dataset '{name}'. "
            f"Available: {available}"
        )
    info = _REGRESSION_REGISTRY[name]
    data_path = resources.files("kd.dataset.data")
    data = np.load(data_path / info["file"])
    X = data[:, info["x_cols"]]
    y = data[:, info["y_col"]]
    meta = {
        "var_names": info["var_names"],
        "target_name": info["target_name"],
        "description": info["description"],
        "source": info["source"],
    }
    return X, y, meta


__all__ = [
    "PDEDataset",
    "load_pde",  # 新增统一接口
    "load_burgers_equation",
    "load_mat_file",
    "load_kdv_equation",
    "load_pde_dataset",
    "list_available_datasets",  # 新增辅助函数
    "get_dataset_sym_true",  # 新增辅助函数
    "Burgers_equation_shock",
    "KdV_equation",
    "KdV_equation_sine",
    "Chaffee_Infante_equation",
    "KG_equation",
    "Allen_Cahn_equation",
    "Convection_diffusion_equation_solution",
    "Convection_diffusion_equation_simulation",
    "Wave_equation",
    "KS_equation",
    "Beam_equation",
    "Heat_equation",
    "Heat_equation_sin",
    "Diffusion_equation",
    "Parametric_convection_diffusion",
    "Parametric_Burgers_equation",
    "Parametric_wave_equation",
    "Burgers_2D",
    "load_regression",
]
