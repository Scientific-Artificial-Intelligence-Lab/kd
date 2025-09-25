"""Quick script to print `load_pde(name).sample()` shapes for DLGA verification."""
from __future__ import annotations

import numpy as np

from kd.dataset import load_pde

DATASETS = [
    'chafee-infante',
    'burgers',
    'kdv',
    'PDE_divide',
    'PDE_compound',
    'fisher',
    'fisher_linear',
]

for name in DATASETS:
    try:
        dataset = load_pde(name)
    except Exception as exc:  # pragma: no cover - manual script
        print(f"load failed: {name} -> {exc}")
        continue
    try:
        X, y = dataset.sample(128)
    except Exception as exc:
        print(f"sample failed: {name} -> {exc}")
        continue
    print(f"{name}: X{X.shape} y{y.shape}")
