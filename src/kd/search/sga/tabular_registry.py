
from __future__ import annotations

import torch

from kd.core.expr.registry import FunctionRegistry

_RAW = {"exp": torch.exp, "log": torch.log}


def build_tabular_registry() -> FunctionRegistry:
    default = FunctionRegistry.create_default()
    registry = FunctionRegistry()
    for name in default.list_names():
        registry.register(
            name,
            _RAW.get(name, default.get_func(name)),
            arity=default.get_arity(name),
            commutative=default.is_commutative(name),
        )
    return registry
