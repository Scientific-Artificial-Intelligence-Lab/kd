
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import torch
from torch import Tensor

from kd.core.safety import safe_div, safe_exp, safe_log





_POWER_CLAMP = 1e6



AnyCallable = Callable[..., Any]



_VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


_DANGEROUS_NAMES = frozenset(
    {
        "__builtins__",
        "__class__",
        "__import__",
        "__globals__",
        "__code__",
        "__dict__",
        "__name__",
        "__module__",
        "__annotations__",
        "__doc__",
        "__slots__",
        "__init__",
        "__new__",
        "__del__",
        "__call__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__getattribute__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
    }
)


@dataclass
class _FunctionInfo:

    func: AnyCallable
    arity: int
    commutative: bool


class FunctionRegistry:

    def __init__(self) -> None:
        self._functions: dict[str, _FunctionInfo] = {}

    def _validate_name(self, name: str) -> None:
        if not name:
            raise ValueError("Function name cannot be empty")


        if not _VALID_NAME_PATTERN.match(name):
            raise ValueError(f"Function name '{name}' must be a valid ASCII identifier")


        if name.startswith("__") and name.endswith("__"):
            raise ValueError(
                f"Function name '{name}' is reserved (dunder names forbidden)"
            )


        if name in _DANGEROUS_NAMES:
            raise ValueError(f"Function name '{name}' is dangerous/forbidden")

    def _validate_func(self, func: AnyCallable) -> None:
        if func is None or not callable(func):
            raise TypeError("func must be callable")

    def _validate_arity(self, arity: int) -> None:
        if arity < 0:
            raise ValueError(f"arity must be >= 0, got {arity}")

    def register(
        self,
        name: str,
        func: AnyCallable,
        arity: int,
        commutative: bool = False,
    ) -> None:

        self._validate_name(name)
        self._validate_func(func)
        self._validate_arity(arity)


        if name in self._functions:
            raise ValueError(f"Function '{name}' already exists in registry")

        self._functions[name] = _FunctionInfo(
            func=func,
            arity=arity,
            commutative=commutative,
        )

    def get_context(self) -> dict[str, AnyCallable]:
        return {name: info.func for name, info in self._functions.items()}

    def is_commutative(self, name: str) -> bool:
        if name not in self._functions:
            raise KeyError(f"Function '{name}' not found in registry")
        return self._functions[name].commutative

    def get_arity(self, name: str) -> int:
        if name not in self._functions:
            raise KeyError(f"Function '{name}' not found in registry")
        return self._functions[name].arity

    def get_func(self, name: str) -> AnyCallable:
        if name not in self._functions:
            raise KeyError(f"Function '{name}' not found in registry")
        return self._functions[name].func

    def has(self, name: str) -> bool:
        return name in self._functions

    def list_names(self) -> list[str]:
        return list(self._functions.keys())

    def get_by_arity(self, arity: int) -> list[str]:
        return [name for name, info in self._functions.items() if info.arity == arity]

    @classmethod
    def create_default(cls) -> "FunctionRegistry":
        reg = cls()


        reg.register("add", torch.add, arity=2, commutative=True)
        reg.register("mul", torch.mul, arity=2, commutative=True)
        reg.register("sub", torch.sub, arity=2, commutative=False)
        reg.register("div", _safe_div_wrapper, arity=2, commutative=False)


        reg.register("sin", torch.sin, arity=1)
        reg.register("cos", torch.cos, arity=1)
        reg.register("exp", safe_exp, arity=1)
        reg.register("log", safe_log, arity=1)
        reg.register("neg", _neg, arity=1)
        reg.register("n2", _square, arity=1)
        reg.register("n3", _cube, arity=1)
        reg.register("recip", _recip, arity=1)
        reg.register("lap", _lap_stub, arity=1)

        return reg







PROTECTED_OPERATORS: Final[frozenset[str]] = frozenset(
    {"div", "exp", "log", "n2", "n3", "recip"}
)





def _safe_div_wrapper(a: Tensor, b: Tensor) -> Tensor:
    return safe_div(a, b)


def _recip(x: Tensor) -> Tensor:
    return safe_div(torch.ones_like(x), x)


def _neg(x: Tensor) -> Tensor:
    return -x


def _square(x: Tensor) -> Tensor:
    x_c = torch.clamp(x, min=-_POWER_CLAMP, max=_POWER_CLAMP)
    return x_c * x_c


def _cube(x: Tensor) -> Tensor:
    x_c = torch.clamp(x, min=-_POWER_CLAMP, max=_POWER_CLAMP)
    return x_c * x_c * x_c


def _lap_stub(_x: Tensor) -> Tensor:
    raise NotImplementedError(
        "lap requires a context-aware executor and cannot run via registry"
    )
