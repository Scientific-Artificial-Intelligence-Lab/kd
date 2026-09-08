
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from types import CodeType
from typing import TYPE_CHECKING, Any, Final

import torch
from torch import Tensor

from kd.core.expr.naming import parse_compound_derivative, parse_derivative_name
from kd.core.expr.registry import FunctionRegistry, _lap_stub

if TYPE_CHECKING:
    from kd.core.executor.context import ExecutionContext








DIFF_OPERATOR_PATTERN = re.compile(r"^diff([0-9]*)_([a-z]+)$")

_SPECIAL_OPERATOR_STUBS: dict[str, Any] = {"lap": _lap_stub}

_SPECIAL_OPERATORS: frozenset[str] = frozenset(_SPECIAL_OPERATOR_STUBS)
_UNITY_TOKEN: Final = "one"


def _resolve_unity(name: str, context: ExecutionContext) -> Tensor | None:
    if name == _UNITY_TOKEN:
        return context.unity_column()
    return None


@dataclass(eq=False)
class _ParsedExpression:

    tree: ast.Expression
    depth: int
    has_open_form_diff: bool
    compiled: CodeType


@lru_cache(maxsize=512)
def _parse_expression(code: str) -> _ParsedExpression:
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in expression: {e}") from e
    return _ParsedExpression(
        tree=tree,
        depth=_get_ast_depth(tree.body),
        has_open_form_diff=_tree_has_open_form_diff(tree),
        compiled=compile(tree, "<expr>", "eval"),
    )


@dataclass
class ExecutorResult:

    value: Tensor
    used_diff: bool


class PythonExecutor:

    def __init__(
        self,
        registry: FunctionRegistry,
        max_depth: int = 1000,
    ) -> None:
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")

        self._registry = registry
        self._max_depth = max_depth

    @property
    def registry(self) -> FunctionRegistry:
        return self._registry

    def execute(
        self,
        code: str,
        context: ExecutionContext,
        force_diff_path: bool = False,
    ) -> ExecutorResult:

        if not code or not code.strip():
            raise ValueError("Expression cannot be empty")


        parsed = _parse_expression(code)


        if parsed.depth > self._max_depth:
            raise RuntimeError(f"Maximum recursion depth ({self._max_depth}) exceeded")

        use_full_path = _should_use_full_path(
            parsed, context, force_diff_path=force_diff_path
        )

        if use_full_path:

            value = self._execute_with_diff(parsed.tree.body, context, depth=0)
        else:

            value = self._execute_simple(parsed.compiled, context)

        return ExecutorResult(value=value, used_diff=use_full_path)

    def _execute_simple(
        self,
        compiled: CodeType,
        context: ExecutionContext,
    ) -> Tensor:



        eval_ctx: dict[str, Any] = {"__builtins__": {}}
        eval_ctx.update(self._registry.get_context())



        known_fields, known_axes = _context_name_sets(context)
        eval_ctx = _VariableAccessDict(
            eval_ctx,
            context,
            known_fields=known_fields,
            known_axes=known_axes,
        )




        try:
            result = eval(compiled, {"__builtins__": {}}, eval_ctx)
        except NameError as e:


            raise KeyError(str(e)) from e


        if not isinstance(result, Tensor):
            result = torch.tensor(result, device=context.device, dtype=torch.float32)

        return result

    def _execute_with_diff(
        self,
        node: ast.expr,
        context: ExecutionContext,
        depth: int,
    ) -> Tensor:

        if depth >= self._max_depth:
            raise RuntimeError(f"Maximum recursion depth ({self._max_depth}) exceeded")

        if isinstance(node, ast.Call):
            return self._execute_call(node, context, depth)

        elif isinstance(node, ast.Name):

            return self._resolve_name_for_diff(node.id, context)

        elif isinstance(node, ast.Constant):

            return torch.tensor(node.value, device=context.device, dtype=torch.float32)

        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):

                operand = self._execute_with_diff(node.operand, context, depth + 1)
                return -operand
            elif isinstance(node.op, ast.UAdd):

                return self._execute_with_diff(node.operand, context, depth + 1)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        elif isinstance(node, ast.BinOp):


            left = self._execute_with_diff(node.left, context, depth + 1)
            right = self._execute_with_diff(node.right, context, depth + 1)

            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                from kd.core.safety import safe_div

                return safe_div(left, right)
            elif isinstance(node.op, ast.Pow):





                return left**right
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def _execute_call(
        self,
        node: ast.Call,
        context: ExecutionContext,
        depth: int,
    ) -> Tensor:

        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")

        func_name = node.func.id

        if _is_diff_operator(func_name):





            if node.keywords:
                raise ValueError(
                    f"Diff operator '{func_name}' does not accept keyword arguments"
                )
            if len(node.args) != 1:
                raise ValueError(
                    f"Diff operator '{func_name}' expects exactly 1 "
                    f"argument, got {len(node.args)}"
                )

            inner = self._execute_with_diff(node.args[0], context, depth + 1)
            axis, order = _parse_diff_name(func_name)
            return context.diff(inner, axis, order)

        if _is_special_operator(func_name):
            return self._dispatch_special_operator(func_name, node, context, depth)



        args = [self._execute_with_diff(arg, context, depth + 1) for arg in node.args]


        try:
            func = self._registry.get_func(func_name)
        except KeyError as e:
            raise KeyError(f"Function '{func_name}' not found in registry") from e

        result: Tensor = func(*args)
        return result

    def _dispatch_special_operator(
        self,
        name: str,
        node: ast.Call,
        context: ExecutionContext,
        depth: int,
    ) -> Tensor:
        self._raise_if_registry_conflicts_with_special_operator(name)

        if name == "lap":
            if node.keywords:
                raise ValueError("lap operator does not accept keyword arguments")
            if len(node.args) != 1:
                raise ValueError(f"lap expects 1 argument, got {len(node.args)}")

            spatial_axes = context.spatial_axes
            if not spatial_axes:
                raise ValueError(
                    "lap operator requires non-empty spatial_axes; "
                    "ensure dataset.lhs_axis is set and axis_order has spatial axes"
                )

            inner = self._execute_with_diff(node.args[0], context, depth + 1)
            result = context.diff(inner, spatial_axes[0], 2)
            for axis in spatial_axes[1:]:
                result = result + context.diff(inner, axis, 2)
            return result

        raise ValueError(f"Unknown special operator: {name}")

    def _raise_if_registry_conflicts_with_special_operator(self, name: str) -> None:
        expected_stub = _SPECIAL_OPERATOR_STUBS.get(name)
        if expected_stub is None:
            return

        try:
            registered_func = self._registry.get_func(name)
        except KeyError:
            return

        if registered_func is not expected_stub:
            raise ValueError(
                f"{name} registry entry conflicts with context-aware special "
                f"operator '{name}'"
            )

    def _resolve_name_for_diff(
        self,
        name: str,
        context: ExecutionContext,
    ) -> Tensor:
        unity = _resolve_unity(name, context)
        if unity is not None:
            return unity





        provider = context.derivative_provider
        get_field = getattr(provider, "get_field", None)
        known_fields, known_axes = _context_name_sets(context)
        if get_field is not None and callable(get_field):
            try:
                result = get_field(name)

                if isinstance(result, Tensor):
                    return result
            except (KeyError, AttributeError, NotImplementedError):
                pass

















        dataset_axes = context.dataset.axes
        provider_coords = provider.coords
        if name in provider_coords:
            coord_t = provider_coords[name]
            in_dataset_axes = dataset_axes is not None and name in dataset_axes
            if not in_dataset_axes or coord_t.requires_grad:
                return coord_t


        try:
            return context.get_variable(name)
        except KeyError:
            pass


        derivative = _try_parse_terminal_derivative(
            name,
            context,
            known_fields=known_fields,
            known_axes=known_axes,
        )
        if derivative is not None:
            return derivative


        try:
            value = context.get_constant(name)
            return torch.tensor(value, device=context.device, dtype=torch.float32)
        except KeyError:
            pass

        raise KeyError(f"Unknown symbol: {name}")


class _VariableAccessDict(dict[str, Any]):

    def __init__(
        self,
        base: dict[str, Any],
        context: ExecutionContext,
        *,
        known_fields: set[str] | None = None,
        known_axes: set[str] | None = None,
    ) -> None:
        super().__init__(base)
        self._context = context
        self._known_fields = known_fields
        self._known_axes = known_axes

    def __missing__(self, key: str) -> Tensor:
        unity = _resolve_unity(key, self._context)
        if unity is not None:
            return unity


        try:
            return self._context.get_variable(key)
        except KeyError:
            pass


        derivative = _try_parse_terminal_derivative(
            key,
            self._context,
            known_fields=self._known_fields,
            known_axes=self._known_axes,
        )
        if derivative is not None:
            return derivative


        try:
            value = self._context.get_constant(key)

            return torch.tensor(value, device=self._context.device, dtype=torch.float32)
        except KeyError:
            pass


        raise KeyError(f"Unknown symbol: {key}")


def _try_parse_terminal_derivative(
    name: str,
    context: ExecutionContext,
    *,
    known_fields: set[str] | None = None,
    known_axes: set[str] | None = None,
) -> Tensor | None:
    parsed = parse_derivative_name(
        name,
        known_fields=known_fields,
        known_axes=known_axes,
    )
    if parsed is not None:
        field, axis, order = parsed
        try:
            return context.get_derivative(field, axis, order)
        except (KeyError, ValueError):
            return None

    compound = parse_compound_derivative(
        name,
        known_fields=known_fields,
        known_axes=known_axes,
    )
    if compound is None:
        return None
    field, segments = compound
    try:
        first_axis, first_order = segments[0]
        result = context.get_derivative(field, first_axis, first_order)
        for axis, order in segments[1:]:
            result = context.diff(result, axis, order)
        return result
    except (AttributeError, KeyError, NotImplementedError, ValueError):
        return None


def _context_name_sets(
    context: ExecutionContext,
) -> tuple[set[str] | None, set[str] | None]:
    dataset = context.dataset
    known_fields = set(dataset.fields) if dataset.fields is not None else None
    known_axes = set(dataset.axes) if dataset.axes is not None else None
    return known_fields, known_axes


def _should_use_full_path(
    parsed: _ParsedExpression,
    context: ExecutionContext,
    *,
    force_diff_path: bool,
) -> bool:
    return (
        force_diff_path or _context_fields_missing(context) or parsed.has_open_form_diff
    )


def _context_fields_missing(context: ExecutionContext) -> bool:
    return context.dataset.fields is None


def _get_ast_depth(node: ast.AST) -> int:
    max_depth = 0

    stack: list[tuple[ast.AST, int]] = [(node, 1)]

    while stack:
        current, depth = stack.pop()
        max_depth = max(max_depth, depth)


        for child in ast.iter_child_nodes(current):
            stack.append((child, depth + 1))

    return max_depth


def _is_diff_operator(name: str) -> bool:
    return DIFF_OPERATOR_PATTERN.match(name) is not None


def _is_special_operator(name: str) -> bool:
    return name in _SPECIAL_OPERATORS


def _parse_diff_name(name: str) -> tuple[str, int]:
    match = DIFF_OPERATOR_PATTERN.match(name)
    if not match:
        raise ValueError(f"Invalid diff operator name: {name}")

    order_str, axis = match.groups()
    order = int(order_str) if order_str else 1


    if order < 1:
        raise ValueError(
            f"Diff order must be >= 1, got {order} in '{name}'. "
            f"Order 0 (identity) is not a valid derivative operation."
        )

    return axis, order


def has_open_form_diff(code: str) -> bool:

    if not code or not code.strip():
        raise ValueError("Expression cannot be empty")

    return _parse_expression(code).has_open_form_diff


def _tree_has_open_form_diff(tree: ast.Expression) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if _is_special_operator(func_name):
                return True



            if _is_diff_operator(func_name):
                return True

    return False
