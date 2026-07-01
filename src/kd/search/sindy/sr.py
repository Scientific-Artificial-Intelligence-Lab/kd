
from __future__ import annotations

import keyword
import re
from collections.abc import Sequence
from typing import Self, cast

import numpy as np
import numpy.typing as npt
import torch

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve import STRidgeSolver
from kd.data.derivatives.base import DerivativeProvider
from kd.data.schema import FieldData, PDEDataset, TaskType

FloatArray = npt.NDArray[np.float64]

_COEFFICIENT_FORMAT = "+.4g"
_DATASET_NAME = "sindy-tabular"
_NO_DERIVATIVE_MESSAGE = "SINDyRegressor does not support derivative terms"
_TORCH_DTYPE = torch.float64
_ZERO_EXPRESSION = "0"





_KD_IR_RESERVED_NAMES: frozenset[str] = frozenset(
    FunctionRegistry.create_default().list_names()
)
_DIFF_RESERVED_PATTERN = re.compile(r"^diff[0-9]*_[a-z]+$")


class SINDyRegressor:

    def __init__(self) -> None:
        self.selected_terms_: list[str] | None = None
        self.coefficients_: FloatArray | None = None
        self.expression_: str | None = None
        self.nmse_: float | None = None
        self.var_names_: list[str] | None = None
        self.n_features_in_: int | None = None

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        terms: Sequence[str],
        var_names: Sequence[str] | None = None,
    ) -> Self:
        x_arr, y_arr, names = _validate_fit_inputs(X, y, var_names)
        term_list = _validate_terms(terms)

        context = _build_context(x_arr, names)




        _execute_terms(term_list, context, x_arr.shape[0])
        lhs = _to_tensor(y_arr)
        evaluator = Evaluator(
            PythonExecutor(FunctionRegistry.create_default()),
            STRidgeSolver(),
            context,
            lhs=lhs,
        )
        result = evaluator.evaluate_terms(term_list, skip_invalid=False)
        if not result.is_valid:
            raise ValueError(result.error_message)
        if result.coefficients is None:
            raise ValueError("Evaluator returned no coefficients")

        result_terms = result.terms if result.terms is not None else term_list
        selected_indices = _resolve_selected_indices(
            result.selected_indices,
            result_terms,
        )
        selected_terms = [result_terms[index] for index in selected_indices]
        full_coefficients = _tensor_to_array(result.coefficients)
        selected_coefficients = _select_coefficients(
            full_coefficients,
            selected_indices,
        )
        expression = _render_expression(selected_terms, selected_coefficients)
        nmse = float(result.nmse)
        n_features = x_arr.shape[1]

        self.selected_terms_ = selected_terms
        self.coefficients_ = selected_coefficients
        self.expression_ = expression
        self.nmse_ = nmse
        self.var_names_ = names
        self.n_features_in_ = n_features
        return self

    def predict(self, X: npt.ArrayLike) -> FloatArray:
        selected_terms = self.selected_terms_
        coefficients = self.coefficients_
        var_names = self.var_names_
        if (
            selected_terms is None
            or coefficients is None
            or var_names is None
            or self.n_features_in_ is None
        ):
            raise RuntimeError("SINDyRegressor.predict was called before fit(X, y).")

        x_arr = self._validate_predict_inputs(X)
        if not selected_terms:
            return np.zeros(x_arr.shape[0], dtype=np.float64)

        context = _build_context(x_arr, var_names)
        theta = _execute_terms(selected_terms, context, x_arr.shape[0])
        coefficient_tensor = _to_tensor(coefficients)
        prediction = theta @ coefficient_tensor
        return _tensor_to_array(prediction)

    def _validate_predict_inputs(self, X: npt.ArrayLike) -> FloatArray:



        assert self.n_features_in_ is not None
        x_arr = _as_float_array(X)
        if x_arr.ndim != 2:
            raise ValueError(f"SINDyRegressor.predict expects 2-D X, got {x_arr.shape}")
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "SINDyRegressor.predict received mismatched feature count: "
                f"X.shape[1]={x_arr.shape[1]}, expected={self.n_features_in_}"
            )
        _raise_if_not_finite(x_arr, name="X")
        return x_arr


class _NoDerivativeProvider(DerivativeProvider):

    def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
        raise NotImplementedError(_NO_DERIVATIVE_MESSAGE)

    def diff(self, expression: torch.Tensor, axis: str, order: int) -> torch.Tensor:
        raise NotImplementedError(_NO_DERIVATIVE_MESSAGE)

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        raise NotImplementedError(_NO_DERIVATIVE_MESSAGE)


def _validate_fit_inputs(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    var_names: Sequence[str] | None,
) -> tuple[FloatArray, FloatArray, list[str]]:
    x_arr = _as_float_array(X)
    y_raw = _as_float_array(y)
    if y_raw.ndim >= 2 and (y_raw.ndim != 2 or y_raw.shape[1] != 1):
        raise ValueError(
            f"SINDyRegressor.fit expects y with shape (n,) or (n, 1), got {y_raw.shape}"
        )
    y_arr = y_raw.reshape(-1)
    if x_arr.ndim != 2:
        raise ValueError(f"SINDyRegressor.fit expects 2-D X, got {x_arr.shape}")
    if x_arr.shape[0] == 0 or x_arr.shape[1] == 0:
        raise ValueError(f"SINDyRegressor.fit expects non-empty X, got {x_arr.shape}")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            "SINDyRegressor.fit received mismatched sample counts: "
            f"X.shape[0]={x_arr.shape[0]}, y.shape[0]={y_arr.shape[0]}"
        )
    _raise_if_not_finite(x_arr, name="X")
    _raise_if_not_finite(y_arr, name="y")
    if isinstance(var_names, str):
        raise ValueError("var_names must be a sequence of names, not a single string")
    names = (
        _generic_feature_names(x_arr.shape[1]) if var_names is None else list(var_names)
    )
    if len(names) != x_arr.shape[1]:
        raise ValueError(
            "var_names length must equal the number of features: "
            f"len(var_names)={len(names)}, n_features={x_arr.shape[1]}"
        )
    _validate_var_names(names)
    return x_arr, y_arr, names


def _validate_terms(terms: Sequence[str]) -> list[str]:
    if isinstance(terms, str):
        raise ValueError("terms must be a sequence of term strings, not a string")
    term_list: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not isinstance(term, str):
            raise ValueError(f"terms must contain strings, got {term!r}")
        normalized = term.strip()
        if not normalized:
            raise ValueError("terms must not contain empty strings")
        if normalized in seen:
            raise ValueError(f"Duplicate term {normalized!r}")
        seen.add(normalized)
        term_list.append(normalized)
    if not term_list:
        raise ValueError("terms must contain at least one candidate")
    return term_list


def _generic_feature_names(n_features: int) -> list[str]:
    return [f"x{index + 1}" for index in range(n_features)]


def _validate_var_names(names: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"var_names must be valid Python identifiers, got {name!r}"
            )
        if keyword.iskeyword(name):
            raise ValueError(f"var_name {name!r} is a Python keyword")
        if name in _KD_IR_RESERVED_NAMES or _is_diff_reserved_name(name):
            raise ValueError(f"var_name {name!r} collides with a reserved kd-IR token")
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        joined = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"var_names must be unique; duplicate names: {joined}")


def _is_diff_reserved_name(name: str) -> bool:
    return _DIFF_RESERVED_PATTERN.match(name) is not None


def _build_context(x_arr: FloatArray, var_names: Sequence[str]) -> ExecutionContext:
    columns = {
        name: FieldData(name=name, values=_to_tensor(x_arr[:, index]))
        for index, name in enumerate(var_names)
    }
    dataset = PDEDataset(
        name=_DATASET_NAME,
        task_type=TaskType.REGRESSION,
        fields=columns,
    )
    return ExecutionContext(
        dataset=dataset,
        derivative_provider=_NoDerivativeProvider(),
    )


def _execute_terms(
    terms: Sequence[str],
    context: ExecutionContext,
    n_samples: int,
) -> torch.Tensor:
    executor = PythonExecutor(FunctionRegistry.create_default())
    columns: list[torch.Tensor] = []
    with torch.no_grad():
        for term in terms:
            try:
                raw = executor.execute(term, context).value
            except Exception as exc:
                raise ValueError(f"Execution error for '{term}': {exc}") from exc




            if raw.dim() == 0:
                raise ValueError(
                    f"Term {term!r} is a scalar constant; SINDy has no intercept "
                    f"column (pure-constant terms are unsupported)"
                )
            values = raw.flatten()
            if values.shape[0] != n_samples:
                raise ValueError(
                    f"Term {term!r} produced {values.shape[0]} values; "
                    f"expected {n_samples}"
                )
            if not torch.isfinite(values).all():
                raise ValueError(f"Term {term!r} produced NaN or Inf values")
            columns.append(values)
    return torch.stack(columns, dim=1)


def _resolve_selected_indices(
    selected_indices: list[int] | None,
    terms: Sequence[str],
) -> list[int]:
    if selected_indices is None:
        return list(range(len(terms)))
    return list(selected_indices)


def _select_coefficients(
    coefficients: FloatArray,
    selected_indices: Sequence[int],
) -> FloatArray:
    return _as_float_array(coefficients[list(selected_indices)])


def _render_expression(terms: Sequence[str], coefficients: FloatArray) -> str:
    if not terms:
        return _ZERO_EXPRESSION
    parts = [
        f"{float(coef):{_COEFFICIENT_FORMAT}}*{term}"
        for term, coef in zip(terms, coefficients, strict=True)
    ]
    return " ".join(parts)


def _as_float_array(values: npt.ArrayLike) -> FloatArray:
    return cast(FloatArray, np.asarray(values, dtype=np.float64))


def _to_tensor(values: npt.ArrayLike) -> torch.Tensor:
    return torch.as_tensor(values, dtype=_TORCH_DTYPE)


def _tensor_to_array(values: torch.Tensor) -> FloatArray:
    array = values.detach().cpu().numpy().astype(np.float64, copy=False)
    return cast(FloatArray, array.reshape(-1))


def _raise_if_not_finite(values: FloatArray, *, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


__all__ = ["SINDyRegressor"]
