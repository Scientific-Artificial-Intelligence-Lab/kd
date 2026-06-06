
from __future__ import annotations

import numpy as np
import pytest
import torch

from kd.core.executor.context import (
    ExecutionContext,
)
from kd.core.expr import (
    FunctionRegistry,
    PythonExecutor,
)
from kd.data.derivatives.finite_diff import (
    FiniteDiffProvider,
)
from kd.data.schema import (
    AxisInfo,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.discover.builder import build_library, build_prior_system
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.controller.tree_state import BatchTracker
from kd.search.discover.tokens.library import Library, LibraryConfig, TokenType
from kd.search.discover.tokens.validator import CandidateValidator




_NX: int = 32
_NT: int = 16
_X_MIN: float = 1.0
_X_MAX: float = 2.0
_T_MIN: float = 0.0
_T_MAX: float = 1.0




_FD_TOLERANCE: float = 1e-3
_INTERIOR_TRIM: int = 2


_OPERATORS: tuple[str, ...] = (
    "add", "mul", "sub", "div", "diff_x", "diff2_x", "diff3_x", "n2", "n3",
)



_GT_PREORDER: tuple[str, ...] = (
    "add", "div", "diff_x", "u", "x", "diff2_x", "u",
)


@pytest.fixture(scope="module")
def divide_dataset() -> PDEDataset:
    x_vals = torch.linspace(_X_MIN, _X_MAX, _NX, dtype=torch.float64)
    t_vals = torch.linspace(_T_MIN, _T_MAX, _NT, dtype=torch.float64)
    grid_x, grid_t = torch.meshgrid(x_vals, t_vals, indexing="ij")
    u = (grid_x * grid_t).contiguous()
    return PDEDataset(
        name="pde_divide_token_routing_test",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x_vals),
            "t": AxisInfo(name="t", values=t_vals),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture(scope="module")
def divide_executor_context(
    divide_dataset: PDEDataset,
) -> tuple[PythonExecutor, ExecutionContext]:
    provider = FiniteDiffProvider(divide_dataset, max_order=2)
    context = ExecutionContext(
        dataset=divide_dataset, derivative_provider=provider,
    )
    executor = PythonExecutor(FunctionRegistry.create_default())
    return executor, context







@pytest.mark.unit
def test_executor_routes_div_diff_x_u_by_x(
    divide_dataset: PDEDataset,
    divide_executor_context: tuple[PythonExecutor, ExecutionContext],
) -> None:
    executor, context = divide_executor_context
    result = executor.execute("div(diff_x(u), x)", context)
    actual = result.value
    assert isinstance(actual, torch.Tensor)
    assert torch.isfinite(actual).all(), "executor produced non-finite values"


    assert divide_dataset.axes is not None
    x_vals = divide_dataset.axes["x"].values
    t_vals = divide_dataset.axes["t"].values
    grid_x, grid_t = torch.meshgrid(x_vals, t_vals, indexing="ij")
    expected = (grid_t / grid_x).to(torch.float64)




    actual_64 = actual.to(torch.float64).reshape(_NX, _NT)
    interior = (slice(_INTERIOR_TRIM, -_INTERIOR_TRIM), slice(None))
    err = (actual_64[interior] - expected[interior]).abs().max().item()
    assert err < _FD_TOLERANCE, (
        f"executor `div(diff_x(u), x)` interior max abs error {err:.3e} "
        f"exceeds FD tolerance {_FD_TOLERANCE}; either `x` is mis-routed "
        f"as a leaf token or the coord broadcast does not align with the "
        f"`u_x` field shape (axis_order=['x','t'])."
    )







@pytest.mark.unit
def test_library_registers_x_as_leaf_and_div_as_binary() -> None:
    library: Library = Library.from_config(
        LibraryConfig(
            operators=list(_OPERATORS),
            state_vars=["u"],
            coord_vars=["x", "t"],
        ),
    )

    x_idx = library.name_to_index("x")
    assert library.tokens[x_idx].arity == 0, (
        f"`x` should be arity-0 leaf, got arity={library.tokens[x_idx].arity}"
    )
    assert library.tokens[x_idx].token_type == TokenType.COORDINATE
    assert x_idx in library.terminal_tokens.tolist(), (
        "`x` index missing from `Library.terminal_tokens`; SR controller "
        "won't be able to leaf-sample it."
    )

    div_idx = library.name_to_index("div")
    assert library.tokens[div_idx].arity == 2, (
        f"`div` should be arity-2 binary, got "
        f"arity={library.tokens[div_idx].arity}"
    )
    assert library.tokens[div_idx].token_type == TokenType.OPERATOR







@pytest.mark.unit
def test_validator_and_prior_allow_div_x_preorder() -> None:
    config = DiscoverConfig(
        n_iterations=1,
        batch_size=1,
        max_length=30,
        library=LibraryConfig(
            operators=list(_OPERATORS),
            state_vars=["u"],
            coord_vars=["x", "t"],
        ),
        num_units=16,
        num_layers=1,
        embedding_dim=4,
    )
    library = build_library(config)
    prior_system = build_prior_system(library, config)
    validator = CandidateValidator(
        library,
        max_length=config.max_length,
        max_diff_order=config.max_diff_order,
        min_length=config.min_length,
    )

    token_ids = np.asarray(
        [library.name_to_index(name) for name in _GT_PREORDER],
        dtype=np.int32,
    )

    assert validator.validate_single(token_ids), (
        f"CandidateValidator rejected paper PDE_divide GT preorder "
        f"{_GT_PREORDER}; check max_length / max_diff_order / min_length."
    )

    actions = token_ids[np.newaxis,:]
    obs = BatchTracker(library).compute_obs(actions)
    priors = prior_system.compute_batch(actions, obs)
    for step_idx, tok_id in enumerate(token_ids.tolist()):
        logit_adjust = float(priors[0, step_idx, int(tok_id)])
        assert np.isfinite(logit_adjust), (
            f"PriorSystem forbids paper GT token "
            f"'{_GT_PREORDER[step_idx]}' at step {step_idx} "
            f"(logit adjustment = -inf); the GT preorder is unreachable."
        )
