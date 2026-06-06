
from __future__ import annotations

from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from kd.core.evaluator import Evaluator
from kd.core.executor import ExecutionContext
from kd.core.expr import (
    FunctionRegistry,
    PythonExecutor,
)
from kd.core.linear_solve import LeastSquaresSolver
from kd.data.derivatives.autograd import (
    AutogradProvider,
)
from kd.search.discover.pinn.executor import make_pinn_dataset



_N_COLLOC = 32
_HIDDEN = 8
_OOM_MESSAGE = "CUDA out of memory. Tried to allocate 526.00 MiB."


class _TinyField(nn.Module):

    def __init__(self, hidden: int = _HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([x, t], dim=-1)
        return cast(torch.Tensor, self.net(coords)).squeeze(-1)


def _build_evaluator() -> Evaluator:
    dataset = make_pinn_dataset(
        axis_names=["x", "t"],
        field_names=["u"],
        lhs_field="u",
        lhs_axis="t",
    )
    coords = {
        "x": torch.rand(_N_COLLOC, requires_grad=True),
        "t": torch.rand(_N_COLLOC, requires_grad=True),
    }
    model = _TinyField()
    provider = AutogradProvider(model, coords, dataset)
    context = ExecutionContext(
        dataset=dataset,
        derivative_provider=provider,
        device=coords["x"].device,
    )
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()



    with torch.enable_grad():
        lhs = provider.get_derivative("u", "t", 1).detach().reshape(-1)

    return Evaluator(executor, solver, context, lhs=lhs)


@pytest.mark.unit
def test_deep_diff_candidate_returns_invalid_on_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _build_evaluator()

    def oom_grad(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        raise torch.OutOfMemoryError(_OOM_MESSAGE)

    monkeypatch.setattr(torch.autograd, "grad", oom_grad)



    result = evaluator.evaluate_expression("diff2_x(diff_x(u))")

    assert result.is_valid is False
    assert "autograd OOM" in result.error_message
    assert result.expression == "diff2_x(diff_x(u))"


@pytest.mark.unit
def test_search_loop_continues_after_candidate_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _build_evaluator()

    call_counter = {"diffs": 0}
    real_grad = torch.autograd.grad

    def oom_then_ok(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        call_counter["diffs"] += 1


        if call_counter["diffs"] <= 2:
            raise torch.OutOfMemoryError(_OOM_MESSAGE)
        return real_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", oom_then_ok)

    oom_result = evaluator.evaluate_expression("diff2_x(diff_x(u))")
    assert oom_result.is_valid is False
    assert "autograd OOM" in oom_result.error_message



    shallow_result = evaluator.evaluate_expression("u")
    assert shallow_result.is_valid is True
