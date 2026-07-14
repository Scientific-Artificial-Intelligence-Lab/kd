
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from unittest.mock import MagicMock

import numpy as np
import torch

from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.llm import LLMRequest, LLMResponse
from kd.search.llm4ed.config import Llm4edConfig
from kd.search.llm4ed.fd import build_operand_columns
from kd.search.llm4ed.plugin import Llm4edPlugin
from kd.search.llm4ed.score import score_equation
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder




ResponseSpec = str | Sequence[str] | Callable[[LLMRequest, int], str]







def res(*equations: str) -> str:
    return "\n".join(f"<res>{eq}</res>" for eq in equations)



GOOD = res("u_xx")


MIX = res("u_xx", "u", "sin(u)", "u_xxxx")



SPREAD = res("u + u_xx", "u", "u_xx", "x", "sin(u)", "u_xxxx")


SPREAD3 = res("u + u_xx", "u", "u_xx", "sin(u)", "u_xxxx")


ALL_INVALID = res("sin(u)", "u_xxxx", "u^6")



DIRTY = (
    "<select>{u_xx}</select>\n<cross>u_x + u_xx</cross>"
    "\n<res>u_xx</res>\n<res>u_x</res>"
)


def raises(exc: BaseException) -> Callable[[LLMRequest, int], str]:

    def _raise(_request: LLMRequest, _index: int) -> str:
        raise exc

    return _raise







class FakeProvider:

    def __init__(self, response: ResponseSpec, *, model: str = "fake") -> None:
        self._response = response
        self.model = model
        self.requests: list[LLMRequest] = []
        self.prepared = False

    def prepare(self) -> None:
        self.prepared = True

    def complete(self, request: LLMRequest) -> LLMResponse:
        index = len(self.requests)
        self.requests.append(request)
        if callable(self._response):
            text = self._response(request, index)
        elif isinstance(self._response, str):
            text = self._response
        else:
            sequence = list(self._response)
            text = sequence[index] if index < len(sequence) else sequence[-1]
        return LLMResponse(text=text, model=self.model, usage=None)

    @property
    def seeds(self) -> list[int]:
        return [request.seed for request in self.requests]

    @property
    def prompts(self) -> list[str]:
        return [request.prompt for request in self.requests]







def heat_dataset(n_x: int = 48, n_t: int = 30) -> PDEDataset:
    x = torch.linspace(0.0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0.0, 0.4, n_t, dtype=torch.float64)
    grid_x, grid_t = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(grid_x) * torch.exp(-grid_t)
    return PDEDataset(
        name="llm4ed_heat",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def two_mode_dataset(n_x: int = 60, n_t: int = 40) -> PDEDataset:
    x = torch.linspace(0.0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0.0, 0.4, n_t, dtype=torch.float64)
    grid_x, grid_t = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(grid_x) * torch.exp(-grid_t) + torch.cos(2 * grid_x) * torch.exp(
        -0.5 * grid_t
    )
    return PDEDataset(
        name="llm4ed_two_mode",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _operand_columns(
    dataset: PDEDataset,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    u = dataset.fields["u"].values.detach().cpu().numpy().astype(np.float64)
    x = dataset.axes["x"].values.detach().cpu().numpy().astype(np.float64)
    t = dataset.axes["t"].values.detach().cpu().numpy().astype(np.float64)
    return build_operand_columns(u, x, t)


def reward_of(equation: str, dataset: PDEDataset) -> float | None:
    lhs, features = _operand_columns(dataset)
    return score_equation(equation, lhs, features).reward


def make_config(**overrides: object) -> Llm4edConfig:
    defaults: dict[str, object] = {
        "seed": 0,
        "temperature": 0.8,
        "max_tokens": 64,
        "init_num": 4,
        "pool_size": 5,
        "max_llm_calls_per_propose": 4,
        "max_llm_calls_per_run": 200,
    }
    defaults.update(overrides)
    return Llm4edConfig(**defaults)


def components_for(dataset: PDEDataset | None = None) -> PlatformComponents:
    return PlatformComponents(
        dataset=dataset if dataset is not None else heat_dataset(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
        recorder=VizRecorder(),
    )


def prepared(
    *,
    config: Llm4edConfig | None = None,
    provider: object | None = None,
    dataset: PDEDataset | None = None,
) -> tuple[Llm4edPlugin, PlatformComponents]:
    components = components_for(dataset)
    plugin = Llm4edPlugin(config or make_config(), provider=provider)
    plugin.prepare(components)
    return plugin, components


def run(plugin: Llm4edPlugin, rounds: int, *, n: int = 4) -> None:
    for _ in range(rounds):
        candidates = plugin.propose(n)
        plugin.update(plugin.evaluate(candidates))
