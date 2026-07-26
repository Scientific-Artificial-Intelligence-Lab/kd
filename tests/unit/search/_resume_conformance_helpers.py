
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.llm4ed.config import Llm4edConfig
from kd.search.protocol import PlatformComponents


def make_plugin(plugin_cls: type) -> Any:
    if plugin_cls.config_cls is EqGPTConfig:
        return plugin_cls(EqGPTConfig(sparsity_alpha=0.02))
    if plugin_cls.config_cls is Llm4edConfig:
        return plugin_cls(Llm4edConfig())
    return plugin_cls()







def make_fingerprint_plugin(algorithm: str, terms: tuple[str, ...]) -> Any:
    if algorithm == "pysr":
        from kd.search.pysr.config import PySRConfig
        from kd.search.pysr.plugin import PySRPlugin

        return PySRPlugin(PySRConfig(terms=terms, seed=0))
    if algorithm == "pysindy":
        from kd.search.pysindy.config import PySINDyConfig
        from kd.search.pysindy.plugin import PySINDyPlugin

        return PySINDyPlugin(PySINDyConfig(terms=terms, seed=0))
    raise ValueError(f"no fingerprint plugin for {algorithm!r}")


def plugin_fingerprint(algorithm: str, terms: tuple[str, ...]) -> str:
    return str(make_fingerprint_plugin(algorithm, terms)._library.fingerprint)






_SGA_GRID = 10
_SGA_TIME = 5


def _sga_dataset() -> PDEDataset:
    generator = torch.Generator().manual_seed(0)
    return PDEDataset(
        name="resume-conf-sga",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=torch.linspace(0.0, 1.0, _SGA_GRID)),
            "t": AxisInfo(name="t", values=torch.linspace(0.0, 1.0, _SGA_TIME)),
        },
        axis_order=["x", "t"],
        fields={
            "u": FieldData(
                name="u",
                values=torch.randn(_SGA_GRID, _SGA_TIME, generator=generator),
            )
        },
        lhs_field="u",
        lhs_axis="t",
    )


def _sga_mock_context(dataset: PDEDataset) -> MagicMock:
    context = MagicMock()
    context.dataset = dataset
    provider = MagicMock()

    def get_derivative(field_name: str, axis: str, order: int) -> torch.Tensor:
        seed = hash((field_name, axis, order)) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(_SGA_GRID, _SGA_TIME, generator=gen)

    def get_variable(name: str) -> torch.Tensor:
        if dataset.fields is not None and name in dataset.fields:
            return dataset.fields[name].values
        if dataset.axes is not None and name in dataset.axes:
            return dataset.axes[name].values
        raise KeyError(name)

    provider.get_derivative = get_derivative
    context.derivative_provider = provider
    context.get_variable = get_variable
    context.get_derivative = get_derivative
    return context


def sga_components() -> PlatformComponents:
    dataset = _sga_dataset()
    return PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=_sga_mock_context(dataset),
        registry=MagicMock(),
    )


def make_sga_plugin(num: int) -> Any:
    from kd.search.sga.config import SGAConfig
    from kd.search.sga.plugin import SGAPlugin

    config = SGAConfig(
        num=num,
        depth=3,
        width=3,
        p_var=0.6,
        p_mute=0.3,
        p_cro=0.5,
        p_rep=1.0,
        seed=42,
        maxit=3,
        str_iters=3,
        d_tol=0.5,
    )
    return SGAPlugin(config=config)







class _ExactQuadraticModel(nn.Module):

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": 1.0 + x * x + t * t}


def _dlga_dataset() -> PDEDataset:
    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    return PDEDataset(
        name="resume-conf-dlga",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", 1.0 + xg * xg + tg * tg)},
        lhs_field="u",
        lhs_axis="t",
    )


def make_dlga_plugin(pop_size: int) -> Any:
    from kd.search.dlga import DLGAConfig, DLGAPlugin

    return DLGAPlugin(
        DLGAConfig(pop_size=pop_size, seed=13),
        surrogate_model=_ExactQuadraticModel(),
    )


def dlga_components(pop_size: int) -> PlatformComponents:
    from kd.core.platform.builder import PlatformBuilder

    reqs = make_dlga_plugin(pop_size).derivative_requirements
    return PlatformBuilder(_dlga_dataset(), reqs).build()






_EQGPT_BATCH = 2
_EQGPT_SEED = 7


def _eqgpt_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    grid_x, grid_t = torch.meshgrid(x, t, indexing="ij")
    return PDEDataset(
        name="resume-conf-eqgpt",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(grid_x) * torch.cos(grid_t))},
        lhs_field="u",
        lhs_axis="t",
    )


def eqgpt_components() -> PlatformComponents:
    from kd.core.platform.builder import PlatformBuilder
    from kd.core.platform.requirements import DerivativeReqs

    return PlatformBuilder(_eqgpt_dataset(), DerivativeReqs()).build()


def make_eqgpt_plugin(
    *, samples_per_epoch: int = _EQGPT_BATCH, finetune_lr: float = 1e-3
) -> Any:
    from kd.search.eqgpt.backend import FakeGPTBackend
    from kd.search.eqgpt.plugin import EqGPTPlugin
    from kd.search.eqgpt.vocab import load_vocab

    words = load_vocab().word2id
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=_EQGPT_SEED,
        samples_per_epoch=samples_per_epoch,
        finetune_lr=finetune_lr,
        top_k=4,
        max_length=12,
        variables=("t", "x"),
        masked_tokens=frozenset({words["uxxxx"], words["uxxxxx"]}),
    )
    return EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=_EQGPT_SEED))


def run_eqgpt_epochs(plugin: Any, n_epochs: int) -> None:
    batch = plugin.runner_batch_size
    for _ in range(n_epochs):
        candidates = plugin.propose(batch)
        plugin.update(plugin.evaluate(candidates))








DISCOVER_CHAMPION_IR = "diff_x(u)"


@dataclass(frozen=True)
class DiscoverStubSpec:

    nmse: float = 0.04
    complexity: int = 2
    coefficient: float = 1.5


def _discover_result(spec: DiscoverStubSpec, expr: str) -> Any:
    from kd.core.evaluator import EvaluationResult

    return EvaluationResult(
        mse=spec.nmse,
        nmse=spec.nmse,
        r2=1.0 - spec.nmse,
        complexity=spec.complexity,
        is_valid=True,
        expression=expr,
        terms=[expr],
        coefficients=torch.tensor([spec.coefficient]),
        selected_indices=None,
    )


class DiscoverCountingEvaluator:

    def __init__(self, spec: DiscoverStubSpec | None = None) -> None:
        self._spec = spec or DiscoverStubSpec()
        self.calls: list[str] = []

    def evaluate_expression(self, expr: str) -> Any:
        self.calls.append(expr)
        return _discover_result(self._spec, expr)


class DiscoverVaryingEvaluator:

    def evaluate_expression(self, expr: str) -> Any:
        nmse = 0.01 + (len(expr) % 5) * 0.02
        spec = DiscoverStubSpec(nmse=nmse, complexity=1 + len(expr) % 3)
        return _discover_result(spec, expr)


def discover_components(evaluator: Any) -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=evaluator,
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )


def discover_donor_payload(
    config: Any, *, champion_reward: float, champion_expr: str
) -> dict[str, Any]:
    from kd.search.discover.plugin import DISCOVERPlugin

    donor = DISCOVERPlugin(config)
    donor.prepare(discover_components(DiscoverCountingEvaluator()))
    payload: dict[str, Any] = donor.state
    payload["engine_state"]["best_reward"] = champion_reward
    payload["engine_state"]["best_expression"] = champion_expr
    return payload


def discover_expected_reprice(config: Any, spec: DiscoverStubSpec) -> float:
    from kd.search.discover.evaluation.reward import compute_reward

    result = _discover_result(spec, DISCOVER_CHAMPION_IR)
    return float(np.float32(compute_reward(result, alpha=config.reward_alpha)))


def discover_restore_then_prepare(
    config: Any, payload: dict[str, Any], stub: Any
) -> Any:
    from kd.search.discover.plugin import DISCOVERPlugin

    subject = DISCOVERPlugin(config)
    subject.state = payload
    subject.prepare(discover_components(stub))
    return subject


def discover_donor_with_trained_optimizer(config: Any) -> dict[str, Any]:
    from kd.search.discover.plugin import DISCOVERPlugin

    donor = DISCOVERPlugin(config)
    donor.prepare(discover_components(DiscoverVaryingEvaluator()))
    candidates = donor.propose(donor.runner_batch_size)
    donor.update(donor.evaluate(candidates))
    return donor.state








def resume_symmetry_pair(algorithm: str) -> tuple[Any, Any]:
    if algorithm == "sga":
        donor = make_sga_plugin(num=6)
        donor.prepare(sga_components())
        donor.update(donor.evaluate(donor.propose(donor.runner_batch_size)))
        return donor, make_sga_plugin(num=6)
    if algorithm == "dlga":
        donor = make_dlga_plugin(pop_size=4)
        donor.prepare(dlga_components(4))
        donor.update(donor.evaluate(donor.propose(donor.runner_batch_size)))
        return donor, make_dlga_plugin(pop_size=4)
    if algorithm == "eqgpt":
        donor = make_eqgpt_plugin()
        donor.prepare(eqgpt_components())
        run_eqgpt_epochs(donor, 1)
        return donor, make_eqgpt_plugin()
    if algorithm == "discover":
        from kd.search.discover.config import DiscoverConfig
        from kd.search.discover.plugin import DISCOVERPlugin

        config = DiscoverConfig()
        donor = DISCOVERPlugin(config)
        donor.prepare(discover_components(DiscoverCountingEvaluator()))
        donor.update(donor.evaluate(donor.propose(donor.runner_batch_size)))
        return donor, DISCOVERPlugin(config)
    if algorithm == "llm4ed":
        from kd.search.llm4ed.plugin import Llm4edPlugin
        from tests.unit.search.llm4ed._plugin_helpers import (
            GOOD,
            FakeProvider,
            make_config,
            prepared,
            run,
        )

        donor, _ = prepared(config=make_config(), provider=FakeProvider(GOOD))
        run(donor, rounds=1)
        fresh = Llm4edPlugin(make_config(), provider=FakeProvider(GOOD))
        return donor, fresh
    if algorithm in ("pysr", "pysindy"):
        terms = ("u", "u_x")
        return (
            make_fingerprint_plugin(algorithm, terms),
            make_fingerprint_plugin(algorithm, terms),
        )
    raise ValueError(f"no resume symmetry pair for {algorithm!r}")


__all__ = [
    "make_plugin",
    "make_fingerprint_plugin",
    "resume_symmetry_pair",
    "plugin_fingerprint",
    "sga_components",
    "make_sga_plugin",
    "dlga_components",
    "make_dlga_plugin",
    "eqgpt_components",
    "make_eqgpt_plugin",
    "run_eqgpt_epochs",
    "DiscoverStubSpec",
    "DiscoverCountingEvaluator",
    "DiscoverVaryingEvaluator",
    "DISCOVER_CHAMPION_IR",
    "discover_components",
    "discover_donor_payload",
    "discover_expected_reprice",
    "discover_restore_then_prepare",
    "discover_donor_with_trained_optimizer",
]
