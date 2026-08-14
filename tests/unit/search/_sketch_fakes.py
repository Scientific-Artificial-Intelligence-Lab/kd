
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor

from kd.core.equation import Form
from kd.core.equation.sketch import (
    AnchoredTerm,
    PinnedTerm,
    Sketch,
    SketchMatchPolicy,
    TermConstraint,
    TermHole,
)
from kd.core.equation.types import LhsSpec
from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.expr.term_features import TermVocabulary
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.core.platform.sketch_compile import SketchClauseLevels
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.data.synthetic import generate_burgers_data
from kd.search.descriptor import InstrumentDescriptor, InstrumentMode
from kd.search.protocol import DiscoveryTask, PlatformComponents
from kd.search.result import ExperimentResult, default_final_result
from kd.search.runner import ExperimentRunner


BURGERS_NU = 0.1
PINNED_ADVECTION = "mul(u,u_x)"
PINNED_ADVECTION_VALUE = -1.0
HOLE_TERM = "u_xx"



SUPPORTING_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="exit_checked",
    operator_set="exit_checked",
    field_axis_set="exit_checked",
)





PYSINDY_DECLARED_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)




PYSINDY_NU = 0.3


def sketch_vocabulary(*, fields: tuple[str, ...] = ("u",)) -> TermVocabulary:
    return TermVocabulary(fields=frozenset(fields), coordinates=frozenset({"x", "t"}))


def match_policy(*, support_threshold: float = 0.0) -> SketchMatchPolicy:
    return SketchMatchPolicy(
        coeff_atol=1e-9,
        coeff_rtol=1e-9,
        support_threshold=support_threshold,
    )


def default_hole() -> TermHole:
    return TermHole(
        id="diffusion",
        min_count=1,
        max_count=1,
        constraint=TermConstraint(max_deriv_order=2),
    )


def burgers_sketch(
    *,
    pinned: tuple[PinnedTerm, ...] = (
        PinnedTerm(PINNED_ADVECTION, PINNED_ADVECTION_VALUE),
    ),
    anchored: tuple[AnchoredTerm, ...] = (),
    holes: tuple[TermHole, ...] | None = None,
    fields: tuple[str, ...] = ("u",),
    lhs_spec: LhsSpec | None = None,
    support_threshold: float = 0.0,
) -> Sketch:
    return Sketch(
        lhs_spec=LhsSpec("u", "t", 1) if lhs_spec is None else lhs_spec,
        vocabulary=sketch_vocabulary(fields=fields),
        pinned=pinned,
        anchored=anchored,
        holes=(default_hole(),) if holes is None else holes,
        match_policy=match_policy(support_threshold=support_threshold),
    )


def tiny_burgers_dataset(*, nx: int = 64, nt: int = 51) -> PDEDataset:
    return generate_burgers_data(nx=nx, nt=nt, nu=BURGERS_NU, seed=0)


def pysindy_burgers_dataset(*, nx: int = 64, nt: int = 51) -> PDEDataset:
    return generate_burgers_data(nx=nx, nt=nt, nu=PYSINDY_NU, seed=0)


def constant_field_dataset(value: float = 2.0) -> PDEDataset:
    x = torch.linspace(-1.0, 1.0, 8)
    t = torch.linspace(0.0, 1.0, 6)
    return PDEDataset(
        name="constant",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", torch.full((8, 6), value))},
        lhs_field="u",
        lhs_axis="t",
        lhs_order=1,
    )


def fake_descriptor(
    *,
    algorithm: str = "sketch_fake",
    sketch: SketchClauseLevels = SUPPORTING_LEVELS,
) -> InstrumentDescriptor:
    return InstrumentDescriptor(
        algorithm=algorithm,
        summary="In-test platform-evaluator plugin for the sketch exit contract.",
        cost_class="light",
        modes=(
            InstrumentMode(
                name="grid",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
                sketch=sketch,
            ),
        ),
        knobs=(),
    )


class _BaseSketchFakePlugin:

    score_kind: ClassVar[str] = "AIC"
    score_direction: ClassVar[Literal["min", "max"]] = "min"

    def __init__(
        self,
        terms: Sequence[str] = (HOLE_TERM,),
        *,
        algorithm: str = "sketch_fake",
        form: Form = Form.EVOLUTION,
    ) -> None:
        self._terms = tuple(terms)
        self._algorithm = algorithm
        self._form = form
        self._evaluator: Evaluator | None = None
        self._best_score = float("inf")
        self._best_expression = ""
        self._state: dict[str, Any] = {}

    @property
    def derivative_requirements(self) -> DerivativeReqs:
        return DerivativeReqs(
            provider_kind="finite_diff", max_atomic_order=2, lhs_order=1
        )

    def prepare(self, components: PlatformComponents) -> None:
        if components.evaluator is None:
            raise ValueError("sketch fake plugin requires a platform evaluator")
        self._evaluator = components.evaluator

    def propose(self, n: int) -> list[str]:
        return [" + ".join(self._terms)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        evaluator = self._require_evaluator()
        return [evaluator.evaluate_expression(expr) for expr in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        for result in results:
            score = result.score
            if result.is_valid and score is not None and score < self._best_score:
                self._best_score = score
                self._best_expression = result.expression

    def build_final_result(self) -> EvaluationResult:
        result = default_final_result(self._best_expression, self._require_evaluator())
        result.form = self._form
        return result

    def build_result_target(self) -> Tensor:
        return self._require_evaluator().lhs_target.detach().clone()

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_expression(self) -> str:
        return self._best_expression

    @property
    def config(self) -> dict[str, Any]:
        return {
            "algorithm": self._algorithm,
            "terms": list(self._terms),
            "seed": 0,
        }

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = dict(value)

    def _require_evaluator(self) -> Evaluator:
        if self._evaluator is None:
            raise ValueError("sketch fake plugin was not prepared")
        return self._evaluator


class SketchFakePlugin(_BaseSketchFakePlugin):

    descriptor: ClassVar[InstrumentDescriptor] = fake_descriptor()


class UnsupportedSketchPlugin(_BaseSketchFakePlugin):

    descriptor: ClassVar[InstrumentDescriptor] = fake_descriptor(
        algorithm="sketch_fake_unsupported", sketch=SketchClauseLevels()
    )


class UndeclaredSketchPlugin(_BaseSketchFakePlugin):
    pass


def build_components(
    dataset: PDEDataset,
    *,
    task: DiscoveryTask | None = None,
    provider_kind: Literal["finite_diff", "autograd", "none"] = "finite_diff",
    sketch_lower_owner: Literal["platform", "native"] = "platform",
) -> PlatformComponents:
    reqs = DerivativeReqs(provider_kind=provider_kind, max_atomic_order=2, lhs_order=1)
    return PlatformBuilder(
        dataset, reqs, task=task, sketch_lower_owner=sketch_lower_owner
    ).build()


def run_plugin(
    components: PlatformComponents,
    plugin: _BaseSketchFakePlugin,
) -> ExperimentResult:
    runner = ExperimentRunner(algorithm=plugin, max_iterations=1, batch_size=1)
    return runner.run(components)
