
from __future__ import annotations

import pytest
import torch

from kd.core.equation.sketch import PinnedTerm, Sketch
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import PDEDataset
from kd.search.discover.runners.sampled_evaluator import SampledEvaluator
from kd.search.protocol import DiscoveryTask, PlatformComponents
from tests.unit.search._sketch_fakes import (
    BURGERS_NU,
    HOLE_TERM,
    PINNED_ADVECTION,
    PINNED_ADVECTION_VALUE,
    build_components,
    burgers_sketch,
    constant_field_dataset,
    default_hole,
    tiny_burgers_dataset,
)


def column(components: PlatformComponents, term_ir: str) -> torch.Tensor:
    assert components.context is not None
    return (
        components.executor.execute(term_ir, components.context)
        .value.detach()
        .flatten()
    )


@pytest.fixture(scope="module")
def burgers_platform() -> tuple[PlatformComponents, PlatformComponents]:
    dataset = tiny_burgers_dataset()
    task = DiscoveryTask.from_sketch(burgers_sketch())
    return build_components(dataset), build_components(dataset, task=task)


def test_lowered_target_is_the_lhs_minus_the_pinned_column(
    burgers_platform: tuple[PlatformComponents, PlatformComponents],
) -> None:
    raw, lowered = burgers_platform
    assert raw.evaluator is not None
    assert lowered.evaluator is not None

    expected = raw.evaluator.lhs_target - PINNED_ADVECTION_VALUE * column(
        lowered, PINNED_ADVECTION
    )

    torch.testing.assert_close(
        lowered.evaluator.lhs_target, expected, rtol=0.0, atol=0.0
    )


def test_lowered_target_lets_least_squares_recover_the_remaining_law(
    burgers_platform: tuple[PlatformComponents, PlatformComponents],
) -> None:
    _raw, lowered = burgers_platform
    assert lowered.evaluator is not None

    result = lowered.evaluator.evaluate_terms([HOLE_TERM])

    assert result.is_valid
    assert result.coefficients is not None


    torch.testing.assert_close(
        float(result.coefficients.flatten()[0]), BURGERS_NU, rtol=0.0, atol=5e-3
    )


def test_lowered_target_lands_on_the_context_device(
    burgers_platform: tuple[PlatformComponents, PlatformComponents],
) -> None:
    _raw, lowered = burgers_platform
    assert lowered.evaluator is not None
    assert lowered.context is not None
    assert lowered.evaluator.lhs_target.device == lowered.context.device


def test_pinned_complete_deduction_leaves_no_regression_target() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("u", 1.0),), holes=(default_hole(),))
    task = DiscoveryTask.from_sketch(sketch)

    with pytest.raises(ValueError, match="variance"):
        build_components(constant_field_dataset(), task=task)


def test_light_bundle_cannot_carry_a_platform_lower() -> None:
    task = DiscoveryTask.from_sketch(burgers_sketch())

    with pytest.raises(NotImplementedError, match="lower"):
        build_components(
            tiny_burgers_dataset(), task=task, provider_kind="none"
        )


def test_unexecutable_pinned_term_fails_loud_naming_the_term() -> None:
    sketch = burgers_sketch(
        pinned=(PinnedTerm("v_xx", 1.0),), fields=("u", "v")
    )
    task = DiscoveryTask.from_sketch(sketch)

    with pytest.raises(ValueError, match="v_xx"):
        build_components(tiny_burgers_dataset(), task=task)


def test_pinned_execution_resource_failure_is_not_relabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = DiscoveryTask.from_sketch(burgers_sketch())

    def explode(self: object, term: str, context: object) -> object:
        raise RuntimeError(
            "DefaultCPUAllocator: can't allocate memory: you tried to "
            "allocate 8 bytes"
        )

    monkeypatch.setattr(
        "kd.core.expr.executor.PythonExecutor.execute", explode
    )

    with pytest.raises(RuntimeError, match="DefaultCPUAllocator"):
        build_components(tiny_burgers_dataset(), task=task)


def test_closed_sketch_build_skips_the_lower_seat() -> None:
    dataset = tiny_burgers_dataset()
    closed = DiscoveryTask.from_sketch(burgers_sketch(holes=()))

    raw = build_components(dataset)
    closed_build = build_components(dataset, task=closed)

    assert raw.evaluator is not None
    assert closed_build.evaluator is not None
    torch.testing.assert_close(
        closed_build.evaluator.lhs_target,
        raw.evaluator.lhs_target,
        rtol=0.0,
        atol=0.0,
    )


def test_sampled_evaluator_inherits_the_single_deduction(
    burgers_platform: tuple[PlatformComponents, PlatformComponents],
) -> None:
    _raw, lowered = burgers_platform
    assert lowered.evaluator is not None
    indices = torch.tensor([0, 5, 9, 17])

    sampled = SampledEvaluator(lowered.evaluator, indices)




    torch.testing.assert_close(
        sampled._lhs,
        lowered.evaluator.lhs_target.index_select(0, indices),
        rtol=0.0,
        atol=0.0,
    )


def test_builder_without_a_task_keeps_the_pre_sketch_call_form() -> None:
    dataset = tiny_burgers_dataset()
    reqs = DerivativeReqs(provider_kind="finite_diff", max_atomic_order=2)

    components = PlatformBuilder(dataset, reqs).build()

    assert components.evaluator is not None
    assert components.task is None







def _native_build(dataset: PDEDataset, sketch: Sketch) -> PlatformComponents:
    reqs = DerivativeReqs(
        provider_kind="finite_diff", max_atomic_order=2, lhs_order=1
    )
    return PlatformBuilder(
        dataset,
        reqs,
        task=DiscoveryTask.from_sketch(sketch),
        sketch_lower_owner="native",
    ).build()


def test_a_native_lower_owner_keeps_the_platform_target_raw() -> None:
    dataset = tiny_burgers_dataset()

    native = _native_build(dataset, burgers_sketch())
    raw = build_components(dataset)

    assert native.evaluator is not None
    assert raw.evaluator is not None
    torch.testing.assert_close(
        native.evaluator.lhs_target, raw.evaluator.lhs_target, rtol=0.0, atol=0.0
    )


def test_a_native_lower_owner_carries_the_task_without_executing_it() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("v_xx", 1.0),), fields=("u", "v"))

    components = _native_build(tiny_burgers_dataset(), sketch)

    assert components.evaluator is not None
    assert components.task is not None
