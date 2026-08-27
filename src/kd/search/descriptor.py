
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from kd.core.equation import Form
from kd.core.equation.sketch import Sketch
from kd.core.platform.sketch_compile import (
    SKETCH_CLAUSES,
    SketchClauseLevels,
    used_clauses,
)
from kd.data.schema import DataTopology
from kd.search.config_fields import field_specs
from kd.search.resume_policy import SCIENCE_AXIS_DENYLISTS

if TYPE_CHECKING:
    from kd.search.protocol import FacadeWiringContract

_CLAIMABLE_FORMS = frozenset(
    {Form.EVOLUTION, Form.HOMOGENEOUS, Form.REGRESSION}
)


@dataclass(frozen=True)
class InstrumentMode:

    name: str
    forms: frozenset[Form]
    topologies: frozenset[DataTopology]
    provider_kind: Literal["finite_diff", "autograd", "none"]
    description: str = ""
    sketch: SketchClauseLevels = SketchClauseLevels()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mode name must be non-empty")
        reserved = self.forms - _CLAIMABLE_FORMS
        if reserved:
            names = ", ".join(sorted(form.value for form in reserved))
            raise ValueError(f"reserved forms are not claimable: {names}")
        if not self.topologies:
            raise ValueError("mode topologies must be non-empty")


ResumeTier = Literal["init_only", "resume_safe", "identity_breaking"]

_RESUME_TIERS: frozenset[str] = frozenset(
    {"init_only", "resume_safe", "identity_breaking"}
)


@dataclass(frozen=True)
class Knob:

    name: str
    kind: Literal["int", "float", "str", "bool"]
    description: str = ""
    resume_tier: ResumeTier = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.resume_tier not in _RESUME_TIERS:
            raise ValueError(f"unknown resume_tier: {self.resume_tier!r}")


_SEGMENTATION_ARCHIVES: frozenset[str] = frozenset(
    {"progress", "conclusion", "none"}
)


@dataclass(frozen=True)
class Segmentation:

    archive: str
    unit: str
    reseed: bool = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.archive not in _SEGMENTATION_ARCHIVES:
            raise ValueError(f"unknown segmentation archive: {self.archive!r}")
        if not self.unit:
            raise ValueError("segmentation unit must be non-empty")
        if self.reseed and self.archive != "progress":
            raise ValueError(
                "segmentation reseed=True requires archive='progress'; "
                f"got archive={self.archive!r} (a conclusion-only or absent "
                "archive carries no search stream to re-derive)"
            )


@dataclass(frozen=True)
class InstrumentDescriptor:

    algorithm: str
    summary: str
    cost_class: Literal["light", "medium", "heavy"]
    modes: tuple[InstrumentMode, ...]
    knobs: tuple[Knob, ...]


    segmentation: Segmentation = field(kw_only=True)

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("descriptor modes must be non-empty")
        mode_names = [mode.name for mode in self.modes]
        if len(mode_names) != len(set(mode_names)):
            raise ValueError("descriptor mode names must be distinct")
        knob_names = [knob.name for knob in self.knobs]
        if len(knob_names) != len(set(knob_names)):
            raise ValueError("descriptor knob names must be distinct")


def mode_for_topology(
    descriptor: InstrumentDescriptor,
    topology: DataTopology,
) -> InstrumentMode | None:
    matches = [
        mode for mode in descriptor.modes if topology in mode.topologies
    ]
    if len(matches) > 1:
        names = ", ".join(mode.name for mode in matches)
        raise ValueError(
            f"descriptor {descriptor.algorithm!r} has multiple modes for "
            f"topology {topology.value!r}: {names}"
        )
    return matches[0] if matches else None


def assert_sketch_supported(
    descriptor: InstrumentDescriptor,
    sketch: Sketch,
    *,
    algorithm: str,
) -> None:
    used = used_clauses(sketch)
    offending = [
        (mode.name, clause, mode.sketch.level(clause))
        for mode in descriptor.modes
        for clause in SKETCH_CLAUSES
        if clause in used
        if mode.sketch.level(clause) == "unsupported"
    ]
    if not offending:
        return
    details = ", ".join(
        f"(mode={mode!r}, clause={clause!r}, level={level!r})"
        for mode, clause, level in offending
    )
    raise ValueError(
        f"algorithm {algorithm!r} cannot accept this sketch: {details}; "
        "this algorithm declares no sketch support for these clauses"
    )


def _config_cls_ref(plugin_cls: type[FacadeWiringContract]) -> str:
    config_cls = plugin_cls.config_cls
    return f"{config_cls.__module__}.{config_cls.__qualname__}"


def tool_schema(plugin_cls: type[FacadeWiringContract]) -> dict[str, Any]:
    descriptor = plugin_cls.descriptor
    return {
        "algorithm": descriptor.algorithm,
        "summary": descriptor.summary,
        "cost_class": descriptor.cost_class,
        "config_cls": _config_cls_ref(plugin_cls),
        "modes": [
            {
                "name": mode.name,
                "forms": sorted(form.value for form in mode.forms),
                "topologies": sorted(topology.value for topology in mode.topologies),
                "provider_kind": mode.provider_kind,
                "description": mode.description,
            }
            for mode in descriptor.modes
        ],
        "knobs": [
            {
                "name": knob.name,
                "kind": knob.kind,
                "description": knob.description,
                "resume_tier": knob.resume_tier,
            }
            for knob in descriptor.knobs
        ],
        "segmentation": {
            "archive": descriptor.segmentation.archive,
            "unit": descriptor.segmentation.unit,
            "reseed": descriptor.segmentation.reseed,
        },
        "fields": field_specs(plugin_cls),
        "score_kind": plugin_cls.score_kind,
        "score_direction": plugin_cls.score_direction,
        "one_shot": plugin_cls.one_shot,
        "identity_breaking_fields": sorted(
            SCIENCE_AXIS_DENYLISTS.get(descriptor.algorithm, frozenset())
        ),
    }


__all__ = [
    "InstrumentDescriptor",
    "InstrumentMode",
    "Knob",
    "ResumeTier",
    "Segmentation",
    "assert_sketch_supported",
    "mode_for_topology",
    "tool_schema",
]
