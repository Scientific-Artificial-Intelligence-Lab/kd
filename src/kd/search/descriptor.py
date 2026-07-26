
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from kd.core.equation import Form
from kd.data.schema import DataTopology

if TYPE_CHECKING:
    from kd.search.protocol import FacadeWiringContract

_CLAIMABLE_FORMS = frozenset({Form.EVOLUTION, Form.HOMOGENEOUS})


@dataclass(frozen=True)
class InstrumentMode:

    name: str
    forms: frozenset[Form]
    topologies: frozenset[DataTopology]
    provider_kind: Literal["finite_diff", "autograd", "none"]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mode name must be non-empty")
        if not self.forms:
            raise ValueError("mode forms must be non-empty")
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


@dataclass(frozen=True)
class InstrumentDescriptor:

    algorithm: str
    summary: str
    cost_class: Literal["light", "medium", "heavy"]
    modes: tuple[InstrumentMode, ...]
    knobs: tuple[Knob, ...]

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("descriptor modes must be non-empty")
        mode_names = [mode.name for mode in self.modes]
        if len(mode_names) != len(set(mode_names)):
            raise ValueError("descriptor mode names must be distinct")
        knob_names = [knob.name for knob in self.knobs]
        if len(knob_names) != len(set(knob_names)):
            raise ValueError("descriptor knob names must be distinct")


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
        "score_kind": plugin_cls.score_kind,
        "score_direction": plugin_cls.score_direction,
        "one_shot": plugin_cls.one_shot,
    }


__all__ = [
    "InstrumentDescriptor",
    "InstrumentMode",
    "Knob",
    "ResumeTier",
    "tool_schema",
]
