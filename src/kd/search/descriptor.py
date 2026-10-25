
from __future__ import annotations

from collections.abc import Mapping, Sequence
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

if TYPE_CHECKING:
    from kd.search.protocol import FacadeWiringContract

_CLAIMABLE_FORMS = frozenset({Form.EVOLUTION, Form.HOMOGENEOUS, Form.REGRESSION})
















SCORE_FRAME_PLATFORM_FULL_GRID = "kd-evaluate-terms-full-grid-v1"



SCORE_FRAME_SURROGATE_FIELD = "kd-frame-surrogate-field-v1"



SCORE_FRAME_NATIVE_INTERNAL = "kd-frame-native-internal-v1"


@dataclass(frozen=True)
class ModeSelector:

    field: str
    present: bool


@dataclass(frozen=True)
class InstrumentMode:

    name: str
    forms: frozenset[Form]
    topologies: frozenset[DataTopology]
    provider_kind: Literal["finite_diff", "autograd", "none"]
    description: str = ""
    sketch: SketchClauseLevels = SketchClauseLevels()
    lhs_orders: frozenset[int] | None = None
    score_frame: str = "undeclared"


    selected_by: Mapping[str, Any] = field(default_factory=dict)
    order_params: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    required_params: frozenset[str] = frozenset()

    default_overrides: Mapping[str, Any] = field(default_factory=dict)

    refused_fields: Mapping[str, Literal["any", "non_default"]] = field(
        default_factory=dict
    )

    field_names: tuple[str, ...] | None = None
    spatial_axes: tuple[str, ...] | None = None
    names_from: str | None = None
    selectable_from_data: bool = True

    selector: ModeSelector | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mode name must be non-empty")
        reserved = self.forms - _CLAIMABLE_FORMS
        if reserved:
            names = ", ".join(sorted(form.value for form in reserved))
            raise ValueError(f"reserved forms are not claimable: {names}")
        if not self.topologies:
            raise ValueError("mode topologies must be non-empty")
        overlap = set(self.selected_by).intersection(
            key for params in self.order_params.values() for key in params
        )
        if overlap:
            raise ValueError(f"selected_by and order_params overlap: {sorted(overlap)}")
        if any(
            value not in {"any", "non_default"}
            for value in self.refused_fields.values()
        ):
            raise ValueError("refused_fields values must be 'any' or 'non_default'")
        if self.lhs_orders is None:
            return



        if not self.lhs_orders:
            raise ValueError(
                "declared lhs_orders must be non-empty (None declares nothing)"
            )
        offending = sorted(
            repr(order)
            for order in self.lhs_orders
            if type(order) is not int or order < 0
        )
        if offending:
            raise ValueError(
                f"lhs_orders must be non-negative ints: {', '.join(offending)}"
            )


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


_SEGMENTATION_ARCHIVES: frozenset[str] = frozenset({"progress", "conclusion", "none"})


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


    identity_breaking_fields: frozenset[str] = field(kw_only=True)
    config_artifact_keys: frozenset[str] = field(kw_only=True)


    surrogate_fields: frozenset[str] = field(kw_only=True)

    facade_inert_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("descriptor modes must be non-empty")
        mode_names = [mode.name for mode in self.modes]
        if len(mode_names) != len(set(mode_names)):
            raise ValueError("descriptor mode names must be distinct")
        knob_names = [knob.name for knob in self.knobs]
        if len(knob_names) != len(set(knob_names)):
            raise ValueError("descriptor knob names must be distinct")
        if bool(self.surrogate_fields) != bool(self.config_artifact_keys):
            raise ValueError(
                f"descriptor {self.algorithm!r}: surrogate_fields must be "
                "non-empty exactly when config_artifact_keys is non-empty "
                "(an instrument trains a surrogate iff it accepts one injected); "
                f"got surrogate_fields={sorted(self.surrogate_fields)}, "
                f"config_artifact_keys={sorted(self.config_artifact_keys)}"
            )








        identity = self.identity_breaking_fields | self.config_artifact_keys
        for knob in self.knobs:
            if knob.name in identity and knob.resume_tier != "identity_breaking":
                raise ValueError(
                    f"knob {knob.name!r} is an identity_breaking field or "
                    f"config-artifact key of {self.algorithm!r}; declare "
                    f"resume_tier='identity_breaking' (got {knob.resume_tier!r})"
                )


def mode_for_topology(
    descriptor: InstrumentDescriptor,
    topology: DataTopology,
) -> InstrumentMode | None:
    matches = [mode for mode in descriptor.modes if topology in mode.topologies]
    if len(matches) > 1:
        names = ", ".join(mode.name for mode in matches)
        raise ValueError(
            f"descriptor {descriptor.algorithm!r} has multiple modes for "
            f"topology {topology.value!r}: {names}"
        )
    return matches[0] if matches else None


def sketch_eligible_modes(
    descriptor: InstrumentDescriptor,
    sketch: Sketch,
    *,
    sketch_lower_owner: Literal["platform", "native"],
    config: Mapping[str, Any] | None = None,
) -> tuple[InstrumentMode, ...]:
    return tuple(
        mode
        for mode in descriptor.modes
        if Form.EVOLUTION in mode.forms
        if mode.lhs_orders is None or sketch.lhs_spec.order in mode.lhs_orders
        if not (mode.provider_kind == "none" and sketch_lower_owner == "platform")
        if config is None
        or mode.selector is None
        or (config.get(mode.selector.field) is not None) == mode.selector.present
    )


def assert_sketch_supported(
    descriptor: InstrumentDescriptor,
    sketch: Sketch,
    *,
    algorithm: str,
    sketch_lower_owner: Literal["platform", "native"],
    config: Mapping[str, Any] | None = None,
) -> None:
    modes = sketch_eligible_modes(
        descriptor, sketch, sketch_lower_owner=sketch_lower_owner, config=config
    )
    if not modes:
        raise ValueError(
            f"algorithm {algorithm!r} has no mode that can run an evolution "
            f"sketch with LHS order {sketch.lhs_spec.order}"
        )
    used = used_clauses(sketch)
    offending = [
        (mode.name, clause, mode.sketch.level(clause))
        for mode in modes
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


def _spatial_dims(mode: InstrumentMode) -> list[int] | None:
    if mode.spatial_axes is None or mode.names_from is not None:
        return None
    return list(range(1, len(mode.spatial_axes) + 1))


def _field_counts(mode: InstrumentMode) -> list[int] | None:
    if mode.field_names is None or mode.names_from is not None:
        return None
    return [len(mode.field_names)]


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
                "sketch": {
                    clause: mode.sketch.level(clause) for clause in SKETCH_CLAUSES
                },
                "lhs_orders": (
                    None if mode.lhs_orders is None else sorted(mode.lhs_orders)
                ),
                "score_frame": mode.score_frame,
                "selected_by": dict(mode.selected_by),
                "order_params": {
                    str(order): dict(params)
                    for order, params in mode.order_params.items()
                },
                "required_params": sorted(mode.required_params),
                "default_overrides": dict(mode.default_overrides),
                "refused_fields": dict(mode.refused_fields),
                "field_names": None
                if mode.field_names is None
                else list(mode.field_names),
                "field_counts": _field_counts(mode),
                "spatial_axes": None
                if mode.spatial_axes is None
                else list(mode.spatial_axes),
                "spatial_dims": _spatial_dims(mode),
                "names_from": mode.names_from,
                "selectable_from_data": mode.selectable_from_data,
                "selector": (
                    None
                    if mode.selector is None
                    else {
                        "field": mode.selector.field,
                        "present": mode.selector.present,
                    }
                ),
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
        "sketch_lower_owner": plugin_cls.sketch_lower_owner,
        "identity_breaking_fields": sorted(descriptor.identity_breaking_fields),
        "config_artifact_keys": sorted(descriptor.config_artifact_keys),
        "surrogate_fields": sorted(descriptor.surrogate_fields),
        "facade_inert_fields": sorted(descriptor.facade_inert_fields),
    }


@dataclass(frozen=True)
class ModeVerdict:

    accepts: bool
    reason: str | None
    params: dict[str, Any]


def mode_accepts(
    mode: Mapping[str, Any],
    *,
    topology: str,
    form: str,
    lhs_order: int,
    field_names: Sequence[str],
    spatial_axes: Sequence[str],
) -> ModeVerdict:
    if topology not in {item.value for item in DataTopology}:
        raise ValueError(f"unknown topology: {topology!r}")
    if form not in {item.value for item in _CLAIMABLE_FORMS}:
        raise ValueError(f"unknown form: {form!r}")
    if type(lhs_order) is not int or lhs_order not in {0, 1, 2}:
        raise ValueError(f"unknown lhs_order: {lhs_order!r}; expected 0, 1 or 2")
    if not mode["selectable_from_data"]:
        return ModeVerdict(False, "This mode is not selectable from input data.", {})
    for name, value in (
        ("topologies", topology),
        ("forms", form),
        ("lhs_orders", lhs_order),
    ):
        if mode[name] is not None and value not in mode[name]:
            return ModeVerdict(
                False,
                f"Mode {mode['name']} requires {name} in {mode[name]}; got {value!r}.",
                {},
            )
    params = dict(mode["selected_by"])
    params.update(mode["order_params"].get(str(lhs_order), {}))
    params.update({name: "<set>" for name in mode["required_params"]})
    fields_match = mode["field_names"] is None or set(field_names) == set(
        mode["field_names"]
    )
    axes_match = (
        mode["spatial_axes"] is None
        or bool(spatial_axes)
        and set(spatial_axes) <= set(mode["spatial_axes"])
    )
    if not fields_match or not axes_match:
        reason = (
            f"Input fields {list(field_names)} and spatial axes "
            f"{list(spatial_axes)} differ from mode fields {mode['field_names']} "
            f"and axes {mode['spatial_axes']}."
        )
        if mode["names_from"] is None:
            return ModeVerdict(False, reason, params)
        params[mode["names_from"]] = "<build from this input's fields and axes>"
        return ModeVerdict(
            True, f"{reason} Set {mode['names_from']} from this input.", params
        )
    return ModeVerdict(True, None, params)


def mode_inputs(report: Mapping[str, Any]) -> dict[str, Any]:
    topology = report["topology"]
    order = report["lhs_order"]
    form = (
        "REGRESSION"
        if topology == "tabular"
        else "HOMOGENEOUS"
        if order == 0
        else "EVOLUTION"
    )
    return {
        "topology": topology,
        "form": form,
        "lhs_order": order,
        "field_names": [entry["name"] for entry in report["fields"]],
        "spatial_axes": []
        if topology == "tabular"
        else [
            entry["name"]
            for entry in report["axes"]
            if order == 0 or entry["name"] != report["lhs_axis"]
        ],
    }


__all__ = [
    "SCORE_FRAME_NATIVE_INTERNAL",
    "SCORE_FRAME_PLATFORM_FULL_GRID",
    "SCORE_FRAME_SURROGATE_FIELD",
    "InstrumentDescriptor",
    "InstrumentMode",
    "Knob",
    "ModeSelector",
    "ModeVerdict",
    "ResumeTier",
    "Segmentation",
    "assert_sketch_supported",
    "mode_accepts",
    "mode_for_topology",
    "mode_inputs",
    "tool_schema",
]
