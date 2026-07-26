
from __future__ import annotations

import dataclasses
import importlib
import types
from typing import Any, get_args, get_type_hints

import pytest

from kd.api import _PLUGIN_CLASS_BY_ALGORITHM
from kd.core.equation import Form
from kd.data.schema import DataTopology
from kd.models.field_model import FieldModel
from kd.search.descriptor import (
    InstrumentDescriptor,
    InstrumentMode,
    Knob,
    tool_schema,
)
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.protocol import FacadeWiringContract
from kd.search.resume_policy import SCIENCE_AXIS_DENYLISTS as _SCIENCE_AXIS_DENYLISTS

pytestmark = pytest.mark.unit

_PLUGIN_CASES = tuple(_PLUGIN_CLASS_BY_ALGORITHM.items())
_MODE_CASES = tuple(
    (algorithm, plugin_cls, mode)
    for algorithm, plugin_cls in _PLUGIN_CASES
    for mode in plugin_cls.descriptor.modes
)
_CLAIMABLE_FORMS = frozenset({Form.EVOLUTION, Form.HOMOGENEOUS})
_KIND_TYPES: dict[str, type[Any]] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}
_VALID_RESUME_TIERS = frozenset(
    {"init_only", "resume_safe", "identity_breaking"}
)








_EXPECTED_RESUME_TIERS: dict[str, dict[str, str]] = {
    "sga": {
        "num": "resume_safe",
        "depth": "init_only",
        "width": "init_only",
        "aic_ratio": "init_only",
        "lam": "init_only",
    },
    "dlga": {
        "pop_size": "resume_safe",
        "epsilon": "init_only",
        "mutation_rate": "resume_safe",
        "crossover_rate": "resume_safe",
        "surrogate_lr": "init_only",
    },
    "discover": {
        "batch_size": "resume_safe",
        "learning_rate": "resume_safe",
        "entropy_weight": "resume_safe",
        "reward_alpha": "resume_safe",
        "max_length": "init_only",
    },
    "pysr": {
        "population_size": "init_only",
        "populations": "init_only",
        "maxsize": "init_only",
    },
    "eqgpt": {
        "samples_per_epoch": "resume_safe",
        "top_k": "init_only",
        "sparsity_alpha": "init_only",
        "finetune_lr": "resume_safe",
        "exploration_rate": "resume_safe",
    },
    "llm4ed": {
        "temperature": "resume_safe",
        "max_tokens": "resume_safe",
        "stop_threshold": "resume_safe",
        "reward_limit": "init_only",
        "pool_size": "init_only",
    },
    "pysindy": {
        "threshold": "init_only",
        "max_iter": "init_only",
        "normalize_columns": "init_only",
    },
}











def _config_for_mode(
    algorithm: str, plugin_cls: type[FacadeWiringContract], mode_name: str
) -> Any:
    if algorithm == "eqgpt":
        if mode_name == "single_wave":
            return EqGPTConfig(sparsity_alpha=0.02)
        if mode_name == "wave_multicase":
            return EqGPTConfig(sparsity_alpha=0.02, case_filter="N")
        if mode_name == "steady":
            return EqGPTConfig(
                sparsity_alpha=1.0,
                steady=True,
                steady_activation="sin",
                start_words=("S",),
            )
        raise AssertionError(f"unrecognized EqGPT descriptor mode: {mode_name}")




    if len(plugin_cls.descriptor.modes) > 1:
        raise NotImplementedError(
            f"{algorithm} declares {len(plugin_cls.descriptor.modes)} modes but "
            "has no per-mode config fixture; add one to _config_for_mode"
        )
    return plugin_cls.config_cls()


def _plugin_for_mode(
    algorithm: str, plugin_cls: type[FacadeWiringContract], mode_name: str
) -> Any:
    return plugin_cls(_config_for_mode(algorithm, plugin_cls, mode_name))


def _unwrap_optional(annotation: Any) -> Any:
    if isinstance(annotation, types.UnionType):
        args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
        if len(args) == 1:
            return args[0]
    return annotation


class TestDescriptorValidation:

    def test_mode_requires_non_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            InstrumentMode(
                name="",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
            )

    def test_mode_requires_forms(self) -> None:
        with pytest.raises(ValueError, match="forms must be non-empty"):
            InstrumentMode(
                name="default",
                forms=frozenset(),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
            )

    @pytest.mark.parametrize("reserved", (Form.PARAMETRIC, Form.WEAK))
    def test_reserved_forms_are_not_claimable(self, reserved: Form) -> None:
        with pytest.raises(ValueError, match="reserved forms are not claimable"):
            InstrumentMode(
                name="default",
                forms=frozenset({reserved}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
            )

    def test_mode_requires_topologies(self) -> None:
        with pytest.raises(ValueError, match="topologies must be non-empty"):
            InstrumentMode(
                name="default",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset(),
                provider_kind="finite_diff",
            )

    def test_descriptor_requires_modes(self) -> None:
        with pytest.raises(ValueError, match="modes must be non-empty"):
            InstrumentDescriptor("fake", "", "light", (), ())

    def test_descriptor_rejects_duplicate_mode_names(self) -> None:
        mode = InstrumentMode(
            "default",
            frozenset({Form.EVOLUTION}),
            frozenset({DataTopology.GRID}),
            "finite_diff",
        )
        with pytest.raises(ValueError, match="mode names must be distinct"):
            InstrumentDescriptor("fake", "", "light", (mode, mode), ())

    def test_descriptor_rejects_duplicate_knob_names(self) -> None:
        mode = InstrumentMode(
            "default",
            frozenset({Form.EVOLUTION}),
            frozenset({DataTopology.GRID}),
            "finite_diff",
        )
        knobs = (
            Knob("size", "int", resume_tier="resume_safe"),
            Knob("size", "float", resume_tier="resume_safe"),
        )
        with pytest.raises(ValueError, match="knob names must be distinct"):
            InstrumentDescriptor("fake", "", "light", (mode,), knobs)

    def test_dataclasses_are_frozen(self) -> None:
        mode = InstrumentMode(
            "default",
            frozenset({Form.EVOLUTION}),
            frozenset({DataTopology.GRID}),
            "finite_diff",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            mode.name = "changed"


class TestRegistryDescriptorContract:

    @pytest.mark.parametrize(("algorithm", "plugin_cls"), _PLUGIN_CASES)
    def test_registry_entry_declares_descriptor(
        self, algorithm: str, plugin_cls: type[FacadeWiringContract]
    ) -> None:
        assert isinstance(plugin_cls.descriptor, InstrumentDescriptor)
        assert plugin_cls.descriptor.algorithm == algorithm
        assert isinstance(plugin_cls, FacadeWiringContract)

    @pytest.mark.parametrize(("algorithm", "plugin_cls"), _PLUGIN_CASES)
    def test_descriptor_matches_runtime_algorithm_stamp(
        self, algorithm: str, plugin_cls: type[FacadeWiringContract]
    ) -> None:
        first_mode = plugin_cls.descriptor.modes[0]
        plugin = _plugin_for_mode(algorithm, plugin_cls, first_mode.name)
        assert plugin.config["algorithm"] == algorithm

        module = importlib.import_module(plugin_cls.__module__)
        declared_name = getattr(module, "ALGORITHM_NAME", algorithm)
        assert declared_name == algorithm

    def test_protocol_declares_descriptor_classvar(self) -> None:
        annotations = FacadeWiringContract.__annotations__
        assert "descriptor" in annotations


class TestModeAndFormHonesty:

    @pytest.mark.parametrize(
        ("algorithm", "plugin_cls", "mode"),
        _MODE_CASES,
        ids=lambda value: getattr(value, "name", str(value)),
    )
    def test_mode_matches_live_requirements(
        self,
        algorithm: str,
        plugin_cls: type[FacadeWiringContract],
        mode: InstrumentMode,
    ) -> None:
        plugin = _plugin_for_mode(algorithm, plugin_cls, mode.name)
        reqs = plugin.derivative_requirements
        assert reqs.supported_topologies == mode.topologies
        assert reqs.provider_kind == mode.provider_kind

    @pytest.mark.parametrize(
        ("algorithm", "plugin_cls", "mode"),
        _MODE_CASES,
        ids=lambda value: getattr(value, "name", str(value)),
    )
    def test_mode_forms_are_non_empty_and_claimable(
        self,
        algorithm: str,
        plugin_cls: type[FacadeWiringContract],
        mode: InstrumentMode,
    ) -> None:
        del algorithm, plugin_cls


        assert mode.forms
        assert mode.forms <= _CLAIMABLE_FORMS

    def test_homogeneous_is_declared_only_by_the_known_emitters(self) -> None:






        homogeneous = {
            (algorithm, mode.name)
            for algorithm, plugin_cls in _PLUGIN_CASES
            for mode in plugin_cls.descriptor.modes
            if Form.HOMOGENEOUS in mode.forms
        }
        assert homogeneous == {("eqgpt", "steady")}


class TestKnobHonesty:

    @pytest.mark.parametrize(("algorithm", "plugin_cls"), _PLUGIN_CASES)
    def test_knobs_match_config_dataclass_fields_and_annotations(
        self, algorithm: str, plugin_cls: type[FacadeWiringContract]
    ) -> None:
        del algorithm
        field_names = {
            field.name for field in dataclasses.fields(plugin_cls.config_cls)
        }
        annotations = get_type_hints(
            plugin_cls.config_cls, localns={"FieldModel": FieldModel}
        )
        for knob in plugin_cls.descriptor.knobs:
            assert knob.name in field_names
            annotation = _unwrap_optional(annotations[knob.name])
            assert annotation is _KIND_TYPES[knob.kind]

    @pytest.mark.parametrize(("algorithm", "plugin_cls"), _PLUGIN_CASES)
    def test_every_knob_declares_a_valid_resume_tier(
        self, algorithm: str, plugin_cls: type[FacadeWiringContract]
    ) -> None:
        del algorithm



        for knob in plugin_cls.descriptor.knobs:
            assert knob.resume_tier in _VALID_RESUME_TIERS

    def test_discover_excludes_facade_rejected_and_loop_fields(self) -> None:


        excluded = {
            "n_iterations",
            "pinn",
            "stability_selection",
            "stability_queue_capacity",
        }
        names = {
            knob.name
            for knob in _PLUGIN_CLASS_BY_ALGORITHM["discover"].descriptor.knobs
        }
        assert names.isdisjoint(excluded)

    def test_discover_controller_structural_fields_never_enter_as_resume_safe(
        self,
    ) -> None:





        structural = {
            "num_units",
            "num_layers",
            "embedding_dim",
            "observe_parent",
            "observe_sibling",
            "observe_action",
            "observe_dangling",
            "use_embedding",
            "attention",
            "attn_length",
        }
        for knob in _PLUGIN_CLASS_BY_ALGORITHM["discover"].descriptor.knobs:
            if knob.name in structural:
                assert knob.resume_tier != "resume_safe"


class TestSchemaComposition:

    @pytest.mark.parametrize(("algorithm", "plugin_cls"), _PLUGIN_CASES)
    def test_schema_composes_score_contract_and_one_shot(
        self, algorithm: str, plugin_cls: type[FacadeWiringContract]
    ) -> None:
        schema = tool_schema(plugin_cls)
        assert schema["algorithm"] == algorithm
        assert schema["score_kind"] == plugin_cls.score_kind
        assert schema["score_direction"] == plugin_cls.score_direction
        assert schema["one_shot"] is plugin_cls.one_shot

    @pytest.mark.parametrize(("algorithm", "plugin_cls"), _PLUGIN_CASES)
    def test_schema_knobs_carry_resume_tier(
        self, algorithm: str, plugin_cls: type[FacadeWiringContract]
    ) -> None:
        del algorithm



        schema_knobs = tool_schema(plugin_cls)["knobs"]
        declared = plugin_cls.descriptor.knobs
        assert len(schema_knobs) == len(declared)
        for entry, knob in zip(schema_knobs, declared, strict=True):
            assert entry["resume_tier"] == knob.resume_tier

    @pytest.mark.parametrize(("algorithm", "plugin_cls"), _PLUGIN_CASES)
    def test_schema_emits_config_cls_as_resolvable_ref(
        self, algorithm: str, plugin_cls: type[FacadeWiringContract]
    ) -> None:
        del algorithm


        config_cls = plugin_cls.config_cls
        ref = tool_schema(plugin_cls)["config_cls"]

        assert ref == f"{config_cls.__module__}.{config_cls.__qualname__}"




        module_path, _, qualname = ref.rpartition(".")
        resolved = getattr(importlib.import_module(module_path), qualname)
        assert dataclasses.is_dataclass(resolved)


class TestResumeTierDeclarations:

    @pytest.mark.parametrize(("algorithm", "plugin_cls"), _PLUGIN_CASES)
    def test_per_plugin_tier_map_equals_golden(
        self, algorithm: str, plugin_cls: type[FacadeWiringContract]
    ) -> None:
        actual = {
            knob.name: knob.resume_tier
            for knob in plugin_cls.descriptor.knobs
        }
        assert actual == _EXPECTED_RESUME_TIERS[algorithm]

    def test_golden_table_covers_exactly_the_registry(self) -> None:


        assert set(_EXPECTED_RESUME_TIERS) == set(_PLUGIN_CLASS_BY_ALGORITHM)

    def test_summary_counts_match_adjudicated_totals(self) -> None:



        tiers = [
            tier
            for knobs in _EXPECTED_RESUME_TIERS.values()
            for tier in knobs.values()
        ]
        assert len(tiers) == 31
        assert tiers.count("resume_safe") == 14
        assert tiers.count("init_only") == 17
        assert tiers.count("identity_breaking") == 0


class TestScienceAxisSeal:

    @pytest.mark.parametrize(("algorithm", "plugin_cls"), _PLUGIN_CASES)
    def test_science_axis_fields_never_declared_below_identity_breaking(
        self, algorithm: str, plugin_cls: type[FacadeWiringContract]
    ) -> None:
        denylist = _SCIENCE_AXIS_DENYLISTS[algorithm]
        for knob in plugin_cls.descriptor.knobs:
            if knob.name in denylist:


                assert knob.resume_tier == "identity_breaking"

    def test_denylist_covers_exactly_the_registry(self) -> None:
        assert set(_SCIENCE_AXIS_DENYLISTS) == set(_PLUGIN_CLASS_BY_ALGORITHM)
