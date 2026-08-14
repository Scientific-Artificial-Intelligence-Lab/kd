
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

__all__ = ["SCHEMA_REGISTRY", "SchemaEntry"]


@dataclass(frozen=True)
class SchemaEntry:

    version: int
    module: str


SCHEMA_REGISTRY: Final[Mapping[str, SchemaEntry]] = {

    "kd-term-v1": SchemaEntry(1, "kd.core.equation.library"),
    "kd-termlib-v1": SchemaEntry(1, "kd.core.equation.library"),
    "kd-lawsig-v1": SchemaEntry(1, "kd.core.equation.signature"),
    "kd-sketch-v1": SchemaEntry(1, "kd.core.equation.sketch"),
    "kd-verification-v1": SchemaEntry(1, "kd.core.verify"),

    "kd-config-v1": SchemaEntry(1, "kd.search.run_spec"),
    "kd-runspec-v1": SchemaEntry(1, "kd.search.run_spec"),
    "kd-rundir-v1": SchemaEntry(1, "kd.search.run_dir"),
    "kd-runcat-v1": SchemaEntry(1, "kd.search.run_catalog"),
    "kd-iterevent-v1": SchemaEntry(1, "kd.search.iteration_events"),
    "kd-runphase-v1": SchemaEntry(1, "kd.search.iteration_events"),
    "kd-ckptman-v1": SchemaEntry(2, "kd.search._checkpoint_manifest_verify"),
    "kd-confighash-v1": SchemaEntry(1, "kd.search._checkpoint_manifest_verify"),
    "kd-record-v1": SchemaEntry(2, "kd.search.records"),
    "kd-evidence-v1": SchemaEntry(2, "kd.search.records"),
    "kd-torch-module-v1": SchemaEntry(1, "kd.search._torch_module_artifact"),
    "kd-field-model-v1": SchemaEntry(1, "kd.search.sga.plugin"),
    "kd-sketch-outcome-v1": SchemaEntry(1, "kd.search.sketch_outcome"),

    "kd-plan-v1": SchemaEntry(1, "kd.harness.plan"),
    "kd-consensus-v1": SchemaEntry(1, "kd.harness.consensus"),
    "kd-stratum-rules-v1": SchemaEntry(1, "kd.harness.consensus"),
    "kd-eligibility-rules-v1": SchemaEntry(1, "kd.harness.consensus"),
    "kd-dispatch-v1": SchemaEntry(2, "kd.harness._dispatch_schema"),
    "kd-dispatch-log-v1": SchemaEntry(1, "kd.harness.dispatch_log"),
}
