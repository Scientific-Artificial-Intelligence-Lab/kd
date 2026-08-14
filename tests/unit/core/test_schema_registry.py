
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import kd
from kd.core.schema_registry import SCHEMA_REGISTRY

pytestmark = pytest.mark.unit




CONVERGENCE_TABLE: tuple[tuple[str, str, str | None], ...] = (

    ("kd-term-v1", "TERM_FINGERPRINT_DOMAIN", None),
    ("kd-termlib-v1", "CATALOG_FINGERPRINT_DOMAIN", None),
    ("kd-lawsig-v1", "LAWSIG_DOMAIN", None),
    ("kd-sketch-v1", "SKETCH_SCHEMA_TAG", None),
    ("kd-verification-v1", "VERIFICATION_ARTIFACT_TAG", None),

    ("kd-config-v1", "CONFIG_CANON_SCHEME", None),
    ("kd-runspec-v1", "RUN_SPEC_HASH_SCHEME", None),
    ("kd-rundir-v1", "RUNDIR_SCHEME", "RUNDIR_SCHEMA_VERSION"),
    ("kd-runcat-v1", "RUNCAT_SCHEME", "RUNCAT_SCHEMA_VERSION"),
    ("kd-iterevent-v1", "ITEREVENT_SCHEME", "ITEREVENT_SCHEMA_VERSION"),
    ("kd-runphase-v1", "PHASE_SCHEME", "PHASE_SCHEMA_VERSION"),
    ("kd-ckptman-v1", "CKPTMAN_SCHEME", "CKPTMAN_SCHEMA_VERSION"),
    ("kd-confighash-v1", "CKPTMAN_CONFIG_HASH_SCHEME", None),
    ("kd-record-v1", "RECORD_HASH_SCHEME", "RUN_RECORD_SCHEMA_VERSION"),
    ("kd-evidence-v1", "EVIDENCE_HASH_SCHEME", "RUN_RECORD_SCHEMA_VERSION"),
    ("kd-torch-module-v1", "TORCH_MODULE_ARTIFACT_FORMAT", None),

    ("kd-field-model-v1", "_FIELD_MODEL_ARTIFACT_FORMAT", None),
    ("kd-sketch-outcome-v1", "SKETCH_OUTCOME_ARTIFACT_TAG", None),

    ("kd-plan-v1", "PLAN_HASH_SCHEME", "PLAN_SCHEMA_VERSION"),
    ("kd-consensus-v1", "CONSENSUS_ARTIFACT_TAG", None),
    ("kd-stratum-rules-v1", "STRATUM_RULES_VERSION", None),
    ("kd-eligibility-rules-v1", "ELIGIBILITY_RULES_VERSION", None),
    ("kd-dispatch-v1", "DISPATCH_ARTIFACT_TAG", "DISPATCH_SCHEMA_VERSION"),
    ("kd-dispatch-log-v1", "DISPATCH_LOG_ARTIFACT_TAG", "DISPATCH_LOG_SCHEMA_VERSION"),
)


_TAG_LITERAL = re.compile(r"[\"'](kd-[a-z0-9-]+-v\d+)[\"']")


def _package_root() -> Path:
    return Path(kd.__file__).parent


def _tags_in_source() -> set[str]:
    root = _package_root()
    registry_file = root / "core" / "schema_registry.py"
    return {
        match.group(1)
        for path in root.rglob("*.py")
        if path != registry_file
        for match in _TAG_LITERAL.finditer(path.read_text(encoding="utf-8"))
    }


@pytest.mark.parametrize(
    ("tag", "tag_const", "version_const"),
    CONVERGENCE_TABLE,
    ids=[row[0] for row in CONVERGENCE_TABLE],
)
def test_owner_module_agrees_with_registry(
    tag: str, tag_const: str, version_const: str | None
) -> None:
    entry = SCHEMA_REGISTRY[tag]


    module = importlib.import_module(entry.module)

    assert getattr(module, tag_const) == tag
    if version_const is not None:
        assert getattr(module, version_const) == entry.version


def test_convergence_table_covers_registry() -> None:
    assert {row[0] for row in CONVERGENCE_TABLE} == set(SCHEMA_REGISTRY)


def test_registry_covers_every_tag_in_src() -> None:
    assert _tags_in_source() == set(SCHEMA_REGISTRY)
