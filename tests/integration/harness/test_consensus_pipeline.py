
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from kd.harness.consensus import build_consensus
from kd.harness.consensus_report import (
    consensus_to_dict,
    render_consensus_markdown,
)

from kd.core.equation import LhsSpec, Scalar, make_evolution, to_dict
from kd.data.schema import PDEDataset, compute_dataset_fingerprint
from kd.harness.plan import PlanEntry
from tests.unit.harness._helpers import build_sealed_store, make_record

_REF = "kdv_tiny"
_UT = LhsSpec(field="u", axis="t", order=1)


def _evo(terms: list[tuple[str, float]]) -> dict[str, Any]:
    return to_dict(make_evolution(_UT, [(ir, Scalar(v)) for ir, v in terms]))



CF_KDV = _evo([("mul(u, u_x)", -6.0), ("u_xxx", -1.0)])

CF_HEAT = _evo([("u_xx", 0.1)])

_SCHEMAS = [
    {
        "algorithm": "sga",
        "modes": [
            {
                "name": "m0",
                "forms": ["EVOLUTION"],
                "topologies": ["grid"],
                "provider_kind": "finite_diff",
            }
        ],
    },
    {
        "algorithm": "discover",
        "modes": [
            {
                "name": "m0",
                "forms": ["EVOLUTION"],
                "topologies": ["grid"],
                "provider_kind": "finite_diff",
            }
        ],
    },
]


def _kdv_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 40)
    t = torch.linspace(0.0, 0.2, 20)
    u = torch.sin(2 * torch.pi * x).reshape(-1, 1) * torch.exp(-t).reshape(1, -1)
    return PDEDataset.from_arrays(
        name=_REF, coords={"x": x, "t": t}, fields={"u": u}, lhs="u_t"
    )


@pytest.fixture(scope="module")
def consensus_slice(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Any, PDEDataset]:
    dataset = _kdv_dataset()
    fp = compute_dataset_fingerprint(dataset)

    def rec(instr: str, seed: int, catalog_fit: dict[str, Any], **kw: Any) -> Any:
        return make_record(
            instr,
            seed=seed,
            dataset_name=_REF,
            dataset_cache_fingerprint=fp,
            catalog_fit=catalog_fit,
            **kw,
        )

    entries = [
        PlanEntry(instrument="sga", dataset_ref=_REF, seed=0, model_kwargs={}),
        PlanEntry(instrument="sga", dataset_ref=_REF, seed=1, model_kwargs={}),
        PlanEntry(instrument="discover", dataset_ref=_REF, seed=0, model_kwargs={}),
        PlanEntry(instrument="sga", dataset_ref=_REF, seed=2, model_kwargs={}),
    ]
    records = {

        0: rec("sga", 0, CF_KDV, config={"use_autograd": False}),
        1: rec("sga", 1, CF_KDV, config={"use_autograd": True}),
        2: rec("discover", 0, CF_KDV),

        3: rec("sga", 2, CF_HEAT, config={"use_autograd": False}),
    }
    root = tmp_path_factory.mktemp("consensus_slice") / "store"
    store = build_sealed_store(root, entries=entries, records=records)


    report = build_consensus(
        store, datasets={_REF: dataset}, instrument_schemas=_SCHEMAS
    )
    return report, dataset


@pytest.mark.integration
def test_single_dataset_node_with_two_classes(consensus_slice: Any) -> None:
    report, _dataset = consensus_slice
    assert len(report.datasets) == 1
    node = report.datasets[0]
    assert node.dataset_ref == _REF
    assert node.empirical_state == "available"
    structures = {c.terms for c in node.classes}
    assert ("mul(u,u_x)", "u_t", "u_xxx") in structures
    assert ("u_t", "u_xx") in structures


@pytest.mark.integration
def test_kdv_class_is_cross_boundary_with_three_members(consensus_slice: Any) -> None:
    report, _dataset = consensus_slice
    kdv = next(
        c
        for c in report.datasets[0].classes
        if c.terms == ("mul(u,u_x)", "u_t", "u_xxx")
    )
    assert kdv.stratification.n_members == 3
    assert kdv.stratification.n_instruments == 2
    assert kdv.stratification.corroboration_level == "multi_instrument_cross_boundary"


@pytest.mark.integration
def test_third_order_members_verify_under_max_atomic_order_3(
    consensus_slice: Any,
) -> None:
    report, _dataset = consensus_slice
    kdv = next(
        c
        for c in report.datasets[0].classes
        if c.terms == ("mul(u,u_x)", "u_t", "u_xxx")
    )
    assert "u_xxx" in kdv.terms
    for member in kdv.members:
        assert member.verification.status == "verified", (
            member.instrument,
            member.verification.stage,
            member.verification.error_type,
        )
        assert member.verification.report is not None

    assert kdv.axes.empirical.state == "evaluated"
    assert kdv.axes.empirical.subgroups


@pytest.mark.integration
def test_slice_has_adjacency_and_verifier_provenance(consensus_slice: Any) -> None:
    report, dataset = consensus_slice
    node = report.datasets[0]
    assert len(node.classes) == 2
    assert len(node.adjacency) == 1
    assert node.adjacency[0].relation in {"subset", "superset", "overlap", "disjoint"}
    assert report.provenance.verifier is not None
    assert _REF in report.provenance.datasets_provided
    assert report.provenance.dataset_fingerprints
    fp = compute_dataset_fingerprint(dataset)
    assert node.dataset_fingerprint == fp


@pytest.mark.integration
def test_slice_renders_and_is_byte_deterministic(consensus_slice: Any) -> None:
    report, dataset = consensus_slice
    rendered = render_consensus_markdown(report)
    assert rendered.startswith("# consensus ")
    assert "kd-consensus-v1" in rendered
    assert rendered.endswith("\n")


    import json

    second = build_consensus(
        report_store(report), datasets={_REF: dataset}, instrument_schemas=_SCHEMAS
    )
    assert json.dumps(
        consensus_to_dict(report), indent=2, allow_nan=False
    ) == json.dumps(consensus_to_dict(second), indent=2, allow_nan=False)


def _heat_dataset(name: str) -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 24)
    t = torch.linspace(0.0, 0.1, 12)
    u = torch.sin(2 * torch.pi * x).reshape(-1, 1) * torch.exp(-t).reshape(1, -1)
    return PDEDataset.from_arrays(
        name=name, coords={"x": x, "t": t}, fields={"u": u}, lhs="u_t"
    )


@pytest.mark.integration
def test_native_nmse_divergence_flag_fires_in_real_pipeline(tmp_path: Path) -> None:
    ref = "heat_flag"
    dataset = _heat_dataset(ref)
    fp = compute_dataset_fingerprint(dataset)
    entries = [PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={})]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=CF_HEAT,
            nmse=999.0,
        )
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    report = build_consensus(
        store, datasets={ref: dataset}, instrument_schemas=_SCHEMAS
    )
    cls = report.datasets[0].classes[0]
    member = cls.members[0]
    assert member.verification.status == "verified", (
        member.verification.stage,
        member.verification.error_type,
    )
    div = [f for f in cls.flags if f.flag == "native_nmse_divergence"]
    assert div, "native nmse 999.0 must diverge from the real verified nmse"
    assert div[0].entry_index == member.entry_index


def report_store(report: Any) -> Any:
    from kd.harness.store import EvidenceStore

    return EvidenceStore.load(Path(report.provenance.store_root))
