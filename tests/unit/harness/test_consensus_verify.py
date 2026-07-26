
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from kd.harness.consensus import build_consensus
from kd.harness.consensus_report import render_consensus_markdown
from kd.harness.consensus_verify import (
    MemberVerification,
    VerifyExecution,
    verify_members,
)

from kd.core.equation import (
    LhsSpec,
    Scalar,
    from_dict,
    make_evolution,
    make_homogeneous,
    to_dict,
)
from kd.core.platform.builder import resolve_lhs_defaults
from kd.core.verify import VerifyPolicy
from kd.data._factory import build_axes_dict, build_fields_dict
from kd.data.schema import (
    DataTopology,
    PDEDataset,
    TaskType,
    compute_dataset_fingerprint,
)
from kd.harness.plan import PlanEntry
from kd.search.records import RunRecord

from ._helpers import build_sealed_store, make_entry, make_record

_UT = LhsSpec(field="u", axis="t", order=1)


def _evo(terms: list[tuple[str, float]]) -> dict[str, Any]:
    return to_dict(make_evolution(_UT, [(ir, Scalar(v)) for ir, v in terms]))


def _hom(terms: list[tuple[str, float]]) -> dict[str, Any]:
    return to_dict(make_homogeneous([(ir, Scalar(v)) for ir, v in terms]))


CF_BURGERS = _evo([("u_xx", 0.1), ("mul(u, u_x)", -1.0)])
CF_HEAT = _evo([("u_xx", 3.0)])
CF_HOM = _hom([("u_t", 1.0), ("u_xx", -1.0)])

_SENTINEL = object()


def _sentinel_execution(provider_kind: str = "finite_diff") -> VerifyExecution:
    return VerifyExecution(
        executor=_SENTINEL,
        context=_SENTINEL,
        provider_kind=provider_kind,
    )


def _tiny_dataset(name: str = "ds_evo") -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 8)
    u = torch.sin(x).reshape(-1, 1) * torch.cos(t).reshape(1, -1)
    return PDEDataset.from_arrays(
        name=name, coords={"x": x, "t": t}, fields={"u": u}, lhs="u_t"
    )


def _hom_dataset(name: str = "ds_hom") -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 8)
    u = torch.sin(x).reshape(-1, 1) * torch.cos(t).reshape(1, -1)
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes=build_axes_dict({"x": x, "t": t}, dtype=torch.float64, periodic=set()),
        axis_order=["x", "t"],
        fields=build_fields_dict({"u": u}, dtype=torch.float64),
        lhs_field="",
        lhs_axis="",
        lhs_order=0,
    )


def _unresolved_evo_dataset(name: str = "ds_default_lhs") -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 8)
    u = torch.sin(x).reshape(-1, 1) * torch.cos(t).reshape(1, -1)
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes=build_axes_dict({"x": x, "t": t}, dtype=torch.float64, periodic=set()),
        axis_order=["x", "t"],
        fields=build_fields_dict({"u": u}, dtype=torch.float64),
        lhs_field="",
        lhs_axis="",
        lhs_order=1,
    )





def test_verify_members_none_execution_marks_all_not_evaluated() -> None:
    equations = {0: from_dict(CF_HEAT), 3: from_dict(CF_BURGERS)}
    result = verify_members(equations, None, policy=VerifyPolicy())
    assert set(result) == {0, 3}
    for member in result.values():
        assert isinstance(member, MemberVerification)
        assert member.status == "not_evaluated"
        assert member.stage is None
        assert member.error_type is None
        assert member.report is None





def test_verify_members_passes_coefficients_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    def fake_verify(eq: Any, **kwargs: Any) -> Any:
        captured.append(eq)
        return _SENTINEL

    monkeypatch.setattr("kd.harness.consensus_verify.verify_equation", fake_verify)

    equation = from_dict(CF_BURGERS)
    result = verify_members({0: equation}, _sentinel_execution(), policy=VerifyPolicy())
    assert result[0].status == "verified"
    assert len(captured) == 1
    passed_eq = captured[0]


    got = {ir: float(coeff.value) for ir, coeff in passed_eq.terms}


    assert got == {"u_xx": 0.1, "mul(u, u_x)": -1.0}


def test_verify_members_forwards_policy_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[VerifyPolicy] = []

    def fake_verify(eq: Any, *, policy: Any, **kwargs: Any) -> Any:
        seen.append(policy)
        return _SENTINEL

    monkeypatch.setattr("kd.harness.consensus_verify.verify_equation", fake_verify)

    policy = VerifyPolicy(coeff_atol=0.05, empirical_atol=0.07)
    verify_members({0: from_dict(CF_HEAT)}, _sentinel_execution(), policy=policy)
    assert seen == [policy]





def test_verify_members_failure_is_totalized_and_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def fake_verify(eq: Any, **kwargs: Any) -> Any:


        if len(eq.terms) == 1:
            raise ValueError("degenerate normalizer")
        return _SENTINEL

    monkeypatch.setattr("kd.harness.consensus_verify.verify_equation", fake_verify)

    result = verify_members(
        {0: from_dict(CF_HEAT), 1: from_dict(CF_BURGERS)},
        _sentinel_execution(),
        policy=VerifyPolicy(),
    )
    assert result[0].status == "failed"
    assert result[0].stage == "verify"
    assert result[0].error_type == "ValueError"
    assert result[0].error_message is not None
    assert result[1].status == "verified"





def _dataset_store(
    tmp_path: Path,
    *,
    instruments: list[tuple[str, int]],
    catalog_fits: dict[int, dict[str, Any] | None],
    dataset: PDEDataset,
    raised: dict[int, tuple[str, str]] | None = None,
) -> Any:
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name
    entries = [
        PlanEntry(instrument=instr, dataset_ref=ref, seed=seed, model_kwargs={})
        for instr, seed in instruments
    ]
    records: dict[int, RunRecord] = {}
    for index, (instr, seed) in enumerate(instruments):
        if index in catalog_fits and index not in (raised or {}):
            records[index] = make_record(
                instr,
                seed=seed,
                dataset_name=ref,
                dataset_cache_fingerprint=fp,
                catalog_fit=catalog_fits[index],
            )
    return build_sealed_store(
        tmp_path / "store", entries=entries, records=records, raised=raised
    )


def test_datasets_none_makes_empirical_no_dataset(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    report = build_consensus(store, datasets=None)
    dataset = report.datasets[0]
    assert dataset.empirical_state == "no_dataset"
    member = dataset.classes[0].members[0]
    assert member.verification.status == "not_evaluated"
    assert report.provenance.verifier is None


def test_verifier_provenance_non_none_iff_datasets_provided(tmp_path: Path) -> None:
    dataset = _tiny_dataset()
    store = _dataset_store(
        tmp_path,
        instruments=[("sga", 0)],
        catalog_fits={0: CF_HEAT},
        dataset=dataset,
    )
    report = build_consensus(
        store,
        datasets={dataset.name: dataset},
        context_factory=lambda ds: _sentinel_execution("finite_diff"),
    )
    assert report.provenance.verifier is not None
    assert report.provenance.verifier.provider_kind == "finite_diff"
    assert dataset.name in report.provenance.datasets_provided


def test_context_build_failure_totalizes_dataset(tmp_path: Path) -> None:
    dataset = _tiny_dataset()
    store = _dataset_store(
        tmp_path,
        instruments=[("sga", 0), ("discover", 1)],
        catalog_fits={0: CF_HEAT, 1: CF_HEAT},
        dataset=dataset,
    )

    def boom(ds: PDEDataset) -> VerifyExecution:
        raise NotImplementedError("homogeneous dataset has no provider")

    report = build_consensus(
        store, datasets={dataset.name: dataset}, context_factory=boom
    )
    dnode = report.datasets[0]
    assert dnode.empirical_state == "context_build_failed"
    for member in dnode.classes[0].members:
        assert member.verification.status == "failed"
        assert member.verification.stage == "context_build"

    assert report.provenance.verifier is not None


def test_factory_called_once_per_ref_only_with_signable_members(tmp_path: Path) -> None:
    ds_a = _tiny_dataset(name="ds_a")
    ds_b = _tiny_dataset(name="ds_b")
    fp_a = compute_dataset_fingerprint(ds_a)
    fp_b = compute_dataset_fingerprint(ds_b)
    entries = [
        PlanEntry(instrument="sga", dataset_ref="ds_a", seed=0, model_kwargs={}),
        PlanEntry(instrument="sga", dataset_ref="ds_b", seed=0, model_kwargs={}),
    ]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name="ds_a",
            dataset_cache_fingerprint=fp_a,
            catalog_fit=CF_HEAT,
        ),
        1: make_record(
            "sga",
            seed=0,
            dataset_name="ds_b",
            dataset_cache_fingerprint=fp_b,
            catalog_fit=None,
        ),
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)

    calls: list[str] = []

    def counting_factory(dataset: PDEDataset) -> VerifyExecution:
        calls.append(dataset.name)
        return _sentinel_execution()

    report = build_consensus(
        store,
        datasets={"ds_a": ds_a, "ds_b": ds_b},
        context_factory=counting_factory,
    )

    assert calls == ["ds_a"]
    node_b = next(d for d in report.datasets if d.dataset_ref == "ds_b")
    assert node_b.empirical_state == "available"
    assert node_b.classes == ()


def test_provider_kind_disagreement_across_refs_raises(tmp_path: Path) -> None:
    ds_a = _tiny_dataset(name="ds_a")
    ds_b = _tiny_dataset(name="ds_b")
    fp_a = compute_dataset_fingerprint(ds_a)
    fp_b = compute_dataset_fingerprint(ds_b)
    entries = [
        PlanEntry(instrument="sga", dataset_ref="ds_a", seed=0, model_kwargs={}),
        PlanEntry(instrument="sga", dataset_ref="ds_b", seed=0, model_kwargs={}),
    ]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name="ds_a",
            dataset_cache_fingerprint=fp_a,
            catalog_fit=CF_HEAT,
        ),
        1: make_record(
            "sga",
            seed=0,
            dataset_name="ds_b",
            dataset_cache_fingerprint=fp_b,
            catalog_fit=CF_HEAT,
        ),
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)

    def diverging_factory(dataset: PDEDataset) -> VerifyExecution:
        kind = "finite_diff" if dataset.name == "ds_a" else "autograd"
        return _sentinel_execution(kind)

    with pytest.raises(ValueError):
        build_consensus(
            store,
            datasets={"ds_a": ds_a, "ds_b": ds_b},
            context_factory=diverging_factory,
        )


def test_default_factory_declares_kind_when_context_build_fails(
    tmp_path: Path,
) -> None:
    dataset = _hom_dataset()
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name
    entries = [PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={})]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=CF_HOM,
        )
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)


    report = build_consensus(store, datasets={ref: dataset})
    assert report.datasets[0].empirical_state == "context_build_failed"
    verifier = report.provenance.verifier
    assert verifier is not None
    assert verifier.provider_kind == "finite_diff"


def test_bare_factory_all_builds_fail_yields_none_and_na_markdown(
    tmp_path: Path,
) -> None:
    dataset = _tiny_dataset()
    store = _dataset_store(
        tmp_path,
        instruments=[("sga", 0)],
        catalog_fits={0: CF_HEAT},
        dataset=dataset,
    )

    def bare(ds: PDEDataset) -> VerifyExecution:
        raise NotImplementedError("no context available")

    report = build_consensus(
        store, datasets={dataset.name: dataset}, context_factory=bare
    )
    verifier = report.provenance.verifier
    assert verifier is not None
    assert verifier.provider_kind is None
    rendered = render_consensus_markdown(report)
    assert "provider_kind=n/a" in rendered
    assert "provider_kind=," not in rendered
    assert "provider_kind= " not in rendered


def test_declared_kind_disagreeing_with_observed_raises(tmp_path: Path) -> None:
    dataset = _tiny_dataset()
    store = _dataset_store(
        tmp_path,
        instruments=[("sga", 0)],
        catalog_fits={0: CF_HEAT},
        dataset=dataset,
    )

    class LyingFactory:
        provider_kind = "autograd"

        def __call__(self, ds: PDEDataset) -> VerifyExecution:
            return _sentinel_execution("finite_diff")

    with pytest.raises(ValueError, match="provider_kind"):
        build_consensus(
            store, datasets={dataset.name: dataset}, context_factory=LyingFactory()
        )


def test_build_consensus_rejects_dataset_fingerprint_mismatch(tmp_path: Path) -> None:
    dataset = _tiny_dataset()
    entries = [
        PlanEntry(instrument="sga", dataset_ref=dataset.name, seed=0, model_kwargs={})
    ]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name=dataset.name,
            dataset_cache_fingerprint="sha256:not-the-real-fingerprint",
            catalog_fit=CF_HEAT,
        )
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    with pytest.raises(ValueError):
        build_consensus(
            store,
            datasets={dataset.name: dataset},
            context_factory=lambda ds: _sentinel_execution(),
        )


def test_default_lhs_dataset_does_not_false_mismatch(tmp_path: Path) -> None:
    dataset = _unresolved_evo_dataset()
    ref = dataset.name


    resolved_fp = compute_dataset_fingerprint(resolve_lhs_defaults(dataset))
    entries = [PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={})]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name=ref,
            dataset_cache_fingerprint=resolved_fp,
            catalog_fit=CF_HEAT,
        )
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)

    report = build_consensus(
        store,
        datasets={ref: dataset},
        context_factory=lambda ds: _sentinel_execution(),
    )
    assert report.datasets[0].dataset_fingerprint == resolved_fp
