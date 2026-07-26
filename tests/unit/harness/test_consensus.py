
from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from kd.harness.consensus import (
    CONSENSUS_ARTIFACT_TAG,
    CORROBORATION_ORDER,
    COUNT_FIELD_LEVELS,
    DEPENDENCY_STRATA,
    ELIGIBILITY_RULES_VERSION,
    STRATUM_RULES_VERSION,
    UNSIGNABLE_REASONS,
    ConsensusPolicy,
    build_consensus,
)
from kd.harness.consensus_report import consensus_to_dict, render_consensus_markdown
from kd.harness.consensus_verify import VerifyExecution

from kd.core.equation import (
    Form,
    LhsSpec,
    Scalar,
    from_dict,
    make_evolution,
    make_homogeneous,
    to_dict,
)
from kd.core.equation.signature import law_signature
from kd.core.verify import VerificationReport, VerifyPolicy
from kd.data._factory import build_axes_dict, build_fields_dict
from kd.data.schema import (
    DataTopology,
    PDEDataset,
    TaskType,
    compute_dataset_fingerprint,
)
from kd.harness.plan import PlanEntry
from kd.search.records import RunRecord
from tests.adversarial.lawsig.corpus import CORPUS, KNOWN_SPLIT_FAMILIES, Expected

from ._helpers import build_sealed_store, make_entry, make_record



_UT = LhsSpec(field="u", axis="t", order=1)


def _evo(
    terms: list[tuple[str, float]],
    *,
    lhs: LhsSpec = _UT,
    active: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    eq = make_evolution(
        lhs, [(ir, Scalar(v)) for ir, v in terms], active_indices=active
    )
    return to_dict(eq)


def _hom(terms: list[tuple[str, float]]) -> dict[str, Any]:
    return to_dict(make_homogeneous([(ir, Scalar(v)) for ir, v in terms]))


CF_HEAT = _evo([("u_xx", 0.1)])
CF_BURGERS = _evo([("u_xx", 0.1), ("mul(u, u_x)", -1.0)])
CF_BURGERS_STRONG = _evo([("u_xx", 0.1), ("mul(u, u_x)", -5.0)])
CF_ADVEC = _evo([("u_x", -1.0)])
CF_VHEAT = _evo(
    [("v_xx", 0.1)], lhs=LhsSpec(field="v", axis="t", order=1)
)





def _schema(
    algorithm: str, modes: list[tuple[list[str], list[str], str]]
) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "modes": [
            {
                "name": f"m{i}",
                "forms": forms,
                "topologies": topologies,
                "provider_kind": provider_kind,
            }
            for i, (forms, topologies, provider_kind) in enumerate(modes)
        ],
    }


def _store(
    tmp_path: Path,
    *,
    records: dict[int, RunRecord] | None = None,
    entries: list[PlanEntry],
    raised: dict[int, tuple[str, str]] | None = None,
    no_record: list[int] | None = None,
) -> Any:
    return build_sealed_store(
        tmp_path / "store",
        entries=entries,
        records=records,
        raised=raised,
        no_record=no_record,
    )


def _only_dataset(report: Any) -> Any:
    assert len(report.datasets) == 1
    return report.datasets[0]


def _tiny_evo_dataset(name: str = "ds_evo", lhs: str = "u_t") -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 8)
    u = torch.sin(x).reshape(-1, 1) * torch.cos(t).reshape(1, -1)
    return PDEDataset.from_arrays(
        name=name, coords={"x": x, "t": t}, fields={"u": u}, lhs=lhs
    )


def _tiny_hom_dataset(name: str = "ds_hom") -> PDEDataset:
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


_SENTINEL = object()


def _sentinel_factory(
    provider_kind: str = "finite_diff",
) -> Callable[[PDEDataset], Any]:
    def factory(dataset: PDEDataset) -> Any:
        return VerifyExecution(
            executor=_SENTINEL,
            context=_SENTINEL,
            provider_kind=provider_kind,
        )

    return factory





def test_vocabulary_constants_are_frozen() -> None:
    assert CONSENSUS_ARTIFACT_TAG == "kd-consensus-v1"
    assert STRATUM_RULES_VERSION == "kd-stratum-rules-v1"
    assert ELIGIBILITY_RULES_VERSION == "kd-eligibility-rules-v1"
    assert (
        frozenset({"finite_diff", "autograd", "none", "undeclared"})
        == DEPENDENCY_STRATA
    )
    assert (
        frozenset({"invalid_record", "no_catalog_fit", "signature_error"})
        == UNSIGNABLE_REASONS
    )

    assert CORROBORATION_ORDER == (
        "multi_instrument_cross_boundary",
        "multi_instrument_correlated",
        "single_instrument",
        "single_run",
    )




_MERGE_PAIRS = [
    p
    for p in CORPUS
    if p.expected in (Expected.MERGE_EXACT, Expected.MERGE_STRUCT)
    and p.family not in KNOWN_SPLIT_FAMILIES
]
_SPLIT_PAIRS = [p for p in CORPUS if p.expected == Expected.SPLIT]
_RAISE_PAIRS = [p for p in CORPUS if p.expected == Expected.RAISES]


@pytest.mark.parametrize("pair", _MERGE_PAIRS, ids=[p.pair_id for p in _MERGE_PAIRS])
def test_merge_pairs_form_one_class(tmp_path: Path, pair: Any) -> None:
    entries = [make_entry("sga", seed=0), make_entry("discover", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=to_dict(pair.eq_a)),
        1: make_record("discover", seed=1, catalog_fit=to_dict(pair.eq_b)),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    dataset = _only_dataset(report)
    assert len(dataset.classes) == 1
    assert dataset.classes[0].stratification.n_members == 2


@pytest.mark.parametrize("pair", _SPLIT_PAIRS, ids=[p.pair_id for p in _SPLIT_PAIRS])
def test_split_pairs_form_two_classes(tmp_path: Path, pair: Any) -> None:
    entries = [make_entry("sga", seed=0), make_entry("discover", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=to_dict(pair.eq_a)),
        1: make_record("discover", seed=1, catalog_fit=to_dict(pair.eq_b)),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    dataset = _only_dataset(report)
    assert len(dataset.classes) == 2
    assert all(cls.stratification.n_members == 1 for cls in dataset.classes)


@pytest.mark.parametrize("pair", _RAISE_PAIRS, ids=[p.pair_id for p in _RAISE_PAIRS])
def test_raises_family_is_signature_error(tmp_path: Path, pair: Any) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=to_dict(pair.eq_a))}
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    assert _only_dataset(report).classes == ()
    unsignable = report.store_level.unsignable
    assert len(unsignable) == 1
    assert unsignable[0].reason == "signature_error"
    assert unsignable[0].error_type is not None


def test_cross_form_pair_splits_support_axis(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("discover", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=_evo([("u_xx", 3.0)])),
        1: make_record(
            "discover", seed=1, catalog_fit=_hom([("u_t", 1.0), ("u_xx", -3.0)])
        ),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    cls = _only_dataset(report).classes[0]
    assert cls.stratification.n_members == 2
    assert cls.axes.support.state == "evaluated"
    assert cls.axes.support.all_agree is False
    variant_forms = {v.native_form.value for v in cls.axes.support.variants}
    assert variant_forms == {"EVOLUTION", "HOMOGENEOUS"}


def test_catalog_active_indices_drive_grouping(tmp_path: Path) -> None:
    catalog = [("u", 0.5), ("u_x", -1.0), ("u_xx", 0.1), ("mul(u, u_x)", -2.0)]
    entries = [make_entry("sga", seed=0), make_entry("discover", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=_evo(catalog, active=(2,))),
        1: make_record("discover", seed=1, catalog_fit=_evo(catalog, active=(1, 3))),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    assert len(_only_dataset(report).classes) == 2





def test_unsignable_invalid_record_wins_over_missing_catalog(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {
        0: make_record(
            "sga",
            seed=0,
            is_valid=False,
            invalid_reason="no_candidate",
            catalog_fit=None,
        )
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    entry = report.store_level.unsignable[0]
    assert entry.reason == "invalid_record"
    assert entry.invalid_reason == "no_candidate"


def test_unsignable_no_catalog_fit(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, is_valid=True, catalog_fit=None)}
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    assert report.store_level.unsignable[0].reason == "no_catalog_fit"





def test_stratum_r1_unknown_instrument_is_undeclared(tmp_path: Path) -> None:
    entries = [make_entry("ghost", seed=0)]
    records = {0: make_record("ghost", seed=0, catalog_fit=CF_HEAT)}
    schemas = [_schema("other", [(["EVOLUTION"], ["grid"], "finite_diff")])]
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=schemas,
    )
    member = _only_dataset(report).classes[0].members[0]
    assert member.dependency.stratum == "undeclared"


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"use_autograd": True}, "autograd"),
        ({"use_autograd": False}, "finite_diff"),
        ({}, "undeclared"),
    ],
)
def test_stratum_r2_sga_use_autograd_tristate(
    tmp_path: Path, config: dict[str, Any], expected: str
) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT, config=config)}
    schemas = [_schema("sga", [(["EVOLUTION"], ["grid"], "finite_diff")])]
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=schemas,
    )
    member = _only_dataset(report).classes[0].members[0]
    assert member.dependency.stratum == expected


def test_stratum_r2_wins_over_lying_provider_kind(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {
        0: make_record(
            "sga",
            seed=0,
            catalog_fit=CF_HEAT,
            config={"use_autograd": True, "provider_kind": "finite_diff"},
        )
    }
    schemas = [_schema("sga", [(["EVOLUTION"], ["grid"], "finite_diff")])]
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=schemas,
    )
    assert _only_dataset(report).classes[0].members[0].dependency.stratum == "autograd"


def test_stratum_r3_single_mode_provider_kind(tmp_path: Path) -> None:
    entries = [make_entry("toolfd", seed=0)]
    records = {0: make_record("toolfd", seed=0, catalog_fit=CF_HEAT)}
    schemas = [_schema("toolfd", [(["EVOLUTION"], ["grid"], "finite_diff")])]
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=schemas,
    )
    assert (
        _only_dataset(report).classes[0].members[0].dependency.stratum == "finite_diff"
    )


def test_stratum_r4_multi_mode_reads_config_provider_kind(tmp_path: Path) -> None:
    entries = [make_entry("multi", seed=0)]
    records = {
        0: make_record(
            "multi", seed=0, catalog_fit=CF_HEAT, config={"provider_kind": "none"}
        )
    }
    schemas = [
        _schema(
            "multi",
            [
                (["EVOLUTION"], ["grid"], "finite_diff"),
                (["HOMOGENEOUS"], ["grid"], "none"),
            ],
        )
    ]
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=schemas,
    )
    assert _only_dataset(report).classes[0].members[0].dependency.stratum == "none"


def test_stratum_r5_multi_mode_missing_config_is_undeclared(tmp_path: Path) -> None:
    entries = [make_entry("multi", seed=0)]
    records = {0: make_record("multi", seed=0, catalog_fit=CF_HEAT)}
    schemas = [
        _schema(
            "multi",
            [
                (["EVOLUTION"], ["grid"], "finite_diff"),
                (["HOMOGENEOUS"], ["grid"], "none"),
            ],
        )
    ]
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=schemas,
    )
    assert (
        _only_dataset(report).classes[0].members[0].dependency.stratum == "undeclared"
    )





def _fd_schema(algorithm: str) -> dict[str, Any]:
    return _schema(algorithm, [(["EVOLUTION"], ["grid"], "finite_diff")])


def test_corroboration_single_run(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=[_fd_schema("sga")],
    )
    assert _only_dataset(report).classes[0].stratification.corroboration_level == (
        "single_run"
    )


def test_corroboration_single_instrument_multi_seed(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=s) for s in range(3)]
    records = {i: make_record("sga", seed=i, catalog_fit=CF_HEAT) for i in range(3)}
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=[_fd_schema("sga")],
    )
    cls = _only_dataset(report).classes[0]



    assert cls.stratification.corroboration_level == "single_instrument"


def test_corroboration_multi_instrument_correlated_same_stratum(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("toolfd", seed=0)]
    records = {
        0: make_record(
            "sga", seed=0, catalog_fit=CF_HEAT, config={"use_autograd": False}
        ),
        1: make_record("toolfd", seed=0, catalog_fit=CF_HEAT),
    }
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=[_fd_schema("sga"), _fd_schema("toolfd")],
    )
    assert _only_dataset(report).classes[0].stratification.corroboration_level == (
        "multi_instrument_correlated"
    )


def test_corroboration_multi_instrument_cross_boundary(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("toolfd", seed=0)]
    records = {
        0: make_record(
            "sga", seed=0, catalog_fit=CF_HEAT, config={"use_autograd": True}
        ),
        1: make_record("toolfd", seed=0, catalog_fit=CF_HEAT),
    }
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=[_fd_schema("sga"), _fd_schema("toolfd")],
    )
    assert _only_dataset(report).classes[0].stratification.corroboration_level == (
        "multi_instrument_cross_boundary"
    )


def test_corroboration_undeclared_does_not_cross_boundary(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("ghost", seed=0)]
    records = {
        0: make_record(
            "sga", seed=0, catalog_fit=CF_HEAT, config={"use_autograd": False}
        ),
        1: make_record("ghost", seed=0, catalog_fit=CF_HEAT),
    }
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=[
            _fd_schema("sga")
        ],
    )
    assert _only_dataset(report).classes[0].stratification.corroboration_level == (
        "multi_instrument_correlated"
    )





def test_eligibility_present_instruments_are_eligible_present(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=[_fd_schema("sga")],
    )
    elig = _only_dataset(report).classes[0].stratification.eligibility
    assert elig.rules_version == ELIGIBILITY_RULES_VERSION
    assert "sga" in elig.eligible_present


def test_eligibility_unknown_when_topology_unknown(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("disc", seed=0)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=CF_HEAT),
        1: make_record("disc", seed=0, catalog_fit=CF_BURGERS),
    }
    schemas = [
        _fd_schema("sga"),
        _schema("disc", [(["HOMOGENEOUS", "EVOLUTION"], ["grid"], "finite_diff")]),
    ]
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records),
        instrument_schemas=schemas,
    )
    heat_cls = next(
        c for c in _only_dataset(report).classes if c.terms == ("u_t", "u_xx")
    )
    elig = heat_cls.stratification.eligibility
    assert "disc" in elig.unknown
    absent_names = {a.instrument for a in elig.eligible_absent}
    assert "disc" not in absent_names


@pytest.mark.parametrize(
    "outcome_kind,expected_reason",
    [
        ("other_class", "other_class"),
        ("unsignable", "unsignable"),
        ("failed", "failed"),
    ],
)
def test_eligibility_absent_attribution_trichotomy(
    tmp_path: Path, outcome_kind: str, expected_reason: str
) -> None:
    dataset = _tiny_evo_dataset()
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name

    entries = [
        PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={}),
        PlanEntry(instrument="probe", dataset_ref=ref, seed=0, model_kwargs={}),
    ]

    def rec(instr: str, **kw: Any) -> RunRecord:
        return make_record(
            instr, seed=0, dataset_name=ref, dataset_cache_fingerprint=fp, **kw
        )

    records = {0: rec("sga", catalog_fit=CF_HEAT)}
    raised: dict[int, tuple[str, str]] = {}
    if outcome_kind == "other_class":
        records[1] = rec("probe", catalog_fit=CF_BURGERS)
    elif outcome_kind == "unsignable":
        records[1] = rec("probe", catalog_fit=None)
    else:
        raised[1] = ("RuntimeError", "boom")

    schemas = [
        _fd_schema("sga"),
        _schema("probe", [(["HOMOGENEOUS", "EVOLUTION"], ["grid"], "finite_diff")]),
    ]
    store = build_sealed_store(
        tmp_path / "store", entries=entries, records=records, raised=raised
    )
    report = build_consensus(
        store,
        datasets={ref: dataset},
        instrument_schemas=schemas,
        context_factory=_sentinel_factory(),
    )
    heat_cls = next(c for c in report.datasets[0].classes if c.terms == ("u_t", "u_xx"))
    elig = heat_cls.stratification.eligibility
    absent = {a.instrument: a.reason for a in elig.eligible_absent}
    assert absent.get("probe") == expected_reason


def test_eligibility_present_overrides_rules_and_records_conflict(
    tmp_path: Path,
) -> None:
    dataset = _tiny_evo_dataset()
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name
    entries = [PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={})]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=CF_HEAT,
        )
    }

    schemas = [_schema("sga", [(["EVOLUTION"], ["scattered"], "finite_diff")])]
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    report = build_consensus(
        store,
        datasets={ref: dataset},
        instrument_schemas=schemas,
        context_factory=_sentinel_factory(),
    )
    elig = report.datasets[0].classes[0].stratification.eligibility
    assert "sga" in elig.eligible_present
    assert "sga" in elig.rule_conflicts


def test_homogeneous_dataset_makes_evolution_only_instrument_ineligible(
    tmp_path: Path,
) -> None:
    dataset = _tiny_hom_dataset()
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name
    entries = [
        PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={}),
        PlanEntry(instrument="evoonly", dataset_ref=ref, seed=0, model_kwargs={}),
    ]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=_hom([("u_t", 1.0), ("u_xx", -1.0)]),
        )
    }
    schemas = [
        _schema("sga", [(["HOMOGENEOUS"], ["grid"], "finite_diff")]),
        _schema("evoonly", [(["EVOLUTION"], ["grid"], "finite_diff")]),
    ]
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    report = build_consensus(
        store,
        datasets={ref: dataset},
        instrument_schemas=schemas,
        context_factory=_sentinel_factory(),
    )
    elig = report.datasets[0].classes[0].stratification.eligibility
    assert "evoonly" in elig.ineligible


def test_eligibility_lhs_double_source_conflict_is_unknown(tmp_path: Path) -> None:
    dataset = _tiny_evo_dataset(name="ds_utt", lhs="u_tt")
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name
    entries = [
        PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={}),
        PlanEntry(instrument="evoonly", dataset_ref=ref, seed=0, model_kwargs={}),
    ]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=CF_HEAT,
        )
    }
    schemas = [
        _schema("sga", [(["HOMOGENEOUS", "EVOLUTION"], ["grid"], "finite_diff")]),
        _schema("evoonly", [(["EVOLUTION"], ["grid"], "finite_diff")]),
    ]
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    report = build_consensus(
        store,
        datasets={ref: dataset},
        instrument_schemas=schemas,
        context_factory=_sentinel_factory(),
    )
    elig = report.datasets[0].classes[0].stratification.eligibility
    assert "evoonly" in elig.unknown





def test_adjacency_all_four_relations_and_jaccard(tmp_path: Path) -> None:
    entries = [make_entry(f"i{k}", seed=0) for k in range(4)]
    records = {
        0: make_record("i0", seed=0, catalog_fit=CF_HEAT),
        1: make_record("i1", seed=0, catalog_fit=CF_BURGERS),
        2: make_record("i2", seed=0, catalog_fit=CF_ADVEC),
        3: make_record("i3", seed=0, catalog_fit=CF_VHEAT),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    dataset = _only_dataset(report)
    key_by_terms = {c.terms: c.structure_key for c in dataset.classes}
    heat = key_by_terms[("u_t", "u_xx")]
    burgers = key_by_terms[("mul(u,u_x)", "u_t", "u_xx")]
    advec = key_by_terms[("u_t", "u_x")]
    vheat = key_by_terms[("v_t", "v_xx")]

    adj = {
        frozenset({a.structure_key_a, a.structure_key_b}): a for a in dataset.adjacency
    }

    hb = adj[frozenset({heat, burgers})]
    assert hb.relation in {"subset", "superset"}
    assert math.isclose(hb.jaccard, 2 / 3)

    ha = adj[frozenset({heat, advec})]
    assert ha.relation == "overlap"
    assert math.isclose(ha.jaccard, 1 / 3)

    hv = adj[frozenset({heat, vheat})]
    assert hv.relation == "disjoint"
    assert hv.jaccard == 0.0

    relations = {a.relation for a in dataset.adjacency}
    assert relations <= {"subset", "superset", "overlap", "disjoint"}
    assert {"overlap", "disjoint"} <= relations


def test_adjacency_pairs_use_display_order_a_before_b(tmp_path: Path) -> None:
    entries = [make_entry(f"i{k}", seed=0) for k in range(3)]
    records = {
        0: make_record("i0", seed=0, catalog_fit=CF_HEAT),
        1: make_record("i1", seed=0, catalog_fit=CF_BURGERS),
        2: make_record("i2", seed=0, catalog_fit=CF_ADVEC),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    dataset = _only_dataset(report)
    order = [c.structure_key for c in dataset.classes]
    for a in dataset.adjacency:
        assert order.index(a.structure_key_a) < order.index(a.structure_key_b)





def test_instrument_stability_counts_and_modal_share(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=s) for s in range(3)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=CF_HEAT),
        1: make_record("sga", seed=1, catalog_fit=CF_HEAT),
        2: make_record("sga", seed=2, catalog_fit=CF_BURGERS),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    stability = _only_dataset(report).instrument_stability
    sga = next(s for s in stability if s.instrument == "sga")
    assert sga.n_completed == 3
    assert sga.n_signable == 3
    assert sga.modal_structure_key is not None
    assert math.isclose(sga.modal_share, 2 / 3)


def test_instrument_stability_excludes_unsignable_from_signable(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("sga", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=CF_HEAT),
        1: make_record("sga", seed=1, catalog_fit=None),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    sga = next(
        s for s in _only_dataset(report).instrument_stability if s.instrument == "sga"
    )
    assert sga.n_completed == 2
    assert sga.n_signable == 1


def test_instrument_stability_modal_tie_breaks_by_display_order(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("sga", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=CF_HEAT),
        1: make_record("sga", seed=1, catalog_fit=CF_ADVEC),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    dataset = _only_dataset(report)
    display_order = [c.structure_key for c in dataset.classes]
    sga = next(s for s in dataset.instrument_stability if s.instrument == "sga")

    assert sga.modal_structure_key == display_order[0]





def _walk(node: Any):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_every_count_field_is_registered_in_count_field_levels(tmp_path: Path) -> None:

    entries = [
        make_entry("sga", seed=0),
        make_entry("discover", seed=1),
        make_entry("pysindy", seed=2),
    ]
    records = {
        0: make_record("sga", seed=0, catalog_fit=CF_HEAT),
        1: make_record("discover", seed=1, catalog_fit=CF_HEAT),
        2: make_record("pysindy", seed=2, catalog_fit=None),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    payload = consensus_to_dict(report)
    n_keys = {key for key, _ in _walk(payload) if key.startswith("n_")}
    assert n_keys, "expected some count fields in the serialized tree"
    unregistered = n_keys - set(COUNT_FIELD_LEVELS)
    assert not unregistered, f"unregistered count fields: {sorted(unregistered)}"

    assert set(COUNT_FIELD_LEVELS.values()) <= {
        "member",
        "member_pair",
        "instrument",
        "attempt",
        "sample",
    }





def test_policy_rejects_empty_version_strings() -> None:
    with pytest.raises(ValueError):
        ConsensusPolicy(stratum_rules_version="")
    with pytest.raises(ValueError):
        ConsensusPolicy(eligibility_rules_version="")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"native_nmse_flag_atol": -1.0},
        {"native_nmse_flag_atol": math.inf},
        {"inactive_mass_flag_max": -1e-9},
        {"inactive_mass_flag_max": math.nan},
    ],
)
def test_policy_rejects_nonfinite_or_negative_thresholds(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="consensus policy"):
        ConsensusPolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stratum_rules_version": "kd-stratum-rules-v2"},
        {"eligibility_rules_version": "kd-eligibility-rules-v9"},
    ],
)
def test_build_consensus_rejects_unknown_rules_version(
    tmp_path: Path, kwargs: dict[str, str]
) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    store = _store(tmp_path, entries=entries, records=records)
    with pytest.raises(ValueError):
        build_consensus(store, policy=ConsensusPolicy(**kwargs))


def test_build_consensus_rejects_datasets_outside_plan(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    store = _store(tmp_path, entries=entries, records=records)
    with pytest.raises(ValueError):
        build_consensus(store, datasets={"not_a_plan_ref": _tiny_evo_dataset()})


def test_build_consensus_rejects_schema_missing_required_keys(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    store = _store(tmp_path, entries=entries, records=records)
    with pytest.raises(ValueError):
        build_consensus(store, instrument_schemas=[{"algorithm": "sga"}])





def _d4_no_completed(tmp_path: Path, monkeypatch: Any) -> None:
    entries = [make_entry("sga", seed=0), make_entry("sga", seed=1)]
    store = build_sealed_store(
        tmp_path / "store",
        entries=entries,
        records={},
        raised={0: ("RuntimeError", "boom")},
        no_record=[1],
    )
    report = build_consensus(store)
    dataset = _only_dataset(report)
    assert dataset.classes == ()
    assert dataset.adjacency == ()
    assert all(s.n_completed == 0 for s in dataset.instrument_stability)
    assert report.store_level.n_attempts == 2


def _d4_all_unsignable(tmp_path: Path, monkeypatch: Any) -> None:
    entries = [make_entry("sga", seed=0), make_entry("sga", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=None),
        1: make_record("sga", seed=1, catalog_fit=None),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    assert _only_dataset(report).classes == ()
    assert len(report.store_level.unsignable) == 2


def _d4_single_member_class(tmp_path: Path, monkeypatch: Any) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    cls = _only_dataset(report).classes[0]
    assert cls.stratification.corroboration_level == "single_run"
    assert cls.axes.structure.state == "unreplicated"
    assert cls.axes.structure.agreed is None
    assert cls.axes.support.state == "not_applicable"
    assert cls.axes.support.variants == ()
    assert cls.axes.support.n_pairs is None
    assert cls.axes.coefficient.state == "not_applicable"
    assert cls.axes.coefficient.pairs == ()
    assert cls.axes.coefficient.max_delta is None


def _d4_all_singletons(tmp_path: Path, monkeypatch: Any) -> None:
    entries = [make_entry("i0", seed=0), make_entry("i1", seed=0)]
    records = {
        0: make_record("i0", seed=0, catalog_fit=CF_HEAT),
        1: make_record("i1", seed=0, catalog_fit=CF_VHEAT),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    dataset = _only_dataset(report)
    assert len(dataset.classes) == 2
    assert all(c.stratification.n_members == 1 for c in dataset.classes)
    assert len(dataset.adjacency) == 1
    assert dataset.adjacency[0].relation == "disjoint"
    assert dataset.adjacency[0].jaccard == 0.0


def _d4_coefficients_over_threshold(tmp_path: Path, monkeypatch: Any) -> None:
    entries = [make_entry("sga", seed=0), make_entry("discover", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=CF_BURGERS),
        1: make_record("discover", seed=1, catalog_fit=CF_BURGERS_STRONG),
    }
    report = build_consensus(_store(tmp_path, entries=entries, records=records))
    cls = _only_dataset(report).classes[0]
    assert cls.axes.coefficient.state == "evaluated"
    assert cls.axes.coefficient.all_within_atol is False
    assert len(cls.axes.coefficient.pairs) == 1

    assert not hasattr(cls, "agree")
    assert not hasattr(cls, "verdict")


def _d4_verify_all_fail(tmp_path: Path, monkeypatch: Any) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("verify blew up")

    monkeypatch.setattr("kd.harness.consensus_verify.verify_equation", boom)
    dataset = _tiny_evo_dataset()
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name
    entries = [
        PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={}),
        PlanEntry(instrument="discover", dataset_ref=ref, seed=1, model_kwargs={}),
    ]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=CF_HEAT,
        ),
        1: make_record(
            "discover",
            seed=1,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=CF_HEAT,
        ),
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    report = build_consensus(
        store, datasets={ref: dataset}, context_factory=_sentinel_factory()
    )
    cls = report.datasets[0].classes[0]
    assert cls.axes.empirical.state == "not_evaluated"
    assert cls.axes.structure.state == "evaluated"


    assert cls.axes.empirical.n_unverified_members == cls.stratification.n_members
    for member in cls.members:
        assert member.verification.status == "failed"
        assert member.verification.stage == "verify"


_D4_CASES: list[Callable[[Path, Any], None]] = [
    _d4_no_completed,
    _d4_all_unsignable,
    _d4_single_member_class,
    _d4_all_singletons,
    _d4_coefficients_over_threshold,
    _d4_verify_all_fail,
]


@pytest.mark.parametrize("case", _D4_CASES, ids=[c.__name__ for c in _D4_CASES])
def test_d4_degenerate_table(
    tmp_path: Path, monkeypatch: Any, case: Callable[[Path, Any], None]
) -> None:
    case(tmp_path, monkeypatch)





def test_class_display_order_follows_corroboration_then_counts(tmp_path: Path) -> None:
    entries = [
        make_entry("sga", seed=0),
        make_entry("toolfd", seed=0),
        make_entry("sga", seed=1),
        make_entry("sga", seed=2),
        make_entry("discover", seed=0),
    ]
    records = {
        0: make_record(
            "sga", seed=0, catalog_fit=CF_HEAT, config={"use_autograd": True}
        ),
        1: make_record("toolfd", seed=0, catalog_fit=CF_HEAT),
        2: make_record("sga", seed=1, catalog_fit=CF_ADVEC),
        3: make_record(
            "sga", seed=2, catalog_fit=CF_ADVEC
        ),
        4: make_record("discover", seed=0, catalog_fit=CF_VHEAT),
    }
    schemas = [_fd_schema("sga"), _fd_schema("toolfd"), _fd_schema("discover")]
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records), instrument_schemas=schemas
    )
    ranks = [
        CORROBORATION_ORDER.index(c.stratification.corroboration_level)
        for c in _only_dataset(report).classes
    ]
    assert ranks == sorted(ranks)
    assert _only_dataset(report).classes[0].stratification.corroboration_level == (
        "multi_instrument_cross_boundary"
    )





def test_build_consensus_is_byte_identical_across_runs(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("discover", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=CF_HEAT),
        1: make_record("discover", seed=1, catalog_fit=CF_HEAT),
    }
    store = _store(tmp_path, entries=entries, records=records)
    first = build_consensus(store)
    second = build_consensus(store)
    dumped_first = json.dumps(consensus_to_dict(first), indent=2, allow_nan=False)
    dumped_second = json.dumps(consensus_to_dict(second), indent=2, allow_nan=False)
    assert dumped_first == dumped_second
    assert render_consensus_markdown(first) == render_consensus_markdown(second)





def test_provenance_carries_verify_policy_and_flag_thresholds_without_datasets(
    tmp_path: Path,
) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    report = build_consensus(
        _store(tmp_path, entries=entries, records=records), datasets=None
    )
    prov = consensus_to_dict(report)["provenance"]
    assert prov["verifier"] is None
    assert "coeff_atol" in prov["verify_policy"]
    assert "native_nmse_flag_atol" in prov
    assert "inactive_mass_flag_max" in prov


def test_provenance_distinguishes_policies_differing_only_in_flag_threshold(
    tmp_path: Path,
) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    store = _store(tmp_path, entries=entries, records=records)
    a = build_consensus(store, policy=ConsensusPolicy(native_nmse_flag_atol=1e-2))
    b = build_consensus(store, policy=ConsensusPolicy(native_nmse_flag_atol=2e-2))
    assert consensus_to_dict(a)["provenance"] != consensus_to_dict(b)["provenance"]





def _fake_report(
    *, nmse: float = 0.0, normalizer_term: str = "u_t", inactive_mass: float = 0.0
) -> VerificationReport:
    sig = law_signature(from_dict(CF_HEAT))
    return VerificationReport(
        signature=sig,
        form=Form.EVOLUTION,
        dataset_name="ds_evo",
        dataset_fingerprint="fp",
        mse=0.001,
        nmse=nmse,
        r2=1.0 - nmse,
        residual_mean=0.0,
        residual_std=0.1,
        residual_max_abs=0.2,
        n_samples=42,
        inactive_coefficient_mass=inactive_mass,
        normalizer_term=normalizer_term,
        normalizer_variance=1.0,
        policy=VerifyPolicy(),
        passed=None,
    )


def _verified_store(
    tmp_path: Path, specs: list[tuple[str, int, dict[str, Any], float | None]]
) -> tuple[Any, str, PDEDataset]:
    dataset = _tiny_evo_dataset()
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name
    entries = [
        PlanEntry(instrument=instr, dataset_ref=ref, seed=seed, model_kwargs={})
        for instr, seed, _, _ in specs
    ]
    records = {
        i: make_record(
            instr,
            seed=seed,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=cf,
            nmse=native_nmse,
        )
        for i, (instr, seed, cf, native_nmse) in enumerate(specs)
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    return store, ref, dataset


def test_native_nmse_flag_is_strict_and_value_is_magnitude(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "kd.harness.consensus_verify.verify_equation",
        lambda eq, **kw: _fake_report(nmse=0.0),
    )

    store, ref, dataset = _verified_store(
        tmp_path,
        [("sga", 0, _evo([("u_xx", 0.1)]), 0.25), ("disc", 1, _evo([("u_xx", 0.1)]), 0.5)],
    )
    report = build_consensus(
        store,
        datasets={ref: dataset},
        context_factory=_sentinel_factory(),
        policy=ConsensusPolicy(native_nmse_flag_atol=0.25),
    )
    cls = report.datasets[0].classes[0]
    div = [f for f in cls.flags if f.flag == "native_nmse_divergence"]
    assert [f.entry_index for f in div] == [1]
    assert math.isclose(div[0].value, 0.5)


def test_native_nmse_none_produces_no_divergence_flag(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "kd.harness.consensus_verify.verify_equation",
        lambda eq, **kw: _fake_report(nmse=0.9),
    )
    store, ref, dataset = _verified_store(
        tmp_path, [("sga", 0, _evo([("u_xx", 0.1)]), None)]
    )
    report = build_consensus(
        store, datasets={ref: dataset}, context_factory=_sentinel_factory()
    )
    cls = report.datasets[0].classes[0]
    assert not [f for f in cls.flags if f.flag == "native_nmse_divergence"]


def test_flags_only_on_verified_members(tmp_path: Path, monkeypatch: Any) -> None:

    def fake(eq: Any, **kw: Any) -> Any:
        coeffs = {ir: float(c.value) for ir, c in eq.terms}
        if math.isclose(coeffs.get("u_xx", 0.0), 0.2):
            raise ValueError("verify blew up")
        return _fake_report(nmse=0.0)

    monkeypatch.setattr("kd.harness.consensus_verify.verify_equation", fake)
    store, ref, dataset = _verified_store(
        tmp_path,
        [("sga", 0, _evo([("u_xx", 0.1)]), 0.9), ("disc", 1, _evo([("u_xx", 0.2)]), 0.9)],
    )
    report = build_consensus(
        store, datasets={ref: dataset}, context_factory=_sentinel_factory()
    )
    cls = report.datasets[0].classes[0]
    flagged = {f.entry_index for f in cls.flags}
    assert flagged == {0}


def test_flags_sort_by_entry_index_then_flag_name(
    tmp_path: Path, monkeypatch: Any
) -> None:

    def fake(eq: Any, **kw: Any) -> Any:
        coeffs = {ir: float(c.value) for ir, c in eq.terms}
        if math.isclose(coeffs.get("u_xx", 0.0), 0.1):
            return _fake_report(nmse=0.0, inactive_mass=0.5)
        return _fake_report(nmse=0.0, inactive_mass=0.0)

    monkeypatch.setattr("kd.harness.consensus_verify.verify_equation", fake)
    store, ref, dataset = _verified_store(
        tmp_path,
        [("sga", 0, _evo([("u_xx", 0.1)]), 0.5), ("disc", 1, _evo([("u_xx", 0.2)]), 0.3)],
    )
    report = build_consensus(
        store, datasets={ref: dataset}, context_factory=_sentinel_factory()
    )
    cls = report.datasets[0].classes[0]
    assert [(f.entry_index, f.flag) for f in cls.flags] == [
        (0, "high_inactive_mass"),
        (0, "native_nmse_divergence"),
        (1, "native_nmse_divergence"),
    ]


def test_empirical_axis_splits_normalizer_subgroups(
    tmp_path: Path, monkeypatch: Any
) -> None:
    table = {0.1: ("u_t", 0.30), 0.12: ("u_t", 0.32), 0.15: ("u_x", 0.50)}

    def fake(eq: Any, **kw: Any) -> Any:
        c = round(float({ir: cc.value for ir, cc in eq.terms}["u_xx"]), 3)
        term, nmse = table[c]
        return _fake_report(nmse=nmse, normalizer_term=term)

    monkeypatch.setattr("kd.harness.consensus_verify.verify_equation", fake)
    store, ref, dataset = _verified_store(
        tmp_path,
        [
            ("sga", 0, _evo([("u_xx", 0.1)]), 0.0),
            ("disc", 1, _evo([("u_xx", 0.12)]), 0.0),
            ("pysindy", 2, _evo([("u_xx", 0.15)]), 0.0),
        ],
    )
    report = build_consensus(
        store, datasets={ref: dataset}, context_factory=_sentinel_factory()
    )
    emp = report.datasets[0].classes[0].axes.empirical
    assert emp.state == "evaluated"
    assert {sg.normalizer_term for sg in emp.subgroups} == {"u_t", "u_x"}
    ut = next(sg for sg in emp.subgroups if sg.normalizer_term == "u_t")
    assert len(ut.entry_indices) == 2
    assert len(ut.pairs) == 1
    assert math.isclose(ut.pairs[0].nmse_delta, 0.02)

    assert emp.n_not_comparable_pairs == 2
    assert emp.n_unverified_members == 0





def test_as_stratum_rejects_out_of_vocab_provider_kind() -> None:
    from kd.harness._consensus_partition import _as_stratum





    with pytest.raises(ValueError, match="provider_kind"):
        _as_stratum("spectral")





def test_provenance_records_supplied_fingerprint_for_failed_only_ref(
    tmp_path: Path,
) -> None:
    dataset = _tiny_evo_dataset(name="ds_failed_only")
    ref = dataset.name
    entries = [PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={})]
    store = build_sealed_store(
        tmp_path / "store",
        entries=entries,
        records={},
        raised={0: ("RuntimeError", "boom")},
    )
    report = build_consensus(store, datasets={ref: dataset})
    fingerprints = dict(report.provenance.dataset_fingerprints)
    assert fingerprints[ref] == compute_dataset_fingerprint(dataset)


def test_provenance_fingerprint_none_for_failed_only_ref_without_dataset(
    tmp_path: Path,
) -> None:
    ref = "ds_failed_only"
    entries = [PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={})]
    store = build_sealed_store(
        tmp_path / "store",
        entries=entries,
        records={},
        raised={0: ("RuntimeError", "boom")},
    )
    report = build_consensus(store)
    fingerprints = dict(report.provenance.dataset_fingerprints)
    assert fingerprints[ref] is None





def test_store_root_is_absolute_in_provenance(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    store = build_sealed_store(Path("store_rel"), entries=entries, records=records)
    report = build_consensus(store)
    assert Path(report.provenance.store_root).is_absolute()
