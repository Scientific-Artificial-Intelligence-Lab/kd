
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import kd
from kd.api import _PLUGIN_CLASS_BY_ALGORITHM
from kd.core.equation import Form
from kd.core.equation.signature import law_term_entry, law_term_key
from kd.core.equation.sketch import (
    AnchoredTerm,
    PinnedTerm,
    Sketch,
    TermConstraint,
    TermHole,
    sketch_to_dict,
)
from kd.core.equation.types import Evolution, LhsSpec, Scalar
from kd.core.platform.sketch_compile import (
    SKETCH_CLAUSES,
    SKETCH_CONFIG_KEY,
    SketchClauseLevels,
)
from kd.data.schema import PDEDataset
from kd.search.descriptor import assert_sketch_supported
from kd.search.protocol import DiscoveryTask
from kd.search.pysindy import PySINDyPlugin
from kd.search.result import ExperimentResult
from kd.search.sga import SGAPlugin
from kd.search.sketch_outcome import (
    SKETCH_OUTCOME_ARTIFACT_TAG,
    SketchOutcome,
    write_sketch_artifact,
)
from tests.unit.search._sketch_fakes import (
    HOLE_TERM,
    PINNED_ADVECTION,
    PINNED_ADVECTION_VALUE,
    PYSINDY_DECLARED_LEVELS,
    PYSINDY_NU,
    SketchFakePlugin,
    UndeclaredSketchPlugin,
    UnsupportedSketchPlugin,
    build_components,
    burgers_sketch,
    fake_descriptor,
    pysindy_burgers_dataset,
    run_plugin,
    tiny_burgers_dataset,
)




_NON_DECLARING = sorted(set(_PLUGIN_CLASS_BY_ALGORITHM) - {"pysindy", "sga"})




SGA_DECLARED_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)



_SGA_UNREACHABLE_ANCHOR = "diff2_x(diff2_x(diff2_x(u)))"


_SGA_UNSPELLABLE_PIN = "mul(u,neg(u_x))"



_PYSINDY_TERMS = ("u", "u_x", HOLE_TERM, PINNED_ADVECTION)




_UXX_ABS_TOL = 2e-3


@pytest.fixture(scope="module")
def burgers() -> PDEDataset:
    return tiny_burgers_dataset()


@pytest.fixture(scope="module")
def pysindy_burgers() -> PDEDataset:
    return pysindy_burgers_dataset()


def pysindy_sketch_model(**overrides: Any) -> kd.Model:
    return kd.Model(
        algorithm="pysindy",
        config=kd.PySINDyConfig(terms=_PYSINDY_TERMS),
        verbose=False,
        **overrides,
    )


def outcome_of(model: kd.Model) -> tuple[ExperimentResult, SketchOutcome]:
    result = model.result_
    assert result is not None
    outcome = result.sketch_outcome
    assert outcome is not None
    return result, outcome


def model_for(algorithm: str) -> kd.Model:
    if algorithm == "eqgpt":
        return kd.Model(
            algorithm=algorithm,
            config=kd.EqGPTConfig(sparsity_alpha=0.02),
            verbose=False,
        )
    return kd.Model(algorithm=algorithm, verbose=False)


def run_sketch(
    dataset: PDEDataset, sketch: Sketch
) -> tuple[ExperimentResult, SketchOutcome]:
    task = DiscoveryTask.from_sketch(sketch)
    result = run_plugin(build_components(dataset, task=task), SketchFakePlugin())
    outcome = result.sketch_outcome
    assert outcome is not None
    return result, outcome


def coefficient_of(equation: Evolution, law_key: str) -> float:
    values = [
        coefficient.value
        for term_ir, coefficient in equation.terms
        if term_ir == law_key and isinstance(coefficient, Scalar)
    ]
    assert len(values) == 1
    return values[0]







@pytest.mark.parametrize("algorithm", _NON_DECLARING)
def test_every_non_declaring_mode_declares_no_sketch_support(algorithm: str) -> None:
    descriptor = _PLUGIN_CLASS_BY_ALGORITHM[algorithm].descriptor
    for mode in descriptor.modes:
        assert [mode.sketch.level(clause) for clause in SKETCH_CLAUSES] == [
            "unsupported"
        ] * len(SKETCH_CLAUSES)


def test_pysindy_declares_the_frozen_capability_vector() -> None:
    modes = PySINDyPlugin.descriptor.modes

    assert [mode.name for mode in modes] == ["default"]
    assert modes[0].sketch == PYSINDY_DECLARED_LEVELS


def test_sga_declares_the_frozen_capability_vector() -> None:
    modes = SGAPlugin.descriptor.modes

    assert [mode.name for mode in modes] == ["default", "autograd"]
    assert [mode.sketch for mode in modes] == [SGA_DECLARED_LEVELS] * 2


@pytest.mark.parametrize("algorithm", _NON_DECLARING)
def test_sketch_is_refused_by_every_non_declaring_algorithm(
    algorithm: str, burgers: PDEDataset
) -> None:
    with pytest.raises(ValueError, match="fixed_terms") as excinfo:
        model_for(algorithm).fit(burgers, sketch=burgers_sketch())

    assert algorithm in str(excinfo.value)


def test_the_declaring_algorithms_pass_the_capability_gate() -> None:
    assert_sketch_supported(
        PySINDyPlugin.descriptor, burgers_sketch(), algorithm="pysindy"
    )
    assert_sketch_supported(SGAPlugin.descriptor, burgers_sketch(), algorithm="sga")


def test_fit_without_a_sketch_keeps_its_current_behaviour(
    burgers: PDEDataset,
) -> None:
    model = kd.Model(
        algorithm="pysindy",
        config=kd.PySINDyConfig(
            terms=("u", "u_x", HOLE_TERM, PINNED_ADVECTION), seed=7
        ),
        verbose=False,
    ).fit(burgers)

    result = model.result_
    assert result is not None
    assert result.sketch_outcome is None
    assert result.run_record is not None
    assert SKETCH_CONFIG_KEY not in result.run_record.run_spec.config


def test_non_sketch_argument_is_refused_by_type(burgers: PDEDataset) -> None:
    with pytest.raises(TypeError, match="sketch"):
        model_for("pysindy").fit(burgers, sketch={"pinned": []})


def test_sketch_requires_an_evolution_dataset(
    burgers: PDEDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        PySINDyPlugin, "descriptor", fake_descriptor(algorithm="pysindy")
    )
    homogeneous = dataclasses.replace(burgers, lhs_order=0, lhs_axis="", lhs_field="")

    with pytest.raises(ValueError, match="(?i)evolution"):
        model_for("pysindy").fit(homogeneous, sketch=burgers_sketch())


def test_sketch_lhs_must_match_the_resolved_dataset_lhs(
    burgers: PDEDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        PySINDyPlugin, "descriptor", fake_descriptor(algorithm="pysindy")
    )
    sketch = burgers_sketch(lhs_spec=LhsSpec("u", "t", 2))

    with pytest.raises(ValueError, match="(?i)lhs"):
        model_for("pysindy").fit(burgers, sketch=sketch)







def test_certified_solution_carries_the_exact_pinned_coefficient(
    burgers: PDEDataset,
) -> None:
    result, outcome = run_sketch(burgers, burgers_sketch())

    assert outcome.solution is not None
    assert isinstance(outcome.solution, Evolution)
    pinned = coefficient_of(outcome.solution, law_term_key(PINNED_ADVECTION))
    assert pinned == PINNED_ADVECTION_VALUE
    assert result.equation is outcome.solution
    assert outcome.failure is None
    assert outcome.full_verify is not None


def test_soundness_invariant_holds_on_the_certified_path(
    burgers: PDEDataset,
) -> None:
    sketch = burgers_sketch()

    _result, outcome = run_sketch(burgers, sketch)

    assert outcome.solution is not None
    assert sketch.matches(outcome.solution).overall is True
    assert outcome.verdict is not None
    assert outcome.verdict.overall is True


def test_closed_sketch_runs_end_to_end_and_lifts_from_itself(
    burgers: PDEDataset,
) -> None:
    sketch = burgers_sketch(holes=())

    _result, outcome = run_sketch(burgers, sketch)

    assert outcome.solution is not None
    key, value = law_term_entry(PINNED_ADVECTION, PINNED_ADVECTION_VALUE)
    assert [
        (term_ir, coefficient.value)
        for term_ir, coefficient in outcome.solution.terms
        if isinstance(coefficient, Scalar)
    ] == [(key, value)]
    assert outcome.verdict is not None
    assert outcome.verdict.overall is True


def test_violating_candidate_never_reaches_the_solution_face(
    burgers: PDEDataset,
) -> None:
    sketch = burgers_sketch(anchored=(AnchoredTerm("u"),))

    result, outcome = run_sketch(burgers, sketch)

    assert outcome.solution is None
    assert result.equation is None
    assert outcome.best_candidate is not None
    assert outcome.verdict is not None
    assert outcome.verdict.overall is False
    assert [entry.matched for entry in outcome.verdict.anchored] == [False]


def test_candidate_quality_failure_is_recorded_not_raised(
    burgers: PDEDataset, monkeypatch: pytest.MonkeyPatch
) -> None:

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("residual program could not be executed")

    monkeypatch.setattr("kd.search.runner.verify_equation", explode)

    _result, outcome = run_sketch(burgers, burgers_sketch())

    assert outcome.failure is not None
    assert outcome.failure.startswith("verify")
    assert outcome.solution is None
    assert outcome.best_candidate is not None


def test_outcome_is_written_as_a_versioned_sidecar(
    burgers: PDEDataset, tmp_path: Path
) -> None:
    sketch = burgers_sketch()
    result, outcome = run_sketch(burgers, sketch)
    assert result.run_record is not None

    path = write_sketch_artifact(
        outcome,
        sketch_payload=sketch_to_dict(sketch),
        evidence_hash=result.run_record.evidence_hash,
        path=tmp_path / "sketch_outcome.json",
    )

    stored: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert stored["artifact"] == SKETCH_OUTCOME_ARTIFACT_TAG
    assert stored["evidence_hash"] == result.run_record.evidence_hash
    assert stored["verdict"]["overall"] is True


def test_sealed_result_schema_does_not_carry_the_outcome(
    burgers: PDEDataset,
) -> None:
    result, _outcome = run_sketch(burgers, burgers_sketch())

    assert "sketch_outcome" not in result.to_dict()







def test_runner_backstop_refuses_an_algorithm_without_a_declaration(
    burgers: PDEDataset,
) -> None:
    task = DiscoveryTask.from_sketch(burgers_sketch())
    components = build_components(burgers, task=task)

    with pytest.raises(TypeError, match="descriptor"):
        run_plugin(components, UndeclaredSketchPlugin())


def test_runner_backstop_mirrors_the_facade_capability_gate(
    burgers: PDEDataset,
) -> None:
    task = DiscoveryTask.from_sketch(burgers_sketch())
    components = build_components(burgers, task=task)

    with pytest.raises(ValueError, match="fixed_terms"):
        run_plugin(components, UnsupportedSketchPlugin())


def test_homogeneous_final_evaluation_on_a_task_run_is_refused(
    burgers: PDEDataset,
) -> None:
    task = DiscoveryTask.from_sketch(burgers_sketch())
    components = build_components(burgers, task=task)

    with pytest.raises(TypeError, match="HOMOGENEOUS|homogeneous"):
        run_plugin(components, SketchFakePlugin(form=Form.HOMOGENEOUS))







@pytest.fixture(scope="module")
def certified_run(
    pysindy_burgers: PDEDataset,
) -> tuple[ExperimentResult, SketchOutcome]:
    return outcome_of(
        pysindy_sketch_model().fit(pysindy_burgers, sketch=burgers_sketch())
    )


def test_pysindy_certifies_the_pinned_law_end_to_end(
    certified_run: tuple[ExperimentResult, SketchOutcome],
) -> None:
    result, outcome = certified_run

    assert outcome.solution is not None
    assert isinstance(outcome.solution, Evolution)
    assert (
        coefficient_of(outcome.solution, law_term_key(PINNED_ADVECTION))
        == PINNED_ADVECTION_VALUE
    )
    assert coefficient_of(outcome.solution, HOLE_TERM) == pytest.approx(
        PYSINDY_NU, abs=_UXX_ABS_TOL
    )
    assert outcome.verdict is not None
    assert outcome.verdict.overall is True
    assert outcome.failure is None
    assert result.equation is outcome.solution


def test_the_certified_run_record_speaks_the_full_law_domain(
    certified_run: tuple[ExperimentResult, SketchOutcome],
) -> None:
    result, _outcome = certified_run

    assert result.run_record is not None
    assert list(result.run_record.evidence.support or []) == [
        law_term_key(PINNED_ADVECTION),
        HOLE_TERM,
    ]


    coefficients = result.run_record.evidence.coefficients
    assert coefficients is not None
    assert coefficients[0] == PINNED_ADVECTION_VALUE
    assert coefficients[1] == pytest.approx(PYSINDY_NU, abs=_UXX_ABS_TOL)


def test_the_outcome_carries_the_backends_own_compile_report(
    certified_run: tuple[ExperimentResult, SketchOutcome],
) -> None:
    _result, outcome = certified_run

    assert outcome.compile_report.levels == PYSINDY_DECLARED_LEVELS


def test_pysindy_withholds_a_solution_that_misses_an_anchor(
    pysindy_burgers: PDEDataset,
) -> None:
    sketch = burgers_sketch(anchored=(AnchoredTerm("u"),))

    result, outcome = outcome_of(
        pysindy_sketch_model().fit(pysindy_burgers, sketch=sketch)
    )

    assert outcome.solution is None
    assert result.equation is None
    assert outcome.best_candidate is not None
    assert outcome.verdict is not None
    assert outcome.verdict.overall is False
    assert [entry.matched for entry in outcome.verdict.anchored] == [False]


def test_a_closed_sketch_runs_pysindy_against_the_raw_target(
    pysindy_burgers: PDEDataset,
) -> None:
    result, outcome = outcome_of(
        pysindy_sketch_model().fit(pysindy_burgers, sketch=burgers_sketch(holes=()))
    )

    key, value = law_term_entry(PINNED_ADVECTION, PINNED_ADVECTION_VALUE)
    assert outcome.solution is not None
    assert [
        (term_ir, coefficient.value)
        for term_ir, coefficient in outcome.solution.terms
        if isinstance(coefficient, Scalar)
    ] == [(key, value)]
    assert outcome.verdict is not None
    assert outcome.verdict.overall is True
    assert result.run_record is not None
    assert list(result.run_record.evidence.support or []) == [key]
    assert result.final_eval is not None
    assert result.run_record.evidence.nmse == pytest.approx(result.final_eval.nmse)

    assert key in result.best_expression


@pytest.fixture(scope="module")
def sketch_checkpoint(
    pysindy_burgers: PDEDataset, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    write_dir = tmp_path_factory.mktemp("sketch_write")
    pysindy_sketch_model(checkpoint_dir=write_dir).fit(
        pysindy_burgers, sketch=burgers_sketch()
    )
    final = write_dir / "checkpoint_final.pt"
    assert final.exists()
    return final


def test_resuming_the_same_sketch_recovers_the_certified_run(
    pysindy_burgers: PDEDataset, sketch_checkpoint: Path
) -> None:
    result, outcome = outcome_of(
        pysindy_sketch_model().fit(
            pysindy_burgers, resume_from=sketch_checkpoint, sketch=burgers_sketch()
        )
    )

    assert outcome.solution is not None
    assert (
        coefficient_of(outcome.solution, law_term_key(PINNED_ADVECTION))
        == PINNED_ADVECTION_VALUE
    )
    assert result.run_record is not None


def test_resuming_a_changed_sketch_is_an_identity_rejection(
    pysindy_burgers: PDEDataset, sketch_checkpoint: Path
) -> None:
    changed = burgers_sketch(pinned=(PinnedTerm(PINNED_ADVECTION, -0.5),))

    with pytest.raises(ValueError, match=SKETCH_CONFIG_KEY) as excinfo:
        pysindy_sketch_model().fit(
            pysindy_burgers, resume_from=sketch_checkpoint, sketch=changed
        )

    assert "identity_breaking" in str(excinfo.value)







def sga_sketch(
    *, anchored: tuple[AnchoredTerm, ...] = (), max_count: int = 2
) -> Sketch:
    return burgers_sketch(
        anchored=anchored,
        holes=(
            TermHole(
                id="diffusion",
                min_count=1,
                max_count=max_count,
                constraint=TermConstraint(max_deriv_order=2),
            ),
        ),
    )


def sga_model(*, width: int = 3) -> kd.Model:
    return kd.Model(
        algorithm="sga",
        config=kd.SGAConfig(num=6, depth=2, width=width, seed=0),
        generations=2,
        verbose=False,
    )


@pytest.fixture(scope="module")
def sga_certified_run(
    burgers: PDEDataset,
) -> tuple[ExperimentResult, SketchOutcome]:
    return outcome_of(sga_model().fit(burgers, sketch=sga_sketch()))


def test_sga_certifies_the_pinned_law_end_to_end(
    sga_certified_run: tuple[ExperimentResult, SketchOutcome],
) -> None:
    result, outcome = sga_certified_run

    assert outcome.solution is not None
    assert isinstance(outcome.solution, Evolution)
    assert (
        coefficient_of(outcome.solution, law_term_key(PINNED_ADVECTION))
        == PINNED_ADVECTION_VALUE
    )
    assert outcome.verdict is not None
    assert outcome.verdict.overall is True
    assert outcome.failure is None
    assert result.equation is outcome.solution


def test_the_sga_outcome_carries_the_backends_own_compile_report(
    sga_certified_run: tuple[ExperimentResult, SketchOutcome],
) -> None:
    _result, outcome = sga_certified_run

    assert outcome.compile_report.levels == SGA_DECLARED_LEVELS


def test_an_sga_sketch_run_leaves_the_platform_target_raw(
    burgers: PDEDataset,
) -> None:
    task = DiscoveryTask.from_sketch(sga_sketch())
    model = kd.Model(algorithm="sga", verbose=False)

    raw = model._build_components(burgers)
    with_task = model._build_components(burgers, task=task)

    assert raw.evaluator is not None
    assert with_task.evaluator is not None
    torch.testing.assert_close(
        with_task.evaluator.lhs_target,
        raw.evaluator.lhs_target,
        rtol=0.0,
        atol=0.0,
    )


def test_a_closed_sga_sketch_lifts_from_a_pin_it_cannot_spell(
    burgers: PDEDataset,
) -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm(_SGA_UNSPELLABLE_PIN, -1.0),), holes=())

    _result, outcome = outcome_of(sga_model().fit(burgers, sketch=sketch))

    key, value = law_term_entry(_SGA_UNSPELLABLE_PIN, -1.0)
    assert outcome.solution is not None
    assert [
        (term_ir, coefficient.value)
        for term_ir, coefficient in outcome.solution.terms
        if isinstance(coefficient, Scalar)
    ] == [(key, value)]
    assert outcome.verdict is not None
    assert outcome.verdict.overall is True


def test_the_canonical_max_count_one_sketch_runs_to_the_exit(
    burgers: PDEDataset,
) -> None:
    _result, outcome = outcome_of(
        sga_model().fit(burgers, sketch=sga_sketch(max_count=1))
    )

    assert outcome.verdict is not None
    assert outcome.best_candidate is not None


def test_an_autograd_mode_sketch_runs_the_same_chain(
    burgers: PDEDataset,
) -> None:
    model = kd.Model(
        algorithm="sga",
        config=kd.SGAConfig(
            num=4,
            depth=2,
            width=2,
            seed=0,
            use_autograd=True,
            autograd_train_epochs=30,
        ),
        generations=1,
        verbose=False,
    )

    _result, outcome = outcome_of(model.fit(burgers, sketch=sga_sketch()))

    assert outcome.verdict is not None
    assert outcome.compile_report.levels == SGA_DECLARED_LEVELS


def test_a_statically_infeasible_sga_sketch_is_refused_before_the_search(
    burgers: PDEDataset,
) -> None:
    sketch = burgers_sketch(
        holes=(
            TermHole(
                id="quartet",
                min_count=4,
                max_count=4,
                constraint=TermConstraint(max_deriv_order=2),
            ),
        )
    )

    with pytest.raises(ValueError, match="width"):
        sga_model(width=2).fit(burgers, sketch=sketch)


def test_sga_withholds_a_solution_its_search_cannot_reach(
    burgers: PDEDataset,
) -> None:
    sketch = sga_sketch(anchored=(AnchoredTerm(_SGA_UNREACHABLE_ANCHOR),))

    result, outcome = outcome_of(sga_model().fit(burgers, sketch=sketch))

    assert outcome.solution is None
    assert result.equation is None
    assert outcome.best_candidate is not None
    assert outcome.verdict is not None
    assert outcome.verdict.overall is False
    assert [entry.matched for entry in outcome.verdict.anchored] == [False]
