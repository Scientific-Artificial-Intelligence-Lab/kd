
from __future__ import annotations

import pytest

from kd.core.equation.library import TermLibrarySpec
from kd.core.equation.sketch import (
    AnchoredTerm,
    PinnedTerm,
    Sketch,
    TermConstraint,
    TermHole,
)
from kd.core.platform.sketch_compile import CompileReport
from kd.search.protocol import DiscoveryTask
from kd.search.pysindy.config import PySINDyConfig
from kd.search.pysindy.plugin import PySINDyPlugin
from tests.unit.search._sketch_fakes import (
    HOLE_TERM,
    PINNED_ADVECTION,
    PYSINDY_DECLARED_LEVELS,
    build_components,
    burgers_sketch,
    pysindy_burgers_dataset,
)

_ORDER_3_TERM = "u_xxx"


_CATALOG = ("u", "u_x", HOLE_TERM, _ORDER_3_TERM, PINNED_ADVECTION)
_EFFECTIVE = ("u", "u_x", HOLE_TERM)


def _plugin(terms: tuple[str, ...], sketch: Sketch | None) -> PySINDyPlugin:
    config = PySINDyConfig(terms=terms)
    if sketch is None:
        return PySINDyPlugin(config)
    return PySINDyPlugin(config, task=DiscoveryTask.from_sketch(sketch))


def test_the_published_fingerprint_describes_the_effective_library() -> None:
    plugin = _plugin(_CATALOG, burgers_sketch())

    assert (
        plugin.config["library_fingerprint"]
        == TermLibrarySpec.from_terms(_EFFECTIVE).fingerprint
    )


def test_derivative_requirements_also_serve_the_pinned_spellings() -> None:
    sketch = burgers_sketch(
        pinned=(PinnedTerm(_ORDER_3_TERM, -1.0),),
        holes=(
            TermHole(
                id="low",
                min_count=1,
                max_count=1,
                constraint=TermConstraint(max_deriv_order=1),
            ),
        ),
    )

    plugin = _plugin(("u", "u_x", _ORDER_3_TERM), sketch)


    assert (
        plugin.config["library_fingerprint"]
        == TermLibrarySpec.from_terms(("u", "u_x")).fingerprint
    )
    assert plugin.derivative_requirements.max_atomic_order == 3


def test_the_compile_report_is_published_on_the_runner_seam() -> None:
    report = _plugin(_CATALOG, burgers_sketch()).sketch_compile_report

    assert isinstance(report, CompileReport)
    assert report.levels == PYSINDY_DECLARED_LEVELS


def test_without_a_task_the_plugin_keeps_todays_library_and_no_report() -> None:
    plugin = _plugin(_CATALOG, None)

    assert plugin.sketch_compile_report is None
    assert (
        plugin.config["library_fingerprint"]
        == TermLibrarySpec.from_terms(_CATALOG).fingerprint
    )


def test_an_unrepresentable_anchor_fails_at_plugin_construction() -> None:
    sketch = burgers_sketch(pinned=(), anchored=(AnchoredTerm(_ORDER_3_TERM),))

    with pytest.raises(ValueError, match="anchors"):
        _plugin(("u", "u_x", HOLE_TERM), sketch)


def test_the_fit_builds_theta_from_the_effective_library() -> None:
    task = DiscoveryTask.from_sketch(burgers_sketch())
    components = build_components(pysindy_burgers_dataset(), task=task)
    plugin = PySINDyPlugin(PySINDyConfig(terms=_CATALOG), task=task)

    plugin.prepare(components)
    plugin.propose(1)

    assert plugin.terms == list(_EFFECTIVE)
