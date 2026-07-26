
from __future__ import annotations

import json
import re

import pytest

from kd.core.equation.library import TermLibrarySpec
from kd.search.protocol import PlatformComponents
from kd.search.pysindy.config import PySINDyConfig
from kd.search.pysindy.plugin import PySINDyPlugin
from tests.unit.search.pysindy.conftest import FakeSINDyBackend, make_backend_factory


def _plugin(terms: tuple[str, ...]) -> PySINDyPlugin:
    backend = FakeSINDyBackend()
    return PySINDyPlugin(
        PySINDyConfig(terms=terms),
        backend_factory=make_backend_factory(backend),
    )


def test_duplicate_canonical_terms_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _plugin(("u", " u "))


def test_noncanonical_spellings_are_canonicalized_before_theta_build(
    real_pysindy_components: PlatformComponents,
) -> None:
    plugin = _plugin(("u", "mul(u_x, u)"))
    plugin.prepare(real_pysindy_components)
    plugin.propose(1)
    assert plugin.terms == ["u", "mul(u,u_x)"]


def test_equivalent_spellings_share_fingerprint() -> None:
    first = _plugin(("u", "mul(u, u_x)"))
    second = _plugin(("u", "mul(u_x,u)"))
    assert first.config["library_fingerprint"] == second.config["library_fingerprint"]


def test_fingerprint_is_real_and_order_sensitive() -> None:
    plugin = _plugin(("u", "u_x"))
    expected = TermLibrarySpec.from_terms(("u", "u_x")).fingerprint
    assert plugin.config["library_fingerprint"] == expected
    assert plugin.config["library_fingerprint"] != _plugin(("u_x", "u")).config[
        "library_fingerprint"
    ]


def test_fingerprint_is_json_safe_stable_shape() -> None:
    config = _plugin(("u", "u_x")).config
    assert re.fullmatch(r"[0-9a-f]{16}", config["library_fingerprint"])
    json.dumps(config)
