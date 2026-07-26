
from __future__ import annotations

import json
import re

import pytest

from kd.core.equation.library import TermLibrarySpec
from kd.core.platform.requirements import DerivativeReqs
from kd.search.protocol import PlatformComponents
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.plugin import PySRPlugin
from tests.unit.search.pysr.conftest import (
    FakePySRBackend,
    make_backend_factory,
)

_TERMS = ("u", "u_x", "u_xx")


def _make_plugin(
    backend: FakePySRBackend,
    *,
    terms: tuple[str, ...] = _TERMS,
) -> PySRPlugin:
    config = PySRConfig(terms=terms, seed=0)
    return PySRPlugin(config, backend_factory=make_backend_factory(backend))


def test_duplicate_config_terms_raise_at_construction() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        PySRPlugin(PySRConfig(terms=("u", " u ")))


def test_theta_built_from_canonical_spellings(
    real_pysr_components: PlatformComponents,
) -> None:
    plugin = _make_plugin(
        FakePySRBackend(),
        terms=("u", "mul(u_x, u)"),
    )
    plugin.prepare(real_pysr_components)
    plugin.propose(1)
    assert plugin.terms == ["u", "mul(u,u_x)"]


def test_spelling_variants_identical_fingerprint() -> None:
    first = _make_plugin(
        FakePySRBackend(),
        terms=("u", "mul(u, u_x)"),
    )
    second = _make_plugin(
        FakePySRBackend(),
        terms=("u", "mul(u_x,u)"),
    )
    assert first.config["library_fingerprint"] == second.config["library_fingerprint"]


def test_fingerprint_identifies_configured_catalog() -> None:



    plugin = _make_plugin(FakePySRBackend(), terms=("u", "u_x"))
    expected = TermLibrarySpec.from_terms(("u", "u_x")).fingerprint
    assert plugin.config["library_fingerprint"] == expected

    permuted = _make_plugin(FakePySRBackend(), terms=("u_x", "u"))
    permuted_fingerprint = permuted.config["library_fingerprint"]
    assert plugin.config["library_fingerprint"] != permuted_fingerprint


def test_config_dict_fingerprint_json_safe() -> None:
    plugin = _make_plugin(FakePySRBackend(), terms=("u", "u_x"))
    fingerprint = plugin.config["library_fingerprint"]
    assert re.fullmatch(r"[0-9a-f]{16}", fingerprint)
    json.dumps(plugin.config)


def test_derivative_requirements_spelling_invariant() -> None:
    first = _make_plugin(
        FakePySRBackend(),
        terms=("u", "mul(u_xx, u)"),
    )
    second = _make_plugin(
        FakePySRBackend(),
        terms=("u", "mul(u, u_xx)"),
    )
    first_requirements = first.derivative_requirements
    second_requirements = second.derivative_requirements
    assert isinstance(first_requirements, DerivativeReqs)
    assert first_requirements == second_requirements
