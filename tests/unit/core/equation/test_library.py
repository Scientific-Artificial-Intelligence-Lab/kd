
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from kd.core.equation.canonical import canonicalize_expression
from kd.core.equation.library import TermLibrarySpec, term_fingerprint


class TestFromTerms:

    def test_canonicalizes_spellings(self) -> None:

        spec = TermLibrarySpec.from_terms(["mul(u_x, u)", " u_xx "])
        assert spec.terms == ("mul(u,u_x)", "u_xx")

    def test_preserves_order(self) -> None:
        spec = TermLibrarySpec.from_terms(["u_xx", "u"])
        assert spec.terms == ("u_xx", "u")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            TermLibrarySpec.from_terms([])

    def test_bare_string_rejected(self) -> None:


        with pytest.raises(ValueError, match="bare str"):
            TermLibrarySpec.from_terms("u_x")

    def test_duplicate_after_canonicalization_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            TermLibrarySpec.from_terms(["mul(u, u_x)", "mul(u_x, u)"])

    def test_malformed_term_raises_with_index(self) -> None:
        with pytest.raises(ValueError, match="index 1"):
            TermLibrarySpec.from_terms(["u", "add(u,"])

    def test_constant_term_rejected(self) -> None:
        with pytest.raises(ValueError, match="constant"):
            TermLibrarySpec.from_terms(["1"])


class TestDirectConstruction:

    def test_non_canonical_direct_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="canonical"):
            TermLibrarySpec(terms=("mul(u_x, u)",))

    def test_direct_empty_raises(self) -> None:


        with pytest.raises(ValueError, match="empty"):
            TermLibrarySpec(terms=())

    def test_direct_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            TermLibrarySpec(terms=("u_x", "u_x"))

    def test_direct_bare_string_rejected(self) -> None:


        with pytest.raises(ValueError, match="tuple"):
            TermLibrarySpec(terms="uv")

    def test_direct_list_rejected(self) -> None:


        with pytest.raises(ValueError, match="tuple"):
            TermLibrarySpec(terms=["u", "v"])

    def test_direct_non_string_element_rejected(self) -> None:
        with pytest.raises(ValueError, match="index 1"):
            TermLibrarySpec(terms=("u", 3))

    def test_canonical_direct_equals_from_terms(self) -> None:
        direct = TermLibrarySpec(terms=("u_x",))
        assert direct == TermLibrarySpec.from_terms(["u_x"])

        canonical = "mul(u,u_x)"
        assert canonicalize_expression(canonical) == canonical


class TestFingerprint:

    def test_deterministic_within_process(self) -> None:
        first = TermLibrarySpec.from_terms(["u", "mul(u_x, u)"])
        second = TermLibrarySpec.from_terms(["u", "mul(u_x, u)"])
        assert first.fingerprint == second.fingerprint
        assert first.term_fingerprints == second.term_fingerprints

    def test_hex16_format(self) -> None:
        spec = TermLibrarySpec.from_terms(["u", "u_x"])
        assert re.fullmatch(r"[0-9a-f]{16}", spec.fingerprint)
        assert all(
            re.fullmatch(r"[0-9a-f]{16}", fingerprint)
            for fingerprint in spec.term_fingerprints
        )

    def test_spelling_variants_share_fingerprint(self) -> None:
        first = TermLibrarySpec.from_terms(["mul(u, u_x)"])
        second = TermLibrarySpec.from_terms(["mul(u_x,u)"])
        assert first.fingerprint == second.fingerprint

    def test_order_sensitive(self) -> None:

        first = TermLibrarySpec.from_terms(["u", "u_x"])
        second = TermLibrarySpec.from_terms(["u_x", "u"])
        assert first.fingerprint != second.fingerprint

    def test_set_view_order_insensitive(self) -> None:
        first = TermLibrarySpec.from_terms(["u", "u_x"])
        second = TermLibrarySpec.from_terms(["u_x", "u"])
        assert frozenset(first.term_fingerprints) == frozenset(
            second.term_fingerprints
        )

    def test_term_fingerprints_parallel_and_unique(self) -> None:
        spec = TermLibrarySpec.from_terms(["u", "u_x", "mul(u,u_x)"])
        assert len(spec.term_fingerprints) == len(spec.terms)
        assert len(frozenset(spec.term_fingerprints)) == len(spec.term_fingerprints)

    def test_term_fingerprint_helper_spelling_invariant(self) -> None:
        assert term_fingerprint("mul(u_x, u)") == term_fingerprint("mul(u, u_x)")

    def test_golden_fingerprint_bytes(self) -> None:


        spec = TermLibrarySpec.from_terms(["u", "u_x", "mul(u, u_x)"])
        assert spec.fingerprint == "98d5dfd27dcdb553"

    def test_golden_term_fingerprint_bytes(self) -> None:



        assert term_fingerprint("u") == "d73e6e4e9c666b54"


class TestCrossProcessStability:

    def test_fingerprint_stable_across_process_and_hash_seed(self) -> None:
        in_process = TermLibrarySpec.from_terms(
            ["u", "u_x", "mul(u, u_x)"]
        ).fingerprint
        snippet = (
            "from kd.core.equation.library import TermLibrarySpec; "
            "print(TermLibrarySpec.from_terms("
            "['u', 'u_x', 'mul(u_x,u)']).fingerprint)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            env={**os.environ, "PYTHONHASHSEED": "12345"},
            capture_output=True,
            text=True,
            check=True,
        )
        assert completed.stdout.strip() == in_process
