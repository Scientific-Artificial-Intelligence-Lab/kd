
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Final

from kd.core.equation.canonical import canonicalize_expression




TERM_FINGERPRINT_DOMAIN: Final[str] = "kd-term-v1"
CATALOG_FINGERPRINT_DOMAIN: Final[str] = "kd-termlib-v1"

_FINGERPRINT_HEX_LEN: Final[int] = 16


def term_fingerprint(term: str) -> str:
    canonical_term = canonicalize_expression(term)
    text = f"{TERM_FINGERPRINT_DOMAIN}\n{canonical_term}"
    payload = text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_FINGERPRINT_HEX_LEN]


@dataclass(frozen=True)
class TermLibrarySpec:


    terms: tuple[str, ...]

    def __post_init__(self) -> None:




        terms_obj: object = self.terms
        if not isinstance(terms_obj, tuple):
            raise ValueError("terms must be a tuple of term strings.")
        for index, item in enumerate(terms_obj):
            if not isinstance(item, str):
                raise ValueError(f"Term at index {index} is not a string: {item!r}.")
        if not self.terms:
            raise ValueError("Term catalog cannot be empty.")

        seen: set[str] = set()
        for index, term in enumerate(self.terms):
            try:
                canonical_term = canonicalize_expression(term)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot canonicalize term at index {index}: {term!r}: {exc}"
                ) from exc
            if canonical_term != term:
                raise ValueError(
                    f"Term at index {index} is not canonical; "
                    "use TermLibrarySpec.from_terms()."
                )
            if canonical_term in seen:
                raise ValueError(
                    f"Term catalog contains duplicate term at index {index}: "
                    f"{canonical_term!r}."
                )
            seen.add(canonical_term)

    @classmethod
    def from_terms(cls, terms: Sequence[str]) -> TermLibrarySpec:
        if isinstance(terms, str):


            raise ValueError("terms must be a sequence of term strings, not a bare str")
        if not terms:
            raise ValueError("Term catalog cannot be empty.")

        canonical_terms: list[str] = []
        seen: set[str] = set()
        for index, term in enumerate(terms):
            try:
                canonical_term = canonicalize_expression(term)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot canonicalize term at index {index}: {term!r}: {exc}"
                ) from exc
            if canonical_term in seen:
                raise ValueError(
                    f"Term catalog contains duplicate term at index {index}: "
                    f"{canonical_term!r}."
                )
            canonical_terms.append(canonical_term)
            seen.add(canonical_term)

        return cls(terms=tuple(canonical_terms))

    @cached_property
    def term_fingerprints(self) -> tuple[str, ...]:
        return tuple(term_fingerprint(term) for term in self.terms)

    @cached_property
    def fingerprint(self) -> str:
        serialized_terms = "\n".join(self.terms)
        text = f"{CATALOG_FINGERPRINT_DOMAIN}\n{serialized_terms}"
        payload = text.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:_FINGERPRINT_HEX_LEN]


__all__ = [
    "CATALOG_FINGERPRINT_DOMAIN",
    "TERM_FINGERPRINT_DOMAIN",
    "TermLibrarySpec",
    "term_fingerprint",
]
