
from __future__ import annotations

from pathlib import Path

_DOC_DIR = Path(__file__).resolve().parent / "_assets" / "instruments"
_SECTION_PREFIX = "## "

SECTION_HEADINGS: dict[str, str] = {
    "search_space": "Search space",
    "method": "Method",
    "results": "Result interpretation",
    "references": "References",
}
REQUIRED_SECTIONS: tuple[str, ...] = tuple(SECTION_HEADINGS)


def instrument_doc_path(algorithm: str) -> Path:




    available = available_instrument_docs()
    if algorithm not in available:
        raise ValueError(f"no instrument doc for {algorithm!r}; have {list(available)}")
    return _DOC_DIR / f"{algorithm}.md"


def available_instrument_docs() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in _DOC_DIR.glob("*.md")))


def instrument_doc(algorithm: str, section: str | None = None) -> str:
    if section is not None and section not in REQUIRED_SECTIONS:
        raise ValueError(f"unknown section {section!r}; have {list(REQUIRED_SECTIONS)}")
    body = instrument_doc_path(algorithm).read_text(encoding="utf-8")
    if section is None:
        return body
    sections = _split_sections(body)
    if section not in sections:
        raise ValueError(
            f"{algorithm!r} has no section {section!r}; have {list(sections)}"
        )
    return sections[section]


def instrument_doc_sections(algorithm: str) -> dict[str, str]:
    return _split_sections(instrument_doc(algorithm))


def _split_sections(body: str) -> dict[str, str]:
    keys = {heading: key for key, heading in SECTION_HEADINGS.items()}
    sections: dict[str, str] = {}
    heading: str | None = None
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith(_SECTION_PREFIX):
            if heading is not None:
                sections[keys.get(heading, heading)] = "\n".join(lines).strip()
            heading = line[len(_SECTION_PREFIX):].strip()
            lines = []
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        sections[keys.get(heading, heading)] = "\n".join(lines).strip()
    return sections
