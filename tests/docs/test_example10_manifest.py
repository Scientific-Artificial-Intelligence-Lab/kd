
from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _REPO_ROOT / "examples" / "10_checkpoint_resume.py"




_LITERAL_RESUME_RE = re.compile(r"""resume_from=[^\n]*['"][^'"]*\.pt['"]""")


def _source() -> str:
    return _EXAMPLE.read_text(encoding="utf-8")


def _docstring() -> str:
    doc = ast.get_docstring(ast.parse(_source()))
    assert doc is not None, "example 10 lost its module docstring"
    return doc


def test_example10_does_not_glob_checkpoint_filenames() -> None:
    assert ".glob(" not in _source(), (
        "example 10 must not glob checkpoint filenames: a crashed run leaves a "
        "final-looking checkpoint_final.pt that only the manifest's "
        "final_status distinguishes. Enumerate via kd.load_checkpoint_manifest."
    )


def test_example10_selects_the_resume_point_from_the_manifest() -> None:
    source = _source()
    assert "kd.load_checkpoint_manifest(" in source, (
        "example 10 must demonstrate kd.load_checkpoint_manifest as the way to "
        "enumerate resumable checkpoints."
    )
    assert "resume_from=" in source, "example 10 lost its resume call"
    assert not _LITERAL_RESUME_RE.search(source), (
        "example 10 must pass resume_from a filename taken from a manifest "
        "entry, not a hardcoded checkpoint filename literal."
    )


def test_example10_rejects_a_crashed_final_checkpoint() -> None:
    source = _source()
    for name in ("kd.KIND_FINAL", "kd.FINAL_STATUS_COMPLETED"):
        assert name in source, (
            f"example 10 must select on {name}: a final entry with "
            "final_status='crashed' is not a resume point."
        )


def test_example10_coverage_claim_names_every_registered_algorithm() -> None:
    import kd

    doc = _docstring()
    missing = [
        schema["algorithm"]
        for schema in kd.instrument_schemas()
        if schema["algorithm"] not in doc
    ]
    assert not missing, (
        f"example 10's checkpoint coverage claim omits {missing}; it is the "
        "only place the per-engine resume semantics are stated, so a registered "
        "algorithm left out reads as unsupported."
    )
