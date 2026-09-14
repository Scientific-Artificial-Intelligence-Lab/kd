
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

__all__ = ["annotate", "assert_serves", "features_from_report", "load_prior"]





Prior = dict[str, Any]

FORMAT = "kd-prior-v1"
VERSION = 1




FEATURES: tuple[str, ...] = (
    "topology",
    "lhs_order",
    "n_axes",
    "n_fields",
    "points_bucket",
    "has_nan",
)



CAPABILITIES = frozenset({"capable", "unproven", "topology_mismatch", "form_mismatch"})





INSTRUMENT_KEYS = frozenset({"capability", "cost_class"})


ROW_KEYS = frozenset({"match", "instruments", "basis", "matched_datasets"})

DOCUMENT_KEYS = frozenset(
    {
        "format",
        "version",
        "built_at",
        "builder",
        "features_vocabulary",
        "rows",
        "excluded",
        "min_support",
    }
)


def load_prior(path: Path) -> Prior:
    document: Prior = json.loads(path.read_text(encoding="utf-8"))
    if document["format"] != FORMAT:
        raise ValueError(
            f"{path} declares format {document['format']!r}; this loader reads "
            f"{FORMAT!r} and would misread anything else"
        )
    if document["version"] != VERSION:
        raise ValueError(
            f"{path} declares version {document['version']!r}; this loader reads "
            f"{FORMAT} version {VERSION}"
        )
    vocabulary = document["features_vocabulary"]
    if set(vocabulary) != set(FEATURES):




        raise ValueError(
            f"{path} declares features_vocabulary {sorted(vocabulary)}; "
            f"{FORMAT} is indexed by {sorted(FEATURES)}"
        )
    unknown_document_keys = sorted(set(document) - DOCUMENT_KEYS)
    if unknown_document_keys:
        raise ValueError(
            f"{path} carries document keys {unknown_document_keys} outside "
            f"{sorted(DOCUMENT_KEYS)}: the whitelist that keeps a band out of "
            "an instrument row has to hold at every level of the document"
        )
    excluded = document.get("excluded")
    if excluded is not None and not (
        isinstance(excluded, list) and all(isinstance(name, str) for name in excluded)
    ):
        raise ValueError(
            f"{path} declares excluded {excluded!r}; an evaluation prior names "
            "the datasets it withheld as a list of strings, and a bare string "
            "would iterate as characters and serve nothing"
        )
    min_support = document.get("min_support")





    if min_support is not None and (
        type(min_support) is not int or min_support < 2
    ):
        raise ValueError(
            f"{path} declares min_support {min_support!r}; the support floor a "
            "build applied to its `capable` words is an integer of at least 2, "
            "which is the floor itself"
        )
    for index, row in enumerate(document["rows"]):
        _check_row(row, index, vocabulary)
    return document


def _check_row(row: dict[str, Any], index: int, vocabulary: Sequence[str]) -> None:
    unknown_row_keys = sorted(set(row) - ROW_KEYS)
    if unknown_row_keys:
        raise ValueError(
            f"row {index} carries keys {unknown_row_keys} outside "
            f"{sorted(ROW_KEYS)}: a row-level side channel is the same leak "
            "the instrument-row whitelist refuses"
        )





    matched = row["matched_datasets"]
    if type(matched) is not int:
        raise ValueError(
            f"row {index} records matched_datasets {matched!r}; how many "
            "datasets reached a row is an integer count, and the support gate "
            "on the `capable` word is decided by comparing it"
        )
    unknown_keys = sorted(set(row["match"]) - set(vocabulary))
    if unknown_keys:
        raise ValueError(
            f"row {index} matches on {unknown_keys}, which is outside the "
            f"document's features_vocabulary {sorted(vocabulary)}: a prior is "
            "indexed by data features, and a key outside that vocabulary is "
            "either dead or a dataset identity in disguise"
        )
    claims_a_measurement = False
    for name, declaration in row["instruments"].items():
        extra_keys = sorted(set(declaration) - INSTRUMENT_KEYS)
        if extra_keys:
            raise ValueError(
                f"row {index} declares {extra_keys} for instrument {name!r}; a "
                f"{FORMAT} instrument carries {sorted(INSTRUMENT_KEYS)} and "
                "nothing else"
            )
        capability = declaration["capability"]
        if capability not in CAPABILITIES:
            raise ValueError(
                f"row {index} gives instrument {name!r} the capability word "
                f"{capability!r}, which is outside {sorted(CAPABILITIES)}"
            )
        if capability == "capable":
            claims_a_measurement = True







    if claims_a_measurement and row["matched_datasets"] < 2:
        raise ValueError(
            f"row {index} records matched_datasets={row['matched_datasets']}: a "
            "row that only one dataset reaches is a fingerprint of that dataset, "
            "and its basis sentence does not change that"
        )


def assert_serves(prior: Prior, dataset_id: str) -> None:
    excluded = prior.get("excluded")
    if excluded and dataset_id not in excluded:
        raise ValueError(
            f"this prior withheld {sorted(excluded)} and serves those datasets "
            f"alone; {dataset_id!r} is not one of them, and reading it there "
            "puts that dataset's own measurements back into the run measuring it"
        )


def features_from_report(report: dict[str, Any]) -> dict[str, Any]:
    fields = report["fields"]
    points = math.prod(fields[0]["shape"])
    return {
        "topology": report["topology"],
        "lhs_order": report["lhs_order"],
        "n_axes": len(report["axes"]),
        "n_fields": len(fields),


        "points_bucket": len(str(points)) - 1,
        "has_nan": any(field["nan_count"] > 0 for field in fields),
    }


def annotate(
    prior: Prior, features: dict[str, Any], instruments: Iterable[str]
) -> dict[str, dict[str, Any]]:
    annotation: dict[str, dict[str, Any]] = {
        name: {"capability": "unproven", "matched": []} for name in instruments
    }
    for row in prior["rows"]:
        match = row["match"]
        if any(features.get(key) != value for key, value in match.items()):
            continue
        for name, declaration in row["instruments"].items():
            if name in annotation:
                annotation[name] = {
                    "capability": declaration["capability"],
                    "matched": list(match),
                }
    return annotation
