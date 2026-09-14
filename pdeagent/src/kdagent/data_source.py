
from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import kd

INPUT_FORMAT = "kd-agent-input-v1"
FILE_PREFIX = "file-v1-"
LOAD_OPTIONS = ("coords", "fields", "field_axes", "select", "layout", "lhs", "periodic")


def is_file_ref(dataset_id: str) -> bool:
    return dataset_id.startswith(FILE_PREFIX)


def _options(options: dict[str, Any] | None) -> dict[str, Any]:
    import kd

    supplied = {} if options is None else options
    unknown = supplied.keys() - set(LOAD_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported load options: {sorted(unknown)}")
    parameters = inspect.signature(kd.load).parameters
    return {key: supplied.get(key, parameters[key].default) for key in LOAD_OPTIONS}


def _ordered(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: list(item.items())
            if key in {"coords", "fields"} and item is not None
            else _ordered(item)
            for key, item in value.items()
        }
    return value


def input_ref(source: dict[str, Any], options: dict[str, Any]) -> str:
    semantics = {**options, "periodic": sorted(set(options["periodic"] or []))}
    payload = {
        "format": INPUT_FORMAT,
        "source": {key: value for key, value in source.items() if key != "path"},
        "load_options": semantics,
    }
    encoded = json.dumps(
        _ordered(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return FILE_PREFIX + hashlib.sha256(encoded).hexdigest()


def _load(source: str | Path, options: dict[str, Any]) -> kd.PDEDataset:
    import kd

    if isinstance(source, str):

        kd.get_dataset(source)
    return kd.load(source, **options)


def prepare_input(
    path: Path, options: dict[str, Any] | None
) -> tuple[kd.PDEDataset, dict[str, Any]]:
    resolved = _options(options)
    dataset = _load(path, resolved)
    source = cast("kd.DatasetSource", dataset.source).to_dict()

    if resolved["coords"] is None and source["container"] != "xlsx":
        resolved["layout"] = source["layout"]
    ref = input_ref(source, resolved)
    dataset.name = ref
    return dataset, {
        "format": INPUT_FORMAT,
        "ref": ref,
        "path": source["path"],
        "source": source,
        "load_options": resolved,
    }


def _verify_description(ref: str, description: dict[str, Any]) -> None:
    if description["format"] != INPUT_FORMAT:
        raise ValueError(f"unsupported input format: {description['format']!r}")
    actual = input_ref(description["source"], description["load_options"])
    if actual != ref or description["ref"] != ref:
        raise ValueError(f"input description does not match reference {ref}")


def resolve_input(workspace: Path, dataset_id: str) -> dict[str, Any] | None:
    if not is_file_ref(dataset_id):
        import kd

        kd.get_dataset(dataset_id)
        return None
    path = workspace / "inputs" / f"{dataset_id}.json"
    description: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    _verify_description(dataset_id, description)
    with Path(description["path"]).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != description["source"]["sha256"]:
        raise ValueError(f"input file changed for reference {dataset_id}")
    return description


def load_dataset(dataset_id: str, description: dict[str, Any] | None) -> kd.PDEDataset:
    if description is None:
        return _load(dataset_id, {})
    _verify_description(dataset_id, description)
    dataset = _load(Path(description["path"]), description["load_options"])
    source = cast("kd.DatasetSource", dataset.source).to_dict()
    if input_ref(source, description["load_options"]) != dataset_id:
        raise ValueError(f"loaded input source changed for reference {dataset_id}")


    dataset.name = dataset_id
    return dataset


def save_input(workspace: Path, description: dict[str, Any]) -> None:
    directory = workspace / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False
    ) as stream:
        json.dump(description, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(stream.name, directory / f"{description['ref']}.json")
