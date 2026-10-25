
from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from typing import TypeAlias

    import kd

    Loaded: TypeAlias = kd.PDEDataset | kd.TrajectorySet | kd.TabularDataset

INPUT_FORMAT = "kd-agent-input-v2"
INPUT_FORMAT_V1 = "kd-agent-input-v1"
FILE_PREFIX = "file-v1-"
LOAD_OPTIONS_V1 = (
    "coords",
    "fields",
    "field_axes",
    "select",
    "layout",
    "lhs",
    "periodic",
)
LOAD_OPTIONS = ("kind", *LOAD_OPTIONS_V1, "trajectory_id", "params")
TABLE_TASKS = ("tabular-regression", "ode")
PROJECTION_FORMAT = "kd-agent-trajectory-projection-v1"
PROJECTION_TRIM = 2
CONTROLLER_KEYS = {"task_kind": "--task-kind", "target_state": "--target-state"}
DERIVED_SEPARATOR = "+"


def is_file_ref(dataset_id: str) -> bool:
    return dataset_id.startswith(FILE_PREFIX)


DERIVATION_FORMAT = "kd-agent-derivation-v1"


class InputVerificationError(ValueError):
    pass


def is_derived(dataset_id: str) -> bool:
    return DERIVED_SEPARATOR in dataset_id


def root_ref(dataset_id: str) -> str:
    return dataset_id.split(DERIVED_SEPARATOR, 1)[0]


def is_pinn_derived(dataset_id: str) -> bool:
    return any(
        segment.startswith("pinn")
        for segment in dataset_id.split(DERIVED_SEPARATOR)[1:]
    )


def spell_modes(lowpass_modes: int | Mapping[str, int]) -> str:
    if isinstance(lowpass_modes, int):
        return str(lowpass_modes)
    return ",".join(f"{axis}={lowpass_modes[axis]}" for axis in sorted(lowpass_modes))


def derived_ref(
    parent: str, lowpass_modes: int | Mapping[str, int], time_window: int
) -> str:
    return (
        f"{parent}{DERIVED_SEPARATOR}lowpass{spell_modes(lowpass_modes)}-t{time_window}"
    )


PINN_SETTINGS = (
    "terms",
    "coefficients",
    "epochs",
    "pretrain_epochs",
    "residual_weight",
    "seed",
)


def pinn_ref(parent: str, pinn: Mapping[str, Any]) -> str:
    settings = json.dumps(
        {key: pinn[key] for key in PINN_SETTINGS},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(settings.encode("utf-8")).hexdigest()[:10]
    return f"{parent}{DERIVED_SEPARATOR}pinn{pinn['epochs']}-{digest}"


def field_ref(parent: str, surrogate_id: str) -> str:
    return f"{parent}{DERIVED_SEPARATOR}field-{surrogate_id}"


def derived_reference(derived: Mapping[str, Any]) -> str:
    pinn = derived.get("pinn")
    if pinn is not None:
        return pinn_ref(derived["from"], pinn)
    network = derived.get("network")
    if network is not None:
        return field_ref(derived["from"], network["surrogate_id"])
    return derived_ref(
        derived["from"], derived["lowpass_modes"], derived["time_window"]
    )


def projection_recipe(target_state: str) -> dict[str, Any]:
    return {
        "format": PROJECTION_FORMAT,
        "target_state": target_state,
        "method": "finite_diff",
        "state_source": "observed",
        "trim": PROJECTION_TRIM,
    }


def table_task(description: Mapping[str, Any] | None) -> str | None:
    if description is None:
        return None
    if description["format"] == INPUT_FORMAT_V1:
        options = description["load_options"]
        if options["layout"] != "kd-tabular-npz":
            return None
        return cast(str, options["task_kind"])
    task = cast(str, description["task_kind"])
    return task if task in TABLE_TASKS else None


def is_table(description: Mapping[str, Any] | None) -> bool:
    return table_task(description) is not None


def raw_fingerprint(description: Mapping[str, Any] | None) -> str | None:
    if description is None or description["format"] == INPUT_FORMAT_V1:
        return None
    return cast("str | None", description["raw_fingerprint"])


def _options(options: dict[str, Any] | None) -> dict[str, Any]:
    import kd

    supplied = {} if options is None else options
    for key, flag in CONTROLLER_KEYS.items():
        if key in supplied:
            raise ValueError(
                f"{key} is not a kd load option; pass it as {key}= ({flag} on "
                "the command line)"
            )
    unknown = supplied.keys() - set(LOAD_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported load options: {sorted(unknown)}")
    parameters = inspect.signature(kd.load).parameters
    return {key: supplied.get(key, parameters[key].default) for key in LOAD_OPTIONS}


def _ordered(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: list(item.items())
            if key in {"coords", "fields", "params"} and item is not None
            else _ordered(item)
            for key, item in value.items()
        }
    return value


def input_ref(description: Mapping[str, Any]) -> str:
    options = description["load_options"]
    semantics = dict(options)
    if "periodic" in options:
        semantics["periodic"] = sorted(set(options["periodic"] or []))
    payload = {
        "format": description["format"],
        "source": {
            key: value for key, value in description["source"].items() if key != "path"
        },
        "load_options": semantics,
    }
    if description["format"] != INPUT_FORMAT_V1:
        payload["task_kind"] = description["task_kind"]
        payload["projection"] = description["projection"]
    encoded = json.dumps(
        _ordered(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return FILE_PREFIX + hashlib.sha256(encoded).hexdigest()


def _load_v1(
    source: str | Path, options: dict[str, Any]
) -> kd.PDEDataset | kd.TabularDataset:
    import kd

    if isinstance(source, str):

        kd.get_dataset(source)
    if options.get("layout") == "kd-tabular-npz":
        return kd.load_tabular(Path(source))
    return kd.load(source, kind="pde", **options)


def _source_v1(
    dataset: kd.PDEDataset | kd.TabularDataset, path: Path
) -> dict[str, Any]:
    import kd

    if isinstance(dataset, kd.TabularDataset):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "path": str(path.resolve()),
            "sha256": digest,
            "container": "npz",
            "layout": "kd-tabular-npz",
        }
    return cast("kd.DatasetSource", dataset.source).to_dict()


def _loaded_source(loaded: Loaded) -> dict[str, Any]:
    import kd

    record = loaded.source if isinstance(loaded, kd.PDEDataset) else loaded.file_source
    return cast("kd.DatasetSource", record).to_dict()


def _kind(loaded: Loaded) -> str:
    import kd

    if isinstance(loaded, kd.PDEDataset):
        return "pde"
    if isinstance(loaded, kd.TrajectorySet):
        return "trajectory"
    return "tabular"


def pde_task_kind(dataset: kd.PDEDataset) -> str:
    return "pde-steady" if dataset.lhs_order == 0 else "pde-evolution"


def _task_kind(loaded: Loaded, declared: str | None, path: Path) -> str:
    import kd

    if isinstance(loaded, kd.TrajectorySet):
        if declared not in (None, "ode"):
            raise ValueError(
                f"{path} holds raw trajectories, whose task kind is ode; "
                f"task_kind {declared!r} conflicts with it"
            )
        return "ode"
    if isinstance(loaded, kd.TabularDataset):
        task = "tabular-regression" if declared is None else declared
        if task not in TABLE_TASKS:
            raise ValueError("task_kind must be tabular-regression or ode")
        return task
    if declared is not None:
        raise ValueError(
            f"task_kind {declared!r} is declared for a table or raw "
            f"trajectories, and {path} loads as grid data"
        )
    return pde_task_kind(loaded)


def _target_state(trajectories: kd.TrajectorySet, requested: str | None) -> str:
    states = list(trajectories.state_vars)
    if requested is None:
        if len(states) == 1:
            return states[0]
        raise ValueError(
            f"the trajectories have the states {states}; choose the one whose "
            "time derivative is the target with target_state (--target-state)"
        )
    if requested not in states:
        raise ValueError(
            f"target_state {requested!r} is not a state of the trajectories {states}"
        )
    return requested


def _project(
    trajectories: kd.TrajectorySet, file_sha256: str, target_state: str, trim: int
) -> tuple[kd.TrajectorySet, kd.TabularDataset]:
    import kd

    raw = replace(trajectories, name=file_sha256)
    return raw, kd.trajectories_to_tabular(raw, target_state, trim=trim)


def _named(
    dataset: kd.PDEDataset | kd.TabularDataset, name: str
) -> kd.PDEDataset | kd.TabularDataset:
    import kd

    if isinstance(dataset, kd.TabularDataset):
        return replace(dataset, name=name)
    dataset.name = name
    return dataset


def _resolved_lhs(dataset: kd.PDEDataset) -> str:
    import kd

    if dataset.lhs_order == 0:
        return ""
    return kd.render_lhs_label(
        kd.LhsSpec(
            field=dataset.lhs_field, axis=dataset.lhs_axis, order=dataset.lhs_order
        )
    )


def prepare_input(
    path: Path,
    options: dict[str, Any] | None,
    *,
    task_kind: str | None = None,
    target_state: str | None = None,
    recognized: Sequence[str] | None = None,
) -> tuple[kd.PDEDataset | kd.TabularDataset, dict[str, Any]]:
    import kd

    resolved = _options(options)
    loaded = kd.load(path, **resolved)
    resolved["kind"] = _kind(loaded)
    if isinstance(loaded, kd.PDEDataset) and resolved["lhs"] is None:


        resolved["lhs"] = _resolved_lhs(loaded)
    source = _loaded_source(loaded)
    if recognized is None:
        recognized = kd.inspect_file(path).layouts
    if resolved["coords"] is None and source["layout"] in recognized:
        resolved["layout"] = source["layout"]
    description: dict[str, Any] = {
        "format": INPUT_FORMAT,
        "ref": None,
        "path": source["path"],
        "source": source,
        "load_options": resolved,
        "task_kind": _task_kind(loaded, task_kind, path),
        "projection": None,
        "raw_fingerprint": None,
    }
    dataset: kd.PDEDataset | kd.TabularDataset
    if isinstance(loaded, kd.TrajectorySet):
        dataset = _describe_projection(description, loaded, target_state)
    elif target_state is not None:
        raise ValueError(
            "target_state chooses the state of raw trajectories whose time "
            f"derivative is the target, and {path} loads as {resolved['kind']} "
            "data, not trajectories"
        )
    else:
        dataset = loaded
    description["ref"] = input_ref(description)
    return _named(dataset, description["ref"]), description


def _describe_projection(
    description: dict[str, Any],
    trajectories: kd.TrajectorySet,
    target_state: str | None,
) -> kd.TabularDataset:
    import kd

    state = _target_state(trajectories, target_state)
    raw, table = _project(
        trajectories, description["source"]["sha256"], state, PROJECTION_TRIM
    )
    description["projection"] = projection_recipe(state)
    description["raw_fingerprint"] = kd.compute_dataset_fingerprint(raw)
    description["trajectories"] = {
        "count": len(raw.trajectories),
        "points": [len(trajectory.t) for trajectory in raw.trajectories],
    }
    return table


def _derivation_record(derived: Mapping[str, Any], options: Mapping[str, Any]) -> bytes:
    return json.dumps(
        {"format": DERIVATION_FORMAT, "derived": derived, "load_options": options},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_derivation_record(data: bytes) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            comment = archive.comment
        record = json.loads(comment.decode("utf-8")) if comment else None
    except (zipfile.BadZipFile, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("format") != DERIVATION_FORMAT:
        return None
    return record


def prepare_derived_input(
    path: Path,
    options: dict[str, Any],
    *,
    derived: dict[str, Any],
) -> tuple[kd.PDEDataset | kd.TabularDataset, dict[str, Any]]:
    import kd

    resolved = {**_options(options), "kind": "pde"}
    with zipfile.ZipFile(path, "a") as archive:
        archive.comment = _derivation_record(derived, resolved)
    dataset = cast("kd.PDEDataset", kd.load(path, **resolved))
    source = _loaded_source(dataset)
    if source["layout"] != resolved["layout"]:
        raise ValueError(
            f"derived file {path} loaded as {source['layout']!r}, not as the "
            f"{resolved['layout']!r} its record names"
        )
    ref = derived_reference(derived)
    return _named(dataset, ref), {
        "format": INPUT_FORMAT,
        "ref": ref,
        "path": source["path"],
        "source": source,
        "load_options": resolved,
        "task_kind": pde_task_kind(dataset),
        "projection": None,
        "raw_fingerprint": None,
        "derived": dict(derived),
    }


def _expected_ref(description: dict[str, Any]) -> str:
    derived = description.get("derived")
    if derived is None:
        return input_ref(description)
    return derived_reference(derived)


def _verify_description(ref: str, description: dict[str, Any]) -> None:
    if description["format"] not in (INPUT_FORMAT, INPUT_FORMAT_V1):
        raise InputVerificationError(
            f"unsupported input format: {description['format']!r}"
        )
    projection = (
        None if description["format"] == INPUT_FORMAT_V1 else description["projection"]
    )
    if projection is not None and projection != projection_recipe(
        projection["target_state"]
    ):
        raise InputVerificationError(
            f"input {ref} names a trajectory projection the controller does not "
            f"make: {projection!r}"
        )
    if _expected_ref(description) != ref or description["ref"] != ref:
        raise InputVerificationError(
            f"input description does not match reference {ref}"
        )


def _verify_derivation(
    workspace: Path,
    dataset_id: str,
    description: dict[str, Any],
    *,
    written: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    if written is None or dataset_id not in written:
        raise InputVerificationError(
            f"derived input {dataset_id} was not derived by the controller in "
            "this run, and only an input it derived and registered this run is "
            "used; derive it again with the call that names it"
        )
    data = Path(description["path"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if written[dataset_id] != digest:
        raise InputVerificationError(
            f"derived input {dataset_id}: its file (sha256 {digest}) is not the "
            f"one the controller wrote this run (sha256 {written[dataset_id]})"
        )
    if digest != description["source"]["sha256"]:
        raise InputVerificationError(f"input file changed for reference {dataset_id}")
    record = _read_derivation_record(data)
    if record is None:
        raise InputVerificationError(
            f"derived input {dataset_id}: its file carries no derivation record, "
            "so it is not a file the controller derived"
        )
    derived = description["derived"]
    if record["derived"] != derived:
        raise InputVerificationError(
            f"derived input {dataset_id}: the description's derivation differs "
            "from the one recorded in its file"
        )
    if record["load_options"] != description["load_options"]:
        raise InputVerificationError(
            f"derived input {dataset_id}: the description's load options differ "
            "from the ones recorded in its file"
        )
    network = derived.get("network")
    if network is not None:
        _verify_artifact(workspace, dataset_id, network)
    parent = derived["from"]
    if not is_file_ref(parent):
        raise InputVerificationError(
            f"derived input {dataset_id}: its parent {parent!r} is not a file input"
        )
    chain = derivation_chain(workspace, parent, written=written)
    parent_sha256 = chain[-1]["source"]["sha256"]
    if derived["parent_sha256"] != parent_sha256:
        raise InputVerificationError(
            f"derived input {dataset_id} was written from a parent file with "
            f"sha256 {derived['parent_sha256']}, and the file of {parent} has "
            f"sha256 {parent_sha256}"
        )
    child_lhs = record["load_options"].get("lhs")
    parent_lhs = lhs_label(chain[-1])
    if child_lhs != parent_lhs:
        raise InputVerificationError(
            f"derived input {dataset_id}: its left-hand side {child_lhs!r} is "
            f"not the left-hand side {parent_lhs!r} of its parent {parent}, and "
            "a derived input keeps its parent's"
        )
    return [*chain, description]


def _verify_artifact(
    workspace: Path, dataset_id: str, network: Mapping[str, Any]
) -> None:
    path = workspace / network["artifact"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != network["artifact_sha256"]:
        raise InputVerificationError(
            f"derived input {dataset_id}: the model it was sampled from, "
            f"{network['artifact']}, has sha256 {digest}, not the "
            f"{network['artifact_sha256']} it was sampled with"
        )


def lhs_label(description: Mapping[str, Any]) -> str | None:
    if "derived" in description:
        return cast("str | None", description["load_options"].get("lhs"))
    return cast("str | None", description["summary"]["report"].get("lhs_label"))


def derivation_chain(
    workspace: Path, dataset_id: str, *, written: Mapping[str, str] | None = None
) -> list[dict[str, Any]]:
    if not is_file_ref(dataset_id):
        raise InputVerificationError(f"{dataset_id!r} is not a file input")
    path = workspace / "inputs" / f"{dataset_id}.json"
    description: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    _verify_description(dataset_id, description)
    if "derived" in description:
        return _verify_derivation(workspace, dataset_id, description, written=written)
    with Path(description["path"]).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != description["source"]["sha256"]:
        raise InputVerificationError(f"input file changed for reference {dataset_id}")
    return [description]


def input_record(dataset_id: str, chain: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "file_sha256": chain[-1]["source"]["sha256"] if chain else None,
        "chain": [
            {"dataset_id": link["ref"], "file_sha256": link["source"]["sha256"]}
            for link in chain
        ],
        "derived": None if not chain else deepcopy(chain[-1].get("derived")),
    }


def sealed_input_failure(
    workspace: Path, record: Mapping[str, Any], *, written: Mapping[str, str]
) -> str | None:
    dataset_id = record["dataset_id"]
    if not is_file_ref(dataset_id):
        return None
    try:
        chain = derivation_chain(workspace, dataset_id, written=written)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"
    current = input_record(dataset_id, chain)["chain"]
    for index in range(max(len(current), len(record["chain"]))):
        then = record["chain"][index] if index < len(record["chain"]) else None
        now = current[index] if index < len(current) else None
        if then != now:
            link = (then or now or {}).get("dataset_id")
            return (
                f"link {link}: the search ran on the file with sha256 "
                f"{None if then is None else then['file_sha256']}, and the "
                "workspace now holds "
                f"{None if now is None else now['file_sha256']}"
            )
    return None


def resolve_input(
    workspace: Path, dataset_id: str, *, written: Mapping[str, str] | None = None
) -> dict[str, Any] | None:
    if not is_file_ref(dataset_id):
        import kd

        kd.get_dataset(dataset_id)
        return None
    return derivation_chain(workspace, dataset_id, written=written)[-1]


def _load_described(
    description: dict[str, Any],
) -> tuple[Loaded, dict[str, Any]]:
    import kd

    path = Path(description["path"])
    if description["format"] == INPUT_FORMAT_V1:
        dataset = _load_v1(path, description["load_options"])
        return dataset, _source_v1(dataset, path)
    loaded = kd.load(path, **description["load_options"])
    return loaded, _loaded_source(loaded)


def _verified_raw(
    dataset_id: str,
    description: dict[str, Any],
    trajectories: kd.TrajectorySet,
    file_sha256: str,
) -> kd.TrajectorySet:
    import kd

    raw = replace(trajectories, name=file_sha256)
    fingerprint = kd.compute_dataset_fingerprint(raw)
    if fingerprint != description["raw_fingerprint"]:
        raise ValueError(
            f"the raw trajectories of input {dataset_id} fingerprint as "
            f"{fingerprint}, not as the {description['raw_fingerprint']} its "
            "description records"
        )
    return raw


def _verified_projection(
    dataset_id: str,
    description: dict[str, Any],
    trajectories: kd.TrajectorySet,
    file_sha256: str,
) -> kd.TabularDataset:
    import kd

    projection = description["projection"]
    raw = _verified_raw(dataset_id, description, trajectories, file_sha256)
    return kd.trajectories_to_tabular(
        raw, projection["target_state"], trim=projection["trim"]
    )


def _loaded_input(
    dataset_id: str, description: dict[str, Any]
) -> tuple[Loaded, dict[str, Any]]:
    _verify_description(dataset_id, description)
    try:
        loaded, source = _load_described(description)
    except ValueError as exc:
        raise ValueError(
            f"{exc}. The saved description of input {dataset_id} no longer "
            "loads; prepare the file again with --data"
        ) from exc


    changed = (
        source["sha256"] != description["source"]["sha256"]
        if "derived" in description
        else input_ref({**description, "source": source}) != dataset_id
    )
    if changed:
        raise ValueError(f"loaded input source changed for reference {dataset_id}")
    return loaded, source


def load_dataset(
    dataset_id: str, description: dict[str, Any] | None
) -> kd.PDEDataset | kd.TabularDataset:
    import kd

    if description is None:
        return _load_v1(dataset_id, {})
    loaded, source = _loaded_input(dataset_id, description)
    dataset = (
        _verified_projection(dataset_id, description, loaded, source["sha256"])
        if isinstance(loaded, kd.TrajectorySet)
        else loaded
    )


    return _named(dataset, dataset_id)


def load_trajectories(dataset_id: str, description: dict[str, Any]) -> kd.TrajectorySet:
    import kd

    loaded, source = _loaded_input(dataset_id, description)
    if not isinstance(loaded, kd.TrajectorySet):
        raise ValueError(f"input {dataset_id} holds no raw trajectories")
    return _verified_raw(dataset_id, description, loaded, source["sha256"])


def save_input(workspace: Path, description: dict[str, Any]) -> None:
    directory = workspace / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False
    ) as stream:
        json.dump(description, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(stream.name, directory / f"{description['ref']}.json")
