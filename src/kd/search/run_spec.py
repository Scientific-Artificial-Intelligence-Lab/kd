
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from kd.search.records import RecordSchemaError, StrictDecodeError

CONFIG_CANON_SCHEME: Final[str] = "kd-config-v1"
RUN_SPEC_HASH_SCHEME: Final[str] = "kd-runspec-v1"

_RUN_SPEC_FIELDS = frozenset(
    {
        "kd_version",
        "config",
        "config_canon_scheme",
        "library_fingerprint",
        "dataset_cache_fingerprint",
        "artifacts",
    }
)


class ConfigCanonicalizationError(RecordSchemaError):
    pass


class ConfigCanonSchemeError(RecordSchemaError):
    pass


def _value_type_name(value: object) -> str:
    return type(value).__name__


def _canonicalize_value(value: object, *, path: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not (float("-inf") < value < float("inf")):
            raise ConfigCanonicalizationError(
                f"Non-finite float at {path}: {value!r}"
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if type(value) is dict:
        normalized: dict[str, Any] = {}
        for key in value:
            if not isinstance(key, str):
                raise ConfigCanonicalizationError(
                    f"Non-string key at {path}: {_value_type_name(key)}"
                )
        for key in sorted(value):
            normalized[key] = _canonicalize_value(
                value[key],
                path=f"{path}.{key}",
            )
        return normalized
    if type(value) in (list, tuple):
        items = cast("list[object] | tuple[object, ...]", value)
        return [
            _canonicalize_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(items)
        ]
    raise ConfigCanonicalizationError(
        f"Unsupported config value at {path}: {_value_type_name(value)}"
    )


def canonicalize_config(config: dict[str, object]) -> dict[str, Any]:
    if type(config) is not dict:
        raise ConfigCanonicalizationError(
            f"Unsupported config value at config: {_value_type_name(config)}"
        )
    normalized = _canonicalize_value(config, path="config")
    return cast(dict[str, Any], normalized)


@dataclass(frozen=True)
class RunSpec:

    kd_version: str
    config: dict[str, Any]
    dataset_cache_fingerprint: str
    library_fingerprint: str | None = None
    artifacts: dict[str, Any] | None = None
    config_canon_scheme: str = CONFIG_CANON_SCHEME

    def __post_init__(self) -> None:
        if self.config_canon_scheme != CONFIG_CANON_SCHEME:
            raise ConfigCanonSchemeError(
                "Unsupported config_canon_scheme: "
                f"got {self.config_canon_scheme!r}; "
                f"supported schemes: {[CONFIG_CANON_SCHEME]!r}"
            )
        if self.library_fingerprint is not None and not isinstance(
            self.library_fingerprint, str
        ):
            raise RecordSchemaError(
                "library_fingerprint must be str or None; "
                f"got {_value_type_name(self.library_fingerprint)}"
            )
        object.__setattr__(self, "config", canonicalize_config(self.config))
        if self.artifacts is not None:
            if type(self.artifacts) is not dict:
                raise ConfigCanonicalizationError(
                    "Unsupported config value at artifacts: "
                    f"{_value_type_name(self.artifacts)}"
                )
            artifacts = _canonicalize_value(self.artifacts, path="artifacts")
            object.__setattr__(self, "artifacts", cast(dict[str, Any], artifacts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kd_version": self.kd_version,
            "config": deepcopy(self.config),
            "config_canon_scheme": self.config_canon_scheme,
            "library_fingerprint": self.library_fingerprint,
            "dataset_cache_fingerprint": self.dataset_cache_fingerprint,
            "artifacts": deepcopy(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSpec:
        actual = frozenset(data)
        unknown = actual - _RUN_SPEC_FIELDS
        if unknown:
            keys = ", ".join(repr(key) for key in sorted(unknown))
            raise StrictDecodeError(f"Unknown run_spec field(s): {keys}")
        missing = _RUN_SPEC_FIELDS - actual
        if missing:
            keys = ", ".join(repr(key) for key in sorted(missing))
            raise StrictDecodeError(f"Missing required run_spec field(s): {keys}")
        config = data["config"]
        if not isinstance(config, dict):
            raise StrictDecodeError("run_spec.config must be an object")
        artifacts = data["artifacts"]
        if artifacts is not None and not isinstance(artifacts, dict):
            raise StrictDecodeError("run_spec.artifacts must be an object or null")
        return cls(
            kd_version=data["kd_version"],
            config=cast(dict[str, Any], config),
            config_canon_scheme=data["config_canon_scheme"],
            library_fingerprint=data["library_fingerprint"],
            dataset_cache_fingerprint=data["dataset_cache_fingerprint"],
            artifacts=cast(dict[str, Any] | None, artifacts),
        )

    @property
    def run_spec_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload = f"{RUN_SPEC_HASH_SCHEME}:{canonical}".encode()
        return hashlib.sha256(payload).hexdigest()

    def content_hash(self) -> str:
        return self.run_spec_hash


__all__ = [
    "CONFIG_CANON_SCHEME",
    "RUN_SPEC_HASH_SCHEME",
    "ConfigCanonicalizationError",
    "ConfigCanonSchemeError",
    "RunSpec",
    "canonicalize_config",
]
