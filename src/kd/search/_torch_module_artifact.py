
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import BuiltinFunctionType, FunctionType
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Module

TORCH_MODULE_ARTIFACT_FORMAT = "kd-torch-module-v1"

_CAPTURED_MODULE_ATTRS = frozenset({"_buffers", "_modules", "_parameters"})
_HOOK_ATTRS = frozenset(
    {
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
    }
)


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _callable_identity(value: object, *, path: str) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if (
        isinstance(value, BuiltinFunctionType)
        and getattr(value, "__self__", None) is None
        and isinstance(module, str)
        and isinstance(qualname, str)
    ):
        return f"{module}.{qualname}"
    raise TypeError(
        f"Unsupported torch module attribute at {path}: {_qualified_type(value)}"
    )


def _canonical_attribute(value: object, *, path: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"Non-finite torch module attribute at {path}: {value!r}")
        return value
    if isinstance(value, Enum):
        return {"enum": _qualified_type(value), "name": value.name}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(
        value, (torch.dtype, torch.device, torch.layout, torch.memory_format)
    ):
        return {"torch_value": _qualified_type(value), "value": str(value)}
    if isinstance(value, Tensor):
        raw = _tensor_bytes(value)
        return {
            "tensor": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        }
    if isinstance(value, BuiltinFunctionType):
        return {"callable": _callable_identity(value, path=path)}
    if isinstance(value, FunctionType):
        raise TypeError(
            f"Unsupported torch module attribute at {path}: {_qualified_type(value)}"
        )
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        normalized: dict[str, Any] = {}
        for key in mapping:
            if not isinstance(key, str):
                raise TypeError(
                    f"Unsupported torch module attribute key at {path}: "
                    f"{_qualified_type(key)}"
                )
        string_mapping = cast(dict[str, object], mapping)
        for key in sorted(string_mapping):
            normalized[key] = _canonical_attribute(
                string_mapping[key], path=f"{path}.{key}"
            )
        return {"dict": normalized}
    if type(value) in (list, tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        kind = "list" if type(value) is list else "tuple"
        return {
            kind: [
                _canonical_attribute(item, path=f"{path}[{index}]")
                for index, item in enumerate(sequence)
            ]
        }
    if type(value) in (set, frozenset):
        unordered = cast(set[object] | frozenset[object], value)
        items = [
            _canonical_attribute(item, path=f"{path}[]") for item in unordered
        ]
        items.sort(
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            )
        )
        return {"frozenset" if type(value) is frozenset else "set": items}
    raise TypeError(
        f"Unsupported torch module attribute at {path}: {_qualified_type(value)}"
    )


def _module_attributes(module: Module, *, path: str) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for name, value in sorted(vars(module).items()):
        if name in _CAPTURED_MODULE_ATTRS:
            continue
        attribute_path = f"{path}.{name}"
        if name in _HOOK_ATTRS:
            if value:
                raise TypeError(
                    f"Unsupported registered torch hook at {attribute_path}"
                )
            continue
        if name == "_compiled_call_impl":
            if value is not None:
                raise TypeError(
                    f"Unsupported compiled torch call at {attribute_path}"
                )
            continue
        attributes[name] = _canonical_attribute(value, path=attribute_path)
    return attributes


def _module_topology(model: Module) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    for name, module in model.named_modules(remove_duplicate=False):
        path = f"modules.{name or '<root>'}"
        modules.append(
            {
                "name": name,
                "type": _qualified_type(module),
                "attributes": _module_attributes(module, path=path),
            }
        )
    return modules


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous()
    byte_view = value.reshape(-1).view(torch.uint8)
    return byte_view.numpy().tobytes(order="C")


def _state_items(model: Module) -> list[tuple[str, Tensor]]:
    state: dict[str, Tensor] = {}
    for name, value in model.state_dict().items():
        if not isinstance(value, Tensor):
            raise TypeError(
                "torch module state must contain tensors; "
                f"{name} has type {_qualified_type(value)}"
            )
        state[name] = value
    for module_name, module in model.named_modules(remove_duplicate=False):
        non_persistent = cast(
            set[str], getattr(module, "_non_persistent_buffers_set", set())
        )
        for buffer_name in sorted(non_persistent):
            buffer = module._buffers.get(buffer_name)
            if buffer is not None:
                prefix = f"{module_name}." if module_name else ""
                state[f"<nonpersistent>:{prefix}{buffer_name}"] = buffer
    return sorted(state.items())


def torch_module_artifact(
    model: Module,
    *,
    format_name: str = TORCH_MODULE_ARTIFACT_FORMAT,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, str | int]:
    state_items = _state_items(model)
    tensor_metadata: list[dict[str, object]] = []
    tensor_payloads: list[bytes] = []
    for name, tensor in state_items:
        tensor_metadata.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        )
        tensor_payloads.append(_tensor_bytes(tensor))

    identity_metadata = _canonical_attribute(dict(metadata or {}), path="metadata")[
        "dict"
    ]





    for reserved_key in ("modules", "state"):
        if reserved_key in identity_metadata:
            raise ValueError(
                f"torch module metadata key {reserved_key!r} collides with a "
                "reserved artifact-identity key"
            )
    identity_metadata["modules"] = _module_topology(model)
    identity_metadata["state"] = tensor_metadata
    header = (
        format_name
        + ":"
        + json.dumps(identity_metadata, sort_keys=True, separators=(",", ":"))
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    size = len(header)
    for payload in tensor_payloads:
        digest.update(payload)
        size += len(payload)
    return {"format": format_name, "sha256": digest.hexdigest(), "size": size}


__all__ = ["TORCH_MODULE_ARTIFACT_FORMAT", "torch_module_artifact"]
