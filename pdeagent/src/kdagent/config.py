
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class EndpointPreset:

    model: str
    base_url: str
    api_key_env: str | None
    protocol: str = "openai"


ENDPOINT_PRESETS: dict[str, EndpointPreset] = {
    "local": EndpointPreset(
        model="gemma4:12b",
        base_url="http://localhost:11434/v1",
        api_key_env=None,
    ),
    "nvidia": EndpointPreset(












        model="nvidia/nemotron-3-ultra-550b-a55b",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
    ),




    "bigmodel": EndpointPreset(
        model="glm-5.3",
        base_url="https://open.bigmodel.cn/api/anthropic",
        api_key_env="BIGMODEL_API_KEY",
        protocol="anthropic",
    ),
}

ENDPOINTS = ("openai", "anthropic", *ENDPOINT_PRESETS)
DEFAULT_RECURSION_LIMIT = 100


@dataclass(frozen=True)
class EndpointConfig:

    endpoint: str
    protocol: str
    base_url: str
    model: str
    api_key: str = field(repr=False)


class MissingConfigError(ValueError):
    pass


def config_path() -> Path:
    return Path.home() / ".config" / "kd" / "agent.json"


def load_config(path: Path) -> dict[str, str] | None:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    saved: dict[str, str] = json.loads(content)
    _validate_settings(saved)
    resolve_config(saved, environ={})
    return saved


def _validate_settings(values: Mapping[str, str]) -> None:

    if not isinstance(values, dict):
        raise ValueError("agent configuration must be a JSON object")
    for key, value in values.items():
        if key not in ("endpoint", "base_url", "model", "api_key"):
            raise ValueError(f"unknown configuration field: {key}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
    if "endpoint" in values and values["endpoint"] not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of: {', '.join(ENDPOINTS)}")


def _merge_settings(
    saved: dict[str, str] | None,
    endpoint: str | None,
    base_url: str | None,
    model: str | None,
    env: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    file_values = {} if saved is None else saved
    env_values = {
        name: env[f"KD_AGENT_{name.upper()}"]
        for name in ("endpoint", "base_url", "model", "api_key")
        if f"KD_AGENT_{name.upper()}" in env
    }
    explicit = {
        k: v
        for k, v in (("endpoint", endpoint), ("base_url", base_url), ("model", model))
        if v is not None
    }
    for values in (file_values, env_values, explicit):
        _validate_settings(values)
    selected = {**file_values, **env_values, **explicit}.get("endpoint")
    if selected is None:
        raise MissingConfigError("endpoint is missing; run kd-agent setup")
    preset = ENDPOINT_PRESETS.get(selected)
    values = (
        {}
        if preset is None
        else {
            "base_url": preset.base_url,
            "model": preset.model,
        }
    )
    if file_values.get("endpoint") == selected:
        values.update(file_values)
    if (
        preset is not None
        and preset.api_key_env is not None
        and preset.api_key_env in env
    ):
        values["api_key"] = env[preset.api_key_env]
    values.update(env_values)
    values.update(explicit)
    if selected == "local" and "api_key" not in values:
        values["api_key"] = "ollama"
    return selected, values


def resolve_config(
    saved: dict[str, str] | None = None,
    *,
    endpoint: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> EndpointConfig:
    selected, values = _merge_settings(
        saved,
        endpoint,
        base_url,
        model,
        os.environ if environ is None else environ,
    )
    preset = ENDPOINT_PRESETS.get(selected)
    _validate_settings(values)
    for name in ("base_url", "model", "api_key"):
        if name not in values:
            key_env = preset.api_key_env if preset is not None else "KD_AGENT_API_KEY"
            hint = f" ({key_env})" if name == "api_key" else ""
            raise MissingConfigError(f"{name}{hint} is missing; run kd-agent setup")
    url = urlsplit(values["base_url"])
    if url.scheme not in ("http", "https") or not url.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    return EndpointConfig(
        selected,
        preset.protocol if preset else selected,
        values["base_url"],
        values["model"],
        values["api_key"],
    )


def save_config(config: EndpointConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(
            {
                "endpoint": config.endpoint,
                "base_url": config.base_url,
                "model": config.model,
                "api_key": config.api_key,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
