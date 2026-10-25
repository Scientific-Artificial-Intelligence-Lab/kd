
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
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
    effort: str | None = None


class MissingConfigError(ValueError):
    pass


LLM4ED_KEY = "llm4ed"

LLM4ED_ENDPOINTS: dict[str, str | None] = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openai": None,
}

LLM4ED_LIMITS: dict[str, int] = {
    "max_llm_calls_per_propose": 50,
    "max_llm_calls_per_run": 200,
    "max_tokens": 1024,
    "max_search_seconds": 600,
}

_LLM4ED_STRINGS = ("endpoint", "base_url", "model", "api_key")


@dataclass(frozen=True)
class Llm4edConfig:

    endpoint: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    extra_body: dict[str, Any] | None = None
    max_llm_calls_per_propose: int = LLM4ED_LIMITS["max_llm_calls_per_propose"]
    max_llm_calls_per_run: int = LLM4ED_LIMITS["max_llm_calls_per_run"]
    max_tokens: int = LLM4ED_LIMITS["max_tokens"]
    max_search_seconds: int = LLM4ED_LIMITS["max_search_seconds"]

    def public(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            "model": self.model,
            "extra_body": deepcopy(self.extra_body),
            **{name: getattr(self, name) for name in LLM4ED_LIMITS},
        }


def _validate_llm4ed_block(block: object) -> None:
    if not isinstance(block, dict):
        raise ValueError("the llm4ed configuration must be a JSON object")
    known = {"enabled", "extra_body", *_LLM4ED_STRINGS, *LLM4ED_LIMITS}
    for key, value in block.items():
        if key not in known:
            raise ValueError(f"unknown llm4ed configuration field: {key}")
        if key == "enabled" and not isinstance(value, bool):
            raise ValueError("llm4ed enabled must be true or false")
        if key in _LLM4ED_STRINGS and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"llm4ed {key} must be a non-empty string")
        if key == "extra_body" and not isinstance(value, dict):
            raise ValueError("llm4ed extra_body must be a JSON object")
        if key in LLM4ED_LIMITS and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"llm4ed {key} must be a positive integer")
    if "endpoint" in block and block["endpoint"] not in LLM4ED_ENDPOINTS:
        raise ValueError(
            f"llm4ed endpoint must be one of: {', '.join(LLM4ED_ENDPOINTS)}"
        )


def resolve_llm4ed(
    saved: Mapping[str, Any] | None,
    *,
    enabled: bool | None,
    require_key: bool = True,
) -> Llm4edConfig | None:
    block = {} if saved is None else dict(saved.get(LLM4ED_KEY, {}))
    if not (block.get("enabled", False) if enabled is None else enabled):
        return None
    endpoint = block.get("endpoint")
    if endpoint is None:
        raise MissingConfigError(
            "LLM4ED is enabled but its endpoint is missing; run kd-agent setup --llm4ed"
        )
    base_url: str | None = block.get("base_url", LLM4ED_ENDPOINTS[endpoint])
    if base_url is None:
        raise MissingConfigError(
            "LLM4ED is enabled but its base URL is missing; run kd-agent setup --llm4ed"
        )
    if "model" not in block:
        raise MissingConfigError(
            "LLM4ED is enabled but its model is missing; run kd-agent setup --llm4ed"
        )
    if require_key and "api_key" not in block:
        raise MissingConfigError(
            "LLM4ED is enabled but its API key is missing; run kd-agent setup --llm4ed"
        )
    url = urlsplit(base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username is not None
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "the LLM4ED base URL must be an absolute HTTP(S) URL without "
            "credentials, query or fragment"
        )
    return Llm4edConfig(
        endpoint=endpoint,
        base_url=base_url,
        model=block["model"],
        api_key=block.get("api_key", "fingerprint-placeholder"),
        extra_body=deepcopy(block.get("extra_body")),
        **{name: block.get(name, default) for name, default in LLM4ED_LIMITS.items()},
    )


def check_llm4ed_shell(llm4ed: Llm4edConfig | None, *, shell: bool) -> None:
    if llm4ed is not None and shell:
        raise ValueError(
            "LLM4ED cannot run together with --shell: the shell could read the "
            "LLM4ED key. Run without --shell, or pass --no-llm4ed"
        )


def config_path() -> Path:
    return Path.home() / ".config" / "kd" / "agent.json"


def load_config(path: Path) -> dict[str, Any] | None:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    saved: dict[str, Any] = json.loads(content)
    _validate_settings(saved)
    if LLM4ED_KEY in saved:
        _validate_llm4ed_block(saved[LLM4ED_KEY])
    resolve_config(saved, environ={})
    return saved


def _validate_settings(values: Mapping[str, Any]) -> None:


    if not isinstance(values, dict):
        raise ValueError("agent configuration must be a JSON object")
    for key, value in values.items():
        if key == LLM4ED_KEY:
            continue
        if key not in ("endpoint", "base_url", "model", "api_key", "effort"):
            raise ValueError(f"unknown configuration field: {key}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
    if "endpoint" in values and values["endpoint"] not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of: {', '.join(ENDPOINTS)}")


def _merge_settings(
    saved: dict[str, Any] | None,
    endpoint: str | None,
    base_url: str | None,
    model: str | None,
    effort: str | None,
    env: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    file_values = (
        {}
        if saved is None
        else {key: value for key, value in saved.items() if key != LLM4ED_KEY}
    )
    env_values = {
        name: env[f"KD_AGENT_{name.upper()}"]
        for name in ("endpoint", "base_url", "model", "api_key", "effort")
        if f"KD_AGENT_{name.upper()}" in env
    }
    explicit = {
        k: v
        for k, v in (
            ("endpoint", endpoint),
            ("base_url", base_url),
            ("model", model),
            ("effort", effort),
        )
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
    saved: dict[str, Any] | None = None,
    *,
    endpoint: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> EndpointConfig:
    selected, values = _merge_settings(
        saved,
        endpoint,
        base_url,
        model,
        effort,
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
        values.get("effort"),
    )


def _saved_llm4ed(path: Path) -> dict[str, Any] | None:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    block: dict[str, Any] | None = json.loads(content).get(LLM4ED_KEY)
    return block


def _write_private(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(content, stream, indent=2)
        stream.write("\n")


def save_config(config: EndpointConfig, path: Path) -> None:
    block = _saved_llm4ed(path)
    _write_private(
        path,
        {
            "endpoint": config.endpoint,
            "base_url": config.base_url,
            "model": config.model,
            "api_key": config.api_key,
            **({"effort": config.effort} if config.effort is not None else {}),
            **({} if block is None else {LLM4ED_KEY: block}),
        },
    )


def save_llm4ed(block: dict[str, Any], path: Path) -> None:
    _validate_llm4ed_block(block)
    saved = load_config(path)
    if saved is None:
        raise MissingConfigError(
            f"no saved configuration at {path}; run kd-agent setup first"
        )
    _write_private(path, {**saved, LLM4ED_KEY: block})
