
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from pydantic import Field

from kdagent import experience_text, orchestration, prompt_text
from kdagent.adapter import make_tools
from kdagent.config import EndpointConfig
from kdagent.session import Session

_PACKAGES = (
    "deepagents",
    "langchain",
    "langchain-core",
    "langgraph",
    "langchain-openai",
    "langchain-anthropic",
)


class _Recorder:

    requests: list[dict[str, Any]]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.requests.append(
            {
                "system_messages": [m.content for m in messages if m.type == "system"],
                "tools": kwargs["tools"],
            }
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Recorded."))]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)


class RecordingOpenAI(_Recorder, ChatOpenAI):

    requests: list[dict[str, Any]] = Field(default_factory=list, exclude=True)


class RecordingAnthropic(_Recorder, ChatAnthropic):

    requests: list[dict[str, Any]] = Field(default_factory=list, exclude=True)


def capture_first_request(
    config: EndpointConfig, workspace: Path, system_prompt: str
) -> dict[str, Any]:
    model = orchestration.build_model(
        replace(config, api_key="fingerprint-placeholder", effort=None),
        client_class=RecordingAnthropic
        if config.protocol == "anthropic"
        else RecordingOpenAI,
    )
    agent = orchestration.build_agent(
        model,
        make_tools(Session(workspace)),
        workspace,
        system_prompt=system_prompt,
    )
    with tracing_context(enabled=False):
        agent.invoke(
            {"messages": [{"role": "user", "content": "Inspect this input."}]},
            {"callbacks": [], "max_concurrency": 1},
            interrupt_after=["model"],
        )
    request: dict[str, Any] = model.requests[0]
    return request


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute(config: EndpointConfig) -> dict[str, Any]:
    inputs = json.loads(
        files("kdagent")
        .joinpath("_assets/fingerprint/inputs.json")
        .read_text(encoding="utf-8")
    )
    requests = {}
    with TemporaryDirectory(prefix="kd-agent-fingerprint-") as directory:
        workspace = Path(directory)
        for name, summary in inputs["summaries"].items():
            orchestration.install_skills(workspace, experience_card=False)
            prompt = orchestration.compose_system_prompt(summary, None, None)
            requests[name] = capture_first_request(config, workspace, prompt)
        summary = inputs["summaries"]["grid-ut"]
        retrieval, advice = inputs["retrieval"], inputs["advice"]
        orchestration.install_skills(workspace, experience_card=True)
        prompt = orchestration.compose_system_prompt(summary, retrieval, advice)
        requests["experience"] = capture_first_request(config, workspace, prompt)
        skill_files = {
            str(path.relative_to(workspace)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((workspace / "skills").rglob("*"))
            if path.is_file()
        }
    components = {
        "first_requests": _digest(requests),
        "skill_files": skill_files,
        "experience_render": _digest(
            orchestration.render_experience(summary, retrieval, advice)
        ),



        "render_modules": {
            module.__name__: hashlib.sha256(
                Path(str(module.__file__)).read_bytes()
            ).hexdigest()
            for module in (experience_text, prompt_text)
        },
        "nudges": _digest([orchestration.ANSWER_NUDGE, orchestration.TRUNCATION_NUDGE]),
        "subagent_description": _digest(orchestration.GENERAL_PURPOSE_DESCRIPTION),
        "packages": {name: version(name) for name in _PACKAGES},
        "protocol": config.protocol,
        "model": config.model,
    }
    return {
        "format": "kd-agent-fingerprint-v1",
        "prompt_sha256": _digest(components),
        "components": components,
    }
