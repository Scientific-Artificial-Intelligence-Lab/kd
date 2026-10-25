
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

from kdagent import action_memory_text, experience_text, orchestration, prompt_text
from kdagent.adapter import make_tools
from kdagent.config import EndpointConfig, Llm4edConfig
from kdagent.session import Session

FORMAT = "kd-agent-fingerprint-v1"

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
    config: EndpointConfig,
    workspace: Path,
    system_prompt: str,
    *,
    bare: bool = False,
    shell: bool = False,
    llm4ed: Llm4edConfig | None = None,
) -> dict[str, Any]:
    model = orchestration.build_model(
        replace(config, api_key="fingerprint-placeholder", effort=None),
        client_class=RecordingAnthropic
        if config.protocol == "anthropic"
        else RecordingOpenAI,
    )
    agent = orchestration.build_agent(
        model,
        make_tools(Session(workspace, llm4ed=llm4ed)),
        workspace,
        system_prompt=system_prompt,
        bare=bare,
        shell=shell,
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


def compute(
    config: EndpointConfig,
    *,
    bare: bool = False,
    shell: bool = False,
    llm4ed: Llm4edConfig | None = None,
) -> dict[str, Any]:
    inputs = json.loads(
        files("kdagent")
        .joinpath("_assets/fingerprint/inputs.json")
        .read_text(encoding="utf-8")
    )
    requests = {}
    with TemporaryDirectory(prefix="kd-agent-fingerprint-") as directory:
        workspace = Path(directory)
        for name, summary in inputs["summaries"].items():
            if not bare:
                orchestration.install_skills(
                    workspace, experience_card=False, shell=shell
                )
            prompt = orchestration.compose_system_prompt(
                summary, None, None, bare=bare, shell=shell, llm4ed=llm4ed is not None
            )
            requests[name] = capture_first_request(
                config, workspace, prompt, bare=bare, shell=shell, llm4ed=llm4ed
            )
        summary = inputs["summaries"]["grid-ut"]
        retrieval, advice = inputs["retrieval"], inputs["advice"]
        if not bare:
            orchestration.install_skills(workspace, experience_card=True, shell=shell)
            prompt = orchestration.compose_system_prompt(
                summary, retrieval, advice, shell=shell, llm4ed=llm4ed is not None
            )
            requests["experience"] = capture_first_request(
                config, workspace, prompt, shell=shell, llm4ed=llm4ed
            )
        skill_files = {
            str(path.relative_to(workspace)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((workspace / "skills").rglob("*"))
            if path.is_file()
        }
        if bare and skill_files:
            raise RuntimeError("a bare agent must install no skills")
    components = {
        "first_requests": _digest(requests),
        "skill_files": skill_files,
        "experience_render": _digest(
            orchestration.render_experience(summary, retrieval, advice)
        ),
        "action_memory_render": _digest(
            [
                action_memory_text.render(inputs["action_hits"]),
                action_memory_text.render(inputs["action_effect_hits"]),
            ]
        ),



        "render_modules": {
            module.__name__: hashlib.sha256(
                Path(str(module.__file__)).read_bytes()
            ).hexdigest()
            for module in (experience_text, prompt_text, action_memory_text)
        },
        "nudges": _digest([orchestration.ANSWER_NUDGE, orchestration.TRUNCATION_NUDGE]),
        "subagent_description": _digest(
            orchestration.GENERAL_PURPOSE_DESCRIPTION
            if shell
            else orchestration.GENERAL_PURPOSE_DESCRIPTION_NO_SHELL
        ),
        "bare": bare,
        "shell": shell,
        "llm4ed": None if llm4ed is None else llm4ed.public(),
        "packages": {name: version(name) for name in _PACKAGES},
        "protocol": config.protocol,
        "model": config.model,
    }
    kinds = {summary["task_kind"] for summary in inputs["summaries"].values()}
    by_kind = {
        kind: _digest(
            {
                **components,
                "first_requests": _digest(
                    {
                        name: request
                        for name, request in requests.items()
                        if _request_kind(name, inputs) == kind
                    }
                ),
                "task_kind": kind,
            }
        )
        for kind in sorted(kinds)
    }
    return {
        "format": FORMAT,
        "prompt_sha256": _digest(components),
        "prompt_sha256_by_task_kind": by_kind,
        "components": components,
        "llm4ed": components["llm4ed"],
    }


def _request_kind(name: str, inputs: dict[str, Any]) -> str:
    summary = inputs["summaries"]["grid-ut" if name == "experience" else name]
    kind: str = summary["task_kind"]
    return kind
