
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

from kdagent import cards, experience_text, model_signals
from kdagent.advice import Advice
from kdagent.experience import Retrieval
from kdagent.prompt_text import render_pinned_glossary, render_task_section
from kdagent.resume import schema_of
from kdagent.config import ENDPOINT_PRESETS, EndpointConfig
from kdagent.state import (
    ANSWER,
    ANSWER_NUDGES,
    TRUNCATION_NUDGES,
    Answer,
    KdAgentState,
    RunLoopResult,
)

logger = logging.getLogger(__name__)


SKILLS_DIR = "skills"
TRACE_FILENAME = "trace.jsonl"
RUN_NAME = "kdagent"

ANSWER_NUDGE = (
    "That turn ended without a tool call, so nothing new ran and no answer was "
    "submitted. If you want to keep searching, call run_discovery. If you are "
    "done, finish by calling submit_answer — an equation written as text is not "
    "an answer this run can record."
)

MAX_ANSWER_NUDGES = 2

TRUNCATION_NUDGE = (
    "Your last reply was cut off at the output limit before it reached a tool "
    "call, so nothing ran and nothing was recorded. Keep the reasoning short and "
    "go straight to the call: run_discovery to keep searching, submit_answer to "
    "finish."
)

MAX_TRUNCATION_NUDGES = 2

ANSWER_ENDS_RUN = "SubmittedAnswerEndsRun"

STOP_ANSWER_SUBMITTED = "answer_submitted"

STOP_TRUNCATION_CAP = "truncation_cap"


def is_truncated(message: Any) -> bool:
    from langchain_core.messages import AIMessage

    return isinstance(message, AIMessage) and model_signals.is_truncated(
        message.response_metadata
    )


def classify_stop_reason(
    messages: list[Any], answer: Answer | None, error: Exception | None
) -> str:
    from langchain_core.messages import AIMessage
    from langgraph.errors import GraphRecursionError

    if isinstance(error, GraphRecursionError):
        return "recursion_limit"
    if error is not None:
        return f"crashed:{type(error).__name__}"
    if answer is not None and answer["source"] == "submitted":
        return STOP_ANSWER_SUBMITTED
    if messages and is_truncated(messages[-1]):
        return STOP_TRUNCATION_CAP
    if (
        messages
        and isinstance(messages[-1], AIMessage)
        and model_signals.is_refusal(messages[-1].response_metadata)
    ):
        return "refused"
    return "model_finished"


_RECORD = cards.run_record_layout()
SYSTEM_PROMPT = f"""You drive KD, a symbolic regression platform that discovers equations from data.

Work out which equation governs the data you are given. You have tools to list
the datasets, to run a search, to train a derivative surrogate, to abandon or to
publish a search you have already run, and to submit your answer, plus a shell
and a Python interpreter. When your searches disagree, `prune_node` gives up on
a branch that has gone dead and `select_node` names the search to publish;
without a submission, the search you selected is what this run ends with.

Your workspace holds the skills and the records of your own searches.
`list_datasets` returns the usable IDs: a file input with its task kind, target
and data report, a catalog dataset with its axes, lhs, fmt, tier and tags and no
report. Pass the ID unchanged to both search and surrogate training. File inputs
have content references and descriptions under `inputs/`; KD loads their original
files in the worker.
Changed bytes or loading options require preparing a new input reference.
The file tools are rooted at the workspace, so `/` is its top level; `execute` runs
in the same directory but `pwd` prints the host path, which the file tools do
not accept.

Before the first search, read `/skills/kd-tools/SKILL.md`.
Read the skill for an algorithm before you configure it.

When a tool fails and names what to change, apply that change and call it again
before you move on to something else. When an iterative search ends with a
poor fit, continue it with a larger budget before switching; the applicability
table names each instrument's budget lever.

Every search seals a directory under `{_RECORD["runs_root"]}/` in your workspace
and files one segment row in `{_RECORD["tree"]}`, the lineage of every segment
run in this workspace, including segments another process ran here. A search
run through `run_discovery` also appends one row to `{_RECORD["catalog"]}`, so
that file lists your own searches and not necessarily every segment on the
tree; `run_id` joins the two. Each `run_discovery` result carries the path to
its own directory in `provenance.run_dir`. A run directory holds
`{_RECORD["record"]}` (the sealed result), `{_RECORD["events"]}` (one line per
search iteration, with the best expression and score so far) and
`{_RECORD["phases"]}`. The catalog is append-only: re-running a configuration
adds a row rather than replacing one, so the current row for a run is the one
with the latest `created_at`. The `diagnostics.segment_report` in each search
result summarizes that segment; the shell and Python interpreter let you inspect its sealed files.

Finish by calling submit_answer. Prose is not an answer.
"""


def render_experience(
    summary: dict[str, Any], retrieval: Retrieval, advice: Advice | None
) -> str:
    return experience_text.render(
        task_kind=retrieval["task_kind"],
        rows=retrieval["rows"],
        readings={key: dict(value) for key, value in retrieval["readings"].items()},
        vocabulary=retrieval["vocabulary"],
        templates=retrieval["templates"],
        bins=retrieval["bins"],
        min_support=retrieval["min_support"],
        rules_version=retrieval["rules_version"],
        dataset_id=summary["id"],
        advice_id=None if advice is None else advice["advice_id"],
        usage=[] if advice is None else advice["usage"],
    )


def compose_system_prompt(
    summary: dict[str, Any] | None, retrieval: Retrieval | None, advice: Advice | None
) -> str:
    if summary is None:
        return SYSTEM_PROMPT
    schemas = [
        cast(dict[str, Any], schema_of(brief["algorithm"]))
        for brief in cards.instrument_briefs()
    ]
    prompt = (
        SYSTEM_PROMPT
        + render_task_section(summary, schemas, advice)
        + "\n"
        + render_pinned_glossary(summary["task_kind"])
        + "\nPinned input (use its id for all data tools):\n"
        + json.dumps(summary, ensure_ascii=False)
    )
    if retrieval is not None:
        prompt += (
            "\nRetrieved experience:\n"
            + json.dumps(retrieval["rows"], ensure_ascii=False)
            + "\n"
            + render_experience(summary, retrieval, advice)
        )
    return prompt


def build_model(
    config: EndpointConfig, *, client_class: type[Any] | None = None
) -> Any:
    from pydantic import SecretStr

    options: dict[str, Any] = {}
    preset = config.endpoint in ENDPOINT_PRESETS
    if preset:


        options.update(temperature=1.0, top_p=0.95)







    retries = 6 if preset else 2
    if config.protocol == "anthropic":
        from langchain_anthropic import ChatAnthropic







        if config.effort is not None:
            options["effort"] = config.effort
        client = ChatAnthropic if client_class is None else client_class
        return client(
            model_name=config.model,
            base_url=config.base_url,
            api_key=SecretStr(config.api_key),
            max_tokens_to_sample=16384,
            max_retries=retries,
            timeout=600,
            stop=None,
            **options,
        )

    from langchain_openai import ChatOpenAI

    if preset:
        import httpx




        options["http_client"] = httpx.Client(trust_env=False)


    if config.effort is not None:
        options["reasoning_effort"] = config.effort
    client = ChatOpenAI if client_class is None else client_class
    return client(
        model=config.model,
        base_url=config.base_url,
        api_key=SecretStr(config.api_key),
        max_completion_tokens=16384,
        max_retries=retries,
        timeout=600,
        **options,
    )


def install_skills(workspace: Path, *, experience_card: bool) -> None:
    root = workspace / SKILLS_DIR
    if root.exists():
        shutil.rmtree(root)
    texts = {
        brief["algorithm"]: cards.algorithm_card(brief)
        for brief in cards.instrument_briefs()
    }
    texts["kd-tools"] = cards.common_card()
    if experience_card:
        texts["experience"] = cards.experience_card()
    for name, text in texts.items():
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(text, encoding="utf-8")


def _answer_ends_run_middleware() -> Any:
    from langchain.agents.middleware import before_model

    @before_model(can_jump_to=["end"], name=ANSWER_ENDS_RUN)
    def end_once_the_answer_is_in(state: Any, _runtime: Any) -> dict[str, Any] | None:
        answer = state.get(ANSWER)
        if answer is None or answer["source"] != "submitted":
            return None
        return {"jump_to": "end"}

    return end_once_the_answer_is_in


def _answer_nudge_middleware() -> Any:
    from langchain.agents.middleware import after_model
    from langchain_core.messages import AIMessage, HumanMessage

    @after_model(can_jump_to=["model"], name="AnswerChannelNudge")
    def nudge_toward_the_answer_tool(
        state: Any, _runtime: Any
    ) -> dict[str, Any] | None:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        answer = state.get(ANSWER)
        if answer is not None and answer["source"] == "submitted":
            return None
        if model_signals.is_refusal(last.response_metadata):
            return None
        if is_truncated(last):







            if state[TRUNCATION_NUDGES] >= MAX_TRUNCATION_NUDGES:
                return None
            return {
                "messages": [HumanMessage(content=TRUNCATION_NUDGE)],
                TRUNCATION_NUDGES: 1,
                "jump_to": "model",
            }
        if state[ANSWER_NUDGES] >= MAX_ANSWER_NUDGES:
            return None
        return {
            "messages": [HumanMessage(content=ANSWER_NUDGE)],
            ANSWER_NUDGES: 1,
            "jump_to": "model",
        }

    return nudge_toward_the_answer_tool


GENERAL_PURPOSE_DESCRIPTION = (
    "General-purpose agent for reading files, running shell commands and "
    "multi-step research in an isolated context window. It does not have the "
    "KD verbs (list_datasets, run_discovery, train_surrogate, prune_node, "
    "select_node, submit_answer) "
    "— run a search, train a surrogate and submit the answer yourself."
)


SKILLS_PROMPT = """## Skills

Read the relevant skill before configuring a tool. Skills are available at:
{skills_locations}{skills_load_warnings}

{skills_list}"""

EXECUTE_DESCRIPTION = (
    "Run a shell command on this machine in the workspace directory, without a sandbox. "
    "The command can read and write host files and run installed programs. "
    "Use workspace-relative paths for files shared with the file tools."
)


def build_agent(
    model: Any,
    tools: list[Callable[..., Any]],
    workspace: Path,
    *,
    system_prompt: str,
) -> Any:
    from deepagents import SubAgent, create_deep_agent
    from deepagents.backends import LocalShellBackend
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
    from deepagents.middleware.skills import SkillsMiddleware
    from deepagents.profiles import HarnessProfile, register_harness_profile

    register_harness_profile(
        "openai",
        HarnessProfile(tool_description_overrides={"execute": EXECUTE_DESCRIPTION}),
    )
    register_harness_profile(
        "anthropic",
        HarnessProfile(
            tool_description_overrides={"execute": EXECUTE_DESCRIPTION},
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
        ),
    )
    backend = LocalShellBackend(root_dir=str(workspace), inherit_env=True)

    general_purpose: SubAgent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "description": GENERAL_PURPOSE_DESCRIPTION,
        "tools": [],
    }
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
        skills=[f"/{SKILLS_DIR}/"],





        middleware=[
            SkillsMiddleware(
                backend=backend,
                sources=[f"/{SKILLS_DIR}/"],
                system_prompt=SKILLS_PROMPT,
            ),
            _answer_ends_run_middleware(),
            _answer_nudge_middleware(),
        ],
        subagents=[general_purpose],
        state_schema=KdAgentState,
    )


def _flatten_run(run: Any) -> Iterator[Any]:
    yield run
    for child in run.child_runs:
        yield from _flatten_run(child)


def file_tracer(path: Path) -> Any:
    from langchain_core.tracers.base import BaseTracer

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    class _FileTracer(BaseTracer):
        name = "kdagent_file_tracer"

        def _persist_run(self, run: Any) -> None:
            with path.open("a", encoding="utf-8") as sink:
                for node in _flatten_run(run):
                    sink.write(
                        json.dumps(node.model_dump(), default=str, ensure_ascii=False)
                    )
                    sink.write("\n")

    return _FileTracer()


def run_loop(
    agent: Any,
    request: str,
    trace_path: Path,
    recursion_limit: int = 100,
    metadata: dict[str, Any] | None = None,
) -> RunLoopResult:
    from langgraph.errors import GraphRecursionError

    messages: list[Any] = []
    answer_nudges = 0
    truncation_nudges = 0
    answer: Answer | None = None
    error: Exception | None = None
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": request}]},
            {
                "recursion_limit": recursion_limit,
                "callbacks": [file_tracer(trace_path)],






                "max_concurrency": 1,
                "run_name": RUN_NAME,
                "metadata": metadata or {},
            },
            stream_mode="values",
        ):
            messages = chunk["messages"]
            answer_nudges = chunk[ANSWER_NUDGES]
            truncation_nudges = chunk[TRUNCATION_NUDGES]
            answer = chunk.get(ANSWER)
    except GraphRecursionError as exc:
        error = exc
    except Exception as exc:


        logger.exception("agent.stream was interrupted; returning what was captured")
        error = exc
    return RunLoopResult(
        messages=messages,
        stop_reason=classify_stop_reason(messages, answer, error),
        answer_nudges=answer_nudges,
        truncation_nudges=truncation_nudges,
        answer=answer,
    )
