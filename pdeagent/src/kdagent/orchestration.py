
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from kdagent.adapter import (
    DEFAULT_GENERATIONS,
    DEFAULT_SEED,
    instrument_briefs,
    run_record_layout,
)
from kdagent.config import ENDPOINT_PRESETS, EndpointConfig
from kdagent.instrument_doc import instrument_doc
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

    return (
        isinstance(message, AIMessage)
        and message.response_metadata.get("finish_reason") == "length"
    )


def classify_stop_reason(
    messages: list[Any], answer: Answer | None, error: Exception | None
) -> str:
    from langgraph.errors import GraphRecursionError

    if isinstance(error, GraphRecursionError):
        return "recursion_limit"
    if error is not None:
        return f"crashed:{type(error).__name__}"
    if answer is not None and answer["source"] == "submitted":
        return STOP_ANSWER_SUBMITTED
    if messages and is_truncated(messages[-1]):
        return STOP_TRUNCATION_CAP
    return "model_finished"


_RECORD = run_record_layout()
SYSTEM_PROMPT = f"""You drive KD, a platform that discovers PDEs from data.

Work out which equation governs the data you are given. You have tools to list
the datasets, to run a search, to train a derivative surrogate, to abandon or to
publish a search you have already run, and to submit your answer, plus a shell
and a Python interpreter. When your searches disagree, `prune_node` gives up on
a branch that has gone dead and `select_node` names the search to publish;
without a submission, the search you selected is what this run ends with. An instrument that fits a
surrogate before it searches (its skill card has a "Surrogate" section) can
have that surrogate trained once with `train_surrogate` and injected into every
search of a lineage with `run_discovery(surrogate=...)`, instead of retraining
it inside each search.

Your workspace holds the skills and the records of your own searches.
`list_datasets` returns the usable IDs and data summaries. Pass the ID unchanged
to both search and surrogate training. File inputs have content references and
descriptions under `inputs/`; KD loads their original files in the worker.
Changed bytes or loading options require preparing a new input reference.
The file tools are rooted at the workspace, so `/` is its top level; `execute` runs
in the same directory but `pwd` prints the host path, which the file tools do
not accept.

Read the skill for an algorithm before you configure it.

When a tool fails and names what to change, apply that change and call it again
before you move on to something else. When a search finishes but fits the data
poorly, raise how long it is allowed to search and run it again before you
switch algorithms.

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
with the latest `created_at`. Nothing summarises any of this for you; the shell
and the Python interpreter are how you read it.

Finish by calling submit_answer. Prose is not an answer.
"""


def build_model(config: EndpointConfig) -> Any:
    from pydantic import SecretStr

    options: dict[str, Any] = {}
    preset = config.endpoint in ENDPOINT_PRESETS
    if preset:


        options.update(temperature=1.0, top_p=0.95)







    retries = 6 if preset else 2
    if config.protocol == "anthropic":
        from langchain_anthropic import ChatAnthropic







        return ChatAnthropic(
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


    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=SecretStr(config.api_key),
        max_completion_tokens=16384,
        max_retries=retries,
        timeout=600,
        **options,
    )


PARAMETERS_HEADING = "## Parameters"
NARROWING_HEADING = "## Narrowing with a sketch"
SURROGATE_HEADING = "## Surrogate"
_PARAMETERS_PREAMBLE = (
    "Generated from kd's instrument schema when this skill was installed; the "
    "card prose above is hand-written, this table is not. Pass these as the "
    "`params` mapping of `run_discovery`. `seed` is a separate argument of "
    "that tool, not a `params` key.\n\n"
    "`Resume` is kd's classification of changing that parameter: `resume_safe` "
    "can take effect on a search restored from a checkpoint, `init_only` "
    "cannot, and `identity_breaking` makes it a different run rather than the "
    "same one continued. Blank means kd classifies that name nowhere."
)
_RESUMABLE_ARCHIVE = "progress"
_PROGRESS_RESUME_SENTENCE = (
    " Use `resume_from_run_id` without `resume_iteration` to continue from the "
    "latest checkpoint, or add `resume_iteration` to roll back and fork; kd's "
    "tier gate names any parameter it refuses."
)
_CONCLUSION_RESUME_SENTENCE = (
    " This instrument's checkpoint archive carries a finished conclusion rather "
    "than search state, so `run_discovery` refuses `resume_from_run_id` "
    "pointing at one of its runs; run it again with different parameters "
    "instead."
)


def _parameters_preamble(segmentation_archive: str) -> str:
    sentence = (
        _PROGRESS_RESUME_SENTENCE
        if segmentation_archive == _RESUMABLE_ARCHIVE
        else _CONCLUSION_RESUME_SENTENCE
    )
    return f"{_PARAMETERS_PREAMBLE}{sentence}"




_ASSETS_DIR = Path(__file__).resolve().parent / "_assets"


def install_skills(workspace: Path) -> None:
    run_guidance = (
        (_ASSETS_DIR / "run-guidance.md")
        .read_text(encoding="utf-8")
        .format(DEFAULT_GENERATIONS=DEFAULT_GENERATIONS, DEFAULT_SEED=DEFAULT_SEED)
    )
    for brief in instrument_briefs():
        algorithm = brief["algorithm"]
        instrument_guidance = (_ASSETS_DIR / "guidance" / f"{algorithm}.md").read_text(
            encoding="utf-8"
        )
        skill_dir = workspace / SKILLS_DIR / algorithm
        skill_dir.mkdir(parents=True, exist_ok=True)
        surrogate = (
            f"\n{SURROGATE_HEADING}\n\n{brief['surrogate']}\n"
            if brief["surrogate"]
            else ""
        )
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {algorithm}\n"
            f"description: {brief['summary']} "
            f"Runs synchronously; kd rates its cost {brief['cost_class']}.\n---\n\n"
            f"## Capabilities\n\n{brief['capabilities']}\n\n"
            f"{instrument_doc(algorithm)}\n\n"
            f"{run_guidance}\n\n{instrument_guidance}\n\n"
            f"{PARAMETERS_HEADING}\n\n"
            f"{_parameters_preamble(brief['segmentation_archive'])}\n\n"
            f"{brief['parameters']}\n\n"
            f"{NARROWING_HEADING}\n\n"
            f"{brief['narrowing']}\n"
            f"{surrogate}",
            encoding="utf-8",
        )


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


def build_agent(
    model: Any,
    tools: list[Callable[..., Any]],
    workspace: Path,
    *,
    data_summary: dict[str, Any] | None = None,
) -> Any:
    from deepagents import SubAgent, create_deep_agent
    from deepagents.backends import LocalShellBackend
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

    general_purpose: SubAgent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "description": GENERAL_PURPOSE_DESCRIPTION,
        "tools": [],
    }
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
        + (
            ""
            if data_summary is None
            else "\nPinned input (use its id for all data tools):\n"
            + json.dumps(data_summary, ensure_ascii=False)
        ),
        backend=LocalShellBackend(root_dir=str(workspace), inherit_env=True),
        skills=[f"/{SKILLS_DIR}/"],





        middleware=[_answer_ends_run_middleware(), _answer_nudge_middleware()],
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
