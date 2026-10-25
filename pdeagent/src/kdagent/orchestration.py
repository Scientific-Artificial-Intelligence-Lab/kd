
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
    TERMINAL_SOURCES,
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
    "done, finish by calling submit_answer, or finish_without_answer when your "
    "evidence supports no law; an equation written as text is not an answer "
    "this run can record."
)

MAX_ANSWER_NUDGES = 2

TRUNCATION_NUDGE = (
    "Your last reply was cut off at the output limit before it reached a tool "
    "call, so nothing ran and nothing was recorded. Keep the reasoning short and "
    "go straight to the call: run_discovery to keep searching, submit_answer or "
    "finish_without_answer to finish."
)

MAX_TRUNCATION_NUDGES = 2

ANSWER_ENDS_RUN = "SubmittedAnswerEndsRun"

STOP_ANSWER_SUBMITTED = "answer_submitted"

STOP_NOT_CONVERGED = "not_converged"

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
    if answer is not None and answer["source"] == "abstained":
        return STOP_NOT_CONVERGED
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
publish a search you have already run, to refit a search's law with its terms
held fixed or score a variant term set without a new search
(`refine_candidate`), to smooth a noisy input (`smooth_input`), to measure a
law in a declared frame (`ablate_candidate`, `score_term_sets`,
`verify_candidate`), to integrate one law per state of raw ODE trajectories
together (`verify_system`), to count the terms that recur across searches
(`vote_terms`) and narrow a search to them (`make_sketch`), to make the
structure the researcher states the constraint of every later search
(`set_sketch`), and to submit your answer or end the run without one
(`finish_without_answer`), plus a shell and
a Python interpreter. When your searches disagree, `prune_node` gives up on
a branch that has gone dead and `select_node` names the search (or the ODE
system) to publish; without a submission, what you selected is what this run
ends with.

With neither, the run ends with the default selection, which
`primary.now_selected` names after every search (null when there is none).
After each search KD integrates the sealed law, with its own coefficients and
left-hand side, forward from the first time slice of the input the search
ran on, in a worker of at most 30 s (10 s of integration) that is charged to
your time budget, and reports it as `diagnostics.selection_forward`. Searches
of one algorithm are compared on that algorithm's own score. Across
algorithms the default selection takes the fewest terms among the laws whose
forward error is at most twice the smallest; a search with no forward value
(a timeout, a divergence, a refusal, no time left) is left out rather than
ranked last, and when none has a value there is no default answer. Laws with
different left-hand sides are not ranked against each other.

On raw ODE trajectories (an input whose summary has a `projection`), each
search fits one state's time derivative and `run_discovery(target_state=...)`
searches another state of the same trajectories. `verify_system` takes one
such search per state and integrates their sealed laws together, forward from
each trajectory's first observation, and compares the result with the
observations. Once a system of these trajectories has a result (a timeout or
a failed worker is none), the systems decide the default answer: among the
systems whose error is at most twice the smallest, the one with the fewest
terms over all states, and its law for this run's target is published with
the system; when none of them has a finite error there is no default answer.
Before that, the default answer comes from the searches of this run's target;
a projection's searches have no forward value of their own, so one
algorithm's searches are compared on its own score and several algorithms'
by the same fewest-terms rule on `platform_nmse`.

`verify_candidate` checks a law three ways on one input: whether each term can
be evaluated, how much the coefficients change when refitted on separate grid
blocks, and how far integrating the law from the first time slice ends from
the data. The default selection does not read its numbers; use them to submit
or select. In 58 stored benchmark cases where the final answers of fixed
pipelines disagreed (seven problems at relative noise up to 5e-2, every
disagreement at noise 0 to 1e-2; terms with |coefficient| at most 1e-4 of the
largest dropped first), the fewest terms within twice the smallest forward
error was the right answer 49 times, the smallest block dispersion 38 times
and the same fewest-terms rule on `platform_nmse` 32 times.

The Python interpreter can `import kd`: `kd.validate_terms(dataset, terms)`,
`kd.block_dispersion(dataset, terms)` and
`kd.forward_error(dataset, terms, coefficients)` are the same three checks.
The kd-tools skill gives a loading line for grid file inputs; these three
functions take that loaded dataset.
`forward_error(..., max_seconds=10)` stops at a cooperative deadline and
returns `reason="timeout"` (10 s per candidate was used there); a single long
solver step is not interrupted, so keep `timeout` as the hard limit.

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
before you move on to something else. When a search ends with a poor fit,
compare legal continuation, restart, parameter changes and switching against
current feedback and remaining budget. Historical action evidence, when
present, reports observations and costs rather than a command to follow.
The applicability table names each instrument's budget lever.

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

Finish by calling submit_answer. Prose is not an answer. When your searches
and checks do not support any law, finish by calling finish_without_answer
with the reason and the searches it rests on: the run then records an
unresolved result instead of a law.
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


_SHELL_SENTENCES = (
    (
        "(`finish_without_answer`), plus a shell and\n"
        "a Python interpreter. When your searches disagree,",
        "(`finish_without_answer`). When your searches disagree,",
    ),
    (
        "and the same fewest-terms rule on `platform_nmse` 32 times.\n\n"
        "The Python interpreter can `import kd`: `kd.validate_terms(dataset, terms)`,\n"
        "`kd.block_dispersion(dataset, terms)` and\n"
        "`kd.forward_error(dataset, terms, coefficients)` are the same three checks.\n"
        "The kd-tools skill gives a loading line for grid file inputs; these three\n"
        "functions take that loaded dataset.\n"
        "`forward_error(..., max_seconds=10)` stops at a cooperative deadline and\n"
        'returns `reason="timeout"` (10 s per candidate was used there); a single long\n'
        "solver step is not interrupted, so keep `timeout` as the hard limit.\n\n",
        "and the same fewest-terms rule on `platform_nmse` 32 times; `verify_candidate`\n"
        "stops the forward integration at its `max_seconds` and reports\n"
        '`reason="timeout"`, which is not measured.\n\n',
    ),
    (
        "The file tools are rooted at the workspace, so `/` is its top level; `execute` runs\n"
        "in the same directory but `pwd` prints the host path, which the file tools do\n"
        "not accept.\n",
        "The file tools are rooted at the workspace, so `/` is its top level.\n",
    ),
    (
        "result summarizes that segment; the shell and Python interpreter let you inspect its sealed files.",
        "result summarizes that segment; the file tools let you inspect its sealed files.",
    ),
)


def _without_shell(prompt: str) -> str:
    for old, new in _SHELL_SENTENCES:
        if prompt.count(old) != 1:
            raise ValueError(
                f"shell sentence not found exactly once in the prompt: {old[:50]!r}"
            )
        prompt = prompt.replace(old, new)
    return prompt


SYSTEM_PROMPT_NO_SHELL = _without_shell(SYSTEM_PROMPT)

BARE_PROMPT = "Find the equation that governs the data `{input_id}`."

BARE_PROMPT_UNPINNED = "Find the equation that governs the data you are given."


def compose_system_prompt(
    summary: dict[str, Any] | None,
    retrieval: Retrieval | None,
    advice: Advice | None,
    *,
    bare: bool = False,
    shell: bool = False,
    llm4ed: bool = False,
) -> str:
    if bare:
        if retrieval is not None or advice is not None:
            raise ValueError("a bare prompt takes no experience or advice document")
        if summary is None:
            return BARE_PROMPT_UNPINNED
        return BARE_PROMPT.format(input_id=summary["id"])
    base = SYSTEM_PROMPT if shell else SYSTEM_PROMPT_NO_SHELL
    if summary is None:
        return base
    schemas = [
        cast(dict[str, Any], schema_of(brief["algorithm"]))
        for brief in cards.instrument_briefs()
    ]
    prompt = (
        base
        + render_task_section(summary, schemas, advice, llm4ed=llm4ed)
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


def install_skills(
    workspace: Path, *, experience_card: bool, shell: bool = False
) -> None:
    root = workspace / SKILLS_DIR
    if root.exists():
        shutil.rmtree(root)
    texts = {
        brief["algorithm"]: cards.algorithm_card(brief)
        for brief in cards.instrument_briefs()
    }
    texts["kd-tools"] = cards.common_card(shell=shell)
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
        if answer is None or answer["source"] not in TERMINAL_SOURCES:
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
        if answer is not None and answer["source"] in TERMINAL_SOURCES:
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
    "select_node, submit_answer, refine_candidate, smooth_input, "
    "ablate_candidate, score_term_sets, verify_candidate, verify_system, "
    "vote_terms, make_sketch, set_sketch, finish_without_answer); run a "
    "search, train a "
    "surrogate, measure and submit the answer yourself."
)

GENERAL_PURPOSE_DESCRIPTION_NO_SHELL = GENERAL_PURPOSE_DESCRIPTION.replace(
    "reading files, running shell commands and ", "reading files and "
)
assert GENERAL_PURPOSE_DESCRIPTION_NO_SHELL != GENERAL_PURPOSE_DESCRIPTION


SKILLS_PROMPT = """## Skills

Read the relevant skill before configuring a tool. Skills are available at:
{skills_locations}{skills_load_warnings}

{skills_list}"""

RUN_ARTIFACT_ROOTS = ("runs", "surrogates", "inputs")

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
    bare: bool = False,
    shell: bool = False,
) -> Any:
    from deepagents import FilesystemPermission, SubAgent, create_deep_agent
    from deepagents.backends import FilesystemBackend, LocalShellBackend
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
    backend = (
        LocalShellBackend(root_dir=str(workspace), inherit_env=True)
        if shell
        else FilesystemBackend(root_dir=str(workspace))
    )

    general_purpose: SubAgent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "description": GENERAL_PURPOSE_DESCRIPTION
        if shell
        else GENERAL_PURPOSE_DESCRIPTION_NO_SHELL,
        "tools": [],
    }





    middleware: list[Any] = [_answer_ends_run_middleware(), _answer_nudge_middleware()]
    if not bare:
        middleware.insert(
            0,
            SkillsMiddleware(
                backend=backend,
                sources=[f"/{SKILLS_DIR}/"],
                system_prompt=SKILLS_PROMPT,
            ),
        )
    permissions = (
        None
        if shell
        else [
            FilesystemPermission(
                operations=["write"],
                paths=[
                    path
                    for root in RUN_ARTIFACT_ROOTS
                    for path in (f"/{root}", f"/{root}/**")
                ],
                mode="deny",
            )
        ]
    )
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
        skills=None if bare else [f"/{SKILLS_DIR}/"],
        middleware=middleware,
        subagents=[general_purpose],
        state_schema=KdAgentState,
        permissions=permissions,
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
