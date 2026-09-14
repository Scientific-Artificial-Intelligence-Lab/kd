
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from kdagent import orchestration
from kdagent.config import DEFAULT_RECURSION_LIMIT, EndpointConfig
from kdagent.run import finish_run, prepare_run
from kdagent.state import (
    ANSWER,
    ANSWER_NUDGES,
    TRUNCATION_NUDGES,
    Answer,
    RunLoopResult,
)


def _write(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def _preview(value: Any, limit: int = 200) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _show(message: Any) -> None:
    if type(message).__name__ == "HumanMessage":
        return
    for call in getattr(message, "tool_calls", None) or []:
        _write(f" -> {call['name']}({_preview(call.get('args', ''), 160)})")
    text = _message_text(getattr(message, "content", None))
    if text.strip():
        _write(f" {text.strip()}")
    elif not getattr(message, "tool_calls", None):
        raw = getattr(message, "content", "")
        if raw:
            _write(f" <- {_preview(raw)}")


def chat(
    workspace: Path,
    dataset_id: str | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    time_budget_seconds: float | None = None,
    *,
    config: EndpointConfig,
    data: Path | None = None,
    load_options: dict[str, Any] | None = None,
) -> None:
    session, agent, metadata, _prior_record = prepare_run(
        workspace,
        config=config,
        dataset_id=dataset_id,
        recursion_limit=recursion_limit,
        time_budget_seconds=time_budget_seconds,
        data=data,
        load_options=load_options,
    )



    tracer = orchestration.file_tracer(workspace / orchestration.TRACE_FILENAME)
    captured = RunLoopResult(
        messages=[],
        answer=None,
        answer_nudges=0,
        truncation_nudges=0,
        stop_reason="chat_ready",
    )
    finish_run(captured, session, config=config)

    _write(
        f"kdagent | endpoint={config.endpoint} model={config.model} "
        f"dataset={metadata['dataset_id'] or 'unpinned (the agent lists datasets itself)'}"
    )
    _write(f"workspace={workspace} | /exit quits, /reset clears the conversation")
    _write(f"report={workspace / 'report.md'}")

    history: list[Any] = []
    printed = 0


    latest_answer: Answer | None = None
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            _write()
            return
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return
        if line == "/reset":
            history.clear()
            printed = 0
            latest_answer = None
            captured.update(
                {"messages": [], "answer": None, "stop_reason": "conversation_reset"}
            )
            finish_run(captured, session, config=config)
            _write("conversation cleared (the pinned dataset and run ledger stay)")
            continue
        history.append({"role": "user", "content": line})
        captured = RunLoopResult(
            messages=history,
            answer=None,
            answer_nudges=0,
            truncation_nudges=0,
            stop_reason="model_finished",
        )
        error: Exception | None = None
        try:
            for chunk in agent.stream(
                {"messages": history},
                {
                    "recursion_limit": recursion_limit,
                    "callbacks": [tracer],
                    "max_concurrency": 1,
                    "run_name": "kdagent-chat",
                    "metadata": metadata,
                },
                stream_mode="values",
            ):
                history = chunk["messages"]
                captured.update(
                    {
                        "messages": history,
                        "answer": chunk.get(ANSWER),
                        "answer_nudges": chunk[ANSWER_NUDGES],
                        "truncation_nudges": chunk[TRUNCATION_NUDGES],
                    }
                )
                while printed < len(history):
                    _show(history[printed])
                    printed += 1
        except Exception as exc:


            error = exc
        answer = captured[ANSWER]
        captured["stop_reason"] = orchestration.classify_stop_reason(
            history, answer, error
        )
        if answer is None:
            captured[ANSWER] = latest_answer
        else:
            latest_answer = answer
        if captured["stop_reason"] == "recursion_limit":
            _write(" !! recursion limit reached; the turn was cut short")
        elif error is not None:
            _write(f" !! turn crashed: {type(error).__name__}: {error}")
        if answer is not None and answer["source"] == "submitted":
            _write(f"=== answer submitted: {answer}")
        finish_run(captured, session, config=config)
