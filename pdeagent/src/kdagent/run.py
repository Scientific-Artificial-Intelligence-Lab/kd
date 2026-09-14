
from __future__ import annotations

from pathlib import Path
from typing import Any

from kdagent.adapter import make_tools, prime_prior
from kdagent.config import DEFAULT_RECURSION_LIMIT, EndpointConfig
from kdagent.data_source import save_input
from kdagent.episode_worker import prepare_file
from kdagent.report import write_report
from kdagent.session import Session
from kdagent.state import ANSWER, ANSWER_NUDGES, TRUNCATION_NUDGES, RunLoopResult

_PUBLIC_ANSWER_SOURCE = {"submitted": "submitted", "selected": "fallback_selected"}

_TRACE_PRIOR_KEYS = ("prior_type", "prior_sha256")

_EXPLICITLY_SELECTED = "selected"


def _answer_result(loop_result: RunLoopResult, session: Session) -> dict[str, Any]:
    answer = loop_result[ANSWER]
    if answer is None:
        return {"answer": None, "answer_source": None, "answer_provenance": None}
    source = answer["source"]
    if source == "selected":
        provenance = answer["provenance"]
        run_id = None if provenance is None else provenance.get("run_id")
        if run_id is not None and session.ledger.is_pruned(run_id):
            return {"answer": None, "answer_source": None, "answer_provenance": None}
        if run_id is not None and session.ledger.selected == run_id:
            return {
                "answer": answer["law"],
                "answer_source": _EXPLICITLY_SELECTED,
                "answer_provenance": provenance,
            }
    return {
        "answer": answer["law"],
        "answer_source": _PUBLIC_ANSWER_SOURCE[source],
        "answer_provenance": answer["provenance"],
    }


def run(
    request: str,
    workspace: Path,
    dataset_id: str | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    time_budget_seconds: float | None = None,
    *,
    config: EndpointConfig,
    prior: Path | None = None,
    data: Path | None = None,
    load_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from kdagent import orchestration

    session, agent, metadata, prior_record = prepare_run(
        workspace,
        config=config,
        dataset_id=dataset_id,
        recursion_limit=recursion_limit,
        time_budget_seconds=time_budget_seconds,
        prior=prior,
        data=data,
        load_options=load_options,
    )
    loop_result = orchestration.run_loop(
        agent,
        request,
        workspace / orchestration.TRACE_FILENAME,
        recursion_limit,
        metadata=metadata,
    )
    result = finish_run(loop_result, session, config=config)

    result.update(prior_record)
    return result


def finish_run(
    loop_result: RunLoopResult,
    session: Session,
    *,
    config: EndpointConfig,
) -> dict[str, Any]:
    from kdagent import orchestration



    result = {
        "messages": loop_result["messages"],
        "stop_reason": loop_result["stop_reason"],
        "answer_nudges": loop_result[ANSWER_NUDGES],
        "truncation_nudges": loop_result[TRUNCATION_NUDGES],
        **_answer_result(loop_result, session),
        "endpoint": config.endpoint,
        "model": config.model,
    }
    result["report_path"] = str(
        write_report(
            session.workspace,
            result=result,
            ledger=session.ledger,
            trace_path=session.workspace / orchestration.TRACE_FILENAME,
            elapsed_seconds=session.budget()["elapsed_seconds"],
        )
    )
    return result


def prepare_run(
    workspace: Path,
    *,
    config: EndpointConfig,
    dataset_id: str | None,
    recursion_limit: int,
    time_budget_seconds: float | None,
    prior: Path | None = None,
    data: Path | None = None,
    load_options: dict[str, Any] | None = None,
) -> tuple[Session, Any, dict[str, Any], dict[str, Any]]:
    from kdagent import orchestration

    if data is not None and dataset_id is not None:
        raise ValueError("data and dataset_id are mutually exclusive")
    if load_options is not None and data is None:
        raise ValueError("load_options requires data")
    if data is not None and prior is not None:
        raise ValueError("prior supports catalog datasets only, not file inputs")
    workspace.mkdir(parents=True, exist_ok=True)
    if data is not None:
        description = prepare_file(
            data, load_options, time_cap_seconds=time_budget_seconds
        )
        save_input(workspace, description)
        dataset_id = description["ref"]
    session = Session(
        workspace,
        dataset_id=dataset_id,
        time_budget_seconds=time_budget_seconds,
        recursion_limit=recursion_limit,
    )
    prior_record = {} if prior is None else prime_prior(session, prior)
    orchestration.install_skills(workspace)
    model = orchestration.build_model(config)
    agent = orchestration.build_agent(
        model,
        make_tools(session),
        workspace,
        data_summary=None
        if session.input_source is None
        else session.input_source["summary"],
    )
    metadata = {
        "endpoint": config.endpoint,
        "model": config.model,
        "dataset_id": dataset_id,
        "recursion_limit": recursion_limit,

        **{key: prior_record[key] for key in _TRACE_PRIOR_KEYS if prior is not None},
    }
    return session, agent, metadata, prior_record
