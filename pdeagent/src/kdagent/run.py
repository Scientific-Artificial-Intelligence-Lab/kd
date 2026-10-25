
from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import kd

from kdagent.adapter import (
    make_tools,
    prime_action_experience,
    prime_experience,
    prime_prior,
    register_user_sketch,
)
from kdagent import advice as advice_document, fingerprint
from kdagent.config import (
    DEFAULT_RECURSION_LIMIT,
    EndpointConfig,
    Llm4edConfig,
    check_llm4ed_shell,
    config_path,
)
from kdagent.data_source import (
    is_derived,
    is_file_ref,
    resolve_input,
    root_ref,
    save_input,
    sealed_input_failure,
)
from kdagent.episode_worker import prepare_catalog, prepare_file
from kdagent.report import write_report
from kdagent.session import Session
from kdagent.state import ANSWER, ANSWER_NUDGES, TRUNCATION_NUDGES, RunLoopResult

_PUBLIC_ANSWER_SOURCE = {
    "submitted": "submitted",
    "selected": "fallback_selected",
    "abstained": "abstained",
}

_TRACE_PRIOR_KEYS = ("prior_type", "prior_sha256")

_TRACE_MEMORY_KEYS = ("memory_sha256",)

_TRACE_ADVICE_KEYS = ("advice_sha256",)

_EXPLICITLY_SELECTED = "selected"


_NO_ANSWER = {
    "answer": None,
    "answer_source": None,
    "answer_provenance": None,
    "answer_input": None,
    "answer_system": None,
}


def _answer_input(
    session: Session, provenance: dict[str, Any] | None
) -> dict[str, Any] | None:
    if provenance is None:
        return None
    producer = provenance.get("producer_ref", {"kind": "search"})
    if producer["kind"] == "edited":
        return None
    if producer["kind"] == "refinement":
        refinement_id = producer["refinement_id"]
        record = session.receipts[refinement_id]["evaluation_input_record"]
        return _input_columns(
            session, record["dataset_id"], record, f"refinement {refinement_id!r}"
        )
    run_id = provenance.get("run_id")
    record = None if run_id is None else session.search_inputs.get(run_id)
    return _input_columns(session, provenance["dataset"], record, f"search {run_id!r}")


def _input_columns(
    session: Session,
    dataset: str,
    record: dict[str, Any] | None,
    operation: str,
) -> dict[str, Any]:

    def unverified(failure: str) -> dict[str, Any]:
        return {
            "dataset_id": dataset,
            "derived_from": None,
            "root": None,
            "lowpass_modes": None,
            "time_window": None,
            "pinn": None,
            "verified": False,
            "verification_failure": failure,
        }

    if record is None:
        if is_derived(dataset):
            return unverified(
                f"{operation} has no record from this run of the input it ran on, "
                f"so the derived input {dataset} cannot be bound to it"
            )
        if is_file_ref(dataset):
            try:
                resolve_input(session.workspace, dataset)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                return unverified(f"{type(exc).__name__}: {exc}")
        derived = None
    else:
        if record["dataset_id"] != dataset:
            return unverified(
                f"{operation} was launched on {record['dataset_id']}, not on {dataset}"
            )
        failure = sealed_input_failure(
            session.workspace, record, written=session.derived_inputs
        )
        if failure is not None:
            return unverified(failure)
        derived = record["derived"]
    return {
        "dataset_id": dataset,
        "derived_from": None if derived is None else derived["from"],
        "root": root_ref(dataset),
        "lowpass_modes": None if derived is None else derived["lowpass_modes"],
        "time_window": None if derived is None else derived["time_window"],
        "pinn": None if derived is None else derived.get("pinn"),
        "verified": True,
        "verification_failure": None,
    }


def _answer_receipt(loop_result: RunLoopResult, session: Session) -> dict[str, Any]:
    answer = loop_result[ANSWER]
    if answer is None or answer["provenance"] is None:
        return {}
    producer = answer["provenance"].get("producer_ref", {"kind": "search"})
    if producer["kind"] != "refinement":
        return {}
    return {"answer_receipt": deepcopy(session.receipts[producer["refinement_id"]])}


def _answer_result(loop_result: RunLoopResult, session: Session) -> dict[str, Any]:
    answer = loop_result[ANSWER]
    if answer is None:
        return dict(_NO_ANSWER)
    source = answer["source"]
    if source == "abstained":
        return {**_NO_ANSWER, "answer_source": _PUBLIC_ANSWER_SOURCE[source]}
    system = answer.get("system")
    if source != "selected":
        return {
            "answer": answer["law"],
            "answer_source": _PUBLIC_ANSWER_SOURCE[source],
            "answer_provenance": answer["provenance"],
            "answer_input": _answer_input(session, answer["provenance"]),
            "answer_system": system,
        }
    provenance = answer["provenance"]
    if provenance is None:

        return {**_NO_ANSWER, "answer_selection": answer["selection"]}
    run_id = provenance["run_id"]
    if session.ledger.is_pruned(run_id):
        return dict(_NO_ANSWER)
    explicit = (
        session.ledger.selected_system == system["system_id"]
        if system is not None
        else session.ledger.selected == run_id
    )
    published = {
        "answer": answer["law"],
        "answer_source": (
            _EXPLICITLY_SELECTED if explicit else _PUBLIC_ANSWER_SOURCE[source]
        ),
        "answer_provenance": provenance,
        "answer_input": _answer_input(session, provenance),
        "answer_system": system,
    }
    return (
        published
        if explicit
        else {**published, "answer_selection": answer["selection"]}
    )


def _abstention(loop_result: RunLoopResult) -> dict[str, Any]:
    answer = loop_result[ANSWER]
    if answer is None or answer["source"] != "abstained":
        return {}
    return {
        "reason": answer["reason"],
        "evidence_run_ids": list(answer["evidence_run_ids"]),
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
    memory: Path | None = None,
    action_memory: Path | None = None,
    advice: Path | None = None,
    data: Path | None = None,
    load_options: dict[str, Any] | None = None,
    task_kind: str | None = None,
    target_state: str | None = None,
    bare: bool = False,
    shell: bool = False,
    device: str = "cpu",
    training_dtype: str | None = None,
    initial_sketch: dict[str, Any] | None = None,
    llm4ed: Llm4edConfig | None = None,
) -> dict[str, Any]:
    from kdagent import orchestration

    session, agent, metadata, record = prepare_run(
        workspace,
        config=config,
        dataset_id=dataset_id,
        recursion_limit=recursion_limit,
        time_budget_seconds=time_budget_seconds,
        prior=prior,
        memory=memory,
        action_memory=action_memory,
        advice=advice,
        data=data,
        load_options=load_options,
        task_kind=task_kind,
        target_state=target_state,
        bare=bare,
        shell=shell,
        device=device,
        training_dtype=training_dtype,
        initial_sketch=initial_sketch,
        llm4ed=llm4ed,
    )
    loop_result = orchestration.run_loop(
        agent,
        request,
        workspace / orchestration.TRACE_FILENAME,
        recursion_limit,
        metadata=metadata,
    )
    return finish_run(loop_result, session, config=config, record=record)


def finish_run(
    loop_result: RunLoopResult,
    session: Session,
    *,
    config: EndpointConfig,
    record: dict[str, Any],
) -> dict[str, Any]:
    from kdagent import orchestration



    result = {
        "messages": loop_result["messages"],
        "stop_reason": loop_result["stop_reason"],
        "answer_nudges": loop_result[ANSWER_NUDGES],
        "truncation_nudges": loop_result[TRUNCATION_NUDGES],
        **_answer_result(loop_result, session),
        **_answer_receipt(loop_result, session),
        **_abstention(loop_result),
        "endpoint": config.endpoint,
        "model": config.model,
    }
    if session.user_sketch_id is not None:
        result["sketch_sha256"] = session.sketches[session.user_sketch_id][
            "sketch_sha256"
        ]
    if session.sketches:
        result["sketches"] = {
            "active_sketch_id": session.active_sketch_id,
            "registered": deepcopy(list(session.sketches.values())),
        }
    if session.llm4ed is not None:
        result["llm4ed"] = session.llm4ed.public()
    result["report_path"] = str(
        write_report(
            session.workspace,
            result=result,
            ledger=session.ledger,
            trace_path=session.workspace / orchestration.TRACE_FILENAME,
            elapsed_seconds=session.budget()["elapsed_seconds"],
        )
    )

    return {**result, **record}


def prepare_run(
    workspace: Path,
    *,
    config: EndpointConfig,
    dataset_id: str | None,
    recursion_limit: int,
    time_budget_seconds: float | None,
    prior: Path | None = None,
    memory: Path | None = None,
    action_memory: Path | None = None,
    advice: Path | None = None,
    data: Path | None = None,
    load_options: dict[str, Any] | None = None,
    task_kind: str | None = None,
    target_state: str | None = None,
    bare: bool = False,
    shell: bool = False,
    device: str = "cpu",
    training_dtype: str | None = None,
    initial_sketch: dict[str, Any] | None = None,
    llm4ed: Llm4edConfig | None = None,
) -> tuple[Session, Any, dict[str, Any], dict[str, Any]]:
    from kdagent import orchestration

    _check_workspace(workspace)
    check_llm4ed_shell(llm4ed, shell=shell)
    _check_inputs(
        dataset_id,
        prior=prior,
        memory=memory,
        action_memory=action_memory,
        advice=advice,
        data=data,
    )
    if bare and any(
        document is not None for document in (prior, memory, action_memory, advice)
    ):
        raise ValueError(
            "a bare run takes no prior, memory, action memory or advice document"
        )
    if data is None:
        for name, value in (
            ("load_options", load_options),
            ("task_kind", task_kind),
            ("target_state", target_state),
        ):
            if value is not None:
                raise ValueError(f"{name} requires data")
    workspace.mkdir(parents=True, exist_ok=True)
    if data is not None:
        description = prepare_file(
            data,
            load_options,
            task_kind=task_kind,
            target_state=target_state,
            time_cap_seconds=time_budget_seconds,
        )
        save_input(workspace, description)
        dataset_id = description["ref"]
    session = Session(
        workspace,
        dataset_id=dataset_id,
        time_budget_seconds=time_budget_seconds,
        recursion_limit=recursion_limit,
        device=device,
        training_dtype=training_dtype,
        llm4ed=llm4ed,
    )
    precision_record = (
        {} if training_dtype is None else {"training_dtype": training_dtype}
    )
    sketch_record = (
        {}
        if initial_sketch is None
        else {
            "sketch_sha256": register_user_sketch(session, initial_sketch)[
                "sketch_sha256"
            ]
        }
    )
    prior_record = {} if prior is None else prime_prior(session, prior)
    memory_record, retrieval = (
        ({}, None) if memory is None else prime_experience(session, memory)
    )
    action_record = (
        {} if action_memory is None else prime_action_experience(session, action_memory)
    )
    advice_record, advice_content = (
        ({}, None) if advice is None else advice_document.prime(advice, workspace)
    )
    summary = None
    if session.input_source is not None:
        summary = session.input_source["summary"]
    elif dataset_id is not None:
        summary = prepare_catalog(dataset_id, time_cap_seconds=time_budget_seconds)
    system_prompt = orchestration.compose_system_prompt(
        summary,
        retrieval,
        advice_content,
        bare=bare,
        shell=shell,
        llm4ed=llm4ed is not None,
    )
    if not bare:
        orchestration.install_skills(
            workspace, experience_card=retrieval is not None, shell=shell
        )
    model = orchestration.build_model(config)
    agent = orchestration.build_agent(
        model,
        make_tools(session),
        workspace,
        system_prompt=system_prompt,
        bare=bare,
        shell=shell,
    )
    llm4ed_record = {} if llm4ed is None else {"llm4ed": llm4ed.public()}
    metadata = {
        "endpoint": config.endpoint,
        "model": config.model,
        "dataset_id": dataset_id,
        "recursion_limit": recursion_limit,
        "bare": bare,
        "shell": shell,
        "device": device,
        **precision_record,

        **{key: prior_record[key] for key in _TRACE_PRIOR_KEYS if prior is not None},
        **{key: memory_record[key] for key in _TRACE_MEMORY_KEYS if memory is not None},
        **{key: advice_record[key] for key in _TRACE_ADVICE_KEYS if advice is not None},
        **(
            {}
            if action_memory is None
            else {"action_memory_sha256": action_record["action_memory_sha256"]}
        ),
        **sketch_record,
        **llm4ed_record,
    }
    record = {
        **prior_record,
        **memory_record,
        **advice_record,
        **action_record,
        **sketch_record,
        "device": device,
        **precision_record,
        "prompt_sha256": fingerprint.compute(
            config, bare=bare, shell=shell, llm4ed=llm4ed
        )["prompt_sha256"],
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "task_kind": None if summary is None else summary["task_kind"],
        "target": None if summary is None else summary["target"],
    }
    return session, agent, metadata, record


def _check_workspace(workspace: Path) -> None:
    root = workspace.resolve()
    credentials = config_path().resolve()
    if credentials.is_relative_to(root):
        raise ValueError(
            f"workspace {workspace} holds the saved kd-agent configuration "
            f"{credentials}, which the model's file tools would then read; "
            "choose a workspace outside it"
        )
    source = Path(kd.__file__).resolve().parent
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError(
            f"workspace {workspace} overlaps the installed kd package {source}, "
            "whose source the model's file tools would then read; choose a "
            "workspace outside it"
        )


def _check_inputs(
    dataset_id: str | None,
    *,
    prior: Path | None,
    memory: Path | None,
    action_memory: Path | None = None,
    advice: Path | None,
    data: Path | None,
) -> None:
    if data is not None and dataset_id is not None:
        raise ValueError("data and dataset_id are mutually exclusive")
    if data is not None and prior is not None:
        raise ValueError("prior supports catalog datasets only, not file inputs")
    for name, document in (
        ("memory", memory),
        ("action_memory", action_memory),
        ("advice", advice),
    ):
        if (
            document is not None
            and data is None
            and (dataset_id is None or not is_file_ref(dataset_id))
        ):
            raise ValueError(
                f"{name} supports file inputs only: pass data or a dataset_id naming "
                "a saved input"
            )
