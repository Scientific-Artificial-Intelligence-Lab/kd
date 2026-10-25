
from __future__ import annotations

import operator
from typing import Annotated, Any, Final, Literal, NotRequired, TypedDict

from deepagents import DeepAgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import BaseMessage

ANSWER: Final = "answer"
ANSWER_NUDGES: Final = "answer_nudges"
TRUNCATION_NUDGES: Final = "truncation_nudges"


class Answer(TypedDict):

    law: dict[str, Any] | None
    source: Literal["submitted", "selected", "abstained"]
    provenance: dict[str, Any] | None
    reason: NotRequired[str]
    evidence_run_ids: NotRequired[list[str]]
    selection: NotRequired[dict[str, Any]]
    system: NotRequired[dict[str, Any]]


TERMINAL_SOURCES: Final = frozenset({"submitted", "abstained"})

_ANSWER_RANK: Final = {"selected": 0, **dict.fromkeys(TERMINAL_SOURCES, 1)}


def _answer_precedence(current: Answer | None, incoming: Answer) -> Answer:
    if current is None:
        return incoming
    if _ANSWER_RANK[incoming["source"]] >= _ANSWER_RANK[current["source"]]:
        return incoming
    return current


class KdAgentState(DeepAgentState):

    answer: Annotated[Answer | None, PrivateStateAttr, _answer_precedence]
    answer_nudges: Annotated[int, PrivateStateAttr, operator.add]
    truncation_nudges: Annotated[int, PrivateStateAttr, operator.add]


class RunLoopResult(TypedDict):

    messages: list[BaseMessage]
    stop_reason: str
    answer: Answer | None
    answer_nudges: int
    truncation_nudges: int
