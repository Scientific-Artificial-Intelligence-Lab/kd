
from __future__ import annotations

from pathlib import Path

import pytest

from kd.llm import openai_compat as oc_mod
from kd.llm.budget import BudgetedProvider
from kd.llm.openai_compat import OpenAICompatProvider
from kd.llm.protocol import LLMBudgetExhausted, LLMProvider
from kd.llm.tape import TapeRecordingProvider, TapeReplayProvider
from tests.unit.llm._fakes import FakeProvider, make_request


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


class TestCanonicalChain:
    def test_chain_conforms_to_protocol(self, tmp_path: Path) -> None:
        chain = BudgetedProvider(
            TapeRecordingProvider(FakeProvider(), path=tmp_path / "t.jsonl"),
            max_calls=2,
        )
        assert isinstance(chain, LLMProvider)

    def test_budget_rejected_call_is_not_taped(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        chain = BudgetedProvider(
            TapeRecordingProvider(FakeProvider(), path=path), max_calls=2
        )
        chain.complete(make_request(prompt="p", seed=0))
        chain.complete(make_request(prompt="p", seed=1))
        with pytest.raises(LLMBudgetExhausted):
            chain.complete(make_request(prompt="p", seed=2))
        assert _line_count(path) == 2

    def test_chain_responses_match_bare_inner(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        chain = BudgetedProvider(
            TapeRecordingProvider(FakeProvider(), path=path), max_calls=3
        )
        request = make_request(prompt="chained", seed=5, temperature=0.4)
        through_chain = chain.complete(request)
        bare = FakeProvider().complete(request)
        assert through_chain == bare

    def test_recorded_chain_replays_identically(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        chain = BudgetedProvider(
            TapeRecordingProvider(FakeProvider(), path=path), max_calls=3
        )
        requests = [make_request(prompt="p", seed=s) for s in range(3)]
        recorded = [chain.complete(req) for req in requests]

        replay = TapeReplayProvider(path=path)
        replayed = [replay.complete(req) for req in requests]
        assert replayed == recorded

    def test_chain_prepare_propagates_innermost_fail_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        def _raise_missing_sdk() -> object:
            raise ImportError(oc_mod._MISSING_SDK_MESSAGE)

        monkeypatch.setattr(oc_mod, "_import_sdk", _raise_missing_sdk)
        inner = OpenAICompatProvider(model="m", base_url="http://x/v1")
        chain = BudgetedProvider(
            TapeRecordingProvider(inner, path=tmp_path / "t.jsonl"),
            max_calls=2,
        )
        with pytest.raises(ImportError, match="llm4ed"):
            chain.prepare()
