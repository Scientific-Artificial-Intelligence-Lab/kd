
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kd.llm.protocol import LLMTapeMismatchError
from kd.llm.tape import TapeRecordingProvider, TapeReplayProvider, request_hash
from tests.unit.llm._fakes import FakeProvider, make_request


def _record(path: Path, requests: list) -> list:
    recorder = TapeRecordingProvider(FakeProvider(), path=path)
    return [recorder.complete(req) for req in requests]







class TestRequestHash:
    @pytest.mark.numerical
    def test_deterministic_and_discriminating(self) -> None:
        req = make_request(prompt="p", seed=1, temperature=0.2, max_tokens=32)
        same = make_request(prompt="p", seed=1, temperature=0.2, max_tokens=32)
        h = request_hash(req)
        assert request_hash(same) == h

        assert request_hash(make_request(prompt="q", seed=1)) != h
        assert request_hash(make_request(prompt="p", seed=2)) != h
        assert (
            request_hash(make_request(prompt="p", seed=1, temperature=0.9)) != h
        )
        assert (
            request_hash(make_request(prompt="p", seed=1, max_tokens=64)) != h
        )

    @pytest.mark.numerical
    def test_int_and_float_temperatures_hash_identically(self) -> None:
        req_int = make_request(prompt="p", seed=1, temperature=1)
        req_float = make_request(prompt="p", seed=1, temperature=1.0)
        assert req_int == req_float
        assert request_hash(req_int) == request_hash(req_float)

    @pytest.mark.numerical
    def test_pythonhashseed_independent(self) -> None:
        first = _hash_in_subprocess("0")
        second = _hash_in_subprocess("123456789")
        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert first.stdout.strip()
        assert first.stdout.strip() == second.stdout.strip()


def _hash_in_subprocess(hashseed: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hashseed
    env["MPLBACKEND"] = "Agg"
    code = (
        "import kd.llm as m; "
        "req = m.LLMRequest(prompt='hash me', seed=42, "
        "params=m.LLMParams(temperature=0.3, max_tokens=64)); "
        "print(m.request_hash(req))"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )







class TestRoundTrip:
    def test_replay_returns_field_for_field_identical_response(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "tape.jsonl"
        requests = [make_request(prompt="p", seed=s) for s in range(3)]
        recorded = _record(path, requests)

        replay = TapeReplayProvider(path=path)
        replayed = [replay.complete(req) for req in requests]
        assert replayed == recorded

    def test_replay_has_no_inner_provider(self) -> None:
        params = set(inspect.signature(TapeReplayProvider.__init__).parameters)
        assert "inner" not in params
        assert "provider" not in params
        assert {"path", "initial_position"} <= params

    def test_usage_none_round_trips_as_whole_object(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        recorder = TapeRecordingProvider(
            FakeProvider(emit_usage=False), path=path
        )
        req = make_request(prompt="no usage", seed=0)
        recorded = recorder.complete(req)
        assert recorded.usage is None
        entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert "usage" in entry["response"]
        assert entry["response"]["usage"] is None

        replay = TapeReplayProvider(path=path)
        replayed = replay.complete(req)
        assert replayed.usage is None
        assert replayed == recorded







class TestFailLoud:
    def test_mismatched_request_fails_loud_with_position(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "tape.jsonl"
        _record(path, [make_request(prompt="recorded", seed=0)])

        replay = TapeReplayProvider(path=path)
        with pytest.raises(LLMTapeMismatchError) as excinfo:
            replay.complete(make_request(prompt="DIFFERENT", seed=0))
        assert "position" in str(excinfo.value).lower()

    def test_exhausted_tape_fails_loud(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        req = make_request(prompt="only", seed=0)
        _record(path, [req])

        replay = TapeReplayProvider(path=path)
        replay.complete(req)
        with pytest.raises(LLMTapeMismatchError):
            replay.complete(req)

    def test_missing_tape_file_fails_loud_with_not_found(
        self, tmp_path: Path
    ) -> None:
        replay = TapeReplayProvider(path=tmp_path / "missing.jsonl")
        with pytest.raises(LLMTapeMismatchError, match="not found"):
            replay.complete(make_request(seed=0))

    def test_malformed_tape_line_raises_typed_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "tape.jsonl"
        path.write_text('{"kind": "llm_call"\n', encoding="utf-8")

        replay = TapeReplayProvider(path=path)
        with pytest.raises(LLMTapeMismatchError, match="invalid JSON"):
            replay.complete(make_request(seed=0))







class TestPositional:
    def test_replay_is_positional_not_content_addressed(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "tape.jsonl"
        req_a = make_request(prompt="ALPHA", seed=0)
        req_b = make_request(prompt="BETA", seed=1)
        _record(path, [req_a, req_b])

        replay = TapeReplayProvider(path=path)
        with pytest.raises(LLMTapeMismatchError):
            replay.complete(req_b)

    def test_duplicate_entries_advance_position_not_collapse(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "tape.jsonl"
        req = make_request(prompt="DUP", seed=0)
        _record(path, [req, req])

        replay = TapeReplayProvider(path=path)
        replay.complete(req)
        replay.complete(req)
        with pytest.raises(LLMTapeMismatchError):
            replay.complete(req)







class TestPrepare:
    def test_recording_prepare_forwards_to_inner(self, tmp_path: Path) -> None:
        inner = FakeProvider()
        TapeRecordingProvider(inner, path=tmp_path / "tape.jsonl").prepare()
        assert inner.prepared is True

    def test_replay_prepare_is_noop(self, tmp_path: Path) -> None:
        TapeReplayProvider(path=tmp_path / "missing.jsonl").prepare()







class TestNoWallClock:
    @pytest.mark.numerical
    def test_differing_recorded_at_still_replays(self, tmp_path: Path) -> None:
        src = tmp_path / "src.jsonl"
        req = make_request(prompt="wall clock", seed=1)
        [recorded] = _record(src, [req])

        entry = json.loads(src.read_text().splitlines()[0])
        tape_a = tmp_path / "a.jsonl"
        tape_b = tmp_path / "b.jsonl"
        tape_a.write_text(
            json.dumps({**entry, "recorded_at": "2000-01-01T00:00:00Z"}) + "\n"
        )
        tape_b.write_text(
            json.dumps({**entry, "recorded_at": "2099-12-31T23:59:59Z"}) + "\n"
        )

        resp_a = TapeReplayProvider(path=tape_a).complete(req)
        resp_b = TapeReplayProvider(path=tape_b).complete(req)
        assert resp_a == recorded
        assert resp_b == recorded







class TestCursorSeam:
    def test_negative_initial_position_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            TapeReplayProvider(
                path=tmp_path / "tape.jsonl", initial_position=-1
            )

    def test_initial_position_resumes_equivalently(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        requests = [make_request(prompt="p", seed=s) for s in range(3)]
        _record(path, requests)


        direct = TapeReplayProvider(path=path)
        full = [direct.complete(req) for req in requests]


        rebuilt = TapeReplayProvider(path=path, initial_position=1)
        assert rebuilt.position == 1
        tail = [rebuilt.complete(req) for req in requests[1:]]
        assert tail == full[1:]

    def test_position_advances_with_each_replay(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        requests = [make_request(prompt="p", seed=s) for s in range(2)]
        _record(path, requests)

        replay = TapeReplayProvider(path=path)
        assert replay.position == 0
        replay.complete(requests[0])
        assert replay.position == 1
        replay.complete(requests[1])
        assert replay.position == 2

    def test_replay_reads_tape_file_once(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        requests = [make_request(prompt="p", seed=s) for s in range(2)]
        recorded = _record(path, requests)

        replay = TapeReplayProvider(path=path)
        assert replay.complete(requests[0]) == recorded[0]
        path.unlink()
        assert replay.complete(requests[1]) == recorded[1]







class TestExportSurface:
    def test_wire_codec_exported_from_kd_llm(self) -> None:
        import kd.llm

        for name in (
            "load_tape_entries",
            "request_from_json",
            "request_to_json",
            "response_from_json",
            "response_to_json",
            "usage_to_json",
        ):
            assert hasattr(kd.llm, name), f"kd.llm.{name} is missing"
            assert name in kd.llm.__all__, f"kd.llm.__all__ missing '{name}'"

    def test_manifest_vocabulary_exported_from_kd(self) -> None:
        import kd

        for name in ("KIND_FINAL", "FINAL_STATUS_COMPLETED"):
            assert hasattr(kd, name), f"kd.{name} is missing"
            assert name in kd.__all__, f"kd.__all__ missing '{name}'"
