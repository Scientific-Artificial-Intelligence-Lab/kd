
from __future__ import annotations

import pytest

from kd.llm import LLMParams, LLMProvider, LLMRequest, LLMResponse
from kd.search.llm4ed.parse import Llm4edParseError, parse_equation
from kd.search.llm4ed.prompts import (
    build_evolution_prompt,
    build_initialization_prompt,
    build_optimize_prompt,
    classify_prompt,
    parse_response,
)
from tests.unit.search.llm4ed._fake_backend import FakeLlm4edBackend

OPERANDS = ("x", "u_x", "u_xx", "u_xxx", "u")


def _text(fake: FakeLlm4edBackend, prompt: str, seed: int) -> str:
    request = LLMRequest(
        prompt=prompt, seed=seed, params=LLMParams(temperature=0.8, max_tokens=16)
    )
    return fake.complete(request).text


def test_fake_backend_satisfies_llmprovider_by_name_and_call_shape() -> None:
    fake = FakeLlm4edBackend()

    assert isinstance(fake, LLMProvider)



    request = LLMRequest(
        prompt="1. u_xx\n2. u",
        seed=3,
        params=LLMParams(temperature=0.8, max_tokens=16),
    )
    response = fake.complete(request)
    assert isinstance(response, LLMResponse)
    assert response.model == "fake"
    assert response.usage is None
    assert isinstance(response.text, str) and response.text


def test_fake_backend_is_deterministic_in_prompt_and_seed() -> None:


    request = LLMRequest(
        prompt="1. u_xx", seed=7, params=LLMParams(temperature=0.8, max_tokens=16)
    )
    first = FakeLlm4edBackend().complete(request)
    second = FakeLlm4edBackend().complete(request)
    assert first.text == second.text
    other_seed = LLMRequest(
        prompt="1. u_xx", seed=8, params=LLMParams(temperature=0.8, max_tokens=16)
    )
    assert FakeLlm4edBackend().complete(other_seed).text != first.text


class TestFakeBackendBehavior:

    def test_initialization_response_is_numbered_list_no_res(self) -> None:
        out = _text(FakeLlm4edBackend(), build_initialization_prompt(20), seed=0)
        assert "<res>" not in out
        assert "<select>" not in out
        assert out.splitlines()[0].startswith("1.")

    def test_optimize_response_is_res_wrapped(self) -> None:
        out = _text(
            FakeLlm4edBackend(), build_optimize_prompt("0: u_x score: 0.5", 8), seed=0
        )
        assert "<res>" in out and "</res>" in out
        assert "<select>" not in out

    def test_evolution_response_has_select_cross_res_triples(self) -> None:
        out = _text(FakeLlm4edBackend(), build_evolution_prompt("0: {u_x}", 4), seed=0)
        assert "<select>" in out and "</select>" in out
        assert "<cross>" in out and "</cross>" in out
        assert "<res>" in out and "</res>" in out

    @pytest.mark.parametrize(
        "prompt",
        [
            build_initialization_prompt(20),
            build_optimize_prompt("0: u_x score: 0.5", 8),
            build_evolution_prompt("0: {u_x}", 4),
        ],
    )
    def test_response_yields_both_valid_and_invalid_candidates(
        self, prompt: str
    ) -> None:
        out = _text(FakeLlm4edBackend(), prompt, seed=0)
        candidates = parse_response(out)
        assert candidates
        n_valid = 0
        n_invalid = 0
        for eq in candidates:
            try:
                parse_equation(eq, OPERANDS)
                n_valid += 1
            except Llm4edParseError:
                n_invalid += 1
        assert n_valid >= 4, f"need enough valid to terminate: {candidates}"
        assert n_invalid >= 1, f"need an invalid to exercise drop: {candidates}"

    def test_varies_with_prompt_content(self) -> None:
        fake = FakeLlm4edBackend()
        a = _text(fake, build_optimize_prompt("0: u_x score: 0.5", 8), seed=0)
        b = _text(fake, build_optimize_prompt("0: u_xxx score: 0.9", 8), seed=0)
        assert a != b

    def test_varies_with_seed(self) -> None:
        fake = FakeLlm4edBackend()
        prompt = build_evolution_prompt("0: {u_x}", 4)
        outs = {_text(fake, prompt, seed=s) for s in range(8)}
        assert len(outs) > 1

    def test_every_valid_line_uses_only_whitelisted_vocabulary(self) -> None:
        fake = FakeLlm4edBackend()
        for seed in range(6):
            out = _text(fake, build_initialization_prompt(20), seed=seed)
            for eq in parse_response(out):
                try:
                    parse_equation(eq, OPERANDS)
                except Llm4edParseError:
                    continue


                assert not eq.lstrip().startswith(("0", "."))

    def test_prepare_is_noop(self) -> None:
        FakeLlm4edBackend().prepare()

    def test_classify_matches_generated_output_shape(self) -> None:
        fake = FakeLlm4edBackend()
        for builder, kind in [
            (build_initialization_prompt(20), "initialization"),
            (build_optimize_prompt("0: u_x", 8), "optimize"),
            (build_evolution_prompt("0: {u_x}", 4), "evolution"),
        ]:
            assert classify_prompt(builder) == kind
            _text(fake, builder, seed=0)
