
from __future__ import annotations

import random

import pytest

from kd.search.llm4ed.parse import parse_equation
from kd.search.llm4ed.prompts import (
    DEFAULT_PDE_OPERANDS,
    DEFAULT_PDE_OPERATORS,
    build_evolution_prompt,
    build_initialization_prompt,
    build_optimize_prompt,
    classify_prompt,
    extract_res_blocks,
    normalize_equation_lines,
    parse_response,
    permute_terms,
)

OPERANDS = ("x", "u_x", "u_xx", "u_xxx", "u")







class TestPromptTemplates:
    def test_default_symbol_libraries_match_edl_pde_scripts(self) -> None:

        assert DEFAULT_PDE_OPERATORS == "[+, -, *, /, ^2, ^3]"
        assert DEFAULT_PDE_OPERANDS == "[u, u_x, u_xx, u_xxx, x]"

    def test_initialization_prompt_requests_numbered_list_no_res(self) -> None:
        prompt = build_initialization_prompt(20)

        assert "<res>" not in prompt
        assert "<select>" not in prompt
        assert "1." in prompt and "2." in prompt
        assert "20" in prompt
        assert DEFAULT_PDE_OPERATORS in prompt
        assert DEFAULT_PDE_OPERANDS in prompt

    def test_optimize_prompt_requests_res_wrapping(self) -> None:
        history = "0: u_x score: 0.51\n1: u_xx score: 0.72"
        prompt = build_optimize_prompt(history, 8)
        assert "<res>" in prompt and "</res>" in prompt
        assert "<select>" not in prompt
        assert history in prompt
        assert "8" in prompt

    def test_evolution_prompt_requests_select_cross_res(self) -> None:
        term_sets = "0: {u_xx, u}\n1: {u*u_x, u_xxx}"
        prompt = build_evolution_prompt(term_sets, 4)
        assert "<select>" in prompt and "</select>" in prompt
        assert "<cross>" in prompt and "</cross>" in prompt
        assert "<res>" in prompt and "</res>" in prompt
        assert term_sets in prompt

    def test_custom_operators_and_operands_flow_through(self) -> None:
        prompt = build_initialization_prompt(
            5, operators="[+, -]", operands="[u, u_x]"
        )
        assert "[+, -]" in prompt
        assert "[u, u_x]" in prompt







class TestClassifyPrompt:
    def test_classifies_initialization(self) -> None:
        assert classify_prompt(build_initialization_prompt(20)) == "initialization"

    def test_classifies_optimize(self) -> None:
        assert classify_prompt(build_optimize_prompt("0: u_x", 8)) == "optimize"

    def test_classifies_evolution(self) -> None:
        assert classify_prompt(build_evolution_prompt("0: {u_x}", 4)) == "evolution"

    def test_evolution_wins_over_res_marker(self) -> None:


        prompt = build_evolution_prompt("0: {u_x}", 4)
        assert "<res>" in prompt
        assert classify_prompt(prompt) == "evolution"







class TestExtractResBlocks:
    def test_extracts_paired_res_blocks(self) -> None:
        text = "noise <res>u_xx - u</res> junk <res>u*u_x</res> tail"
        assert extract_res_blocks(text) == "u_xx - u\nu*u_x"

    def test_strips_curly_braces(self) -> None:
        text = "<res>{u_xx - u}</res>"
        assert extract_res_blocks(text) == "u_xx - u"

    def test_no_res_yields_empty(self) -> None:
        assert extract_res_blocks("nothing tagged here") == ""


class TestNormalizeEquationLines:
    def test_strips_numbered_list_prefix(self) -> None:
        text = "1. u_xx - u\n2. u*u_x"
        assert normalize_equation_lines(text) == ["u_xx - u", "u*u_x"]

    def test_strips_colon_index_prefix(self) -> None:
        text = "0: u_x + u_xx\n1: u_xxx"
        assert normalize_equation_lines(text) == ["u_x + u_xx", "u_xxx"]

    def test_comma_becomes_plus(self) -> None:

        assert normalize_equation_lines("u_x, u_xx")[0] == "u_x+ u_xx"

    def test_untagged_line_without_prefix_is_unchanged(self) -> None:
        assert normalize_equation_lines("u_xx - u + u_x")[0] == "u_xx - u + u_x"


class TestParseResponse:
    def test_parses_initialization_numbered_list(self) -> None:
        text = "1. u_xx - u + u^3\n2. u*u_x + u_xx"
        assert parse_response(text) == ["u_xx - u + u^3", "u*u_x + u_xx"]

    def test_parses_optimize_res_wrapped(self) -> None:
        text = "<res>u_xx - u</res>\n<res>u*u_x</res>"
        assert parse_response(text) == ["u_xx - u", "u*u_x"]

    def test_parses_evolution_only_res_candidates(self) -> None:
        text = (
            "<select>{u_xx, u}</select>\n"
            "<select>{u*u_x}</select>\n"
            "<cross>u_xx + u*u_x</cross>\n"
            "<res>u_xx - u*u_x</res>"
        )

        assert parse_response(text) == ["u_xx - u*u_x"]

    def test_parsed_candidates_feed_the_t1_parser(self) -> None:
        text = "1. u_xx - u + u^3\n2. u*u_x + u_xx"
        for eq in parse_response(text):
            parsed = parse_equation(eq, OPERANDS)
            assert parsed.terms







class TestPermuteTerms:
    def test_is_deterministic_for_a_given_seed(self) -> None:
        terms = ["u_xx", "-u", "u^3"]
        first = permute_terms(terms, random.Random(7))
        second = permute_terms(terms, random.Random(7))
        assert first == second

    def test_preserves_the_term_multiset(self) -> None:

        terms = ["u_xx", "-u", "u^3"]
        result = permute_terms(terms, random.Random(3))
        pieces = result.replace(" + ", "|").replace(" - ", "|").split("|")
        assert sorted(pieces) == sorted(["u_xx", "u", "u^3"])

    def test_uses_only_plus_minus_connectors(self) -> None:
        terms = ["u_xx", "u", "u^3", "u_x"]
        result = permute_terms(terms, random.Random(0))

        assert not result.endswith((" + ", " - "))
        assert "*" not in result.replace("u_x", "").replace("u", "")

    def test_strips_leading_minus_and_converts_power(self) -> None:
        result = permute_terms(["-u**2"], random.Random(0))
        assert result == "u^2"

    def test_different_seeds_can_differ(self) -> None:
        terms = ["u_xx", "-u", "u^3", "u_x", "u_xxx"]
        outs = {permute_terms(terms, random.Random(s)) for s in range(12)}
        assert len(outs) > 1

    def test_does_not_consume_module_global_random(self) -> None:

        random.seed(123)
        expected = random.random()
        random.seed(123)
        permute_terms(["u_xx", "-u", "u^3"], random.Random(0))
        assert random.random() == expected

    def test_single_term(self) -> None:
        assert permute_terms(["u_xx"], random.Random(0)) == "u_xx"

    def test_empty(self) -> None:
        assert permute_terms([], random.Random(0)) == ""


@pytest.mark.smoke
def test_public_prompt_surface_importable() -> None:

    from kd.search.llm4ed import (
        build_evolution_prompt,
        build_initialization_prompt,
        build_optimize_prompt,
        classify_prompt,
        parse_response,
        permute_terms,
    )

    assert all(
        callable(obj)
        for obj in (
            build_evolution_prompt,
            build_initialization_prompt,
            build_optimize_prompt,
            classify_prompt,
            parse_response,
            permute_terms,
        )
    )
