
from __future__ import annotations

import pytest
import torch

from kd.search.dlga.genes import (
    crossover_population,
    dedupe_modules,
    gene_to_token,
    genome_to_expression,
    module_to_term,
    mutate_add_module,
    mutate_delete_module,
    mutate_order,
    random_population,
    select_survivors,
    token_to_gene,
)


class TestGeneEncoding:
    @pytest.mark.smoke
    def test_gene_encoding_round_trip(self) -> None:
        for index, token in enumerate(["u", "u_x", "u_xx", "u_xxx"]):
            assert gene_to_token(index) == token
            assert token_to_gene(token) == index

    @pytest.mark.unit
    def test_module_to_term(self) -> None:
        assert module_to_term([0]) == "u"
        assert module_to_term([0, 1]) == "mul(u, u_x)"
        assert module_to_term([0, 0, 0]) == "mul(mul(u, u), u)"

    @pytest.mark.unit
    def test_genome_to_expression_adds_modules(self) -> None:
        assert genome_to_expression([[0, 1], [2]]) == "add(mul(u, u_x), u_xx)"


class TestMutation:
    @pytest.mark.unit
    def test_order_mutation_keeps_gene_in_library_range(self) -> None:
        rng = torch.Generator().manual_seed(123)
        genome = [[0], [3]]

        mutated = mutate_order(genome, rng=rng, library_size=4)

        assert mutated
        assert all(0 <= gene < 4 for module in mutated for gene in module)

    @pytest.mark.unit
    def test_delete_mutation_preserves_at_least_one_rhs_module(self) -> None:
        rng = torch.Generator().manual_seed(0)

        assert mutate_delete_module([[0]], rng=rng) == [[0]]
        assert len(mutate_delete_module([[0], [1]], rng=rng)) == 1

    @pytest.mark.unit
    def test_add_mutation_adds_legal_module(self) -> None:
        rng = torch.Generator().manual_seed(1)

        mutated = mutate_add_module([[0]], rng=rng, library_size=4, max_length=3)

        assert len(mutated) >= 1
        assert all(0 <= gene < 4 for module in mutated for gene in module)

    @pytest.mark.unit
    def test_dedupe_modules_sorts_and_removes_duplicates(self) -> None:
        assert dedupe_modules([[1, 0], [0, 1], [2]]) == [[0, 1], [2]]

    @pytest.mark.unit
    def test_add_mutation_respects_max_modules_cap(self) -> None:

        starting: list[list[int]] = [[0], [1], [2], [3], [0, 1]]
        max_modules = 5



        for seed in range(100):
            rng = torch.Generator().manual_seed(seed)
            mutated = mutate_add_module(
                [module[:] for module in starting],
                rng=rng,
                library_size=4,
                max_length=3,
                max_modules=max_modules,
            )
            assert len(mutated) <= max_modules, (
                f"seed={seed}: genome grew to {len(mutated)} modules > "
                f"max_modules={max_modules}"
            )

    @pytest.mark.unit
    def test_order_mutation_is_plus_minus_one_for_nonzero_gene(self) -> None:

        starting_nonzero: list[list[int]] = [[2]]
        library_size = 4
        observed_nonzero: set[int] = set()
        for seed in range(200):
            rng = torch.Generator().manual_seed(seed)
            mutated = mutate_order(
                [module[:] for module in starting_nonzero],
                rng=rng,
                library_size=library_size,
            )
            assert mutated, f"seed={seed}: mutation produced empty genome"
            for module in mutated:
                for gene in module:
                    observed_nonzero.add(gene)
                    assert abs(gene - 2) <= 1, (
                        f"seed={seed}: gene {gene} more than ±1 from original 2 "
                        f"violates Xu 2020 Def 2.5 neighborhood rule"
                    )

        assert observed_nonzero.issubset({1, 2, 3})


        starting_zero: list[list[int]] = [[0]]
        observed_zero: set[int] = set()
        for seed in range(200):
            rng = torch.Generator().manual_seed(seed)
            mutated = mutate_order(
                [module[:] for module in starting_zero],
                rng=rng,
                library_size=library_size,
            )
            for module in mutated:
                for gene in module:
                    observed_zero.add(gene)


        assert any(g >= 2 for g in observed_zero), (
            f"observed only {observed_zero} from gene=0 mutations; "
            "expected arbitrary library-wide jumps for the zero-special-case"
        )


class TestCrossoverAndSelection:
    @pytest.mark.unit
    def test_crossover_uses_full_pop_size_not_n_generations(self) -> None:
        population = [
            [[0]],
            [[1]],
            [[2]],
            [[3]],
            [[0, 1]],
            [[2, 3]],
        ]
        original = [[module[:] for module in genome] for genome in population]
        rng = torch.Generator().manual_seed(8)

        crossed = crossover_population(
            population,
            rng=rng,
            pop_size=6,
            crossover_rate=1.0,
        )

        changed_after_first_v1_pair = [
            i for i in range(2, 6) if crossed[i] != original[i]
        ]
        assert changed_after_first_v1_pair

    @pytest.mark.unit
    def test_selection_keeps_distinct_indices_for_duplicate_fitness(self) -> None:
        population = [
            [[0]],
            [[1]],
            [[2]],
            [[3]],
        ]
        selected = select_survivors(population, [1.0, 1.0, 2.0, 3.0], keep=2)

        assert selected == [[[0]], [[1]]]

    @pytest.mark.unit
    def test_random_population_is_seed_reproducible(self) -> None:
        first = random_population(
            pop_size=8,
            rng=torch.Generator().manual_seed(99),
            library_size=4,
            max_modules=4,
            max_module_length=3,
        )
        second = random_population(
            pop_size=8,
            rng=torch.Generator().manual_seed(99),
            library_size=4,
            max_modules=4,
            max_module_length=3,
        )

        assert first == second
