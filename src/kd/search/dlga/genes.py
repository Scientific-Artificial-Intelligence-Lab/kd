
from __future__ import annotations

import copy
from collections.abc import Sequence

import torch

Module = list[int]
Genome = list[Module]

DEFAULT_LIBRARY = ["u", "u_x", "u_xx", "u_xxx"]


def gene_to_token(index: int, library: Sequence[str] = DEFAULT_LIBRARY) -> str:
    if index < 0 or index >= len(library):
        raise ValueError(f"gene index {index} outside library size {len(library)}")
    return library[index]


def token_to_gene(token: str, library: Sequence[str] = DEFAULT_LIBRARY) -> int:
    try:
        return list(library).index(token)
    except ValueError as exc:
        raise ValueError(f"token {token!r} not in library") from exc


def module_to_term(
    module: Sequence[int],
    library: Sequence[str] = DEFAULT_LIBRARY,
) -> str:
    if not module:
        raise ValueError("module must not be empty")
    tokens = [gene_to_token(gene, library) for gene in module]
    term = tokens[0]
    for token in tokens[1:]:
        term = f"mul({term}, {token})"
    return term


def genome_to_terms(
    genome: Sequence[Sequence[int]],
    library: Sequence[str] = DEFAULT_LIBRARY,
) -> list[str]:
    return [module_to_term(module, library) for module in genome if module]


def genome_to_expression(
    genome: Sequence[Sequence[int]],
    library: Sequence[str] = DEFAULT_LIBRARY,
) -> str:
    terms = genome_to_terms(genome, library)
    if not terms:
        return "0.0"
    expr = terms[0]
    for term in terms[1:]:
        expr = f"add({expr}, {term})"
    return expr


def random_module(
    *,
    rng: torch.Generator,
    library_size: int,
    max_length: int,
    partial_prob: float = 0.6,
) -> Module:
    _validate_library_size(library_size)
    length = 1
    genes = [_randint(rng, library_size)]
    while length < max_length and _rand(rng) <= partial_prob:
        genes.append(_randint(rng, library_size))
        length += 1
    return sorted(genes)


def random_genome(
    *,
    rng: torch.Generator,
    library_size: int,
    max_modules: int,
    max_module_length: int,
    partial_prob: float = 0.6,
    genes_prob: float = 0.6,
) -> Genome:
    genome = [
        random_module(
            rng=rng,
            library_size=library_size,
            max_length=max_module_length,
            partial_prob=partial_prob,
        )
    ]
    while len(genome) < max_modules and _rand(rng) <= genes_prob:
        module = random_module(
            rng=rng,
            library_size=library_size,
            max_length=max_module_length,
            partial_prob=partial_prob,
        )
        if module not in genome:
            genome.append(module)
    return dedupe_modules(genome)


def random_population(
    *,
    pop_size: int,
    rng: torch.Generator,
    library_size: int,
    max_modules: int,
    max_module_length: int,
    partial_prob: float = 0.6,
    genes_prob: float = 0.6,
) -> list[Genome]:
    return [
        random_genome(
            rng=rng,
            library_size=library_size,
            max_modules=max_modules,
            max_module_length=max_module_length,
            partial_prob=partial_prob,
            genes_prob=genes_prob,
        )
        for _ in range(pop_size)
    ]


def mutate_order(
    genome: Genome,
    *,
    rng: torch.Generator,
    library_size: int,
) -> Genome:
    result = _clone_genome(genome)
    if not result:
        return result
    module_index = _randint(rng, len(result))
    if not result[module_index]:
        return result
    gene_index = _randint(rng, len(result[module_index]))
    gene = result[module_index][gene_index]
    if gene == 0:
        new_gene = _randint(rng, library_size)
    else:
        direction = 1 if _rand(rng) < 0.5 else -1
        new_gene = gene + direction
        if new_gene < 0 or new_gene >= library_size:
            return dedupe_modules(result)
    result[module_index][gene_index] = new_gene
    return dedupe_modules(result)


def mutate_add_module(
    genome: Genome,
    *,
    rng: torch.Generator,
    library_size: int,
    max_length: int,
    partial_prob: float = 0.6,
    max_modules: int | None = None,
) -> Genome:
    result = _clone_genome(genome)
    if max_modules is not None and len(result) >= max_modules:
        return result
    module = random_module(
        rng=rng,
        library_size=library_size,
        max_length=max_length,
        partial_prob=partial_prob,
    )
    if module not in result:
        result.append(module)
    return dedupe_modules(result)


def mutate_delete_module(genome: Genome, *, rng: torch.Generator) -> Genome:
    result = _clone_genome(genome)
    if len(result) <= 1:
        return result
    result.pop(_randint(rng, len(result)))
    return result


def crossover_population(
    population: Sequence[Genome],
    *,
    rng: torch.Generator,
    pop_size: int,
    crossover_rate: float = 0.8,
) -> list[Genome]:
    result = [_clone_genome(genome) for genome in population]
    pair_count = min(pop_size, len(result)) // 2
    for pair in range(pair_count):
        left_i = 2 * pair
        right_i = left_i + 1
        if _rand(rng) > crossover_rate:
            continue
        if not result[left_i] or not result[right_i]:
            continue
        left_module = _randint(rng, len(result[left_i]))
        right_module = _randint(rng, len(result[right_i]))
        result[left_i][left_module], result[right_i][right_module] = (
            result[right_i][right_module],
            result[left_i][left_module],
        )
        result[left_i] = dedupe_modules(result[left_i])
        result[right_i] = dedupe_modules(result[right_i])
    return result


def select_survivors(
    population: Sequence[Genome],
    fitness: Sequence[float],
    *,
    keep: int,
) -> list[Genome]:
    if len(population) != len(fitness):
        raise ValueError("population and fitness must have the same length")
    ranked = sorted(enumerate(fitness), key=lambda item: (item[1], item[0]))
    return [_clone_genome(population[index]) for index, _ in ranked[:keep]]


def dedupe_modules(genome: Sequence[Sequence[int]]) -> Genome:
    result: Genome = []
    for module in genome:
        normalized = sorted(module)
        if normalized and normalized not in result:
            result.append(normalized)
    return result or [[0]]


def _clone_genome(genome: Sequence[Sequence[int]]) -> Genome:
    return copy.deepcopy([list(module) for module in genome])


def _rand(rng: torch.Generator) -> float:
    return float(torch.rand((), generator=rng).item())


def _randint(rng: torch.Generator, high: int) -> int:
    if high <= 0:
        raise ValueError(f"high must be positive, got {high}")
    return int(torch.randint(0, high, (1,), generator=rng).item())


def _validate_library_size(library_size: int) -> None:
    if library_size <= 0:
        raise ValueError(f"library_size must be positive, got {library_size}")
