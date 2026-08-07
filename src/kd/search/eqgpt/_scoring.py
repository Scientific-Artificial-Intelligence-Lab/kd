
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor, nn

from kd.core.equation import Form
from kd.core.evaluator import EvaluationResult
from kd.search.eqgpt.gates import should_zero_reward
from kd.search.eqgpt.reward import compute_reward
from kd.search.eqgpt.vocab import E_ID, PAD_ID, S_ID
from kd.search.result import invalid_evaluation_result

if TYPE_CHECKING:
    from kd.core.executor.context import ExecutionContext
    from kd.core.expr.executor import PythonExecutor
    from kd.core.term_cache import TermColumnCache
    from kd.search.eqgpt.backend import GPTBackend
    from kd.search.eqgpt.vocab import Vocab




PENALTY: Final[float] = 1e10

FINETUNE_CLIP_NORM: Final[float] = 1.0


def execute_term(
    executor: PythonExecutor,
    context: ExecutionContext,
    term: str,
    cache: TermColumnCache | None = None,
) -> Tensor:
    if cache is not None:
        cached = cache.get(term)
        if cached is not None:
            return cached
    column = executor.execute(term, context).value.reshape(-1).detach()
    if cache is not None:
        cache.put(term, column)
    return column


def score_candidate(
    *,
    candidate: str,
    sentence: list[int] | None,
    vocab: Vocab,
    variables: tuple[str, ...],
    executor: PythonExecutor,
    context: ExecutionContext,
    lhs_flat: Tensor | None,
    sparsity_alpha: float,
    term_cache: TermColumnCache | None = None,
) -> EvaluationResult:
    terms = candidate.split(" + ") if candidate else []
    if sentence is None:
        return invalid_result(
            candidate, terms, "candidate not found in propose() cache"
        )

    words = vocab.decode(sentence)
    if should_zero_reward(words, variables):
        return gate_zero_result(candidate, terms)

    try:
        columns = [
            execute_term(executor, context, term, term_cache) for term in terms
        ]
        if lhs_flat is None:
            if not columns:
                raise ValueError("free-pivot scoring requires at least one term")
            lhs = columns[0]
            matrix = torch.stack(columns, dim=1)
        else:
            lhs = lhs_flat.reshape(-1)
            matrix = torch.stack([lhs, *columns], dim=1)
    except (ValueError, KeyError, RuntimeError, IndexError) as exc:
        return invalid_result(
            candidate, terms, f"execution error: {exc}", is_error=True
        )

    rr = compute_reward(matrix.detach().cpu().numpy(), sparsity_alpha=sparsity_alpha)
    is_free_pivot = lhs_flat is None
    valid = math.isfinite(rr.r2) if is_free_pivot else rr.coefficients.size > 0
    r2 = rr.r2 if math.isfinite(rr.r2) else -float("inf")
    lhs_var = float(lhs.var(correction=0).item()) if lhs.numel() > 1 else 0.0
    nmse_val = (1.0 - r2) if math.isfinite(r2) else PENALTY
    mse_val = nmse_val * lhs_var if lhs_var > 0 else PENALTY



    coeffs = (
        None
        if is_free_pivot
        else torch.as_tensor(-rr.coefficients, dtype=torch.float32)
        if valid
        else None
    )

    return EvaluationResult(
        mse=mse_val,
        nmse=nmse_val,
        r2=r2,
        score=rr.reward,
        complexity=rr.n_terms,
        coefficients=coeffs,
        is_valid=valid,
        error_message="" if valid else "invalid/degenerate reward matrix",
        residuals=None,
        terms=list(terms),
        expression=candidate,
        form=Form.HOMOGENEOUS if is_free_pivot else Form.EVOLUTION,
    )


def gate_zero_result(candidate: str, terms: list[str]) -> EvaluationResult:
    return EvaluationResult(
        mse=float("inf"),
        nmse=float("inf"),
        r2=-float("inf"),
        score=0.0,
        complexity=len(terms),
        coefficients=None,
        is_valid=True,
        error_message="",
        residuals=None,
        terms=list(terms),
        expression=candidate,
    )


def invalid_result(
    candidate: str,
    terms: list[str],
    message: str,
    *,
    is_error: bool = False,
) -> EvaluationResult:
    return EvaluationResult(
        mse=PENALTY,
        nmse=PENALTY,
        r2=-float("inf") if is_error else 0.0,
        score=0.0,
        complexity=len(terms),
        coefficients=None,
        is_valid=not is_error,
        error_message=message,
        residuals=None,
        terms=list(terms) if terms else None,
        expression=candidate,
    )


def invalid_final_result(
    message: str,
    best_reward: float,
    *,
    terms: list[str] | None = None,
    reason: str = "unclassified",
) -> EvaluationResult:

    return invalid_evaluation_result(
        message,
        score=best_reward,
        terms=terms,
        reason=reason,
    )


def refit_final(
    *,
    terms: list[str],
    executor: PythonExecutor,
    context: ExecutionContext,
    target: Tensor,
    best_reward: float,
    term_cache: TermColumnCache | None = None,
) -> EvaluationResult:
    try:
        columns = [
            execute_term(executor, context, term, term_cache) for term in terms
        ]
        theta = torch.stack(columns, dim=1).double()
        target64 = target.double()




        solution = torch.linalg.lstsq(theta, target64.unsqueeze(1)).solution.squeeze(1)
    except (ValueError, KeyError, RuntimeError, IndexError) as exc:
        return invalid_final_result(
            f"execution error: {exc}",
            best_reward,
            terms=terms,
            reason="evaluation_error",
        )

    predicted = (theta @ solution).float()
    residuals = (predicted - target).detach()
    mse = float(torch.mean(residuals**2).item())
    if not math.isfinite(mse):




        return invalid_final_result(
            "non-finite residuals from refit (degenerate theta/target or "
            "rank-deficient lstsq)",
            best_reward,
            terms=terms,
            reason="non_finite",
        )
    lhs_var = float(target.var(correction=0).item()) if target.numel() > 1 else 0.0
    r2 = 1.0 - mse / lhs_var if lhs_var > 0 else -float("inf")
    nmse = 1.0 - r2 if math.isfinite(r2) else PENALTY

    return EvaluationResult(
        mse=mse,
        nmse=nmse,
        r2=r2,
        score=best_reward,
        complexity=len(terms),
        coefficients=solution.float().detach(),
        is_valid=True,
        error_message="",
        residuals=residuals,
        terms=terms,
        expression=" + ".join(terms),
    )


def finetune(
    backend: GPTBackend,
    optimizer: torch.optim.Optimizer,
    pool_sentences: list[list[int]],
    finetune_steps: int,
) -> float | None:
    if not pool_sentences:
        return None

    framed: list[list[int]] = []
    for sentence in pool_sentences:
        body = list(sentence)
        if body and body[-1] == E_ID:
            body = body[:-1]
        framed.append([S_ID, *body, E_ID])
    max_len = max(len(s) for s in framed)
    first_param = next(iter(backend.parameters()), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    padded = torch.tensor(
        [s + [PAD_ID] * (max_len - len(s)) for s in framed],
        dtype=torch.long,
        device=device,
    )
    decoder_input = padded[:, :-1]
    decoder_target = padded[:, 1:]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    losses: list[float] = []
    for _ in range(finetune_steps):
        optimizer.zero_grad()
        logits = backend.forward_logits(decoder_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1)
        )
        losses.append(float(loss.detach().item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backend.parameters(), FINETUNE_CLIP_NORM)
        optimizer.step()
    return sum(losses) / len(losses) if losses else 0.0
