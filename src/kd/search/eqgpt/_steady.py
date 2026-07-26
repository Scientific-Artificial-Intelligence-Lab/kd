from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch import Tensor, nn

from kd.core.equation import Equation, Form, residual_program
from kd.core.evaluator import EvaluationResult
from kd.core.executor.surrogate_context import SurrogateContext
from kd.core.term_cache import TermColumnCache
from kd.data.derivatives import AutogradProvider
from kd.data.schema import PDEDataset, compute_dataset_fingerprint
from kd.models.field_model import FieldModel
from kd.search.eqgpt import _scoring
from kd.search.eqgpt._multicase import assemble_pinned_matrix
from kd.search.eqgpt._steady_domain import build_steady_eval_dataset
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.steady_viz import SteadyVizData
from kd.search.eqgpt.vocab import Vocab, load_vocab
from kd.search.result import invalid_evaluation_result

if TYPE_CHECKING:
    from kd.core.expr.executor import PythonExecutor

logger = logging.getLogger(__name__)

STEADY_MAX_ATOMIC_ORDER: Final[int] = 5
SURROGATE_HIDDEN_SIZES: Final[tuple[int, ...]] = (50, 50, 50, 50, 50)
SURROGATE_CHECKPOINT_EVERY: Final[int] = 500
_STEADY_AXES: Final[tuple[str, str]] = ("x", "y")
_STEADY_FIELD: Final[str] = "u"

_STEADY_ERRORS: Final = (
    ValueError,
    KeyError,
    RuntimeError,
    IndexError,
    np.linalg.LinAlgError,
)
_STEADY_VIZ_ERRORS: Final = _STEADY_ERRORS + (NotImplementedError, TypeError)

class SteadyEvaluator:

    def __init__(
        self,
        *,
        executor: PythonExecutor,
        context: SurrogateContext,
        surrogate: FieldModel,
        config: EqGPTConfig,
        n_points: int,
        dataset_fingerprint: str,
        training_dataset: PDEDataset,
        eval_dataset: PDEDataset,
        vocab: Vocab,
        cache: TermColumnCache | None = None,
    ) -> None:
        self._executor = executor
        self._context = context
        self._surrogate = surrogate
        self._config = config
        self._n_points = n_points
        self._dataset_fingerprint = dataset_fingerprint
        self._training_dataset = training_dataset
        self._eval_dataset = eval_dataset




        self._vocab = vocab
        self._cache = cache if cache is not None else TermColumnCache()

    @classmethod
    def from_components(
        cls,
        dataset: PDEDataset,
        executor: PythonExecutor,
        config: EqGPTConfig,
        *,
        device: torch.device | None = None,
        surrogate: FieldModel | None = None,
    ) -> SteadyEvaluator:
        if not config.is_steady:
            raise ValueError("SteadyEvaluator requires an EqGPTConfig in steady mode")




        axes = list(dataset.axis_order or [])
        if set(axes) != set(_STEADY_AXES):
            raise ValueError(
                f"steady mode requires EXACTLY axes {list(_STEADY_AXES)}; dataset "
                f"{dataset.name!r} has {axes}. The reference-faithful 2-D surrogate "
                f"is hard-wired to (x, y): an extra axis (e.g. z) is included in the "
                f"homogeneous SCATTERED spatial_axes, so lap(u) would differentiate "
                f"along it and every candidate would score invalid."
            )
        if _STEADY_FIELD not in (dataset.fields or {}):
            raise ValueError(
                f"steady mode requires field {_STEADY_FIELD!r}; dataset "
                f"{dataset.name!r} has {list((dataset.fields or {}).keys())}."
            )
        resolved_device = device or torch.device("cpu")
        trained = surrogate
        if trained is None:
            trained = _train_surrogate(dataset, config, resolved_device)
        else:



            trained = trained.to(device=resolved_device)
            trained.eval()

        eval_dataset = build_steady_eval_dataset(dataset, config)
        surrogate_dtype = next(trained.parameters()).dtype
        coords = {
            axis: eval_dataset.get_coords(axis)
            .to(device=resolved_device, dtype=surrogate_dtype)
            .detach()
            .clone()
            .requires_grad_(True)
            for axis in eval_dataset.axis_order or []
        }
        provider = AutogradProvider(
            trained,
            coords,
            eval_dataset,
            max_order=STEADY_MAX_ATOMIC_ORDER,
        )
        context = SurrogateContext(
            eval_dataset,
            provider,
            surrogate_field="u",
            device=resolved_device,
        )
        return cls(
            executor=executor,
            context=context,
            surrogate=trained,
            config=config,
            n_points=eval_dataset.get_shape()[0],
            dataset_fingerprint=compute_dataset_fingerprint(dataset),
            training_dataset=dataset,
            eval_dataset=eval_dataset,
            vocab=load_vocab(),
        )

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def surrogate(self) -> FieldModel:
        return self._surrogate

    @property
    def dataset_fingerprint(self) -> str:
        return self._dataset_fingerprint

    def evaluation_points(self) -> np.ndarray:
        return np.column_stack(
            (
                self._eval_dataset.get_coords("x").detach().cpu().numpy(),
                self._eval_dataset.get_coords("y").detach().cpu().numpy(),
            )
        ).astype(np.float64, copy=False)

    def assemble_terms(self, terms: list[str]) -> np.ndarray:
        return np.asarray(self._assemble(terms), dtype=np.float64)

    def build_viz_data(self, equation: Equation | None) -> SteadyVizData:
        observed: np.ndarray | None = None
        predicted: np.ndarray | None = None
        try:
            observed, predicted = self._surrogate_fit_values()
        except _STEADY_VIZ_ERRORS as exc:
            logger.warning("EqGPT steady surrogate-fit data unavailable: %s", exc)
        if equation is None or equation.form is not Form.HOMOGENEOUS:
            return SteadyVizData(observed=observed, predicted=predicted)
        terms = tuple(term for term, _coefficient in equation.terms)
        points = self.evaluation_points()
        matrix: np.ndarray | None = None
        residual: np.ndarray | None = None
        try:
            matrix = self.assemble_terms(list(terms))
        except _STEADY_VIZ_ERRORS as exc:
            logger.warning("EqGPT steady term-balance data unavailable: %s", exc)
        try:
            result = self._executor.execute(residual_program(equation), self._context)
            residual = (
                result.value.detach().cpu().numpy().reshape(-1).astype(np.float64)
            )
        except _STEADY_VIZ_ERRORS as exc:
            logger.warning("EqGPT steady residual data unavailable: %s", exc)
        return SteadyVizData(
            x=points[:, 0],
            y=points[:, 1],
            residual=residual,
            matrix=matrix,
            terms=terms,
            pivot_index=0,
            observed=observed,
            predicted=predicted,
        )

    def _surrogate_fit_values(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        observed = (
            self._training_dataset.get_field("u")
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
        )
        parameter = next(self._surrogate.parameters())
        coords = {
            axis: self._training_dataset.get_coords(axis).to(
                device=parameter.device, dtype=parameter.dtype
            )
            for axis in ("x", "y")
        }
        with torch.no_grad():
            predicted = self._surrogate(**coords)["u"].detach().cpu().numpy()
        return observed, predicted.reshape(-1).astype(np.float64)

    def _matrix_terms(self, terms: list[str]) -> list[str]:
        constant = self._config.steady_constant_column
        return [*terms, "one"] if constant else list(terms)

    def score_candidate(
        self,
        *,
        candidate: str,
        terms: list[str],
    ) -> EvaluationResult:
        matrix_terms = self._matrix_terms(terms)
        scored = _scoring.score_candidate(
            candidate=" + ".join(matrix_terms),
            sentence=[],
            vocab=self._vocab,
            variables=(),
            executor=self._executor,
            context=self._context,
            lhs_flat=None,
            sparsity_alpha=float(self._config.sparsity_alpha),
            term_cache=self._cache,
        )
        return replace(
            scored,
            coefficients=None,
            terms=list(terms),
            expression=candidate,
            form=Form.HOMOGENEOUS,
        )

    def build_final_result(
        self,
        terms: list[str],
        *,
        best_reward: float,
    ) -> EvaluationResult:
        matrix_terms = self._matrix_terms(terms)
        if not terms:
            return _invalid_final(
                "no terms to fit",
                best_reward,
                terms=terms,
                reason="structural_reject",
            )
        try:
            matrix = self._assemble(matrix_terms)
            theta, target = matrix[:, 1:], -matrix[:, 0]
            coefficients = np.linalg.lstsq(theta, target, rcond=None)[0]
        except _STEADY_ERRORS as exc:
            return _invalid_final(
                f"execution error: {exc}",
                best_reward,
                terms=matrix_terms,
                reason="evaluation_error",
            )

        residuals = theta @ coefficients - target
        mse = float(np.mean(residuals**2))
        if not math.isfinite(mse):
            return _invalid_final(
                "non-finite residuals from steady refit",
                best_reward,
                terms=matrix_terms,
                reason="non_finite",
            )
        target_var = float(target.var()) if target.size > 1 else 0.0
        r2 = 1.0 - mse / target_var if target_var > 0.0 else -float("inf")
        all_coefficients = np.concatenate(([1.0], coefficients))
        return EvaluationResult(
            mse=mse,
            nmse=1.0 - r2 if math.isfinite(r2) else _scoring.PENALTY,
            r2=r2,
            score=best_reward,
            complexity=len(matrix_terms),
            coefficients=torch.as_tensor(all_coefficients, dtype=torch.float32),
            is_valid=True,
            error_message="",
            residuals=torch.from_numpy(residuals).float().detach().clone(),
            terms=matrix_terms,
            expression=" + ".join(matrix_terms),
            form=Form.HOMOGENEOUS,
        )

    def result_target(self, terms: list[str] | None) -> Tensor:
        if not terms:
            return torch.zeros(self._n_points, dtype=torch.float32)
        try:
            matrix = self._assemble([terms[0]])
        except _STEADY_ERRORS:
            return torch.zeros(self._n_points, dtype=torch.float32)
        return torch.from_numpy(-matrix[:, 0]).float().detach().clone()

    def _assemble(self, terms: list[str]) -> np.ndarray:
        return assemble_pinned_matrix(
            terms,
            executor=self._executor,
            context=self._context,
            pinned_lhs=None,
            cache=self._cache,
            cache_generation=0,
        )


def _invalid_final(
    message: str,
    best_reward: float,
    *,
    terms: list[str],
    reason: str = "unclassified",
) -> EvaluationResult:

    result = invalid_evaluation_result(
        message,
        score=best_reward,
        terms=terms,
        reason=reason,
    )
    return replace(result, form=Form.HOMOGENEOUS)


def _train_surrogate(
    dataset: PDEDataset,
    config: EqGPTConfig,
    device: torch.device,
) -> FieldModel:
    activation = str(config.steady_activation)
    seed = config.steady_surrogate_seed
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FieldModel(
            coord_names=["x", "y"],
            field_names=["u"],
            hidden_sizes=list(SURROGATE_HIDDEN_SIZES),
            activation=activation,
        )
        _init_reference_weights(model, activation)
    model = model.to(device=device, dtype=torch.float32)
    train, validate = _split_training_data(dataset, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    snapshots: list[tuple[float, dict[str, Tensor]]] = []
    last_finite = _clone_state(model)
    iterations = config.steady_train_iters
    model.train()
    for iteration in range(iterations):
        optimizer.zero_grad()
        prediction = model(x=train["x"], y=train["y"])["u"]
        loss = torch.mean((prediction - train["u"]) ** 2)
        if not torch.isfinite(loss):
            logger.warning(
                "steady surrogate training hit a non-finite loss at iteration "
                "%d/%d (activation=%s); rolling back to the last finite weights "
                "and stopping early -- the surrogate is under-trained, so "
                "downstream coefficients may be unreliable.",
                iteration,
                iterations,
                activation,
            )
            model.load_state_dict(last_finite)
            break
        last_finite = _clone_state(model)
        with torch.no_grad():
            validation_prediction = model(x=validate["x"], y=validate["y"])["u"]
            validation_loss = torch.mean((validation_prediction - validate["u"]) ** 2)
        loss.backward()
        optimizer.step()
        if (iteration + 1) % SURROGATE_CHECKPOINT_EVERY == 0:
            value = float(validation_loss.detach().item())
            if math.isfinite(value):
                snapshots.append((value, _clone_state(model)))
    best = _best_finite_snapshot(snapshots)
    if best is not None:
        model.load_state_dict(best)
    else:
        with torch.no_grad():
            final_prediction = model(x=train["x"], y=train["y"])["u"]
        if not torch.isfinite(final_prediction).all():
            logger.warning(
                "steady surrogate produced non-finite predictions after "
                "training (activation=%s, no finite validation snapshot was "
                "taken); rolling back to the last finite weights.",
                activation,
            )
            model.load_state_dict(last_finite)
    model.eval()
    return model


def _clone_state(model: FieldModel) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _state_is_finite(state: dict[str, Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in state.values())


def _best_finite_snapshot(
    snapshots: list[tuple[float, dict[str, Tensor]]],
) -> dict[str, Tensor] | None:
    finite = [(value, state) for value, state in snapshots if _state_is_finite(state)]
    if not finite:
        return None
    return min(finite, key=lambda item: item[0])[1]


def _init_reference_weights(model: FieldModel, activation: str) -> None:
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if activation == "rational":
            nn.init.xavier_normal_(module.weight, gain=1.41)
        else:
            bound = 3.0 / math.sqrt(SURROGATE_HIDDEN_SIZES[0])
            nn.init.uniform_(module.weight, -bound, bound)
        nn.init.zeros_(module.bias)


def _split_training_data(
    dataset: PDEDataset,
    config: EqGPTConfig,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    arrays = {
        "x": dataset.get_coords("x").detach().cpu().float(),
        "y": dataset.get_coords("y").detach().cpu().float(),
        "u": dataset.get_field("u").detach().cpu().float(),
    }
    n_points = arrays["u"].numel()
    if n_points == 0:
        raise ValueError("steady surrogate training requires at least one point")
    generator = torch.Generator().manual_seed(config.steady_surrogate_seed)
    permutation = torch.randperm(n_points, generator=generator)
    requested_train = config.steady_train_points
    requested_validate = config.steady_validate_points
    train_count = min(requested_train, n_points if n_points == 1 else n_points - 1)
    validate_count = min(requested_validate, n_points - train_count)
    train_indices = permutation[:train_count]
    validate_indices = permutation[train_count: train_count + validate_count]
    if validate_indices.numel() == 0:
        validate_indices = train_indices

    def select(indices: Tensor) -> dict[str, Tensor]:
        return {name: values[indices].to(device) for name, values in arrays.items()}

    return select(train_indices), select(validate_indices)


__all__ = [
    "STEADY_MAX_ATOMIC_ORDER",
    "SURROGATE_CHECKPOINT_EVERY",
    "SURROGATE_HIDDEN_SIZES",
    "SteadyEvaluator",
]
