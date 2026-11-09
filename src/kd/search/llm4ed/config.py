"""Static configuration for the LLM4ED search plugin.

Design pins:

* NO loop-counter field (no ``generations`` / ``epochs`` / ``max_iterations``):
  the Runner owns the loop; a loop field here would be a silent no-op under the
  Runner. Mirrors ``EqGPTConfig`` / ``SGAConfig``.
* Sampling defaults are the EDL Burgers values, NOT generic LLM defaults:
  ``temperature=0.8`` (``prompt_utils.py:20``; NOT 1.0), ``max_tokens=1024``,
  ``stop_threshold=0.995``, ``reward_limit=0.5``, ``pool_size=5`` (EDL
  PriorityQueue ``k``), ``init_num=20``.
* The sampling params (``temperature`` / ``max_tokens``) are thin knobs the
  plugin packs into ``LLMParams`` on EVERY ``complete()`` call -- their single
  source of truth is ``LLMRequest.params`` (``OpenAICompatProvider`` has no
  sampling params). Config just carries the defaults. Transport options
  (``model`` / ``base_url`` / ``extra_body``) go to the provider constructor.
* Guardrail split: ``max_llm_calls_per_run`` feeds the default chain's
  ``BudgetedProvider`` (whole-run budget); ``max_llm_calls_per_propose`` stays a
  plugin knob (single-propose resample bound). ``max_retries_per_call`` is NOT
  here -- it moved into ``OpenAICompatProvider`` (provider-owned retry).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from kd.search.llm4ed.columns import FUNCTION_NAMES
from kd.search.llm4ed.fd import EDL_MAX_ORDER, MAX_ORDERS
from kd.search.llm4ed.stridge import DEFAULT_L0_PENALTY
















_DEFAULT_TEMPERATURE = 0.8
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_STOP_THRESHOLD = 0.995
_DEFAULT_REWARD_LIMIT = 0.5
_DEFAULT_POOL_SIZE = 5
_DEFAULT_INIT_NUM = 20
_DEFAULT_SAMPLES_PER_EPOCH = 8




_DEFAULT_MAX_LLM_CALLS_PER_PROPOSE = 50
_DEFAULT_MAX_LLM_CALLS_PER_RUN = 200

_ALGORITHM_NAME = "llm4ed"





ALGORITHM_NAME = _ALGORITHM_NAME


@dataclass(frozen=True)
class Llm4edConfig:
    """Static configuration for the LLM4ED plugin.

    Every field defaults to its EDL-faithful value where EDL fixes one; the
    transport / guardrail fields (``model`` / ``base_url`` / ``extra_body`` /
    the two call bounds) carry kd-side defaults. Deliberately loop-free (the
    Runner owns the loop).

    """

    temperature: float = _DEFAULT_TEMPERATURE
    """LLM sampling temperature. Default ``0.8`` (EDL Burgers, NOT the generic 1.0).
    Packed into ``LLMParams`` per call.
    """
    max_tokens: int = _DEFAULT_MAX_TOKENS
    """LLM max decode tokens. Default ``1024`` (kd-side pin; EDL does not fix a value).
    """
    stop_threshold: float = _DEFAULT_STOP_THRESHOLD
    """``is_done`` fires when ``best_reward >= stop_threshold``. Default ``0.995``
    (EDL). Sparse reward includes a per-term penalty: a nonempty exact fit
    scores at most ``0.99``. Use ``fit_stop_threshold`` to stop on fit alone.
    """
    reward_limit: float = _DEFAULT_REWARD_LIMIT
    """``filter_score`` drops candidates with ``score <= reward_limit``. Default
    ``0.5`` (EDL).
    """
    pool_size: int = _DEFAULT_POOL_SIZE
    """Elite pool capacity ``k`` (EDL PriorityQueue). Default ``5``."""
    init_num: int = _DEFAULT_INIT_NUM
    """The number of equations the init PROMPT asks the LLM to generate (EDL
    ``optimzier_utils.py:108``, prompt-text only). Does NOT change the returned
    batch size, which is ``n`` like every round (EDL ``GENERATION_NUM = args.N``
    covers initialization). Default ``20``.
    """
    samples_per_epoch: int = _DEFAULT_SAMPLES_PER_EPOCH
    """Per-round proposal count -- the Runner's per-iteration batch size (EDL ``--N`` /
    ``GENERATION_NUM``; the reference script uses 8). Default ``8``. Read by the
    plugin's ``runner_batch_size`` property.
    """
    max_llm_calls_per_propose: int = _DEFAULT_MAX_LLM_CALLS_PER_PROPOSE
    """Plugin-owned upper bound on resample ``complete()`` calls within one
    ``propose``. kd guardrail.
    """
    max_llm_calls_per_run: int = _DEFAULT_MAX_LLM_CALLS_PER_RUN
    """Whole-run call budget fed to the default chain's ``BudgetedProvider``. kd
    guardrail.
    """
    seed: int = 0
    """Base seed for the monotonic per-call LLM seed counter."""
    model: str = "gpt-4o-mini"
    """LLM model id for the default ``OpenAICompatProvider`` (transport; unused when a
    provider is injected).
    """
    base_url: str | None = None
    """Endpoint URL required by the default provider unless a provider is injected."""
    tape_record_path: str | None = None
    """Optional JSONL path. When set (and no provider is injected), the default chain
    wraps the transport in a ``TapeRecordingProvider`` so every real
    request/response is recorded to this path -- the offline replay fixture for CI +
    the manual real-backend recovery step. ``None`` records nothing.
    """
    extra_body: dict[str, Any] | None = None
    """Optional request body options forwarded unchanged to the default provider.
    ``None`` omits the SDK keyword; unused when a provider is injected.
    """
    functions: tuple[str, ...] = ()
    """Unary functions the tabular mode offers the LLM and parses (``sin``,
    ``cos``, ``exp``, ``log``, ``sqrt``), evaluated without protection: a
    candidate with a non-finite column is dropped. The default PDE mode refuses
    a non-empty value.
    """
    edge_trim: int = 0
    """Grid cells dropped at each end of every axis (space and time) from the
    regression rows after the finite differences, so no scored row uses a
    one-sided boundary stencil. ``0`` (default) keeps every row, as EDL does
    (its loaders crop some datasets before differencing instead). Default
    mode only; it changes the columns every candidate is scored on. The
    residuals then cover the interior rows only, so the report's residual
    panels fall back from heatmaps to line plots (with a warning).
    """
    l0_penalty: float = DEFAULT_L0_PENALTY
    """STRidge's selection penalty per nonzero term, added to the in-sample
    MSE. Default ``1e-5``, EDL's ``sparse_calculate`` value.
    """
    l0_relative: bool = False
    """Multiply ``l0_penalty`` by the population variance of the target, so the
    penalty scales with the data. ``False`` (default) is EDL's absolute penalty,
    which outweighs a real term when ``var(u_t)`` is near ``1e-5``.
    """
    max_derivative_order: int = EDL_MAX_ORDER
    """Highest spatial derivative in the operand vocabulary: ``3`` (default,
    EDL's ``u_x, u_xx, u_xxx``) or ``4``, which adds ``u_xxxx`` (and ``u_yyyy``
    on two axes), computed as ``Diff2(Diff2(u))``, to the columns, the prompts
    and the parser. Default mode only, and not with a sketch: kd's platform
    finite differences stop at third order, so the sketch's exit check, and
    finite-difference refits and consensus verification, cannot evaluate a
    ``u_xxxx`` answer.
    """
    fit_stop_threshold: float | None = field(default=None, metadata={"legacy": None})
    """Optional threshold in ``(0, 1]`` for the best sparse-reward candidate's
    unrounded ``1 / (1 + NRMSE)``. ``None`` (default) disables fit stopping.
    NRMSE is RMSE divided by the target's population standard deviation on
    the native scoring rows, after edge trimming and sketch pin subtraction
    when applicable. Candidate ranking and the sparse reward stay unchanged.
    Either this threshold or ``stop_threshold`` can stop the run, inclusively,
    after a completed round. The record uses ``algorithm_done``; an INFO log
    names the fit threshold and measured fit when this criterion fires.
    Set via ``Model(algorithm="llm4ed", fit_stop_threshold=0.995)`` or this
    config. It may change on resume; the carried best candidate is re-evaluated
    on the native scoring rows. This is an in-sample fit criterion.
    """

    def __post_init__(self) -> None:
        temperature = _validated_float(
            "temperature", self.temperature, min_value=0.0
        )
        stop_threshold = _validated_float(
            "stop_threshold",
            self.stop_threshold,
            min_value=0.0,
            max_value=1.0,
            min_inclusive=False,
        )
        reward_limit = _validated_float(
            "reward_limit", self.reward_limit, min_value=0.0, max_value=1.0
        )
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "stop_threshold", stop_threshold)
        object.__setattr__(self, "reward_limit", reward_limit)
        if self.fit_stop_threshold is not None:
            object.__setattr__(
                self,
                "fit_stop_threshold",
                _validated_float(
                    "fit_stop_threshold",
                    self.fit_stop_threshold,
                    min_value=0.0,
                    max_value=1.0,
                    min_inclusive=False,
                ),
            )

        for name in (
            "max_tokens",
            "pool_size",
            "init_num",
            "samples_per_epoch",
            "max_llm_calls_per_propose",
            "max_llm_calls_per_run",
        ):
            _validate_positive_int(name, getattr(self, name))

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be a non-negative int, got {self.seed!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {self.seed!r}")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError(f"model must be a non-empty str, got {self.model!r}")
        if self.base_url is not None and not isinstance(self.base_url, str):
            raise ValueError(
                f"base_url must be None or a str, got {self.base_url!r}"
            )
        if self.extra_body is not None and not isinstance(self.extra_body, dict):
            raise ValueError(
                f"extra_body must be None or a dict, got {self.extra_body!r}"
            )
        if (
            isinstance(self.edge_trim, bool)
            or not isinstance(self.edge_trim, int)
            or self.edge_trim < 0
        ):
            raise ValueError(
                f"edge_trim must be a non-negative int, got {self.edge_trim!r}"
            )
        object.__setattr__(
            self,
            "l0_penalty",
            _validated_float("l0_penalty", self.l0_penalty, min_value=0.0),
        )
        if not isinstance(self.l0_relative, bool):
            raise ValueError(f"l0_relative must be a bool, got {self.l0_relative!r}")
        if (
            isinstance(self.max_derivative_order, bool)
            or self.max_derivative_order not in MAX_ORDERS
        ):
            raise ValueError(
                f"max_derivative_order must be one of {list(MAX_ORDERS)}, got "
                f"{self.max_derivative_order!r}"
            )
        unknown = sorted(set(self.functions) - FUNCTION_NAMES)
        if unknown:
            raise ValueError(
                f"functions {unknown} are not LLM4ED functions; choose from "
                f"{sorted(FUNCTION_NAMES)}"
            )
        if self.tape_record_path is not None:
            if not isinstance(self.tape_record_path, str):
                raise ValueError(
                    "tape_record_path must be None or a str, got "
                    f"{self.tape_record_path!r}"
                )



            if not self.tape_record_path.strip():
                raise ValueError(
                    "tape_record_path must be a non-empty path or None (set "
                    "None to disable tape recording); got an empty string."
                )

    @property
    def config(self) -> dict[str, Any]:
        """JSON-safe config dict prefixed ``{"algorithm": "llm4ed", ...}``.

        Mirrors the EqGPT convention: the algorithm id leads the dict so
        checkpoint bookkeeping can read ``config["algorithm"]``, followed by the
        JSON-safe field dump from :func:`config_to_json_safe_dict`. The plugin's
        protocol ``config`` property delegates here.
        """
        return {"algorithm": _ALGORITHM_NAME, **config_to_json_safe_dict(self)}


def config_to_json_safe_dict(config: Llm4edConfig) -> dict[str, Any]:
    """Return ``config`` as a JSON-safe dict (EqGPT ``config_to_json_safe_dict``).

    All ``Llm4edConfig`` fields are already JSON-native (float / int / str /
    dict / ``None``, and the ``functions`` tuple of str), so this is
    ``dataclasses.asdict`` with disabled ``fit_stop_threshold`` omitted to
    preserve historical default config payloads;
    it exists as the single serialization seam the ``config`` property and the
    plugin share (and to keep parity with the EqGPT module API).
    """
    result = asdict(config)
    if config.fit_stop_threshold is None:
        del result["fit_stop_threshold"]
    return result


def _validated_float(
    name: str,
    value: object,
    *,
    min_value: float,
    max_value: float | None = None,
    min_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_inclusive:
        if normalized < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value!r}")
    elif normalized <= min_value:
        raise ValueError(f"{name} must be > {min_value}, got {value!r}")
    if max_value is not None and normalized > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value!r}")
    return normalized


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
