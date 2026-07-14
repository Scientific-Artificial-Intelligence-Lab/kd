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
  sampling params). Config just carries the defaults.
* Guardrail split: ``max_llm_calls_per_run`` feeds the default chain's
  ``BudgetedProvider`` (whole-run budget); ``max_llm_calls_per_propose`` stays a
  plugin knob (single-propose resample bound). ``max_retries_per_call`` is NOT
  here -- it moved into ``OpenAICompatProvider`` (provider-owned retry).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any
















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
    transport / guardrail fields (``model`` / ``base_url`` / the two call
    bounds) carry kd-side defaults. Deliberately loop-free (the Runner owns
    the loop).

    Attributes:
        temperature: LLM sampling temperature. Default ``0.8`` (EDL Burgers,
            NOT the generic 1.0). Packed into ``LLMParams`` per call.
        max_tokens: LLM max decode tokens. Default ``1024`` (kd-side pin; EDL
            does not fix a value).
        stop_threshold: ``is_done`` fires when ``best_reward >= stop_threshold``.
            Default ``0.995`` (EDL); reward direction is max, range ~(0, 1].
        reward_limit: ``filter_score`` drops candidates with ``score <=
            reward_limit``. Default ``0.5`` (EDL).
        pool_size: Elite pool capacity ``k`` (EDL PriorityQueue). Default ``5``.
        init_num: The number of equations the init PROMPT asks the LLM to
            generate (EDL ``optimzier_utils.py:108``, prompt-text only). Does
            NOT change the returned batch size, which is ``n`` like every
            round (EDL ``GENERATION_NUM = args.N`` covers initialization).
            Default ``20``.
        samples_per_epoch: Per-round proposal count -- the Runner's per-iteration
            batch size (EDL ``--N`` / ``GENERATION_NUM``, Burgers script Num=8).
            Default ``8``. Read by the plugin's ``runner_batch_size`` property.
        max_llm_calls_per_propose: Plugin-owned upper bound on resample
            ``complete()`` calls within one ``propose``. kd guardrail.
        max_llm_calls_per_run: Whole-run call budget fed to the default chain's
            ``BudgetedProvider``. kd guardrail.
        seed: Base seed for the monotonic per-call LLM seed counter.
        model: LLM model id for the default ``OpenAICompatProvider`` (transport;
            unused when a provider is injected).
        base_url: Optional endpoint override for the default provider.
        tape_record_path: Optional JSONL path. When set (and no provider is
            injected), the default chain wraps the transport in a
            ``TapeRecordingProvider`` so every real request/response is recorded
            to this path -- the offline replay fixture for CI + the manual
            real-backend recovery step. ``None`` records nothing.
    """

    temperature: float = _DEFAULT_TEMPERATURE
    max_tokens: int = _DEFAULT_MAX_TOKENS
    stop_threshold: float = _DEFAULT_STOP_THRESHOLD
    reward_limit: float = _DEFAULT_REWARD_LIMIT
    pool_size: int = _DEFAULT_POOL_SIZE
    init_num: int = _DEFAULT_INIT_NUM
    samples_per_epoch: int = _DEFAULT_SAMPLES_PER_EPOCH
    max_llm_calls_per_propose: int = _DEFAULT_MAX_LLM_CALLS_PER_PROPOSE
    max_llm_calls_per_run: int = _DEFAULT_MAX_LLM_CALLS_PER_RUN
    seed: int = 0
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    tape_record_path: str | None = None

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
    ``None``), so this is ``dataclasses.asdict`` plus a stable key order; it
    exists as the single serialization seam the ``config`` property and the
    plugin share (and to keep parity with the EqGPT module API).
    """
    return asdict(config)


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
