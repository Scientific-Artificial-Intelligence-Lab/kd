"""Example 17 - LLM4ED: an LLM equation proposer (EDL) for diffusion.

LLM4ED (Du et al., "LLM4ED") uses a large language model as the search
operator: the LLM PROPOSES candidate PDE right-hand sides as text, kd scores
each one by an EDL finite-difference + sparse-regression reward, keeps an elite
pool, and feeds the best equations back into the next prompt (an
evolution/optimize alternation). The kd facade plugs it into the same one-line
API as every other packaged engine.

Diffusion (1D): u_t = 1.0 * u_xx

The LLM lives behind the ``kd.llm.LLMProvider`` protocol, so the plugin never
imports an SDK directly. There are two ways to supply it:

* OFFLINE (this script, zero network): inject a ``provider=`` that returns
  canned responses. Below, ``OfflineProvider`` always answers ``<res>u_xx</res>``
  so the example recovers the diffusion law deterministically without an API
  key. The offline path is also how the CI recovery test runs (via a recorded
  tape + ``TapeReplayProvider``).

* REAL BACKEND (commented out): give the config a ``base_url`` and set the
  ``OPENAI_API_KEY`` env var; the plugin builds
  ``BudgetedProvider(OpenAICompatProvider(...))`` for you. Adding
  ``tape_record_path=`` inserts a ``TapeRecordingProvider`` inside the budget
  (``BudgetedProvider(TapeRecordingProvider(OpenAICompatProvider(...)))``) so
  every request/response is recorded to a JSONL tape (the offline replay
  fixture / a reproducibility artifact). A real run needs network + an API key
  and is non-deterministic, so it is never part of CI.

Run: python examples/17_llm4ed.py
"""

import kd
from kd import Llm4edConfig
from kd.llm import LLMRequest, LLMResponse


class OfflineProvider:
    """A canned ``kd.llm.LLMProvider``: always proposes ``u_xx`` (no network).

    Implements the structural protocol (``prepare`` + ``complete(LLMRequest)
    -> LLMResponse``). The ``<res>...</res>`` wrapper is the payload format the
    plugin's response parser expects.
    """

    def prepare(self) -> None:
        return None

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="<res>u_xx</res>", model="offline", usage=None)


# 1. Generate synthetic diffusion data with a known ground truth.
dataset = kd.generate_diffusion_data(
    alpha=1.0, waves=(1.0,), grid_sizes=(64,), nt=40, seed=0
)
print(f"Ground truth: {dataset.ground_truth}")

# 2. Configure the model. Every Llm4edConfig field has an EDL-faithful default,
# so a bare config is runnable; here we only shrink the per-round budget to
# keep the demo fast. ``algorithm="llm4ed"`` selects the LLM-proposer plugin.
model = kd.Model(
    algorithm="llm4ed",
    generations=3,
    config=Llm4edConfig(samples_per_epoch=4, max_llm_calls_per_propose=4),
    provider=OfflineProvider(),
)

# --- REAL BACKEND (needs network + OPENAI_API_KEY; non-deterministic, non-CI):
# model = kd.Model(
# algorithm="llm4ed",
# generations=10,
# config=Llm4edConfig(
# model="gpt-4o-mini",
# base_url="https://api.openai.com/v1",
# # record for offline replay; TapeRecordingProvider APPENDS, so use a
# # fresh path per run (a reused path concatenates runs into one tape).
# tape_record_path="out/llm4ed_run.jsonl",
# ),
# )

# 3. Fit. Progress prints to stdout because verbose=True (default).
model.fit(dataset)

# 4. Inspect the result.
print()
print(f"Discovered: {model.best_expr_}")
print(f"Best reward: {model.best_score_:.4f}")
