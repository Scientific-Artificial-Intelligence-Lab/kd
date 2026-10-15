LLM4ED calls its own OpenAI-compatible endpoint: set `base_url`, `model`,
`max_tokens` and `extra_body` for it in `params`, and the provider key must
be in the environment of the run. The controller's endpoint and key are not
reused. Without `base_url` or without the key the call fails at once, naming
what is missing.

Search makes billed requests up to `max_llm_calls_per_run` (default 200),
roughly four per generation. `max_evaluations` does not limit these calls;
wall time grows with endpoint latency times calls. The coefficient gate
refuses magnitudes outside `[1e-4, 1e4]`. The default `stop_threshold=0.995`
exceeds the attainable reward 0.99, so it cannot trigger early success.
After the call budget is exhausted, later rounds can return no proposals.
Pool admission is not proof of a good fit; inspect the proposal vocabulary
and recorded invalid-candidate reasons.
