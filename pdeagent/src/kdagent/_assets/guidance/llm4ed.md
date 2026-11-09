LLM4ED calls an OpenAI-compatible endpoint of its own, never the controller's.
A run has one only when the operator configured it (`kd-agent setup
--llm4ed`) and did not switch it off; the Availability table and the
`run_discovery` description say whether this run has one, and without it a
search with LLM4ED is refused before it starts. With it, the run supplies
`base_url`, `model` and `extra_body` and hands the key to the search alone; a
call that passes those three is refused, and `max_llm_calls_per_propose`,
`max_llm_calls_per_run` and `max_tokens` may be lowered, not raised above
the run's limits.

Search admits provider calls up to `max_llm_calls_per_run` (default 200),
roughly four per generation. Each call may retry connection, timeout, rate
limit, server or empty-response failures up to five times, so one admitted
call can make six transport attempts; this budget is not a billing count. `max_evaluations` does not limit these calls;
wall time grows with endpoint latency times calls. The coefficient gate
refuses magnitudes outside `[1e-4, 1e4]`. The default `stop_threshold=0.995`
exceeds the attainable reward 0.99, so it cannot trigger early success.
After the call budget is exhausted, later rounds can return no proposals.
Resume carries the completed-call count into the allowance.
`max_llm_calls_per_run` is `init_only`, so changing it on resume is refused;
a new allowance requires a fresh fit.
Pool admission is not proof of a good fit; inspect the proposal vocabulary
and recorded invalid-candidate reasons.
