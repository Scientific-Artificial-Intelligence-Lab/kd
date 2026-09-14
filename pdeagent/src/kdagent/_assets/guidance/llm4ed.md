### LLM4ED tool configuration

The algorithm's `OPENAI_API_KEY` is separate from the key authenticating the
controller. An Anthropic controller configuration does not supply an
OpenAI-compatible algorithm endpoint. Give `base_url` in its parameters and
configure the algorithm key in the environment. Python-only provider injection
is not a JSON parameter of `run_discovery`. The three mirrored llm4ed datasets
need network access on first load.

### LLM4ED observations

The old default limits were 50 completions per round and 200 per run. The
stop_threshold value 0.995 exceeded the attainable reward 0.99, so it could
not trigger early success. Once the call budget was spent, later rounds made
one refused request and returned no proposals. Reward 0.96 on three terms
corresponds to NMSE about 1e-4; 0.55 corresponds to about 0.6. The pool could
admit scores above 0.5, so admission alone was not evidence of a good fit.
A pool flat below 0.9 while admitting candidates is a reason to inspect the
symbol library and proposal quality, not proof of an unreachable target.
R² is in the sealed record rather than the compact diagnostics.

The executor/lambdify contrast measured a 4.4e-16 column difference that could
change rejection for dense polynomial candidates whose normalized ridge
coefficients were all above the sweep threshold. Fidelity evidence included
column and score equivalence plus offline replay; the old card's live-endpoint
recovery check had not been completed at that observation point. Subsequent
website worked runs are separate case evidence and supply no benchmark truth
to this skill.
