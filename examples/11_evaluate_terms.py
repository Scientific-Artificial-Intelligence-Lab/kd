"""Example 11 - Score candidate terms directly, no search loop.

Have a hypothesis about which terms belong in the equation - from a paper,
from intuition, or from an LLM agent proposing candidates? You don't need
to run a full search for that. ``kd.evaluate_terms`` is a stateless one-call
entry: it builds the derivative features, least-squares fits your terms
against the LHS, and scores the fit. And it FAILS LOUD - a bad term is never
silently dropped into a misleading fit.

The two-call agent pattern:

  1. ``kd.validate_terms(dataset, terms)`` -> per-term verdicts, no fitting
  2. ``kd.evaluate_terms(dataset, valid)`` -> coefficients + NMSE / R^2

Both calls never mutate your dataset, and the same inputs give the same
outputs (no hidden state). Terms use canonical funcall IR - ``"u_xx"``,
``"mul(u, diff_x(u))"`` - so an infix string like ``"u * u_x"`` is rejected
with a reason instead of being mis-fitted as one opaque column.

Run: python examples/11_evaluate_terms.py
"""

import json

import kd

# 1. Generate synthetic data with a known ground truth.
dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")

# 2. A candidate list as an agent might propose it: the two real Burgers
# terms plus four flawed entries, one per rejection category.
candidates = [
    "mul(u, diff_x(u))", # valid: the convection term u*u_x
    "u_xx", # valid: the diffusion term
    "u * u_x", # syntax: infix, not canonical funcall IR
    "add(u, u_xx)", # composite: two terms in one string -> split hint
    "u_xxx", # beyond max_order=2 -> actionable retry hint
    "diff_t(u)", # tautology: duplicates the u_t LHS target
]

# 3. Validate: classify every term, fit nothing. The report is JSON-safe
# (report.to_dict()) so an agent can consume it across an MCP boundary.
report = kd.validate_terms(dataset, candidates)
print(
    f"\nValidated {len(candidates)} candidates: "
    f"{len(report.valid)} valid, {len(report.rejected)} rejected"
)
for rejection in report.rejected:
    print(f" [rejected] {rejection.term!r}\n {rejection.reason}")

# 4. Evaluate the survivors: one least-squares fit on the shared derivative
# features. If evaluate_terms returns at all, the result is valid.
# (In an agent loop, check ``report.valid`` is non-empty first: an
# all-rejected candidate list would make evaluate_terms raise ValueError.)
result = kd.evaluate_terms(dataset, report.valid)
assert result.terms is not None and result.coefficients is not None
print("\nLeast-squares fit on the valid terms (target: u_t):")
for term, coef in zip(result.terms, result.coefficients.tolist(), strict=True):
    print(f" {coef:+.4f} * {term}")
print(f"NMSE = {result.nmse:.3e} R^2 = {result.r2:.6f}")

# 5. Results serialize for agent / MCP boundaries too.
payload = result.to_dict(include_residuals=False)
print(f"\nresult.to_dict() keys: {sorted(payload)}")
print(json.dumps({key: payload[key] for key in ("terms", "coefficients", "nmse")}))

# 6. Strict mode is the default: feed the raw list and ANY rejection refuses
# the whole fit - one exception carries the COMPLETE rejection report
# (every bad term in a single round trip, not fail-at-first).
try:
    kd.evaluate_terms(dataset, candidates)
except kd.InvalidTermsError as err:
    print(f"\nStrict mode: InvalidTermsError with {len(err.rejected)} rejections")
    print("(pass skip_invalid=True to fit the valid subset instead - the")
    print(" dropped terms are still named in a single logger warning)")

print("\n[kd] Done.")
