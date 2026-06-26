
import json

import kd


dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")



candidates = [
    "mul(u, diff_x(u))",
    "u_xx",
    "u * u_x",
    "add(u, u_xx)",
    "u_xxx",
    "diff_t(u)",
]



report = kd.validate_terms(dataset, candidates)
print(
    f"\nValidated {len(candidates)} candidates: "
    f"{len(report.valid)} valid, {len(report.rejected)} rejected"
)
for rejection in report.rejected:
    print(f" [rejected] {rejection.term!r}\n {rejection.reason}")





result = kd.evaluate_terms(dataset, report.valid)
assert result.terms is not None and result.coefficients is not None
print("\nLeast-squares fit on the valid terms (target: u_t):")
for term, coef in zip(result.terms, result.coefficients.tolist(), strict=True):
    print(f" {coef:+.4f} * {term}")
print(f"NMSE = {result.nmse:.3e} R^2 = {result.r2:.6f}")


payload = result.to_dict(include_residuals=False)
print(f"\nresult.to_dict() keys: {sorted(payload)}")
print(json.dumps({key: payload[key] for key in ("terms", "coefficients", "nmse")}))




try:
    kd.evaluate_terms(dataset, candidates)
except kd.InvalidTermsError as err:
    print(f"\nStrict mode: InvalidTermsError with {len(err.rejected)} rejections")
    print("(pass skip_invalid=True to fit the valid subset instead - the")
    print(" dropped terms are still named in a single logger warning)")

print("\n[kd] Done.")
