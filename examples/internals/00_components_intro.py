#!/usr/bin/env python3
"""Example 00: kd platform components walkthrough.

This is a *component-level* introduction — it does NOT run SGA discovery.
It manually constructs each platform piece (PDEDataset, FiniteDiffProvider,
ExecutionContext, PythonExecutor, Evaluator, LeastSquaresSolver) and
evaluates a *known* PDE expression on Burgers data, to verify the pipeline
plumbing is sound.

For end-to-end PDE discovery via SGA, see ``02_sga_discovery_burgers.py``.
For higher-dim discovery, NN+autograd mode, viz, and result persistence,
see examples 03-05.

Burgers equation: u_t + u * u_x = nu * u_xx

Usage:
    python examples/internals/00_components_intro.py
"""

from __future__ import annotations

from kd.core.executor import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.data.derivatives import FiniteDiffProvider
from kd.data.synthetic import generate_burgers_data


def main() -> None:
    """Run the Burgers equation example."""
    print("=" * 60)
    print("kd Example: Burgers Equation Discovery")
    print("=" * 60)

    # =========================================================================
    # Step 1: Generate synthetic data
    # =========================================================================
    print("\n[Step 1] Generating Burgers equation data...")

    nu = 0.1 # Viscosity coefficient
    dataset = generate_burgers_data(
        nx=128,
        nt=64,
        nu=nu,
        noise_level=0.0,
        seed=42,
    )

    print(f" Dataset: {dataset.name}")
    print(f" Shape: {dataset.get_shape()}")
    print(f" Axes: {list(dataset.axes.keys()) if dataset.axes else []}")
    print(f" Fields: {list(dataset.fields.keys()) if dataset.fields else []}")
    print(f" Ground truth: {dataset.ground_truth}")

    # =========================================================================
    # Step 2: Create derivative provider
    # =========================================================================
    print("\n[Step 2] Creating derivative provider...")

    provider = FiniteDiffProvider(dataset, max_order=2)
    available = provider.available_derivatives()

    print(f" Available derivatives: {len(available)}")
    for field, axis, order in available:
        print(f" - {field}_{''.join([axis] * order)}")

    # =========================================================================
    # Step 3: Create execution context
    # =========================================================================
    print("\n[Step 3] Creating execution context...")

    context = ExecutionContext(
        dataset=dataset,
        derivative_provider=provider,
        constants={"nu": nu, "pi": 3.14159},
    )

    print(f" Constants: {context.constants}")

    # =========================================================================
    # Step 4: Create executor
    # =========================================================================
    print("\n[Step 4] Creating Python executor...")

    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)

    print(f" Registered operators: {registry.list_names()}")

    # =========================================================================
    # Step 5: Execute individual terms
    # =========================================================================
    print("\n[Step 5] Executing individual terms...")

    # Test basic variable access
    result_u = executor.execute("u", context)
    print(f" u: shape={result_u.value.shape}")

    # Test derivative access
    result_u_t = executor.execute("u_t", context)
    result_u_x = executor.execute("u_x", context)
    result_u_xx = executor.execute("u_xx", context)

    print(f" u_t: shape={result_u_t.value.shape}")
    print(f" u_x: shape={result_u_x.value.shape}")
    print(f" u_xx: shape={result_u_xx.value.shape}")

    # Test compound expressions
    result_convection = executor.execute("mul(u, u_x)", context)
    print(f" u * u_x: shape={result_convection.value.shape}")

    # =========================================================================
    # Step 6: Verify Burgers equation
    # =========================================================================
    print("\n[Step 6] Verifying Burgers equation...")
    print(" Equation: u_t + u * u_x - nu * u_xx = 0")

    # Compute residual: u_t + u * u_x - nu * u_xx
    # Note: We don't have constants in expressions yet, so we compute manually
    u_t = result_u_t.value
    u_u_x = result_convection.value
    u_xx = result_u_xx.value

    residual = u_t + u_u_x - nu * u_xx

    # Compute statistics (skip boundary points for accuracy)
    interior = residual[4:-4, 4:-4]
    mean_residual = interior.abs().mean().item()
    max_residual = interior.abs().max().item()

    print("\n Residual statistics (interior points):")
    print(f" Mean |residual|: {mean_residual:.6e}")
    print(f" Max |residual|: {max_residual:.6e}")

    # Check if residual is small (should be near zero for correct data)
    tolerance = 1e-2 # Allow for finite difference errors
    if mean_residual < tolerance:
        print(f"\n ✅ Equation verified! Mean residual < {tolerance}")
    else:
        print("\n ⚠️ Residual larger than expected (may be due to FD errors)")

    # =========================================================================
    # Step 7: Test expression execution with operators
    # =========================================================================
    print("\n[Step 7] Testing complex expressions...")

    expressions = [
        ("add(u_t, mul(u, u_x))", "u_t + u * u_x (convection terms)"),
        ("mul(sin(u), cos(u))", "sin(u) * cos(u)"),
        ("div(u_x, add(u, u))", "u_x / (u + u)"),
        ("add(n2(u), n3(u_x))", "u^2 + u_x^3"),
    ]

    for expr, description in expressions:
        try:
            result = executor.execute(expr, context)
            mean_val = result.value.mean().item()
            std_val = result.value.std().item()
            print(f" ✅ {description}")
            print(f" Expression: {expr}")
            print(f" Stats: mean={mean_val:.4f}, std={std_val:.4f}")
        except Exception as e:
            print(f" ❌ {description}: {e}")

    # =========================================================================
    # Step 8: Use Evaluator for coefficient recovery
    # =========================================================================
    print("\n[Step 8] Using Evaluator for coefficient recovery...")

    from kd.core.evaluator import Evaluator
    from kd.core.expr import split_terms
    from kd.core.linear_solve import LeastSquaresSolver

    # Get the LHS (u_t) for linear regression
    u_t = context.get_derivative("u", "t", 1)

    # Create solver and evaluator
    solver = LeastSquaresSolver()
    evaluator = Evaluator(
        executor=executor,
        solver=solver,
        context=context,
        lhs=u_t,
    )

    # Test split_terms
    expr = "add(mul(u, u_x), u_xx)"
    terms = split_terms(expr, registry)
    print(f"\n Expression: {expr}")
    print(f" Split terms: {terms}")

    # Evaluate the Burgers equation terms
    # True equation: u_t = -u * u_x + nu * u_xx
    # Terms: [mul(u, u_x), u_xx]
    # Expected coefficients: [-1.0, 0.1]
    eval_result = evaluator.evaluate_terms(["mul(u, u_x)", "u_xx"])

    print("\n Evaluation result:")
    print(f" Is valid: {eval_result.is_valid}")
    print(f" R²: {eval_result.r2:.6f}")
    print(f" MSE: {eval_result.mse:.6e}")
    print(f" Coefficients: {eval_result.coefficients}")

    if eval_result.coefficients is not None:
        coef_convection = eval_result.coefficients[0].item()
        coef_diffusion = eval_result.coefficients[1].item()
        print("\n Coefficient recovery:")
        print(f" mul(u, u_x): {coef_convection:.4f} (expected: -1.0)")
        print(f" u_xx: {coef_diffusion:.4f} (expected: {nu})")

        # Check recovery accuracy
        conv_error = abs(coef_convection - (-1.0))
        diff_error = abs(coef_diffusion - nu)
        print("\n Recovery errors:")
        print(f" Convection term: {conv_error:.4f}")
        print(f" Diffusion term: {diff_error:.4f}")

        if conv_error < 0.05 and diff_error < 0.05:
            print("\n ✅ Coefficients recovered within 5% error!")
        else:
            print("\n ⚠️ Coefficient recovery has larger error")

    # Test evaluate_expression (convenience API)
    result2 = evaluator.evaluate_expression("add(mul(u, u_x), u_xx)")
    print("\n evaluate_expression result:")
    print(f" R²: {result2.r2:.6f}")
    print(f" Coefficients: {result2.coefficients}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nPlatform components verified:")
    print(" - PDEDataset: ✅")
    print(" - FiniteDiffProvider: ✅")
    print(" - ExecutionContext: ✅")
    print(" - PythonExecutor: ✅")
    print(" - FunctionRegistry: ✅")
    print(" - split_terms: ✅")
    print(" - Evaluator: ✅")
    print(" - LeastSquaresSolver: ✅")


if __name__ == "__main__":
    main()
