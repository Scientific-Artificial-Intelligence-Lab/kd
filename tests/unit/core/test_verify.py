
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from kd.core.verify import VerificationReport, VerifyPolicy, verify_equation

from kd.core.equation import (
    Equation,
    LhsSpec,
    Scalar,
    law_signature,
    make_evolution,
    make_homogeneous,
)
from kd.core.executor import ExecutionContext
from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import FunctionRegistry
from kd.data.derivatives import FiniteDiffProvider
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType

_LHS = LhsSpec(field="u", axis="t", order=1)


@pytest.fixture(scope="module")
def exact_context() -> ExecutionContext:
    n_x, n_t = 64, 48
    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt)
    dataset = PDEDataset(
        name="verify-fixture",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )
    return ExecutionContext(
        dataset=dataset,
        derivative_provider=FiniteDiffProvider(dataset, max_order=2),
    )


@pytest.fixture(scope="module")
def executor() -> PythonExecutor:
    return PythonExecutor(FunctionRegistry.create_default())


def _true_law() -> Equation:
    return make_evolution(_LHS, [("diff2_x(u)", Scalar(0.5)), ("u", Scalar(-0.5))])


def _wrong_law() -> Equation:
    return make_evolution(_LHS, [("diff2_x(u)", Scalar(5.0)), ("u", Scalar(-5.0))])







class TestNoRefit:
    @pytest.mark.unit
    def test_true_coefficients_verify_clean(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        report = verify_equation(_true_law(), executor=executor, context=exact_context)

        assert isinstance(report, VerificationReport)
        assert report.nmse < 1e-3
        assert report.r2 > 0.999
        assert report.passed is None

    @pytest.mark.unit
    def test_wrong_coefficients_verify_dirty(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        report = verify_equation(_wrong_law(), executor=executor, context=exact_context)

        assert report.nmse > 1.0

    @pytest.mark.unit
    def test_module_never_touches_a_solver(self) -> None:



        import kd.core.verify as verify_module

        source = Path(verify_module.__file__).read_text(encoding="utf-8")
        assert "lstsq" not in source
        assert "SparseSolver" not in source
        assert "linear_solve" not in source







class TestMeasurementDefinitions:
    @pytest.mark.unit
    def test_nmse_uses_population_variance_of_target(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        y = executor.execute("u_t", exact_context).value
        d2x = executor.execute("diff2_x(u)", exact_context).value
        u = executor.execute("u", exact_context).value
        residual = (0.5 * d2x - 0.5 * u) - y
        expected_nmse = float((residual**2).mean() / y.var(correction=0))

        report = verify_equation(_true_law(), executor=executor, context=exact_context)


        assert report.nmse == pytest.approx(expected_nmse, rel=1e-9)
        assert report.mse == pytest.approx(float((residual**2).mean()), rel=1e-9)
        assert report.r2 == pytest.approx(1.0 - expected_nmse, rel=1e-9)

    @pytest.mark.unit
    def test_residual_sign_positive_is_over_prediction(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:



        over_predictor = make_evolution(_LHS, [("one", Scalar(1.0))])

        report = verify_equation(
            over_predictor, executor=executor, context=exact_context
        )

        assert report.residual_mean > 0.5

    @pytest.mark.unit
    def test_residual_stats_and_sample_count(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        y = executor.execute("u_t", exact_context).value

        report = verify_equation(_true_law(), executor=executor, context=exact_context)

        assert report.n_samples == y.numel()
        assert math.isfinite(report.residual_mean)
        assert math.isfinite(report.residual_std)
        assert report.residual_max_abs >= 0.0







class TestDualForm:
    @pytest.mark.unit
    def test_evolution_matches_f0_homogeneous_rewrite(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        evolution_report = verify_equation(
            _true_law(), executor=executor, context=exact_context
        )
        homogeneous = make_homogeneous(
            (
                ("u_t", Scalar(1.0)),
                ("diff2_x(u)", Scalar(-0.5)),
                ("u", Scalar(0.5)),
            )
        )
        homogeneous_report = verify_equation(
            homogeneous, executor=executor, context=exact_context
        )

        assert abs(evolution_report.nmse - homogeneous_report.nmse) < 1e-6

    @pytest.mark.unit
    def test_homogeneous_normalizes_by_pivot_pseudo_target(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        y = executor.execute("u_t", exact_context).value
        d2x = executor.execute("diff2_x(u)", exact_context).value
        u = executor.execute("u", exact_context).value
        residual = 1.0 * y + (-0.5) * d2x + 0.5 * u
        expected_nmse = float((residual**2).mean() / y.var(correction=0))

        homogeneous = make_homogeneous(
            (
                ("u_t", Scalar(1.0)),
                ("diff2_x(u)", Scalar(-0.5)),
                ("u", Scalar(0.5)),
            )
        )
        report = verify_equation(homogeneous, executor=executor, context=exact_context)

        assert report.nmse == pytest.approx(expected_nmse, rel=1e-9)







class TestActiveLawProjection:
    @pytest.mark.unit
    def test_sparse_pick_verifies_law_and_reports_inactive_mass(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        catalog = make_evolution(
            _LHS,
            [
                ("diff2_x(u)", Scalar(0.5)),
                ("u", Scalar(-0.5)),
                ("mul(u, u_x)", Scalar(-2.0)),
            ],
            active_indices=(0, 1),
        )

        report = verify_equation(catalog, executor=executor, context=exact_context)

        assert report.nmse < 1e-3
        assert report.inactive_coefficient_mass == pytest.approx(2.0)

    @pytest.mark.unit
    def test_dense_fit_has_zero_inactive_mass(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        report = verify_equation(_true_law(), executor=executor, context=exact_context)

        assert report.inactive_coefficient_mass == 0.0

    @pytest.mark.unit
    def test_report_signature_matches_law_signature(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        eq = _true_law()

        report = verify_equation(eq, executor=executor, context=exact_context)

        assert report.signature.structure_key == law_signature(eq).structure_key







class TestPolicyAndReport:
    @pytest.mark.unit
    def test_explicit_threshold_yields_boolean_verdict(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        policy = VerifyPolicy(nmse_max=1e-2)

        clean = verify_equation(
            _true_law(), executor=executor, context=exact_context, policy=policy
        )
        dirty = verify_equation(
            _wrong_law(), executor=executor, context=exact_context, policy=policy
        )

        assert clean.passed is True
        assert dirty.passed is False

    @pytest.mark.unit
    def test_report_json_safe_with_policy_verbatim(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        policy = VerifyPolicy(nmse_max=0.5, coeff_atol=0.03)
        report = verify_equation(
            _true_law(), executor=executor, context=exact_context, policy=policy
        )

        decoded = json.loads(json.dumps(report.to_dict(), allow_nan=False))

        assert decoded["policy"]["nmse_max"] == 0.5
        assert decoded["policy"]["coeff_atol"] == 0.03
        assert decoded["signature"]["structure_key"] == report.signature.structure_key
        assert decoded["dataset_name"] == "verify-fixture"
        assert isinstance(decoded["dataset_fingerprint"], str)
        assert decoded["passed"] is True







class TestFailLoud:
    @pytest.mark.unit
    def test_non_finite_coefficient_raises(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        eq = make_evolution(_LHS, [("u", Scalar(float("nan")))])

        with pytest.raises(ValueError):
            verify_equation(eq, executor=executor, context=exact_context)

    @pytest.mark.unit
    def test_unexecutable_term_raises_value_error_naming_term(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        eq = make_evolution(_LHS, [("totally_bogus_op(u)", Scalar(1.0))])

        with pytest.raises(ValueError, match="totally_bogus_op"):
            verify_equation(eq, executor=executor, context=exact_context)







class TestEmpiricalAgreement:
    @pytest.mark.unit
    def test_same_data_close_nmse_agree(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        from kd.core.verify import empirical_agreement

        near_law = make_evolution(
            _LHS, [("diff2_x(u)", Scalar(0.5001)), ("u", Scalar(-0.5001))]
        )
        rep_a = verify_equation(_true_law(), executor=executor, context=exact_context)
        rep_b = verify_equation(near_law, executor=executor, context=exact_context)

        assert empirical_agreement(rep_a, rep_b) is True

    @pytest.mark.unit
    def test_same_data_far_nmse_disagree(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        from kd.core.verify import empirical_agreement

        rep_a = verify_equation(_true_law(), executor=executor, context=exact_context)
        rep_b = verify_equation(_wrong_law(), executor=executor, context=exact_context)

        assert empirical_agreement(rep_a, rep_b) is False

    @pytest.mark.unit
    def test_different_dataset_identity_not_comparable(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        from kd.core.verify import empirical_agreement

        other = PDEDataset(
            name="other-fixture",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo("x", torch.linspace(0, 1, 8, dtype=torch.float64)),
                "t": AxisInfo("t", torch.linspace(0, 1, 6, dtype=torch.float64)),
            },
            axis_order=["x", "t"],
            fields={
                "u": FieldData(
                    "u",
                    torch.linspace(0, 1, 8, dtype=torch.float64)[:, None]
                    * torch.ones(6, dtype=torch.float64),
                )
            },
            lhs_field="u",
            lhs_axis="t",
        )
        other_context = ExecutionContext(
            dataset=other,
            derivative_provider=FiniteDiffProvider(other, max_order=2),
        )
        rep_a = verify_equation(_true_law(), executor=executor, context=exact_context)
        rep_b = verify_equation(
            make_evolution(_LHS, [("u", Scalar(1.0))]),
            executor=executor,
            context=other_context,
        )

        assert empirical_agreement(rep_a, rep_b) is None







class TestVerificationArtifact:
    @pytest.mark.unit
    def test_artifact_links_report_to_evidence_hash(
        self,
        exact_context: ExecutionContext,
        executor: PythonExecutor,
        tmp_path: Path,
    ) -> None:
        from kd.core.verify import (
            VERIFICATION_ARTIFACT_TAG,
            write_verification_artifact,
        )

        report = verify_equation(_true_law(), executor=executor, context=exact_context)
        target = write_verification_artifact(
            report, evidence_hash="deadbeef" * 8, path=tmp_path / "verif.json"
        )

        decoded = json.loads(target.read_text(encoding="utf-8"))
        assert decoded["artifact"] == VERIFICATION_ARTIFACT_TAG
        assert decoded["evidence_hash"] == "deadbeef" * 8
        assert decoded["report"] == report.to_dict()

    @pytest.mark.unit
    def test_artifact_requires_evidence_hash(
        self,
        exact_context: ExecutionContext,
        executor: PythonExecutor,
        tmp_path: Path,
    ) -> None:
        from kd.core.verify import write_verification_artifact

        report = verify_equation(_true_law(), executor=executor, context=exact_context)

        with pytest.raises(ValueError, match="evidence_hash"):
            write_verification_artifact(
                report, evidence_hash="", path=tmp_path / "verif.json"
            )







class TestNormalizerSelfDescription:

    @pytest.mark.unit
    def test_report_records_normalizer_term_and_variance(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        report = verify_equation(
            _true_law(), executor=executor, context=exact_context
        )
        target = executor.execute("u_t", exact_context).value.double()

        assert report.normalizer_term == "u_t"
        assert report.normalizer_variance == pytest.approx(
            float(target.var(correction=0)), rel=1e-9
        )
        decoded = json.loads(json.dumps(report.to_dict(), allow_nan=False))
        assert decoded["normalizer_term"] == "u_t"

    @pytest.mark.unit
    def test_pivot_rotation_not_comparable_on_empirical_axis(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        from kd.core.verify import empirical_agreement

        pivot_ut = make_homogeneous(
            (
                ("u_t", Scalar(1.0)),
                ("diff2_x(u)", Scalar(-0.5)),
                ("u", Scalar(0.5)),
            )
        )

        pivot_d2x = make_homogeneous(
            (
                ("diff2_x(u)", Scalar(1.0)),
                ("u", Scalar(-1.0)),
                ("u_t", Scalar(-2.0)),
            )
        )
        rep_a = verify_equation(pivot_ut, executor=executor, context=exact_context)
        rep_b = verify_equation(pivot_d2x, executor=executor, context=exact_context)

        assert (
            rep_a.signature.structure_key == rep_b.signature.structure_key
        )
        assert rep_a.normalizer_term != rep_b.normalizer_term
        assert empirical_agreement(rep_a, rep_b) is None


class TestR2Semantics:

    @pytest.mark.unit
    def test_evolution_r2_is_one_minus_nmse(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        report = verify_equation(
            _true_law(), executor=executor, context=exact_context
        )

        assert report.r2 == pytest.approx(1.0 - report.nmse)

    @pytest.mark.unit
    def test_homogeneous_r2_is_none(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        homogeneous = make_homogeneous(
            (("u_t", Scalar(1.0)), ("diff2_x(u)", Scalar(-0.5)))
        )
        report = verify_equation(
            homogeneous, executor=executor, context=exact_context
        )

        assert report.r2 is None
        decoded = json.loads(json.dumps(report.to_dict(), allow_nan=False))
        assert decoded["r2"] is None


class TestPolicyWiring:

    @pytest.mark.unit
    def test_law_agreement_forwards_coeff_atol(self) -> None:
        from kd.core.verify import law_agreement

        sig_a = law_signature(
            make_evolution(_LHS, [("diff2_x(u)", Scalar(0.50)), ("u", Scalar(-0.5))])
        )
        sig_b = law_signature(
            make_evolution(_LHS, [("diff2_x(u)", Scalar(0.52)), ("u", Scalar(-0.5))])
        )

        strict = law_agreement(sig_a, sig_b, policy=VerifyPolicy(coeff_atol=1e-6))
        loose = law_agreement(sig_a, sig_b, policy=VerifyPolicy(coeff_atol=0.5))

        assert strict.coefficient is False
        assert loose.coefficient is True

    @pytest.mark.unit
    def test_pivot_unity_rtol_single_sourced_from_construct(self) -> None:
        from kd.core.equation.construct import PIVOT_UNITY_RTOL

        assert VerifyPolicy().pivot_unity_rtol == PIVOT_UNITY_RTOL


class TestNearConstantNormalizer:

    @pytest.mark.unit
    def test_constant_pivot_column_raises_naming_column(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:

        constant_pivot = make_homogeneous(
            (("one", Scalar(1.0)), ("u", Scalar(0.5)))
        )

        with pytest.raises(ValueError, match="'one'"):
            verify_equation(
                constant_pivot, executor=executor, context=exact_context
            )


class TestResidualCrossLock:

    @pytest.mark.unit
    def test_residual_program_matches_verify_arithmetic(
        self, exact_context: ExecutionContext, executor: PythonExecutor
    ) -> None:
        from kd.core.equation import residual_program

        rendered = (
            executor.execute(residual_program(_true_law()), exact_context)
            .value.detach()
            .double()
        )
        report = verify_equation(
            _true_law(), executor=executor, context=exact_context
        )

        assert float(rendered.mean()) == pytest.approx(
            report.residual_mean, abs=1e-12
        )
        assert float(rendered.std(correction=0)) == pytest.approx(
            report.residual_std, rel=1e-9
        )
        assert float(rendered.abs().max()) == pytest.approx(
            report.residual_max_abs, rel=1e-9
        )
