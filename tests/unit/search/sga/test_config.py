
from __future__ import annotations

import pytest

from kd.search.sga.config import SGAConfig






class TestSGAConfigDefaults:

    @pytest.mark.smoke
    def test_config_is_instantiable(self) -> None:
        config = SGAConfig()
        assert config is not None

    def test_ga_defaults(self) -> None:
        config = SGAConfig()
        assert config.num == 20
        assert config.p_var == 0.5
        assert config.p_mute == 0.3
        assert config.p_cro == 0.5
        assert config.p_rep == 1.0
        assert config.seed == 0

    def test_tree_structure_defaults(self) -> None:
        config = SGAConfig()
        assert config.depth == 4
        assert config.width == 5

    def test_evaluation_defaults(self) -> None:
        config = SGAConfig()
        assert config.aic_ratio == 1.0
        assert config.lam == 0.0
        assert config.d_tol == 1.0
        assert config.maxit == 10
        assert config.str_iters == 10
        assert config.normalize == 2







class TestSGAConfigNoGenerations:

    def test_no_generations_field(self) -> None:
        config = SGAConfig()
        assert not hasattr(config, "generations")

    def test_no_sga_run_field(self) -> None:
        config = SGAConfig()
        assert not hasattr(config, "sga_run")







class TestSGAConfigCustom:

    def test_override_single_field(self) -> None:
        config = SGAConfig(num=50)
        assert config.num == 50

    def test_override_multiple_fields(self) -> None:
        config = SGAConfig(depth=6, width=8, p_mute=0.5)
        assert config.depth == 6
        assert config.width == 8
        assert config.p_mute == 0.5

    def test_other_defaults_preserved_after_override(self) -> None:
        config = SGAConfig(num=100)
        assert config.p_var == 0.5
        assert config.depth == 4
        assert config.seed == 0







class TestSGAConfigDataclass:

    def test_has_dataclass_fields(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(SGAConfig)

    def test_equality(self) -> None:
        a = SGAConfig()
        b = SGAConfig()
        assert a == b

    def test_inequality(self) -> None:
        a = SGAConfig(num=10)
        b = SGAConfig(num=20)
        assert a != b







class TestOperatorPools:

    @pytest.mark.smoke
    def test_ops_pool_importable(self) -> None:
        from kd.search.sga.config import OPS

        assert isinstance(OPS, (list, tuple))
        assert len(OPS) > 0

    def test_ops_contains_all_operators(self) -> None:
        from kd.search.sga.config import OPS

        op_names = {name for name, _ in OPS}
        expected = {"+", "-", "*", "/", "^2", "^3", "d", "d^2"}
        assert expected == op_names

    def test_ops_arities_correct(self) -> None:
        from kd.search.sga.config import OPS

        op_dict = {name: arity for name, arity in OPS}
        assert op_dict["+"] == 2
        assert op_dict["-"] == 2
        assert op_dict["*"] == 2
        assert op_dict["/"] == 2
        assert op_dict["^2"] == 1
        assert op_dict["^3"] == 1
        assert op_dict["d"] == 2
        assert op_dict["d^2"] == 2

    def test_root_excludes_plus_minus(self) -> None:
        from kd.search.sga.config import ROOT

        root_names = {name for name, _ in ROOT}
        assert "+" not in root_names
        assert "-" not in root_names

    def test_root_contains_mul_div_powers_and_deriv(self) -> None:
        from kd.search.sga.config import ROOT

        root_names = {name for name, _ in ROOT}
        assert {"*", "/", "^2", "^3", "d", "d^2"}.issubset(root_names)

    def test_op1_only_unary(self) -> None:
        from kd.search.sga.config import OP1

        for name, arity in OP1:
            assert arity == 1, f"{name} has arity {arity}, expected 1"

    def test_op1_contents(self) -> None:
        from kd.search.sga.config import OP1

        op1_names = {name for name, _ in OP1}
        assert op1_names == {"^2", "^3"}

    def test_op2_only_binary(self) -> None:
        from kd.search.sga.config import OP2

        for name, arity in OP2:
            assert arity == 2, f"{name} has arity {arity}, expected 2"

    def test_op2_contents(self) -> None:
        from kd.search.sga.config import OP2

        op2_names = {name for name, _ in OP2}
        assert op2_names == {"+", "-", "*", "/", "d", "d^2"}







class TestOperatorPoolInvariants:

    def test_op1_and_op2_partition_ops_by_arity(self) -> None:
        from kd.search.sga.config import OP1, OP2, OPS

        ops_names = {name for name, _ in OPS}
        combined = {name for name, _ in OP1} | {name for name, _ in OP2}
        assert combined == ops_names

    def test_op1_and_op2_are_disjoint(self) -> None:
        from kd.search.sga.config import OP1, OP2

        op1_names = {name for name, _ in OP1}
        op2_names = {name for name, _ in OP2}
        assert op1_names.isdisjoint(op2_names)

    def test_root_is_subset_of_ops(self) -> None:
        from kd.search.sga.config import OPS, ROOT

        ops_names = {name for name, _ in OPS}
        root_names = {name for name, _ in ROOT}
        assert root_names.issubset(ops_names)

    def test_all_pool_entries_are_name_arity_tuples(self) -> None:
        from kd.search.sga.config import OP1, OP2, OPS, ROOT

        for pool_name, pool in [
            ("OPS", OPS),
            ("ROOT", ROOT),
            ("OP1", OP1),
            ("OP2", OP2),
        ]:
            for entry in pool:
                assert isinstance(entry, tuple), (
                    f"{pool_name} entry is not a tuple: {entry}"
                )
                assert len(entry) == 2, f"{pool_name} entry has wrong length: {entry}"
                name, arity = entry
                assert isinstance(name, str), f"{pool_name} name is not str: {name}"
                assert isinstance(arity, int), f"{pool_name} arity is not int: {arity}"







class TestSGAConfigNegative:

    def test_probability_fields_accept_boundary_values(self) -> None:
        config = SGAConfig(p_var=0.0, p_mute=0.0, p_cro=0.0, p_rep=0.0)
        assert config.p_var == 0.0
        config = SGAConfig(p_var=1.0, p_mute=1.0, p_cro=1.0, p_rep=1.0)
        assert config.p_var == 1.0

    def test_zero_width_accepted(self) -> None:
        config = SGAConfig(width=0)
        assert config.width == 0

    def test_zero_depth_accepted(self) -> None:
        config = SGAConfig(depth=0)
        assert config.depth == 0







class TestDerivativeOperatorGrammar:

    def test_d_in_ops(self) -> None:
        from kd.search.sga.config import OPS

        op_dict = dict(OPS)
        assert "d" in op_dict
        assert op_dict["d"] == 2

    def test_d2_in_ops(self) -> None:
        from kd.search.sga.config import OPS

        op_dict = dict(OPS)
        assert "d^2" in op_dict
        assert op_dict["d^2"] == 2

    def test_d_in_root(self) -> None:
        from kd.search.sga.config import ROOT

        root_dict = dict(ROOT)
        assert "d" in root_dict
        assert root_dict["d"] == 2

    def test_d2_in_root(self) -> None:
        from kd.search.sga.config import ROOT

        root_dict = dict(ROOT)
        assert "d^2" in root_dict
        assert root_dict["d^2"] == 2

    def test_d_in_op2(self) -> None:
        from kd.search.sga.config import OP2

        op2_dict = dict(OP2)
        assert "d" in op2_dict
        assert op2_dict["d"] == 2

    def test_d2_in_op2(self) -> None:
        from kd.search.sga.config import OP2

        op2_dict = dict(OP2)
        assert "d^2" in op2_dict
        assert op2_dict["d^2"] == 2

    def test_d_not_in_op1(self) -> None:
        from kd.search.sga.config import OP1

        op1_names = {name for name, _ in OP1}
        assert "d" not in op1_names
        assert "d^2" not in op1_names












class TestDedupModeConfig:

    @pytest.mark.unit
    def test_default_dedup_mode_is_pre_prune(self) -> None:
        config = SGAConfig()
        assert config.dedup_mode == "pre_prune"

    @pytest.mark.unit
    def test_dedup_mode_accepts_none(self) -> None:
        config = SGAConfig(dedup_mode="none")
        assert config.dedup_mode == "none"

    @pytest.mark.unit
    def test_dedup_mode_accepts_pre_prune(self) -> None:
        config = SGAConfig(dedup_mode="pre_prune")
        assert config.dedup_mode == "pre_prune"

    @pytest.mark.unit
    def test_dedup_mode_accepts_post_prune(self) -> None:
        config = SGAConfig(dedup_mode="post_prune")
        assert config.dedup_mode == "post_prune"

    @pytest.mark.unit
    def test_dedup_mode_accepts_dual(self) -> None:
        config = SGAConfig(dedup_mode="dual")
        assert config.dedup_mode == "dual"

    @pytest.mark.unit
    def test_dedup_mode_is_dataclass_field(self) -> None:
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(SGAConfig)}
        assert "dedup_mode" in field_names, (
        )

    @pytest.mark.unit
    def test_dedup_mode_field_typed_as_literal(self) -> None:
        import dataclasses
        import typing

        fields_by_name = {f.name: f for f in dataclasses.fields(SGAConfig)}
        assert "dedup_mode" in fields_by_name
        field_type = fields_by_name["dedup_mode"].type





        from kd.models.field_model import FieldModel

        hints = typing.get_type_hints(SGAConfig, localns={"FieldModel": FieldModel})
        resolved = hints.get("dedup_mode", field_type)



        args = typing.get_args(resolved)
        expected = {"none", "pre_prune", "post_prune", "dual"}
        assert set(args) == expected, (
            f"dedup_mode should be Literal[{sorted(expected)}], got {resolved} "
            f"with args {args}"
        )

    @pytest.mark.unit
    def test_dedup_mode_in_equality_check(self) -> None:
        a = SGAConfig(dedup_mode="pre_prune")
        b = SGAConfig(dedup_mode="post_prune")
        assert a != b

    @pytest.mark.unit
    def test_other_defaults_preserved_when_dedup_mode_overridden(self) -> None:
        config = SGAConfig(dedup_mode="dual")

        assert config.num == 20
        assert config.depth == 4
        assert config.aic_ratio == 1.0


class TestBuildDen:

    def test_build_den_importable(self) -> None:
        from kd.search.sga.config import build_den

    def test_excludes_lhs_axis(self) -> None:
        from kd.search.sga.config import build_den

        axes = ["x", "t", "y"]
        den = build_den(axes=axes, lhs_axis="t")
        den_names = {name for name, _ in den}
        assert "t" not in den_names

    def test_includes_non_lhs_axes(self) -> None:
        from kd.search.sga.config import build_den

        axes = ["x", "t", "y"]
        den = build_den(axes=axes, lhs_axis="t")
        den_names = {name for name, _ in den}
        assert den_names == {"x", "y"}

    def test_single_spatial_axis(self) -> None:
        from kd.search.sga.config import build_den

        den = build_den(axes=["x", "t"], lhs_axis="t")
        assert len(den) == 1
        assert den[0] == ("x", 0)

    def test_den_entries_are_leaves(self) -> None:
        from kd.search.sga.config import build_den

        den = build_den(axes=["x", "y", "z", "t"], lhs_axis="t")
        for name, arity in den:
            assert arity == 0, f"den entry '{name}' has arity {arity}, expected 0"

    def test_empty_axes_raises(self) -> None:
        from kd.search.sga.config import build_den

        with pytest.raises(ValueError):
            build_den(axes=["t"], lhs_axis="t")

    def test_returns_tuple(self) -> None:
        from kd.search.sga.config import build_den

        den = build_den(axes=["x", "t"], lhs_axis="t")
        assert isinstance(den, (list, tuple))
        for entry in den:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
