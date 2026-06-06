
from __future__ import annotations

import pytest






import kd.search.discover.tokens.prior
from kd.search.discover.runners.pde_registry import (
    _RAW_PRESETS,
    PDE_REGISTRY,
)

_BURGERS_DEFAULT_OPERATORS = (
    "add", "mul", "sub", "div", "diff_x", "diff2_x", "n2", "n3",
)
_CHAFEE_DEFAULT_OPERATORS = (
    "add", "mul", "sub", "div", "n2", "n3", "diff_x", "diff2_x",
)
_TF1_MODE2_ENTROPY_GAMMA = 0.7
_PAPER_CHAFEE_ALIGNED_N_CYCLES = 2
_BURGERS_ALIGNED_CYCLE_N_ITERATIONS = 20
_CHAFEE_ALIGNED_ATTN_LENGTH = 20
_CHAFEE_ALIGNED_SOFT_LENGTH_LOC = 10.0
_BURGERS_ALIGNED_MAX_LENGTH = 256
_BURGERS_ALIGNED_STABILITY_SELECTION = 3
_DEFAULT_MAX_LENGTH = 30


@pytest.mark.unit
class TestBurgersAlignedTier:

    def test_data_path_uses_burgers2_mat(self) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.data_path_for_tier("aligned").name == "burgers2.mat"

    def test_operators_add_diff3_x(self) -> None:
        spec = PDE_REGISTRY["burgers"]
        settings = spec.presets["aligned"]
        assert settings.operators == (
            *_BURGERS_DEFAULT_OPERATORS, "diff3_x"
        )

    def test_collocation_cut_ratio_is_zero(self) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.presets["aligned"].collocation_cut_ratio == 0.0

    def test_max_length_is_256(self) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.presets["aligned"].max_length == _BURGERS_ALIGNED_MAX_LENGTH

    def test_attention_enabled(self) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.presets["aligned"].attention is True

    def test_stability_selection_is_three(self) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert (
            spec.presets["aligned"].stability_selection
            == _BURGERS_ALIGNED_STABILITY_SELECTION
        )

    def test_cycle_n_iterations_is_20(self) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert (
            spec.presets["aligned"].cycle_n_iterations
            == _BURGERS_ALIGNED_CYCLE_N_ITERATIONS
        )


@pytest.mark.unit
@pytest.mark.parametrize("tier", ["fast", "medium", "full"])
class TestBurgersDefaultTiers:

    def test_data_path_uses_burgers_mat(self, tier: str) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.data_path_for_tier(tier).name == "burgers.mat"

    def test_operators_match_default(self, tier: str) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.presets[tier].operators == _BURGERS_DEFAULT_OPERATORS

    def test_collocation_cut_ratio_is_005(self, tier: str) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.presets[tier].collocation_cut_ratio == 0.05

    def test_max_length_is_30(self, tier: str) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.presets[tier].max_length == _DEFAULT_MAX_LENGTH

    def test_attention_disabled(self, tier: str) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.presets[tier].attention is False

    def test_cycle_n_iterations_is_none(self, tier: str) -> None:
        spec = PDE_REGISTRY["burgers"]
        assert spec.presets[tier].cycle_n_iterations is None


@pytest.mark.unit
class TestChafeeAlignedTier:

    def test_attn_length_is_20(self) -> None:
        spec = PDE_REGISTRY["chafee"]
        assert (
            spec.presets["aligned"].attn_length
            == _CHAFEE_ALIGNED_ATTN_LENGTH
        )

    def test_soft_length_loc_is_10(self) -> None:
        spec = PDE_REGISTRY["chafee"]
        assert (
            spec.presets["aligned"].soft_length_loc
            == _CHAFEE_ALIGNED_SOFT_LENGTH_LOC
        )

    def test_aligned_n_cycles_is_paper_two(self) -> None:
        spec = PDE_REGISTRY["chafee"]
        assert spec.presets["aligned"].n_cycles == _PAPER_CHAFEE_ALIGNED_N_CYCLES

    def test_aligned_operators_include_diff3_x(self) -> None:
        spec = PDE_REGISTRY["chafee"]
        assert "diff3_x" in spec.presets["aligned"].operators

    def test_cycle_n_iterations_is_20(self) -> None:
        spec = PDE_REGISTRY["chafee"]


        assert spec.presets["aligned"].cycle_n_iterations == 20


@pytest.mark.unit
@pytest.mark.parametrize("tier", ["fast", "medium", "full"])
class TestChafeeDefaultTiers:

    def test_operators_match_default(self, tier: str) -> None:
        spec = PDE_REGISTRY["chafee"]
        assert spec.presets[tier].operators == _CHAFEE_DEFAULT_OPERATORS

    def test_max_length_is_30(self, tier: str) -> None:
        spec = PDE_REGISTRY["chafee"]
        assert spec.presets[tier].max_length == _DEFAULT_MAX_LENGTH

    def test_attention_disabled(self, tier: str) -> None:
        spec = PDE_REGISTRY["chafee"]
        assert spec.presets[tier].attention is False

    def test_soft_length_loc_is_12(self, tier: str) -> None:
        spec = PDE_REGISTRY["chafee"]
        assert spec.presets[tier].soft_length_loc == 12.0


@pytest.mark.unit
@pytest.mark.parametrize("pde", ["burgers", "chafee"])
@pytest.mark.parametrize("tier", ["fast", "medium", "full", "aligned"])
class TestEntropyGammaInvariant:

    def test_raw_preset_carries_entropy_gamma(
        self, pde: str, tier: str,
    ) -> None:
        preset = _RAW_PRESETS[pde][tier]
        assert "entropy_gamma" in preset, (
            f"preset {pde!r}/{tier!r} must declare entropy_gamma after B3"
        )
        assert preset["entropy_gamma"] == _TF1_MODE2_ENTROPY_GAMMA

    def test_tier_settings_reads_entropy_gamma(
        self, pde: str, tier: str,
    ) -> None:
        spec = PDE_REGISTRY[pde]
        assert spec.presets[tier].entropy_gamma == _TF1_MODE2_ENTROPY_GAMMA


@pytest.mark.unit
def test_burgers_presets_share_entropy_gamma() -> None:
    gammas = {
        tier: preset["entropy_gamma"]
        for tier, preset in _RAW_PRESETS["burgers"].items()
    }
    assert len(set(gammas.values())) == 1, (
        f"burgers presets disagree on entropy_gamma: {gammas}"
    )


@pytest.mark.unit
def test_chafee_presets_share_entropy_gamma() -> None:
    gammas = {
        tier: preset["entropy_gamma"]
        for tier, preset in _RAW_PRESETS["chafee"].items()
    }
    assert len(set(gammas.values())) == 1, (
        f"chafee presets disagree on entropy_gamma: {gammas}"
    )


@pytest.mark.unit
def test_burgers_data_loader_returns_pde_dataset() -> None:
    spec = PDE_REGISTRY["burgers"]
    dataset = spec.load_data("fast")
    assert dataset.name == "burgers"
    assert dataset.fields is not None
    assert "u" in dataset.fields


@pytest.mark.unit
def test_chafee_data_loader_returns_pde_dataset() -> None:
    spec = PDE_REGISTRY["chafee"]
    dataset = spec.load_data("fast")
    assert dataset.name == "chafee_infante"
    assert dataset.fields is not None
    assert "u" in dataset.fields




























_BURGERS_COEF_PDE_V1SHIP = 1.0
_CHAFEE_COEF_PDE_V1SHIP = 1.0



_FISHER_LINEAR_COEF_PDE_V1SHIP = 1.0
_KDV_COEF_PDE_V1SHIP = 1.0
_FISHER_NONLINEAR_COEF_PDE_V1SHIP = 1.0
_PDE_COMPOUND_COEF_PDE_V1SHIP = 1.0
_PDE_DIVIDE_COEF_PDE_V1SHIP = 1.0


@pytest.mark.unit
class TestCoefPDESilentRegressionGuards:

    def test_burgers_coef_pde_matches_v1ship(self) -> None:
        from kd.search.discover.runners.pde_registry import _BURGERS_COEF_PDE

        assert _BURGERS_COEF_PDE == _BURGERS_COEF_PDE_V1SHIP, (
            f"_BURGERS_COEF_PDE = {_BURGERS_COEF_PDE!r} != v1ship "
            f"{_BURGERS_COEF_PDE_V1SHIP}; see / 512d6dd."
        )

    def test_chafee_coef_pde_matches_v1ship(self) -> None:
        from kd.search.discover.runners.pde_registry import _CHAFEE_COEF_PDE

        assert _CHAFEE_COEF_PDE == _CHAFEE_COEF_PDE_V1SHIP, (
            f"_CHAFEE_COEF_PDE = {_CHAFEE_COEF_PDE!r} != v1ship "
            f"{_CHAFEE_COEF_PDE_V1SHIP}; see / parallel."
        )

    def test_fisher_linear_coef_pde_matches_v1ship(self) -> None:
        from kd.search.discover.runners.pde_registry import _FISHER_LINEAR_COEF_PDE

        assert _FISHER_LINEAR_COEF_PDE == _FISHER_LINEAR_COEF_PDE_V1SHIP, (
            f"_FISHER_LINEAR_COEF_PDE = {_FISHER_LINEAR_COEF_PDE!r} != "
            f"v1ship {_FISHER_LINEAR_COEF_PDE_V1SHIP}; see lesson."
        )

    def test_kdv_coef_pde_matches_v1ship(self) -> None:
        from kd.search.discover.runners.pde_registry import _KDV_COEF_PDE

        assert _KDV_COEF_PDE == _KDV_COEF_PDE_V1SHIP, (
            f"_KDV_COEF_PDE = {_KDV_COEF_PDE!r} != "
            f"v1ship {_KDV_COEF_PDE_V1SHIP}; see lesson."
        )

    def test_fisher_nonlinear_coef_pde_matches_v1ship(self) -> None:
        from kd.search.discover.runners.pde_registry import _FISHER_NONLINEAR_COEF_PDE

        assert (
            _FISHER_NONLINEAR_COEF_PDE
            == _FISHER_NONLINEAR_COEF_PDE_V1SHIP
        ), (
            f"_FISHER_NONLINEAR_COEF_PDE = {_FISHER_NONLINEAR_COEF_PDE!r}"
            f" != v1ship {_FISHER_NONLINEAR_COEF_PDE_V1SHIP}; "
            f"see lesson."
        )

    def test_pde_compound_coef_pde_matches_v1ship(self) -> None:
        from kd.search.discover.runners.pde_registry import _PDE_COMPOUND_COEF_PDE

        assert (
            _PDE_COMPOUND_COEF_PDE == _PDE_COMPOUND_COEF_PDE_V1SHIP
        ), (
            f"_PDE_COMPOUND_COEF_PDE = {_PDE_COMPOUND_COEF_PDE!r} != "
            f"v1ship {_PDE_COMPOUND_COEF_PDE_V1SHIP}; see lesson."
        )

    def test_pde_divide_coef_pde_matches_v1ship(self) -> None:
        from kd.search.discover.runners.pde_registry import _PDE_DIVIDE_COEF_PDE

        assert (
            _PDE_DIVIDE_COEF_PDE == _PDE_DIVIDE_COEF_PDE_V1SHIP
        ), (
            f"_PDE_DIVIDE_COEF_PDE = {_PDE_DIVIDE_COEF_PDE!r} != "
            f"v1ship {_PDE_DIVIDE_COEF_PDE_V1SHIP}; see lesson."
        )

    def test_burgers_coef_pde_threaded_into_pde_registry(self) -> None:
        import inspect

        from kd.search.discover.runners._pde_specs import burgers as burgers_mod
        from kd.search.discover.runners._pde_specs import chafee as chafee_mod
        from kd.search.discover.runners._pde_specs import (
            fisher_linear as fisher_linear_mod,
        )
        from kd.search.discover.runners._pde_specs import (
            fisher_nonlinear as fisher_nonlinear_mod,
        )
        from kd.search.discover.runners._pde_specs import kdv as kdv_mod
        from kd.search.discover.runners._pde_specs import (
            pde_compound as pde_compound_mod,
        )
        from kd.search.discover.runners._pde_specs import (
            pde_divide as pde_divide_mod,
        )

        burgers_src = inspect.getsource(burgers_mod)
        assert "coef_pde=BURGERS_COEF_PDE" in burgers_src, (
            "_pde_specs/burgers.py: BURGERS_SPEC no longer threads "
            "coef_pde=BURGERS_COEF_PDE. Adding silent default would "
            "mask future -style regressions."
        )

        chafee_src = inspect.getsource(chafee_mod)
        assert "coef_pde=CHAFEE_COEF_PDE" in chafee_src, (
            "_pde_specs/chafee.py: CHAFEE_SPEC no longer threads "
            "coef_pde=CHAFEE_COEF_PDE."
        )

        fisher_linear_src = inspect.getsource(fisher_linear_mod)
        assert "coef_pde=FISHER_LINEAR_COEF_PDE" in fisher_linear_src, (
            "_pde_specs/fisher_linear.py: FISHER_LINEAR_SPEC no longer "
            "threads coef_pde=FISHER_LINEAR_COEF_PDE."
        )

        kdv_src = inspect.getsource(kdv_mod)
        assert "coef_pde=KDV_COEF_PDE" in kdv_src, (
            "_pde_specs/kdv.py: KDV_SPEC no longer threads "
            "coef_pde=KDV_COEF_PDE."
        )

        fisher_nonlinear_src = inspect.getsource(fisher_nonlinear_mod)
        assert (
            "coef_pde=FISHER_NONLINEAR_COEF_PDE" in fisher_nonlinear_src
        ), (
            "_pde_specs/fisher_nonlinear.py: FISHER_NONLINEAR_SPEC no "
            "longer threads coef_pde=FISHER_NONLINEAR_COEF_PDE."
        )

        pde_compound_src = inspect.getsource(pde_compound_mod)
        assert "coef_pde=PDE_COMPOUND_COEF_PDE" in pde_compound_src, (
            "_pde_specs/pde_compound.py: PDE_COMPOUND_SPEC no longer "
            "threads coef_pde=PDE_COMPOUND_COEF_PDE."
        )

        pde_divide_src = inspect.getsource(pde_divide_mod)
        assert "coef_pde=PDE_DIVIDE_COEF_PDE" in pde_divide_src, (
            "_pde_specs/pde_divide.py: PDE_DIVIDE_SPEC no longer "
            "threads coef_pde=PDE_DIVIDE_COEF_PDE."
        )


@pytest.mark.unit
class TestNoiseHelperSilentRegressionGuards:

    def test_default_scale_path_uses_unbiased_true(self) -> None:
        import ast
        import inspect

        from kd.search.discover.data import loader





        src = inspect.getsource(loader.add_gaussian_noise)
        tree = ast.parse(src)


        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        body_without_docstring: list[ast.stmt] = list(func_def.body)
        if (
            body_without_docstring
            and isinstance(body_without_docstring[0], ast.Expr)
            and isinstance(body_without_docstring[0].value, ast.Constant)
            and isinstance(body_without_docstring[0].value.value, str)
        ):
            body_without_docstring = body_without_docstring[1:]
        body_only_src = "\n".join(
            ast.unparse(stmt) for stmt in body_without_docstring
        )


        assert "torch.std(values, unbiased=True)" in body_only_src, (
            "add_gaussian_noise std-path must call "
            "torch.std(values, unbiased=True) for v1ship parity. "
            "See / commit 3c8fc5a."
        )


        assert "unbiased=False" not in body_only_src, (
            "add_gaussian_noise body contains unbiased=False — "
            "documents that this matches numpy/TensorFlow ddof=0 but "
            "DIVERGES from v1ship torch tensor.std() default. "
            "Multi-seed Burgers MODE2 chaotic-sensitive."
        )
