
from __future__ import annotations

import pytest






import kd.search.discover.tokens.prior
from kd.search.discover.paths import REFERENCE_DATA_DIR
from kd.search.discover.runners.pde_registry import (
    _RAW_PRESETS,
    PDE_REGISTRY,
)


_REFS_PDE_DATA = REFERENCE_DATA_DIR
_requires_refs_data = pytest.mark.skipif(
    not _REFS_PDE_DATA.exists(),
    reason="requires refs/ reference PDE data (not shipped in the public tree)",
)

_BURGERS_DEFAULT_OPERATORS = (
    "add",
    "mul",
    "sub",
    "div",
    "diff_x",
    "diff2_x",
    "n2",
    "n3",
)
_CHAFEE_DEFAULT_OPERATORS = (
    "add",
    "mul",
    "sub",
    "div",
    "n2",
    "n3",
    "diff_x",
    "diff2_x",
)
_REFERENCE_MODE2_ENTROPY_GAMMA = 0.7
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
        assert settings.operators == (*_BURGERS_DEFAULT_OPERATORS, "diff3_x")

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
        assert spec.presets["aligned"].attn_length == _CHAFEE_ALIGNED_ATTN_LENGTH

    def test_soft_length_loc_is_10(self) -> None:
        spec = PDE_REGISTRY["chafee"]
        assert (
            spec.presets["aligned"].soft_length_loc == _CHAFEE_ALIGNED_SOFT_LENGTH_LOC
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
        self,
        pde: str,
        tier: str,
    ) -> None:
        preset = _RAW_PRESETS[pde][tier]
        assert "entropy_gamma" in preset, (
            f"preset {pde!r}/{tier!r} must declare entropy_gamma after B3"
        )
        assert preset["entropy_gamma"] == _REFERENCE_MODE2_ENTROPY_GAMMA

    def test_tier_settings_reads_entropy_gamma(
        self,
        pde: str,
        tier: str,
    ) -> None:
        spec = PDE_REGISTRY[pde]
        assert spec.presets[tier].entropy_gamma == _REFERENCE_MODE2_ENTROPY_GAMMA


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
        tier: preset["entropy_gamma"] for tier, preset in _RAW_PRESETS["chafee"].items()
    }
    assert len(set(gammas.values())) == 1, (
        f"chafee presets disagree on entropy_gamma: {gammas}"
    )


@pytest.mark.unit
@_requires_refs_data
def test_burgers_data_loader_returns_pde_dataset() -> None:
    spec = PDE_REGISTRY["burgers"]
    dataset = spec.load_data("fast")
    assert dataset.name == "burgers"
    assert dataset.fields is not None
    assert "u" in dataset.fields


@pytest.mark.unit
@_requires_refs_data
def test_chafee_data_loader_returns_pde_dataset() -> None:
    spec = PDE_REGISTRY["chafee"]
    dataset = spec.load_data("fast")
    assert dataset.name == "chafee_infante"
    assert dataset.fields is not None
    assert "u" in dataset.fields




























_BURGERS_COEF_PDE_REFERENCE = 1.0
_CHAFEE_COEF_PDE_REFERENCE = 1.0



_FISHER_LINEAR_COEF_PDE_REFERENCE = 1.0
_KDV_COEF_PDE_REFERENCE = 1.0
_FISHER_NONLINEAR_COEF_PDE_REFERENCE = 1.0
_PDE_COMPOUND_COEF_PDE_REFERENCE = 1.0
_PDE_DIVIDE_COEF_PDE_REFERENCE = 1.0


@pytest.mark.unit
class TestCoefPDESilentRegressionGuards:

    def test_burgers_coef_pde_matches_reference(self) -> None:
        from kd.search.discover.runners.pde_registry import _BURGERS_COEF_PDE

        assert _BURGERS_COEF_PDE == _BURGERS_COEF_PDE_REFERENCE, (
            f"_BURGERS_COEF_PDE = {_BURGERS_COEF_PDE!r} != reference "
            f"{_BURGERS_COEF_PDE_REFERENCE}; see / 512d6dd."
        )

    def test_chafee_coef_pde_matches_reference(self) -> None:
        from kd.search.discover.runners.pde_registry import _CHAFEE_COEF_PDE

        assert _CHAFEE_COEF_PDE == _CHAFEE_COEF_PDE_REFERENCE, (
            f"_CHAFEE_COEF_PDE = {_CHAFEE_COEF_PDE!r} != reference "
            f"{_CHAFEE_COEF_PDE_REFERENCE}; see / parallel."
        )

    def test_fisher_linear_coef_pde_matches_reference(self) -> None:
        from kd.search.discover.runners.pde_registry import _FISHER_LINEAR_COEF_PDE

        assert _FISHER_LINEAR_COEF_PDE == _FISHER_LINEAR_COEF_PDE_REFERENCE, (
            f"_FISHER_LINEAR_COEF_PDE = {_FISHER_LINEAR_COEF_PDE!r} != "
            f"reference {_FISHER_LINEAR_COEF_PDE_REFERENCE}; see lesson."
        )

    def test_kdv_coef_pde_matches_reference(self) -> None:
        from kd.search.discover.runners.pde_registry import _KDV_COEF_PDE

        assert _KDV_COEF_PDE == _KDV_COEF_PDE_REFERENCE, (
            f"_KDV_COEF_PDE = {_KDV_COEF_PDE!r} != "
            f"reference {_KDV_COEF_PDE_REFERENCE}; see lesson."
        )

    def test_fisher_nonlinear_coef_pde_matches_reference(self) -> None:
        from kd.search.discover.runners.pde_registry import _FISHER_NONLINEAR_COEF_PDE

        assert _FISHER_NONLINEAR_COEF_PDE == _FISHER_NONLINEAR_COEF_PDE_REFERENCE, (
            f"_FISHER_NONLINEAR_COEF_PDE = {_FISHER_NONLINEAR_COEF_PDE!r}"
            f" != reference {_FISHER_NONLINEAR_COEF_PDE_REFERENCE}; "
            f"see lesson."
        )

    def test_pde_compound_coef_pde_matches_reference(self) -> None:
        from kd.search.discover.runners.pde_registry import _PDE_COMPOUND_COEF_PDE

        assert _PDE_COMPOUND_COEF_PDE == _PDE_COMPOUND_COEF_PDE_REFERENCE, (
            f"_PDE_COMPOUND_COEF_PDE = {_PDE_COMPOUND_COEF_PDE!r} != "
            f"reference {_PDE_COMPOUND_COEF_PDE_REFERENCE}; see lesson."
        )

    def test_pde_divide_coef_pde_matches_reference(self) -> None:
        from kd.search.discover.runners.pde_registry import _PDE_DIVIDE_COEF_PDE

        assert _PDE_DIVIDE_COEF_PDE == _PDE_DIVIDE_COEF_PDE_REFERENCE, (
            f"_PDE_DIVIDE_COEF_PDE = {_PDE_DIVIDE_COEF_PDE!r} != "
            f"reference {_PDE_DIVIDE_COEF_PDE_REFERENCE}; see lesson."
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
            "_pde_specs/kdv.py: KDV_SPEC no longer threads coef_pde=KDV_COEF_PDE."
        )

        fisher_nonlinear_src = inspect.getsource(fisher_nonlinear_mod)
        assert "coef_pde=FISHER_NONLINEAR_COEF_PDE" in fisher_nonlinear_src, (
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

        from kd.data import noise
        from kd.search.discover.data import loader

        def _body_src_without_docstring(func: object) -> str:




            src = inspect.getsource(func)
            tree = ast.parse(src)
            func_def = tree.body[0]
            assert isinstance(func_def, ast.FunctionDef)
            body: list[ast.stmt] = list(func_def.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return "\n".join(ast.unparse(stmt) for stmt in body)

        recipe_src = _body_src_without_docstring(noise.discover_unnormalized)


        assert "torch.std(values, unbiased=True)" in recipe_src, (
            "kd.data.noise.discover_unnormalized std-path must call "
            "torch.std(values, unbiased=True) for reference parity. "
            "See / commit 3c8fc5a."
        )


        assert "unbiased=False" not in recipe_src, (
            "discover_unnormalized body contains unbiased=False — "
            "documents that this matches numpy/TensorFlow ddof=0 but "
            "DIVERGES from reference torch tensor.std() default. "
            "Multi-seed Burgers MODE2 chaotic-sensitive."
        )


        wrapper_src = _body_src_without_docstring(loader.add_gaussian_noise)
        assert "discover_unnormalized(" in wrapper_src, (
            "add_gaussian_noise no longer delegates to "
            "kd.data.noise.discover_unnormalized — the unbiased=True "
            "lock above would silently stop covering the loader path."
        )
