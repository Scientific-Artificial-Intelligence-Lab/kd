"""Tests for the param_fields feature across the full stack.

Covers:
- PDEDataset param_fields validation
- SGA adapter + config + context param_fields handling
- DSCV Token param_var + Library param_tokens + create_tokens
- DSCV execute.py param_var branch (no dim_flag)
- DSCV adapter param data passing
- LaTeX rendering of param symbols
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_grid():
    """Minimal 1D grid: u(x, t) with param nu."""
    nx, nt = 32, 20
    x = np.linspace(0, 2 * np.pi, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    u = np.sin(X) * np.exp(-T)
    nu = np.full_like(u, 0.1)
    return x, t, u, nu


@pytest.fixture
def nd_grid():
    """2D spatial grid: u(t, x, y) with param nu."""
    nt, nx, ny = 10, 16, 16
    t = np.linspace(0, 1, nt)
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    T, X, Y = np.meshgrid(t, x, y, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.exp(-T)
    nu = np.full_like(u, 0.05)
    return t, x, y, u, nu


# ===========================================================================
# Phase 1: PDEDataset
# ===========================================================================

class TestPDEDatasetParamFields:
    """PDEDataset param_fields parameter validation."""

    def test_param_fields_stored(self, simple_grid):
        from kd.dataset import PDEDataset

        x, t, u, nu = simple_grid
        ds = PDEDataset(
            equation_name="test",
            x=x, t=t, usol=u,
            param_fields={"nu": nu},
        )
        assert "nu" in ds.param_fields
        np.testing.assert_array_equal(ds.param_fields["nu"], nu)

    def test_param_fields_empty_by_default(self, simple_grid):
        from kd.dataset import PDEDataset

        x, t, u, _nu = simple_grid
        ds = PDEDataset(equation_name="test", x=x, t=t, usol=u)
        assert ds.param_fields == {}

    def test_param_fields_shape_mismatch_raises(self, simple_grid):
        from kd.dataset import PDEDataset

        x, t, u, _nu = simple_grid
        wrong_shape = np.zeros((5, 5))
        with pytest.raises(ValueError, match="does not match"):
            PDEDataset(
                equation_name="test",
                x=x, t=t, usol=u,
                param_fields={"bad": wrong_shape},
            )

    def test_param_fields_nan_raises(self, simple_grid):
        from kd.dataset import PDEDataset

        x, t, u, nu = simple_grid
        nu_bad = nu.copy()
        nu_bad[5, 3] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            PDEDataset(
                equation_name="test",
                x=x, t=t, usol=u,
                param_fields={"nu": nu_bad},
            )

    def test_param_fields_inf_raises(self, simple_grid):
        from kd.dataset import PDEDataset

        x, t, u, nu = simple_grid
        nu_bad = nu.copy()
        nu_bad[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            PDEDataset(
                equation_name="test",
                x=x, t=t, usol=u,
                param_fields={"nu": nu_bad},
            )

    def test_param_fields_multiple(self, simple_grid):
        from kd.dataset import PDEDataset

        x, t, u, nu = simple_grid
        kappa = np.full_like(u, 0.5)
        ds = PDEDataset(
            equation_name="test",
            x=x, t=t, usol=u,
            param_fields={"nu": nu, "kappa": kappa},
        )
        assert len(ds.param_fields) == 2
        assert "nu" in ds.param_fields
        assert "kappa" in ds.param_fields

    def test_param_fields_nd_mode(self, nd_grid):
        from kd.dataset import PDEDataset

        t, x, y, u, nu = nd_grid
        ds = PDEDataset(
            equation_name="test_2d",
            fields_data={"u": u},
            coords_1d={"t": t, "x": x, "y": y},
            axis_order=["t", "x", "y"],
            lhs_axis="t",
            param_fields={"nu": nu},
        )
        assert "nu" in ds.param_fields


# ===========================================================================
# Phase 1: SGA adapter + config + context
# ===========================================================================

class TestSGAParamFields:
    """SGA pipeline param_fields integration."""

    def test_adapter_passes_param_fields(self, simple_grid):
        from kd.dataset import PDEDataset
        from kd.model.sga.adapter import SGADataAdapter

        x, t, u, nu = simple_grid
        # u.shape = (nx, nt) from meshgrid(x, t, indexing="ij")
        # axis_order must match: ["x", "t"]
        ds = PDEDataset(
            equation_name="test",
            fields_data={"u": u},
            coords_1d={"t": t, "x": x},
            axis_order=["x", "t"],
            lhs_axis="t",
            param_fields={"nu": nu},
        )
        adapter = SGADataAdapter(ds)
        kwargs = adapter.to_solver_kwargs()
        assert "param_fields" in kwargs
        assert "nu" in kwargs["param_fields"]

    def test_adapter_no_param_fields(self, simple_grid):
        from kd.dataset import PDEDataset
        from kd.model.sga.adapter import SGADataAdapter

        x, t, u, _nu = simple_grid
        ds = PDEDataset(
            equation_name="test",
            fields_data={"u": u},
            coords_1d={"t": t, "x": x},
            axis_order=["x", "t"],
            lhs_axis="t",
        )
        adapter = SGADataAdapter(ds)
        kwargs = adapter.to_solver_kwargs()
        assert "param_fields" not in kwargs

    def test_solver_config_accepts_param_fields(self, simple_grid):
        from kd.model.sga.sgapde.config import SolverConfig

        x, t, u, nu = simple_grid
        config = SolverConfig(
            problem_name="test",
            fields_data={"u": u},
            coords_1d={"t": t, "x": x},
            axis_order=["x", "t"],
            target_field="u",
            lhs_axis="t",
            param_fields={"nu": nu},
        )
        assert config.param_fields is not None
        assert "nu" in config.param_fields

    def test_context_build_vars_includes_param(self, nd_grid):
        """Param fields appear in VARS but NOT in den (no derivatives)."""
        from kd.model.sga.sgapde.config import SolverConfig
        from kd.model.sga.sgapde.context import ProblemContext

        t, x, y, u, nu = nd_grid
        config = SolverConfig(
            problem_name="test_ctx",
            fields_data={"u": u},
            coords_1d={"t": t, "x": x, "y": y},
            axis_order=["t", "x", "y"],
            target_field="u",
            lhs_axis="t",
            param_fields={"nu": nu},
        )
        ctx = ProblemContext(config)

        # "nu" should be in the variable names
        var_names = [v[0] for v in ctx.VARS]
        assert "nu" in var_names

        # "nu" should NOT be in den (derivative denominator list)
        den_names = [d[0] for d in ctx.den]
        assert "nu" not in den_names


# ===========================================================================
# Phase 2: DSCV Token + Library
# ===========================================================================

class TestDSCVTokenParamVar:
    """DSCV Token param_var attribute and Library param_tokens."""

    def test_token_param_var(self):
        from kd.model.discover.library import Token

        tok = Token(name="nu", arity=0, complexity=1, function=None, param_var=0)
        assert tok.param_var == 0
        assert tok.input_var is None
        assert tok.state_var is None
        assert tok._param_data_ref is None

    def test_library_param_tokens(self):
        from kd.model.discover.library import Token, Library

        tokens = [
            Token(name="x1", arity=0, complexity=1, function=None, input_var=0),
            Token(name="u1", arity=0, complexity=1, function=None, state_var=0),
            Token(name="nu", arity=0, complexity=1, function=None, param_var=0),
        ]
        lib = Library(tokens)

        assert len(lib.param_tokens) == 1
        assert lib.tokens[lib.param_tokens[0]].name == "nu"
        # param tokens should NOT be in input_tokens (critical for Prior)
        assert lib.param_tokens[0] not in lib.input_tokens

    def test_create_tokens_with_params(self):
        from kd.model.discover.functions import create_tokens

        tokens = create_tokens(
            n_input_var=1,
            function_set=["add", "mul"],
            protected=False,
            n_state_var=1,
            task_type="pde",
            n_param_var=2,
            param_names=["nu", "kappa"],
        )
        param_tokens = [t for t in tokens if t.param_var is not None]
        assert len(param_tokens) == 2
        assert param_tokens[0].name == "nu"
        assert param_tokens[0].param_var == 0
        assert param_tokens[1].name == "kappa"
        assert param_tokens[1].param_var == 1

    def test_create_tokens_default_param_names(self):
        from kd.model.discover.functions import create_tokens

        tokens = create_tokens(
            n_input_var=1,
            function_set=["add"],
            protected=False,
            n_state_var=1,
            task_type="pde",
            n_param_var=2,
        )
        param_tokens = [t for t in tokens if t.param_var is not None]
        assert param_tokens[0].name == "p1"
        assert param_tokens[1].name == "p2"

    def test_create_tokens_no_params_by_default(self):
        from kd.model.discover.functions import create_tokens

        tokens = create_tokens(
            n_input_var=1,
            function_set=["add"],
            protected=False,
            n_state_var=1,
            task_type="pde",
        )
        param_tokens = [t for t in tokens if t.param_var is not None]
        assert len(param_tokens) == 0

    def test_create_tokens_name_collision_input_raises(self):
        from kd.model.discover.functions import create_tokens

        with pytest.raises(ValueError, match="collides"):
            create_tokens(
                n_input_var=1,
                function_set=["add"],
                protected=False,
                n_state_var=1,
                task_type="pde",
                n_param_var=1,
                param_names=["x1"],
            )

    def test_create_tokens_name_collision_state_raises(self):
        from kd.model.discover.functions import create_tokens

        with pytest.raises(ValueError, match="collides"):
            create_tokens(
                n_input_var=1,
                function_set=["add"],
                protected=False,
                n_state_var=1,
                task_type="pde",
                n_param_var=1,
                param_names=["u1"],
            )

    def test_create_tokens_name_collision_operator_raises(self):
        from kd.model.discover.functions import create_tokens

        with pytest.raises(ValueError, match="collides"):
            create_tokens(
                n_input_var=1,
                function_set=["add", "mul"],
                protected=False,
                n_state_var=1,
                task_type="pde",
                n_param_var=1,
                param_names=["add"],
            )


# ===========================================================================
# Phase 2: execute.py param_var branch
# ===========================================================================

class TestExecuteParamVar:
    """Execute engine param_var branch — no dim_flag set."""

    def test_param_token_returns_data_no_dimflag(self):
        from kd.model.discover.library import Token
        from kd.model.discover.execute import python_execute

        # Build a minimal traversal: just a param token
        nu_data = np.array([0.1, 0.2, 0.3])
        tok = Token(name="nu", arity=0, complexity=1, function=None, param_var=0)
        tok._param_data_ref = nu_data

        result = python_execute([tok], u=[np.ones(3)], x=[np.ones(3)])
        np.testing.assert_array_equal(result, nu_data)

    def test_param_in_expression(self):
        """param * u1 should multiply param data by state data."""
        from kd.model.discover.library import Token
        from kd.model.discover.execute import python_execute

        nu_data = np.array([0.1, 0.2, 0.3])
        u_data = np.array([1.0, 2.0, 3.0])

        tok_mul = Token(np.multiply, "mul", arity=2, complexity=1)
        tok_nu = Token(name="nu", arity=0, complexity=1, function=None, param_var=0)
        tok_nu._param_data_ref = nu_data
        tok_u = Token(name="u1", arity=0, complexity=1, function=None, state_var=0)

        # Traversal: mul(nu, u1)
        traversal = [tok_mul, tok_nu, tok_u]
        result = python_execute(traversal, u=[u_data], x=[])
        expected = nu_data * u_data
        np.testing.assert_allclose(result, expected)


# ===========================================================================
# Phase 2: DSCV adapter param data
# ===========================================================================

class TestDSCVAdapterParamFields:
    """DSCV adapter param_fields extraction."""

    def test_regular_adapter_legacy_param_fields(self, simple_grid):
        from kd.dataset import PDEDataset
        from kd.model.discover.adapter import DSCVRegularAdapter

        x, t, u, nu = simple_grid
        ds = PDEDataset(
            equation_name="test",
            x=x, t=t, usol=u,
            param_fields={"nu": nu},
        )
        adapter = DSCVRegularAdapter(ds)
        data = adapter.get_data()
        assert "param_data" in data
        assert data["n_param_var"] == 1
        assert data["param_names"] == ["nu"]
        assert data["param_data"][0].shape == nu.shape

    def test_regular_adapter_nd_param_fields(self, nd_grid):
        from kd.dataset import PDEDataset
        from kd.model.discover.adapter import DSCVRegularAdapter

        t, x, y, u, nu = nd_grid
        ds = PDEDataset(
            equation_name="test_2d",
            fields_data={"u": u},
            coords_1d={"t": t, "x": x, "y": y},
            axis_order=["t", "x", "y"],
            lhs_axis="t",
            param_fields={"nu": nu},
        )
        adapter = DSCVRegularAdapter(ds)
        data = adapter.get_data()
        assert "param_data" in data
        assert data["n_param_var"] == 1
        assert data["param_names"] == ["nu"]
        # Param data should be same shape as u (after permute)
        assert data["param_data"][0].shape == data["u"].shape

    def test_regular_adapter_no_param_fields(self, simple_grid):
        from kd.dataset import PDEDataset
        from kd.model.discover.adapter import DSCVRegularAdapter

        x, t, u, _nu = simple_grid
        ds = PDEDataset(equation_name="test", x=x, t=t, usol=u)
        adapter = DSCVRegularAdapter(ds)
        data = adapter.get_data()
        assert "param_data" not in data


# ===========================================================================
# Phase 3: LaTeX rendering
# ===========================================================================

class TestParamFieldsLatex:
    """LaTeX rendering for param symbols."""

    def test_sga_render_leaf_greek(self):
        from kd.model.sga.sgapde.equation import _render_leaf

        assert _render_leaf("nu") == "\\nu"
        assert _render_leaf("kappa") == "\\kappa"
        assert _render_leaf("alpha") == "\\alpha"
        # Non-Greek names pass through
        assert _render_leaf("a") == "a"

    def test_dscv_symbol_display(self):
        from kd.viz.discover_eq2latex import _SYMBOL_DISPLAY

        assert "p1" in _SYMBOL_DISPLAY
        assert _SYMBOL_DISPLAY["p1"] == "p_{1}"
        assert _SYMBOL_DISPLAY["p2"] == "p_{2}"

    def test_dscv_symbols_for_sympy(self):
        from kd.viz.discover_eq2latex import DEEPRL_SYMBOLS_FOR_SYMPY

        assert "p1" in DEEPRL_SYMBOLS_FOR_SYMPY
        assert "p2" in DEEPRL_SYMBOLS_FOR_SYMPY
        assert "p3" in DEEPRL_SYMBOLS_FOR_SYMPY


# ===========================================================================
# Phase 4: Sparse mode — DSCVSparseAdapter param_fields
# ===========================================================================

class TestDSCVSparseAdapterParamFields:
    """DSCVSparseAdapter should pass param_fields through to the data dict."""

    def test_sparse_adapter_legacy_param_fields(self, simple_grid):
        """Legacy 1D dataset with param_fields: data dict has param_data."""
        # RED: will fail until DSCVSparseAdapter._prepare_legacy adds param extraction
        from kd.dataset import PDEDataset
        from kd.model.discover.adapter import DSCVSparseAdapter

        x, t, u, nu = simple_grid
        ds = PDEDataset(
            equation_name="test",
            x=x, t=t, usol=u,
            param_fields={"nu": nu},
        )
        adapter = DSCVSparseAdapter(ds, sample=10, random_state=42)
        data = adapter.get_data()

        assert "param_data" in data, "Sparse adapter must include param_data"
        assert data["n_param_var"] == 1
        assert data["param_names"] == ["nu"]
        # Raw param array should be preserved (shape matches original nu)
        assert len(data["param_data"]) == 1
        np.testing.assert_allclose(data["param_data"][0], nu)

    def test_sparse_adapter_nd_param_fields(self, nd_grid):
        """N-D dataset with param_fields: param_data uses Sparse's time-last perm."""
        # RED: will fail until DSCVSparseAdapter._prepare_nd adds param extraction
        from kd.dataset import PDEDataset
        from kd.model.discover.adapter import DSCVSparseAdapter

        t, x, y, u, nu = nd_grid
        ds = PDEDataset(
            equation_name="test_2d",
            fields_data={"u": u},
            coords_1d={"t": t, "x": x, "y": y},
            axis_order=["t", "x", "y"],
            lhs_axis="t",
            param_fields={"nu": nu},
        )
        adapter = DSCVSparseAdapter(ds, sample=10, random_state=42)
        data = adapter.get_data()

        assert "param_data" in data, "Sparse N-D adapter must include param_data"
        assert data["n_param_var"] == 1
        assert data["param_names"] == ["nu"]
        # Sparse perm is time-last: spatial_axes + [lhs_axis] = (x, y, t)
        # u after perm: shape = (nx, ny, nt)
        expected_shape = (len(x), len(y), len(t))
        assert data["param_data"][0].shape == expected_shape, (
            f"Param data shape {data['param_data'][0].shape} should match "
            f"Sparse time-last u shape {expected_shape}"
        )

    def test_sparse_adapter_no_param_fields(self, simple_grid):
        """Dataset without param_fields: data dict should NOT have param_data."""
        from kd.dataset import PDEDataset
        from kd.model.discover.adapter import DSCVSparseAdapter

        x, t, u, _nu = simple_grid
        ds = PDEDataset(equation_name="test", x=x, t=t, usol=u)
        adapter = DSCVSparseAdapter(ds, sample=10, random_state=42)
        data = adapter.get_data()

        assert "param_data" not in data
        assert "n_param_var" not in data
        assert "param_names" not in data


# ===========================================================================
# Phase 4: Sparse mode — PDEPINNTask param token creation
# ===========================================================================

class TestPDEPINNTaskParamFields:
    """PDEPINNTask should accept n_param_var and create param tokens."""

    def test_pinn_task_creates_param_tokens(self):
        """Construct PDEPINNTask with n_param_var=1: library has param tokens."""
        # RED: will fail until PDEPINNTask.__init__ accepts n_param_var
        from kd.model.discover.task.pde.pde_pinn import PDEPINNTask

        task = PDEPINNTask(
            function_set=["add", "mul"],
            dataset="test",
            metric="inv_nrmse",
            metric_params=(1,),
            n_param_var=1,
            param_names=["nu"],
        )
        # Verify library contains the param token
        param_toks = [
            t for t in task.library.tokens if t.param_var is not None
        ]
        assert len(param_toks) == 1, "Expected 1 param token in library"
        assert param_toks[0].name == "nu"
        assert param_toks[0].param_var == 0

    def test_pinn_task_load_data_binds_refs(self):
        """After load_data with param_data, token._param_data_ref is bound."""
        # RED: will fail until PDEPINNTask accepts n_param_var AND load_data stores refs
        from kd.model.discover.task.pde.pde_pinn import PDEPINNTask

        task = PDEPINNTask(
            function_set=["add", "mul"],
            dataset="test",
            metric="inv_nrmse",
            metric_params=(1,),
            n_param_var=1,
            param_names=["nu"],
        )
        # Simulate Sparse adapter output with param_data
        param_arr = np.full((32,), 0.1)
        data = {"param_data": [param_arr]}
        task.load_data(data)

        param_tok = next(
            t for t in task.library.tokens if t.param_var is not None
        )
        assert param_tok._param_data_ref is not None, (
            "load_data should bind _param_data_ref on param tokens"
        )
        np.testing.assert_array_equal(param_tok._param_data_ref, param_arr)

    def test_pinn_task_no_param_by_default(self):
        """Construct PDEPINNTask without n_param_var: no param tokens."""
        from kd.model.discover.task.pde.pde_pinn import PDEPINNTask

        task = PDEPINNTask(
            function_set=["add", "mul"],
            dataset="test",
            metric="inv_nrmse",
            metric_params=(1,),
        )
        param_toks = [
            t for t in task.library.tokens if t.param_var is not None
        ]
        assert len(param_toks) == 0, "Default task should have no param tokens"


# ===========================================================================
# Phase 4: Sparse mode — _refresh_param_refs
# ===========================================================================

class TestRefreshParamRefs:
    """PDEPINNTask._refresh_param_refs matches u shape/dtype/device."""

    def _make_task_with_param(self):
        """Helper: create a PDEPINNTask with 1 param token."""
        from kd.model.discover.task.pde.pde_pinn import PDEPINNTask

        task = PDEPINNTask(
            function_set=["add", "mul"],
            dataset="test",
            metric="inv_nrmse",
            metric_params=(1,),
            n_param_var=1,
            param_names=["nu"],
        )
        return task

    def test_refresh_numpy(self):
        """When u is numpy, _param_data_ref becomes numpy with matching shape."""
        # RED: will fail until _refresh_param_refs is implemented
        task = self._make_task_with_param()
        # Set up raw param data (full grid) and u (after generate_meta_data)
        task._param_fields_raw = [np.full((100,), 0.1)]
        task.u = [np.random.randn(50)]  # u after collocation sampling

        task._refresh_param_refs()

        param_tok = next(
            t for t in task.library.tokens if t.param_var is not None
        )
        assert isinstance(param_tok._param_data_ref, np.ndarray)
        assert param_tok._param_data_ref.shape == task.u[0].shape
        np.testing.assert_allclose(param_tok._param_data_ref, 0.1)

    def test_refresh_torch(self):
        """When u is torch tensor, _param_data_ref becomes torch with matching shape/dtype/device."""
        # RED: will fail until _refresh_param_refs is implemented
        torch = pytest.importorskip("torch")
        task = self._make_task_with_param()
        task._param_fields_raw = [np.full((100,), 0.25)]
        task.u = [torch.randn(30, dtype=torch.float32)]

        task._refresh_param_refs()

        param_tok = next(
            t for t in task.library.tokens if t.param_var is not None
        )
        assert isinstance(param_tok._param_data_ref, torch.Tensor)
        assert param_tok._param_data_ref.shape == task.u[0].shape
        assert param_tok._param_data_ref.dtype == task.u[0].dtype
        # All values should be 0.25
        expected = torch.full((30,), 0.25, dtype=torch.float32)
        torch.testing.assert_close(param_tok._param_data_ref, expected)

    def test_refresh_varying_raises(self):
        """Spatially varying param_data raises NotImplementedError in PINN mode."""
        # RED: will fail until _refresh_param_refs is implemented
        task = self._make_task_with_param()
        # Non-constant param field — values differ across the grid
        varying = np.linspace(0.0, 1.0, 100)
        task._param_fields_raw = [varying]
        task.u = [np.random.randn(50)]

        with pytest.raises(NotImplementedError, match="[Ss]patially.varying"):
            task._refresh_param_refs()


# ===========================================================================
# Phase 4: Library.np_names must exclude param tokens
# ===========================================================================

class TestNpNamesExcludesParam:
    """Library.np_names should NOT include param tokens (bug fix)."""

    def test_np_names_no_param_tokens(self):
        """Param token names must NOT appear in np_names.

        Current code INCLUDES them (bug) which causes switch_tokens KeyError.
        This test verifies the fix: param tokens are excluded.
        """
        # RED: will fail until library.py L137 adds `and t.param_var is None`
        from kd.model.discover.library import Token, Library

        tokens = [
            Token(np.add, "add", arity=2, complexity=1),
            Token(np.multiply, "mul", arity=2, complexity=1),
            Token(name="x1", arity=0, complexity=1, function=None, input_var=0),
            Token(name="u1", arity=0, complexity=1, function=None, state_var=0),
            Token(name="nu", arity=0, complexity=1, function=None, param_var=0),
            Token(name="kappa", arity=0, complexity=1, function=None, param_var=1),
        ]
        lib = Library(tokens)

        # np_names is used by switch_tokens to map numpy->torch tokens.
        # param tokens do NOT have torch counterparts, so they must be excluded.
        assert "nu" not in lib.np_names, (
            "Param token 'nu' should NOT be in np_names "
            "(would cause KeyError in switch_tokens)"
        )
        assert "kappa" not in lib.np_names, (
            "Param token 'kappa' should NOT be in np_names"
        )
        # Operators and state tokens that pass the existing filter should still be there
        assert "add" in lib.np_names
        assert "mul" in lib.np_names

    def test_np_names_backward_compat(self):
        """Without param tokens, np_names behavior is unchanged."""
        from kd.model.discover.library import Token, Library

        tokens = [
            Token(np.add, "add", arity=2, complexity=1),
            Token(np.multiply, "mul", arity=2, complexity=1),
            Token(name="x1", arity=0, complexity=1, function=None, input_var=0),
            Token(name="u1", arity=0, complexity=1, function=None, state_var=0),
        ]
        lib = Library(tokens)

        # Without param tokens, np_names should be exactly as before:
        # excludes 'const', names with 'u' (except 'mul'), and input_var tokens
        assert "add" in lib.np_names
        assert "mul" in lib.np_names
        assert "x1" not in lib.np_names  # input_var is not None
        assert "u1" not in lib.np_names  # 'u' in name


# ===========================================================================
# Phase 4: DiffConstraint_right must exclude param tokens
# ===========================================================================

class TestDiffConstraintRightParam:
    """DiffConstraint_right targets must include param tokens (prohibit diff(u, param))."""

    def test_diff_constraint_excludes_param(self):
        """With param token + const: param index must be in targets (prohibited as diff right child).

        Current code uses [-2:] which only gets [p1, const] when tokens are
        [x1, u1, p1, const], missing u1. After fix, all non-input terminals
        (u1, p1, const) are in targets.
        """
        # RED: will fail until prior.py DiffConstraint_right is fixed
        from kd.model.discover.library import Token, Library, PlaceholderConstant
        from kd.model.discover.dso.prior import DiffConstraint_right
        from kd.model.discover.functions import create_tokens

        # Build tokens: x1, u1, p1, const + operators including diff
        base_tokens = create_tokens(
            n_input_var=1,
            function_set=["add", "mul", "diff", "diff2"],
            protected=False,
            n_state_var=1,
            task_type="pde",
            n_param_var=1,
            param_names=["p1"],
        )
        # Add const token (PlaceholderConstant)
        const_tok = PlaceholderConstant()
        base_tokens.append(const_tok)
        lib = Library(base_tokens)

        constraint = DiffConstraint_right(lib)

        # Find indices for key tokens
        p1_idx = lib.names.index("p1")
        u1_idx = lib.names.index("u1")
        const_idx = lib.names.index("const")

        # All three (u1, p1, const) should be in targets (prohibited as diff right child)
        targets_set = set(constraint.targets.tolist())
        assert p1_idx in targets_set, (
            f"Param token p1 (idx={p1_idx}) must be prohibited as diff right child"
        )
        assert u1_idx in targets_set, (
            f"State token u1 (idx={u1_idx}) must be prohibited as diff right child"
        )
        assert const_idx in targets_set, (
            f"Const token (idx={const_idx}) must be prohibited as diff right child"
        )

        # x1 should NOT be in targets (it IS a valid diff right child)
        x1_idx = lib.names.index("x1")
        assert x1_idx not in targets_set, (
            "Input token x1 should be ALLOWED as diff right child"
        )

    def test_diff_constraint_no_param_compat(self):
        """Without param tokens, DiffConstraint_right behavior is unchanged.

        Targets should include u1 and const (if present), but not x1.
        """
        from kd.model.discover.library import Token, Library, PlaceholderConstant
        from kd.model.discover.dso.prior import DiffConstraint_right
        from kd.model.discover.functions import create_tokens

        base_tokens = create_tokens(
            n_input_var=1,
            function_set=["add", "mul", "diff", "diff2"],
            protected=False,
            n_state_var=1,
            task_type="pde",
        )
        const_tok = PlaceholderConstant()
        base_tokens.append(const_tok)
        lib = Library(base_tokens)

        constraint = DiffConstraint_right(lib)

        u1_idx = lib.names.index("u1")
        const_idx = lib.names.index("const")
        x1_idx = lib.names.index("x1")
        targets_set = set(constraint.targets.tolist())

        assert u1_idx in targets_set, "u1 must be prohibited as diff right child"
        assert const_idx in targets_set, "const must be prohibited as diff right child"
        assert x1_idx not in targets_set, "x1 should be allowed as diff right child"
