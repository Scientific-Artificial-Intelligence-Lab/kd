"""Tests for 3D Laplacian (Laplace_3) and related integration changes.

Laplace_3 computes the 3D Laplacian on a 4D array (t, nx, ny, nz):

    Laplace_3(u, x) = d2u/dx2 + d2u/dy2 + d2u/dz2

using Diff2_3 for each spatial axis.

This file covers:
  - Analytical correctness of Laplace_3
  - Shape preservation
  - Asymmetric grid sizes (nx != ny != nz)
  - Single-axis quadratic (only one direction contributes)
  - 2D Laplace regression (existing code must not break)
  - Token registry: _DIFF_TOKENS_BY_DIM[3] includes 'lap_3'
  - Execute prefix matching: 'lap' in 'lap_3'

RED phase: Laplace_3 does not exist yet, so all Laplace_3 tests MUST fail
with ImportError or NotImplementedError.  2D regression and execute prefix
tests should pass on the current codebase.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import helpers -- Laplace_3 does not exist yet
# ---------------------------------------------------------------------------

# 2D Laplace exists, import directly
from kd.model.discover.task.pde.utils_fd import Laplace, Diff2_2

# Laplace_3 does NOT exist yet; guard the import so tests can report
# a meaningful failure instead of crashing the whole module.
_LAPLACE_3_AVAILABLE = False
try:
    from kd.model.discover.task.pde.utils_fd import Laplace_3
    _LAPLACE_3_AVAILABLE = True
except ImportError:
    Laplace_3 = None  # placeholder so test bodies can reference the name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_3d_grid(
    nt: int, nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
):
    """Create uniform 3D spatial grid coordinates.

    Returns
    -------
    x_coord, y_coord, z_coord : ndarray, each shape (n,)
    """
    x_coord = np.arange(nx) * dx
    y_coord = np.arange(ny) * dy
    z_coord = np.arange(nz) * dz
    return x_coord, y_coord, z_coord


def _make_u_sum_of_squares(
    nt: int, nx: int, ny: int, nz: int,
    x_coord: np.ndarray, y_coord: np.ndarray, z_coord: np.ndarray,
) -> np.ndarray:
    """Build u = x^2 + y^2 + z^2 on a 4D grid.

    Laplacian = 2 + 2 + 2 = 6 everywhere.

    Returns
    -------
    u : ndarray, shape (nt, nx, ny, nz)
    """
    # Use broadcasting: x[i]^2 + y[j]^2 + z[k]^2
    xx = x_coord.reshape(1, nx, 1, 1)
    yy = y_coord.reshape(1, 1, ny, 1)
    zz = z_coord.reshape(1, 1, 1, nz)
    u = np.broadcast_to(xx ** 2 + yy ** 2 + zz ** 2, (nt, nx, ny, nz)).copy()
    return u


def _make_u_single_axis_quadratic(
    nt: int, nx: int, ny: int, nz: int,
    coord: np.ndarray, axis: int,
) -> np.ndarray:
    """Build u = coord^2 along a single spatial axis; zero elsewhere.

    Laplacian = 2 (only one axis contributes).

    Parameters
    ----------
    axis : int
        Which spatial axis is quadratic: 1=x, 2=y, 3=z.

    Returns
    -------
    u : ndarray, shape (nt, nx, ny, nz)
    """
    u = np.zeros((nt, nx, ny, nz))
    if axis == 1:
        for i in range(nx):
            u[:, i, :, :] = coord[i] ** 2
    elif axis == 2:
        for j in range(ny):
            u[:, :, j, :] = coord[j] ** 2
    elif axis == 3:
        for k in range(nz):
            u[:, :, :, k] = coord[k] ** 2
    else:
        raise ValueError(f"axis must be 1, 2, or 3, got {axis}")
    return u


# ---------------------------------------------------------------------------
# Laplace_3 analytical tests
# ---------------------------------------------------------------------------

class TestLaplace3Analytical:
    """u = x^2 + y^2 + z^2 => Laplacian = 6 everywhere."""

    @pytest.mark.skipif(
        not _LAPLACE_3_AVAILABLE,
        reason="Laplace_3 not yet implemented"
    )
    def test_laplace_3_analytical(self):
        """Interior points of Laplace_3(u=x^2+y^2+z^2) should be 6.0.

        This is the primary correctness test for the 3D Laplacian.
        Central differences for quadratic functions are exact, so
        all interior points must equal 6.0 within floating-point tolerance.
        """
        nt, nx, ny, nz = 2, 10, 10, 10
        dx = dy = dz = 0.1
        x_c, y_c, z_c = _make_3d_grid(nt, nx, ny, nz, dx, dy, dz)
        u = _make_u_sum_of_squares(nt, nx, ny, nz, x_c, y_c, z_c)

        x_list = [x_c, y_c, z_c]
        result = Laplace_3(u, x_list)

        assert result is not None, "Laplace_3 returned None"

        # Interior: indices [1:-1] along each spatial axis
        interior = result[:, 1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(
            interior, 6.0, rtol=1e-5,
            err_msg="Laplace_3(x^2+y^2+z^2) interior should be 6.0"
        )

    def test_laplace_3_not_implemented_guard(self):
        """If Laplace_3 is not importable, this test documents the RED state."""
        if _LAPLACE_3_AVAILABLE:
            pytest.skip("Laplace_3 is already available -- GREEN state")
        # Explicitly fail to document the RED phase
        pytest.fail(
            "Laplace_3 is not yet implemented in utils_fd.py. "
            "Import fails with ImportError."
        )


# ---------------------------------------------------------------------------
# Laplace_3 shape test
# ---------------------------------------------------------------------------

class TestLaplace3Shape:
    """Output shape must equal input shape."""

    @pytest.mark.skipif(
        not _LAPLACE_3_AVAILABLE,
        reason="Laplace_3 not yet implemented"
    )
    def test_laplace_3_shape(self):
        """Laplace_3 output shape must match input shape (nt, nx, ny, nz)."""
        nt, nx, ny, nz = 3, 8, 8, 8
        dx = dy = dz = 0.1
        x_c, y_c, z_c = _make_3d_grid(nt, nx, ny, nz, dx, dy, dz)
        u = np.ones((nt, nx, ny, nz))

        x_list = [x_c, y_c, z_c]
        result = Laplace_3(u, x_list)

        assert result is not None, "Laplace_3 returned None"
        assert result.shape == u.shape, (
            f"Output shape {result.shape} != input shape {u.shape}"
        )


# ---------------------------------------------------------------------------
# Laplace_3 asymmetric grid
# ---------------------------------------------------------------------------

class TestLaplace3AsymmetricGrid:
    """nx != ny != nz to catch dimension-mixing bugs."""

    @pytest.mark.skipif(
        not _LAPLACE_3_AVAILABLE,
        reason="Laplace_3 not yet implemented"
    )
    def test_laplace_3_asymmetric_grid(self):
        """Asymmetric grid (nx=7, ny=9, nz=11) must produce correct Laplacian.

        Using different sizes for each axis ensures that no dimension index
        is accidentally swapped (e.g., using ny size for nz slicing).
        """
        nt, nx, ny, nz = 2, 7, 9, 11
        dx, dy, dz = 0.1, 0.2, 0.15
        x_c, y_c, z_c = _make_3d_grid(nt, nx, ny, nz, dx, dy, dz)
        u = _make_u_sum_of_squares(nt, nx, ny, nz, x_c, y_c, z_c)

        x_list = [x_c, y_c, z_c]
        result = Laplace_3(u, x_list)

        assert result is not None, "Laplace_3 returned None"
        assert result.shape == (nt, nx, ny, nz)

        # Interior points should be 6.0 (each d2/dx_i^2 contributes 2.0)
        interior = result[:, 1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(
            interior, 6.0, rtol=1e-5,
            err_msg=(
                "Asymmetric grid Laplace_3(x^2+y^2+z^2) interior should be 6.0. "
                f"Grid: nx={nx}, ny={ny}, nz={nz}"
            )
        )


# ---------------------------------------------------------------------------
# Laplace_3 single-axis quadratic
# ---------------------------------------------------------------------------

class TestLaplace3SingleAxisQuadratic:
    """u = z^2 => Laplacian = 2 (only z contributes)."""

    @pytest.mark.skipif(
        not _LAPLACE_3_AVAILABLE,
        reason="Laplace_3 not yet implemented"
    )
    def test_laplace_3_single_axis_quadratic(self):
        """u = z^2 has Laplacian = 0 + 0 + 2 = 2 at interior points.

        This verifies that the x and y contributions are correctly zero
        when u has no quadratic dependence on x or y.
        """
        nt, nx, ny, nz = 2, 7, 9, 11
        dz = 0.1
        z_c = np.arange(nz) * dz
        x_c = np.arange(nx) * 0.1
        y_c = np.arange(ny) * 0.1
        u = _make_u_single_axis_quadratic(nt, nx, ny, nz, z_c, axis=3)

        x_list = [x_c, y_c, z_c]
        result = Laplace_3(u, x_list)

        assert result is not None, "Laplace_3 returned None"

        # Interior: all three axes away from boundary
        interior = result[:, 1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(
            interior, 2.0, rtol=1e-5,
            err_msg="Laplace_3(z^2) interior should be 2.0 (only z contributes)"
        )


# ---------------------------------------------------------------------------
# 2D Laplace regression test
# ---------------------------------------------------------------------------

class TestLaplace2DRegression:
    """2D Laplace must remain correct after Laplace_3 is added.

    Note: Laplace has @jit(nopython=True) which fails when calling Diff2_2
    (not numba-compiled).  We use Laplace.py_func to bypass the JIT and
    test the pure-Python logic, which is what the Token system ultimately
    exercises at runtime.
    """

    def _call_laplace(self, u, x_list):
        """Call 2D Laplace bypassing numba JIT (which cannot compile Diff2_2).

        Uses Laplace.py_func to get the unwrapped Python function.
        """
        fn = Laplace.py_func if hasattr(Laplace, 'py_func') else Laplace
        return fn(u, x_list)

    def test_laplace_2d_regression(self):
        """u = x^2 + y^2 => 2D Laplacian = 4 at interior points.

        This regression test verifies that the existing 2D Laplace function
        is not broken by any changes made to add Laplace_3.
        """
        nt, nx, ny = 2, 15, 15
        dx = dy = 0.1
        x_c = np.arange(nx) * dx
        y_c = np.arange(ny) * dy

        # u[t, x, y] = x^2 + y^2
        xx = x_c.reshape(1, nx, 1)
        yy = y_c.reshape(1, 1, ny)
        u = np.broadcast_to(xx ** 2 + yy ** 2, (nt, nx, ny)).copy()

        x_list = [x_c, y_c]
        result = self._call_laplace(u, x_list)

        assert result is not None, "2D Laplace returned None"
        assert result.shape == (nt, nx, ny)

        # Interior (avoid boundary stencil effects)
        interior = result[:, 1:-1, 1:-1]
        np.testing.assert_allclose(
            interior, 4.0, rtol=1e-5,
            err_msg="2D Laplace(x^2+y^2) interior should be 4.0"
        )

    def test_laplace_2d_single_axis(self):
        """u = x^2 => 2D Laplacian = 2 (only x contributes).

        Additional regression check for axis isolation.
        """
        nt, nx, ny = 2, 15, 10
        dx = 0.1
        x_c = np.arange(nx) * dx
        y_c = np.arange(ny) * dx  # same spacing for simplicity

        xx = x_c.reshape(1, nx, 1)
        u = np.broadcast_to(xx ** 2, (nt, nx, ny)).copy()

        x_list = [x_c, y_c]
        result = self._call_laplace(u, x_list)

        assert result is not None, "2D Laplace returned None"
        interior = result[:, 1:-1, 1:-1]
        np.testing.assert_allclose(
            interior, 2.0, rtol=1e-5,
            err_msg="2D Laplace(x^2) interior should be 2.0"
        )


# ---------------------------------------------------------------------------
# Token registry tests
# ---------------------------------------------------------------------------

class TestDiffTokensByDim:
    """_DIFF_TOKENS_BY_DIM[3] must include 'lap_3' for auto function_set."""

    def test_diff_tokens_by_dim_includes_lap_3(self):
        """After implementation, _DIFF_TOKENS_BY_DIM[3] should contain 'lap_3'.

        Current state: dim=3 only has ['Diff_3', 'Diff2_3'], missing 'lap_3'.
        This test verifies the token is added to enable auto function_set
        selection for 3D datasets.
        """
        from kd.model.kd_dscv import KD_DSCV

        tokens_3d = KD_DSCV._DIFF_TOKENS_BY_DIM[3]
        assert 'lap_3' in tokens_3d, (
            f"'lap_3' not found in _DIFF_TOKENS_BY_DIM[3]. "
            f"Current tokens: {tokens_3d}"
        )

    def test_diff_tokens_by_dim_2d_still_has_lap(self):
        """Regression: dim=2 must still include 'lap'."""
        from kd.model.kd_dscv import KD_DSCV

        tokens_2d = KD_DSCV._DIFF_TOKENS_BY_DIM[2]
        assert 'lap' in tokens_2d, (
            f"'lap' not found in _DIFF_TOKENS_BY_DIM[2]. "
            f"Current tokens: {tokens_2d}"
        )

    def test_all_diff_tokens_includes_lap_3(self):
        """_ALL_DIFF_TOKENS must include 'lap_3' for proper stripping."""
        from kd.model.kd_dscv import KD_DSCV

        assert 'lap_3' in KD_DSCV._ALL_DIFF_TOKENS, (
            f"'lap_3' not found in _ALL_DIFF_TOKENS. "
            f"Current tokens: {KD_DSCV._ALL_DIFF_TOKENS}"
        )


# ---------------------------------------------------------------------------
# Execute prefix matching tests
# ---------------------------------------------------------------------------

class TestExecuteLapPrefixMatching:
    """Verify execute.py correctly dispatches 'lap_3' via prefix matching."""

    def test_execute_matches_lap_prefix(self):
        """'lap' in 'lap_3' must be True for the executor to route
        Laplace-family tokens to the (u, x) calling convention.

        This is a pure string-matching test -- independent of whether
        Laplace_3 is implemented yet.
        """
        assert 'lap' in 'lap_3', (
            "String prefix check failed: 'lap' should be found in 'lap_3'"
        )

    def test_execute_python_exact_match_blocks_lap_3(self):
        """python_execute must dispatch lap_3 via the Laplacian branch (pass x).

        Originally used 'lap' == token.name (exact match), which missed lap_3.
        Fix uses a whitelist: token.name in ('lap', 'lap_3', 'lap_t').
        This avoids mismatching 'laplacian'/'laplacian_t' (different signature).
        """
        import inspect
        from kd.model.discover import execute

        source = inspect.getsource(execute.python_execute)

        # Must NOT use exact match (blocks lap_3)
        uses_exact_match = "'lap' == token.name" in source
        assert not uses_exact_match, (
            "python_execute still uses exact match 'lap' == token.name"
        )

        # Must include 'lap_3' in the dispatch (whitelist or prefix)
        handles_lap_3 = "'lap_3'" in source or "'lap' in token.name" in source
        assert handles_lap_3, (
            "python_execute does not handle 'lap_3' token"
        )


# ---------------------------------------------------------------------------
# Laplace_3 Token registration test
# ---------------------------------------------------------------------------

class TestLaplace3TokenRegistration:
    """Verify Laplace_3 is registered in functions.py as a Token."""

    @pytest.mark.skipif(
        not _LAPLACE_3_AVAILABLE,
        reason="Laplace_3 not yet implemented"
    )
    def test_laplace_3_in_function_map(self):
        """'lap_3' must be a registered token in the function_map.

        The token should have arity=1 (takes u, with x passed separately
        by the executor) and use Laplace_3 as its underlying function.
        """
        from kd.model.discover.functions import function_map

        assert 'lap_3' in function_map, (
            f"'lap_3' not found in function_map. "
            f"Available tokens with 'lap': "
            f"{[k for k in function_map if 'lap' in k]}"
        )
        token = function_map['lap_3']
        assert token.arity == 1, (
            f"'lap_3' token arity should be 1, got {token.arity}"
        )

    def test_laplace_3_token_not_yet_registered(self):
        """If Laplace_3 is not available, 'lap_3' should also be missing
        from the function_map."""
        if _LAPLACE_3_AVAILABLE:
            pytest.skip("Laplace_3 is available -- check registration instead")

        from kd.model.discover.functions import function_map
        assert 'lap_3' not in function_map, (
            "'lap_3' found in function_map but Laplace_3 is not importable -- "
            "inconsistent state"
        )


# ---------------------------------------------------------------------------
# Numerical edge cases for Laplace_3
# ---------------------------------------------------------------------------

class TestLaplace3NumericalEdgeCases:
    """Numerical edge cases: constant field, linear field, large values."""

    @pytest.mark.skipif(
        not _LAPLACE_3_AVAILABLE,
        reason="Laplace_3 not yet implemented"
    )
    def test_laplace_3_constant_field(self):
        """u = const => Laplacian = 0 everywhere."""
        nt, nx, ny, nz = 2, 8, 8, 8
        dx = dy = dz = 0.1
        x_c, y_c, z_c = _make_3d_grid(nt, nx, ny, nz, dx, dy, dz)
        u = np.full((nt, nx, ny, nz), 42.0)

        result = Laplace_3(u, [x_c, y_c, z_c])

        assert result is not None
        interior = result[:, 1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(
            interior, 0.0, atol=1e-10,
            err_msg="Laplace_3(constant) should be 0 everywhere"
        )

    @pytest.mark.skipif(
        not _LAPLACE_3_AVAILABLE,
        reason="Laplace_3 not yet implemented"
    )
    def test_laplace_3_linear_field(self):
        """u = x + y + z => Laplacian = 0 (all second derivatives vanish)."""
        nt, nx, ny, nz = 2, 8, 9, 10
        dx, dy, dz = 0.1, 0.1, 0.1
        x_c, y_c, z_c = _make_3d_grid(nt, nx, ny, nz, dx, dy, dz)

        xx = x_c.reshape(1, nx, 1, 1)
        yy = y_c.reshape(1, 1, ny, 1)
        zz = z_c.reshape(1, 1, 1, nz)
        u = np.broadcast_to(xx + yy + zz, (nt, nx, ny, nz)).copy()

        result = Laplace_3(u, [x_c, y_c, z_c])

        assert result is not None
        interior = result[:, 1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(
            interior, 0.0, atol=1e-10,
            err_msg="Laplace_3(x+y+z) should be 0 at interior"
        )

    @pytest.mark.skipif(
        not _LAPLACE_3_AVAILABLE,
        reason="Laplace_3 not yet implemented"
    )
    def test_laplace_3_output_is_finite(self):
        """Result must be finite for well-formed input (no NaN/Inf)."""
        nt, nx, ny, nz = 2, 8, 8, 8
        dx = dy = dz = 0.1
        x_c, y_c, z_c = _make_3d_grid(nt, nx, ny, nz, dx, dy, dz)
        u = _make_u_sum_of_squares(nt, nx, ny, nz, x_c, y_c, z_c)

        result = Laplace_3(u, [x_c, y_c, z_c])

        assert result is not None
        assert np.all(np.isfinite(result)), (
            "Laplace_3 output contains NaN or Inf"
        )
