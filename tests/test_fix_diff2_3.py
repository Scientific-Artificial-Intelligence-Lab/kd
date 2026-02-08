"""Tests for Diff2_3 function (name==3 branch bug fix).

Target bugs in kd/model/discover/task/pde/utils_fd.py Diff2_3():

  Bug 1 (line 272): name==3 interior uses `m` (dim-2 size) instead of `p`
         (dim-3 size) for z-axis slicing.

  Bug 2 (line 275): name==3 boundary at z=0 uses 3D index `uxt[:,:,0]`
         instead of 4D index `uxt[:,:,:,0]`, writing to the y=0 plane
         instead of the z=0 boundary.

  Bug 3 (missing return): Diff2_3 has no `return uxt` at the end of the
         function body (compare with utils_v1.py which has it). This
         causes the function to return None for all branches.

All tests use nx != ny != nz to ensure m != p, which is required to
expose Bug 1 (dimension mismatch).

RED phase: ALL tests fail on the current buggy code.
  - name==3 tests fail due to Bugs 1 + 2 + 3
  - name==1 / name==2 regression tests fail due to Bug 3 (missing return)
  After the fix, name==1/2 regression tests should PASS,
  and name==3 tests should also PASS.
"""

import numpy as np
import pytest

from kd.model.discover.task.pde.utils_fd import Diff2_3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quadratic_z_data(nt: int, nx: int, ny: int, nz: int, dz: float):
    """Create 4D data where u = z^2, so d2u/dz2 = 2 everywhere.

    Parameters
    ----------
    nt, nx, ny, nz : int
        Grid sizes. Must satisfy nx != ny != nz to expose Bug 1.
    dz : float
        Uniform spacing along the z-axis.

    Returns
    -------
    u : ndarray, shape (nt, nx, ny, nz)
    z_coord : ndarray, shape (nz,)
    """
    z = np.arange(nz) * dz
    u = np.zeros((nt, nx, ny, nz))
    for k in range(nz):
        u[:, :, :, k] = z[k] ** 2
    return u, z


# ---------------------------------------------------------------------------
# Smoke test: function returns an array (catches missing return)
# ---------------------------------------------------------------------------

class TestDiff2_3Smoke:
    """Smoke test: Diff2_3 must return an ndarray, not None."""

    def test_diff2_3_returns_ndarray(self):
        """Diff2_3 should return an ndarray for all name values.

        Currently fails because the function has no `return uxt` statement
        (Bug 3), so it returns None for every branch.
        """
        nt, nx, ny, nz = 2, 5, 6, 7
        dz = 0.1
        u = np.ones((nt, nx, ny, nz))
        z_1d = (np.arange(nz) * dz).reshape(-1, 1)

        result = Diff2_3(u, z_1d, name=3)

        assert result is not None, (
            "Diff2_3 returned None -- missing `return uxt` at end of function"
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (nt, nx, ny, nz)


# ---------------------------------------------------------------------------
# name==3 analytical tests (Bugs 1 + 2 + 3)
# ---------------------------------------------------------------------------

class TestDiff2_3Name3Analytical:
    """Verify Diff2_3(name=3) computes d2u/dz2 correctly against an
    analytical solution: u = z^2 => d2u/dz2 = 2."""

    def test_diff2_3_name3_analytical(self):
        """Interior points of d2u/dz2 for u=z^2 should equal 2.0.

        Fails on current code due to:
        - Bug 3: returns None (no return statement)
        - Bug 1: interior slice uses m instead of p (truncated computation)
        - Bug 2: z=0 boundary written to wrong axis
        """
        nt, nx, ny, nz = 3, 5, 7, 9  # all different!
        dz = 0.1
        u, z_coord = _make_quadratic_z_data(nt, nx, ny, nz, dz)

        z_1d = z_coord.reshape(-1, 1)
        result = Diff2_3(u, z_1d, name=3)

        assert result is not None, "Diff2_3 returned None (missing return)"
        assert result.shape == (nt, nx, ny, nz), (
            f"Output shape {result.shape} != expected {(nt, nx, ny, nz)}"
        )

        # Interior points (z index 1..nz-2): central difference of z^2
        # with uniform spacing is exact, should be 2.0
        interior = result[:, :, :, 1:nz - 1]
        np.testing.assert_allclose(
            interior, 2.0, atol=1e-10,
            err_msg="Interior d2u/dz2 for u=z^2 should be 2.0"
        )


# ---------------------------------------------------------------------------
# name==3 shape mismatch tests (Bug 1)
# ---------------------------------------------------------------------------

class TestDiff2_3Name3ShapeMismatch:
    """Verify that name==3 does not crash or silently truncate when
    nx != ny != nz (i.e., m != p)."""

    def test_diff2_3_name3_shape_mismatch(self):
        """When m < p (ny=7, nz=9), Bug 1 causes interior computation to
        only cover z-indices 1..5 instead of 1..7, leaving 6..7 as zeros.
        """
        nt, nx, ny, nz = 3, 5, 7, 9
        dz = 0.1
        u, z_coord = _make_quadratic_z_data(nt, nx, ny, nz, dz)
        z_1d = z_coord.reshape(-1, 1)

        result = Diff2_3(u, z_1d, name=3)

        assert result is not None, "Diff2_3 returned None (missing return)"

        # ALL interior z-indices (1 through nz-2) must be computed.
        # With Bug 1 (m=7 used instead of p=9), indices >= m-1=6 are
        # not touched, so result[:,:,:,7] remains 0 instead of 2.0.
        for k in range(1, nz - 1):
            slice_k = result[:, :, :, k]
            np.testing.assert_allclose(
                slice_k, 2.0, atol=1e-10,
                err_msg=f"z-index {k}: d2u/dz2 should be 2.0, got truncation"
            )

    def test_diff2_3_name3_shape_m_gt_p(self):
        """When m > p (ny=9 > nz=5), Bug 1 causes shape mismatch in
        numpy broadcasting because uxt[:,:,:,1:m-1] has 7 elements along
        dim-3 but the RHS only has min(p,m)-2 = 3 elements.

        Uses u=z^2 to also verify numerical correctness, not just shape.
        """
        nt, nx, ny, nz = 3, 5, 9, 5
        dz = 0.1
        u, z_coord = _make_quadratic_z_data(nt, nx, ny, nz, dz)
        z_1d = z_coord.reshape(-1, 1)

        # After fix (using p), this should work without errors.
        # On buggy code, it either crashes or produces wrong results.
        # Let pytest report any IndexError/ValueError directly.
        result = Diff2_3(u, z_1d, name=3)

        assert result is not None, "Diff2_3 returned None (missing return)"
        assert result.shape == (nt, nx, ny, nz)

        # Interior z-indices should give d2u/dz2 = 2.0
        interior = result[:, :, :, 1:nz - 1]
        np.testing.assert_allclose(
            interior, 2.0, atol=1e-10,
            err_msg=(
                "m>p case: interior d2u/dz2 for u=z^2 should be 2.0. "
                f"ny({ny}) > nz({nz}) triggers Bug 1 shape mismatch."
            )
        )

    def test_diff2_3_name3_min_grid_nz4(self):
        """Minimum valid grid size: nz=4 is the smallest grid that supports
        the one-sided boundary stencil (which needs 4 points: indices 0..3).

        Uses nx != ny != nz to ensure m != p and expose Bug 1.
        u = z^2 => d2u/dz2 = 2.0 everywhere (boundary + interior).
        """
        nt, nx, ny, nz = 2, 3, 5, 4
        dz = 0.1
        u, z_coord = _make_quadratic_z_data(nt, nx, ny, nz, dz)
        z_1d = z_coord.reshape(-1, 1)

        result = Diff2_3(u, z_1d, name=3)

        assert result is not None, "Diff2_3 returned None (missing return)"
        assert result.shape == (nt, nx, ny, nz)

        # All points (boundary + interior) should be 2.0 for u=z^2
        np.testing.assert_allclose(
            result, 2.0, atol=1e-10,
            err_msg=(
                "Minimum grid nz=4: d2u/dz2 for u=z^2 should be 2.0 "
                "at all points (boundaries and interior)."
            )
        )


# ---------------------------------------------------------------------------
# name==3 boundary indexing tests (Bug 2)
# ---------------------------------------------------------------------------

class TestDiff2_3Name3BoundaryIndexing:
    """Verify that the z=0 boundary is written to the correct axis."""

    def test_diff2_3_name3_boundary_indexing(self):
        """Bug 2: `uxt[:,:,0] = ...` writes to the y=0 plane (all z),
        instead of `uxt[:,:,:,0]` which writes to the z=0 boundary.

        For u = z^2, the forward one-sided 2nd derivative at z=0 is:
          (2*u0 - 5*u1 + 4*u2 - u3) / dz^2
          = (2*0 - 5*dz^2 + 4*(2dz)^2 - (3dz)^2) / dz^2
          = (0 - 5 + 16 - 9) = 2.0
        """
        nt, nx, ny, nz = 3, 5, 7, 9
        dz = 0.1
        u, z_coord = _make_quadratic_z_data(nt, nx, ny, nz, dz)
        z_1d = z_coord.reshape(-1, 1)

        result = Diff2_3(u, z_1d, name=3)

        assert result is not None, "Diff2_3 returned None (missing return)"

        # z=0 boundary should be 2.0 (one-sided FD of z^2 is exact)
        boundary_z0 = result[:, :, :, 0]
        np.testing.assert_allclose(
            boundary_z0, 2.0, atol=1e-10,
            err_msg=(
                "z=0 boundary d2u/dz2 should be 2.0. "
                "Bug: boundary was written to y=0 plane instead of z=0."
            )
        )

    def test_diff2_3_name3_boundary_z_end(self):
        """Symmetric counterpart of test_diff2_3_name3_boundary_indexing:
        verify the z=nz-1 boundary is computed correctly.

        For u = z^2, the backward one-sided 2nd derivative at z=nz-1 is:
          (2*u[nz-1] - 5*u[nz-2] + 4*u[nz-3] - u[nz-4]) / dz^2
        which is exact for quadratic => 2.0.
        """
        nt, nx, ny, nz = 3, 5, 7, 9
        dz = 0.1
        u, z_coord = _make_quadratic_z_data(nt, nx, ny, nz, dz)
        z_1d = z_coord.reshape(-1, 1)

        result = Diff2_3(u, z_1d, name=3)

        assert result is not None, "Diff2_3 returned None (missing return)"

        # z=nz-1 boundary should be 2.0 (one-sided FD of z^2 is exact)
        boundary_z_end = result[:, :, :, nz - 1]
        np.testing.assert_allclose(
            boundary_z_end, 2.0, atol=1e-10,
            err_msg=(
                "z=nz-1 boundary d2u/dz2 should be 2.0. "
                "Backward one-sided stencil for u=z^2 must be exact."
            )
        )

    def test_diff2_3_name3_boundary_does_not_corrupt_y0(self):
        """Complementary check: the y=0 plane should not be overwritten
        by the z=0 boundary formula.

        For u = z^2, the correct d2u/dz2 is 2.0 everywhere. If Bug 2
        fires, uxt[:,:,0] (y=0 slice across ALL z) gets overwritten with
        the z=0 boundary formula values, corrupting interior z values
        at y=0.
        """
        nt, nx, ny, nz = 3, 5, 7, 9
        dz = 0.1
        u, z_coord = _make_quadratic_z_data(nt, nx, ny, nz, dz)
        z_1d = z_coord.reshape(-1, 1)

        result = Diff2_3(u, z_1d, name=3)

        assert result is not None, "Diff2_3 returned None (missing return)"

        # y=0 plane, for z-interior indices, should be 2.0
        y0_interior_z = result[:, :, 0, 1:nz - 1]
        np.testing.assert_allclose(
            y0_interior_z, 2.0, atol=1e-10,
            err_msg=(
                "y=0 plane at interior z-indices should be 2.0. "
                "Bug 2 corrupts this plane with z=0 boundary values."
            )
        )


# ---------------------------------------------------------------------------
# Regression tests: name==1 and name==2 (should pass after Bug 3 fix)
# ---------------------------------------------------------------------------

class TestDiff2_3Name1Regression:
    """Regression tests: name==1 branch should be unaffected by the fix.

    NOTE: Currently also fails due to Bug 3 (missing return). After adding
    `return uxt`, these tests should pass without any other changes.
    """

    def test_diff2_3_name1_regression(self):
        """d2u/dx2 for u = x^2 should be 2.0 at all points.

        name==1 operates on axis-1 (x-axis). Uses asymmetric shape to
        verify it works independently of m/p.
        """
        nt, nx, ny, nz = 3, 9, 5, 7
        dx = 0.1
        x_coord = np.arange(nx) * dx

        # u[t, x, y, z] = x^2
        u = np.zeros((nt, nx, ny, nz))
        for i in range(nx):
            u[:, i, :, :] = x_coord[i] ** 2

        x_1d = x_coord.reshape(-1, 1)
        result = Diff2_3(u, x_1d, name=1)

        assert result is not None, "Diff2_3 returned None (missing return)"
        assert result.shape == (nt, nx, ny, nz)

        # Interior x-indices should give d2u/dx2 = 2.0
        interior = result[:, 1:nx - 1, :, :]
        np.testing.assert_allclose(
            interior, 2.0, atol=1e-10,
            err_msg="name==1: interior d2u/dx2 for u=x^2 should be 2.0"
        )

        # Boundary x=0 (one-sided FD of x^2 is exact => 2.0)
        boundary_x0 = result[:, 0, :, :]
        np.testing.assert_allclose(
            boundary_x0, 2.0, atol=1e-10,
            err_msg="name==1: x=0 boundary d2u/dx2 for u=x^2 should be 2.0"
        )

        # Boundary x=nx-1
        boundary_xn = result[:, nx - 1, :, :]
        np.testing.assert_allclose(
            boundary_xn, 2.0, atol=1e-10,
            err_msg="name==1: x=end boundary d2u/dx2 for u=x^2 should be 2.0"
        )


class TestDiff2_3Name2Regression:
    """Regression tests: name==2 branch should be unaffected by the fix.

    NOTE: Currently also fails due to Bug 3 (missing return). After adding
    `return uxt`, these tests should pass without any other changes.
    """

    def test_diff2_3_name2_regression(self):
        """d2u/dy2 for u = y^2 should be 2.0 at all points.

        name==2 operates on axis-2 (y-axis).
        """
        nt, nx, ny, nz = 3, 5, 9, 7
        dy = 0.1
        y_coord = np.arange(ny) * dy

        # u[t, x, y, z] = y^2
        u = np.zeros((nt, nx, ny, nz))
        for j in range(ny):
            u[:, :, j, :] = y_coord[j] ** 2

        y_1d = y_coord.reshape(-1, 1)
        result = Diff2_3(u, y_1d, name=2)

        assert result is not None, "Diff2_3 returned None (missing return)"
        assert result.shape == (nt, nx, ny, nz)

        # Interior y-indices should give d2u/dy2 = 2.0
        interior = result[:, :, 1:ny - 1, :]
        np.testing.assert_allclose(
            interior, 2.0, atol=1e-10,
            err_msg="name==2: interior d2u/dy2 for u=y^2 should be 2.0"
        )

        # Boundary y=0
        boundary_y0 = result[:, :, 0, :]
        np.testing.assert_allclose(
            boundary_y0, 2.0, atol=1e-10,
            err_msg="name==2: y=0 boundary d2u/dy2 for u=y^2 should be 2.0"
        )

        # Boundary y=ny-1
        boundary_yn = result[:, :, ny - 1, :]
        np.testing.assert_allclose(
            boundary_yn, 2.0, atol=1e-10,
            err_msg="name==2: y=end boundary d2u/dy2 for u=y^2 should be 2.0"
        )
