"""Equivalence tests: Diff2_3 in utils_fd.py vs reference implementations.

Compares:
  - NEW:  kd/model/discover/task/pde/utils_fd.py  (fixed)
  - REF:  ref_lib/DISCOVER/dso/dso/task/pde/utils_v3.py  (original DISCOVER)
  - V1:   kd/model/discover/task/pde/utils_v1.py  (copy of REF, also buggy)

Expected relationships:
  - name==1: NEW == REF == V1  (no bugs in this branch)
  - name==2: NEW == REF == V1  (no bugs in this branch)
  - name==3: NEW != REF == V1  (REF/V1 share Bug 1 + Bug 2; NEW is fixed)
  - Bug 3 (missing return): REF and V1 both have `return uxt` at module
    level (outside if/else), so they DO return. utils_fd.py was missing it
    before the fix. Now it also has `return uxt`.

NOTE: REF and V1 have identical Diff2_3 code, so we treat them as one
      "reference" for brevity but test both to confirm.

Markers: equivalence
"""

import numpy as np
import pytest
import importlib
import sys
import types


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _import_ref_diff2_3():
    """Import Diff2_3 from ref_lib/DISCOVER/dso/dso/task/pde/utils_v3.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ref_utils_v3",
        "/Users/hao/PhD/project/kd/ref_lib/DISCOVER/dso/dso/task/pde/utils_v3.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # numba may not be needed; stub it if import fails
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip("Cannot import ref_lib utils_v3 (numba or other dep)")
    return mod.Diff2_3


def _import_v1_diff2_3():
    """Import Diff2_3 from kd/model/discover/task/pde/utils_v1.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kd_utils_v1",
        "/Users/hao/PhD/project/kd/kd/model/discover/task/pde/utils_v1.py",
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip("Cannot import utils_v1 (numba or other dep)")
    return mod.Diff2_3


def _import_new_diff2_3():
    """Import Diff2_3 from kd/model/discover/task/pde/utils_fd.py (fixed)."""
    from kd.model.discover.task.pde.utils_fd import Diff2_3
    return Diff2_3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_fn():
    return _import_ref_diff2_3()


@pytest.fixture
def v1_fn():
    return _import_v1_diff2_3()


@pytest.fixture
def new_fn():
    return _import_new_diff2_3()


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------

def _make_random_4d(
    nt: int = 3, nx: int = 8, ny: int = 6, nz: int = 10, seed: int = 42
) -> tuple:
    """Random 4D data with uniform coordinate grids.

    Returns (u, x_coord, y_coord, z_coord) where coords are (N, 1) shaped.
    Uses distinct nx, ny, nz to expose dimension bugs.
    """
    rng = np.random.RandomState(seed)
    u = rng.randn(nt, nx, ny, nz)
    dx, dy, dz = 0.1, 0.15, 0.2
    x_coord = (np.arange(nx) * dx).reshape(-1, 1)
    y_coord = (np.arange(ny) * dy).reshape(-1, 1)
    z_coord = (np.arange(nz) * dz).reshape(-1, 1)
    return u, x_coord, y_coord, z_coord


def _make_quadratic_data(
    nt: int, nx: int, ny: int, nz: int, axis: int, d: float
):
    """Create u = coord^2 along the given spatial axis (1, 2, or 3).

    Returns (u, coord_1d) where coord_1d has shape (N, 1).
    """
    u = np.zeros((nt, nx, ny, nz))
    if axis == 1:
        coords = np.arange(nx) * d
        for i in range(nx):
            u[:, i, :, :] = coords[i] ** 2
    elif axis == 2:
        coords = np.arange(ny) * d
        for j in range(ny):
            u[:, :, j, :] = coords[j] ** 2
    elif axis == 3:
        coords = np.arange(nz) * d
        for k in range(nz):
            u[:, :, :, k] = coords[k] ** 2
    else:
        raise ValueError(f"axis must be 1, 2, or 3, got {axis}")
    return u, coords.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Part A: REF == V1 (both have identical bugs)
# ---------------------------------------------------------------------------

@pytest.mark.equivalence
class TestRefEqualsV1:
    """Confirm that ref_lib/utils_v3.py and kd/utils_v1.py Diff2_3 are
    identical (both have the same bugs in name==3 branch)."""

    @pytest.mark.parametrize("name", [1, 2, 3])
    def test_ref_equals_v1_random(self, ref_fn, v1_fn, name):
        """REF and V1 should produce bit-identical results for all branches."""
        u, x_coord, y_coord, z_coord = _make_random_4d(
            nt=3, nx=8, ny=6, nz=10
        )
        coord_map = {1: x_coord, 2: y_coord, 3: z_coord}
        coord = coord_map[name]

        ref_out = ref_fn(u, coord, name=name)
        v1_out = v1_fn(u, coord, name=name)

        assert ref_out is not None, "REF Diff2_3 returned None"
        assert v1_out is not None, "V1 Diff2_3 returned None"

        np.testing.assert_array_equal(
            ref_out, v1_out,
            err_msg=f"REF and V1 should be identical for name=={name}"
        )


# ---------------------------------------------------------------------------
# Part B: NEW == REF for name==1 and name==2
# ---------------------------------------------------------------------------

@pytest.mark.equivalence
class TestNewEqualsRefName1Name2:
    """For name==1 and name==2, the fixed implementation should produce
    results identical to the reference (these branches had no bugs)."""

    @pytest.mark.parametrize("name", [1, 2])
    def test_new_equals_ref_random(self, new_fn, ref_fn, name):
        """Random data: NEW should match REF for name==1 and name==2."""
        u, x_coord, y_coord, z_coord = _make_random_4d(
            nt=3, nx=8, ny=6, nz=10
        )
        coord_map = {1: x_coord, 2: y_coord}
        coord = coord_map[name]

        new_out = new_fn(u, coord, name=name)
        ref_out = ref_fn(u, coord, name=name)

        assert new_out is not None, "NEW Diff2_3 returned None"
        assert ref_out is not None, "REF Diff2_3 returned None"

        np.testing.assert_allclose(
            new_out, ref_out,
            rtol=1e-12, atol=1e-15,
            err_msg=f"NEW should match REF for name=={name}"
        )

    @pytest.mark.parametrize("name,axis", [(1, 1), (2, 2)])
    def test_new_equals_ref_quadratic(self, new_fn, ref_fn, name, axis):
        """Quadratic data: both NEW and REF should give d2u/d(coord)^2 = 2.0."""
        nt, nx, ny, nz = 3, 9, 7, 11
        d = 0.1
        u, coord = _make_quadratic_data(nt, nx, ny, nz, axis=axis, d=d)

        new_out = new_fn(u, coord, name=name)
        ref_out = ref_fn(u, coord, name=name)

        np.testing.assert_allclose(
            new_out, ref_out,
            rtol=1e-12, atol=1e-15,
            err_msg=f"NEW should match REF for name=={name} on quadratic data"
        )
        # Both should be 2.0
        np.testing.assert_allclose(
            new_out, 2.0, atol=1e-10,
            err_msg=f"d2u/d(axis{axis})^2 of quadratic should be 2.0"
        )


# ---------------------------------------------------------------------------
# Part C: NEW != REF for name==3 (intentional divergence - bug fix)
# ---------------------------------------------------------------------------

@pytest.mark.equivalence
class TestNewDiffersFromRefName3:
    """For name==3, the fixed implementation intentionally differs from
    the reference. The reference has two bugs:

    Bug 1: interior slice uses `m` (dim-2 size) instead of `p` (dim-3 size)
    Bug 2: z=0 boundary uses 3D index `uxt[:,:,0]` instead of `uxt[:,:,:,0]`

    These tests document the divergence and verify the fix is correct.
    """

    def test_new_differs_from_ref_name3(self, new_fn, ref_fn):
        """NEW and REF should produce DIFFERENT results for name==3
        when m != p (ny != nz)."""
        # Use ny=6, nz=10 so m != p
        u, x_coord, y_coord, z_coord = _make_random_4d(
            nt=3, nx=8, ny=6, nz=10
        )

        new_out = new_fn(u, z_coord, name=3)
        ref_out = ref_fn(u, z_coord, name=3)

        assert new_out is not None, "NEW Diff2_3 returned None"
        assert ref_out is not None, "REF Diff2_3 returned None"

        # They should NOT be equal due to Bug 1 + Bug 2 in REF
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(new_out, ref_out)

    def test_new_correct_ref_wrong_name3_analytical(self, new_fn, ref_fn):
        """For u=z^2, NEW should give d2u/dz2=2.0 everywhere.
        REF should NOT (due to Bug 1 + Bug 2)."""
        nt, nx, ny, nz = 3, 5, 7, 9  # ny != nz
        dz = 0.1
        u, z_coord = _make_quadratic_data(nt, nx, ny, nz, axis=3, d=dz)

        new_out = new_fn(u, z_coord, name=3)
        ref_out = ref_fn(u, z_coord, name=3)

        # NEW should be correct (2.0 everywhere)
        np.testing.assert_allclose(
            new_out, 2.0, atol=1e-10,
            err_msg="NEW (fixed) should give d2u/dz2 = 2.0 for u=z^2"
        )

        # REF should be WRONG due to Bug 1 (interior uses m=7 instead of p=9)
        # Interior z-index 7 (= m-1) should be 0 in REF (not computed)
        # because the slice 1:m-1 = 1:6 only covers z-indices 1..5,
        # leaving indices 6 and 7 as zeros.
        ref_interior_missed = ref_out[:, :, :, ny - 1]  # z-index = m-1 = 6
        # This should NOT be 2.0 in the reference (it's either 0 or corrupted)
        assert not np.allclose(ref_interior_missed, 2.0, atol=1e-5), (
            "REF should have Bug 1: z-indices >= m-1 not computed correctly"
        )

    def test_document_bug1_interior_slice_difference(self, new_fn, ref_fn):
        """Document Bug 1: REF uses `m` for z-axis interior slice.

        When ny(=6) < nz(=10), REF computes z-indices 1..4 (= 1:m-1 = 1:5)
        instead of 1..8 (= 1:p-1 = 1:9). The missed indices remain zero.
        """
        nt, nx, ny, nz = 2, 5, 6, 10
        dz = 0.1
        u, z_coord = _make_quadratic_data(nt, nx, ny, nz, axis=3, d=dz)

        new_out = new_fn(u, z_coord, name=3)
        ref_out = ref_fn(u, z_coord, name=3)

        # NEW computes ALL interior z-indices: 1..8
        for k in range(1, nz - 1):
            new_val = new_out[:, :, :, k]
            np.testing.assert_allclose(
                new_val, 2.0, atol=1e-10,
                err_msg=f"NEW z-index {k} should be 2.0"
            )

        # REF only computes z-indices 1..4 (= 1:m-1 where m=ny=6).
        # Additionally, Bug 2 overwrites the y=0 plane (uxt[:,:,0] = ...),
        # which corrupts z values at y=0. So we check y>0 only.
        for k in range(1, ny - 1):
            # Exclude y=0 plane which is corrupted by Bug 2
            ref_val = ref_out[:, :, 1:, k]
            np.testing.assert_allclose(
                ref_val, 2.0, atol=1e-10,
                err_msg=(
                    f"REF z-index {k} (within m range, y>0) should be 2.0"
                )
            )

        # z-indices >= m-1 = 5 are NOT computed by REF (remain 0)
        # Again exclude y=0 which is corrupted by Bug 2
        for k in range(ny - 1, nz - 1):
            ref_val = ref_out[:, :, 1:, k]
            assert np.allclose(ref_val, 0.0, atol=1e-10), (
                f"REF z-index {k} (outside m range, y>0) should be 0 (Bug 1)"
            )

    def test_document_bug2_boundary_wrong_axis(self, new_fn, ref_fn):
        """Document Bug 2: REF writes z=0 boundary to y=0 plane.

        `uxt[:,:,0] = ...` (3D index) writes to ALL z values at y=0,
        instead of `uxt[:,:,:,0]` (4D index) which targets z=0.

        This overwrites previously computed values at y=0 for all z.
        """
        nt, nx, ny, nz = 2, 5, 6, 10
        dz = 0.1
        u, z_coord = _make_quadratic_data(nt, nx, ny, nz, axis=3, d=dz)

        ref_out = ref_fn(u, z_coord, name=3)
        new_out = new_fn(u, z_coord, name=3)

        # In NEW (fixed): y=0 plane at interior z should be 2.0
        new_y0_interior = new_out[:, :, 0, 1:nz - 1]
        np.testing.assert_allclose(
            new_y0_interior, 2.0, atol=1e-10,
            err_msg="NEW: y=0 plane at interior z should be 2.0 (not corrupted)"
        )

        # In REF (buggy): y=0 plane is overwritten by z=0 boundary formula.
        # For u=z^2, the boundary formula at z=0 gives:
        #   (2*0 - 5*dz^2 + 4*(2dz)^2 - (3dz)^2) / dz^2 = 2.0
        # So for this particular test data, the y=0 plane might accidentally
        # look correct at some indices. But the key difference is that
        # REF's uxt[:,:,0] is a 2D slice (all z), NOT just z=0.
        # The REF writes the boundary formula (evaluated at z=0 data)
        # to the ENTIRE y=0 plane, which is semantically wrong.

        # To make the corruption visible, use data where the boundary formula
        # at z=0 gives a different value than d2u/dz2 at interior z.
        # For u = z^3: d2u/dz2 = 6*z. At z=0 -> 0, at z=dz -> 6*dz, etc.
        # But the boundary stencil gives a different value.
        u_cubic = np.zeros((nt, nx, ny, nz))
        z_vals = np.arange(nz) * dz
        for k in range(nz):
            u_cubic[:, :, :, k] = z_vals[k] ** 3

        ref_cubic = ref_fn(u_cubic, z_coord, name=3)
        new_cubic = new_fn(u_cubic, z_coord, name=3)

        # NEW: z=0 boundary (4D index) should be separate from y=0 interior
        # For z^3: d2u/dz2 = 6*z, so at z=k*dz -> 6*k*dz
        # Interior z=1: expected = 6*dz = 0.6
        new_y0_z1 = new_cubic[:, :, 0, 1]  # y=0, z=1
        np.testing.assert_allclose(
            new_y0_z1, 6 * dz, atol=1e-8,
            err_msg="NEW: y=0, z=1 should be d2(z^3)/dz2 = 6*dz"
        )

        # REF: y=0 plane is overwritten by the z=0 boundary formula
        # applied with uxt[:,:,0] = formula(u[:,:,0], u[:,:,1], ...)
        # This means ref_cubic[:,:,0, ALL_Z] gets the boundary value
        # The 3D index [:,:,0] collapses y and z into one axis.
        # Actually, uxt[:,:,0] in 4D means uxt[:,:,0,:] = y=0 plane (all z).
        # So it overwrites the y=0 plane with:
        #   (2*u[:,:,0] - 5*u[:,:,1] + 4*u[:,:,2] - u[:,:,3]) / dz^2
        # where u[:,:,j] = u[:,:,j,:] in 4D.
        # This is the formula applied to the Y-axis, not Z-axis!
        # For u=z^3, u[:,:,j,:] = z^3 for all j, so u[:,:,0,:] = z^3,
        # meaning the formula gives:
        #   (2*z^3 - 5*z^3 + 4*z^3 - z^3) / dz^2 = 0
        # So the y=0 plane gets all zeros in REF.
        ref_y0_z1 = ref_cubic[:, :, 0, 1]  # y=0, z=1
        # This value should be WRONG in REF (0 instead of 6*dz)
        assert not np.allclose(ref_y0_z1, 6 * dz, atol=1e-5), (
            "REF should have Bug 2: y=0 plane corrupted by wrong-axis boundary"
        )


# ---------------------------------------------------------------------------
# Part D: NEW matches REF when m == p (bugs are hidden)
# ---------------------------------------------------------------------------

@pytest.mark.equivalence
class TestNewEqualsRefName3WhenMEqualsP:
    """When ny == nz (m == p), Bug 1 is hidden (slice ranges are the same).
    Bug 2 is still present but may produce same values for some data patterns.

    For random data, even with m==p, Bug 2 (wrong axis indexing) still
    causes different results. This test documents this subtlety.
    """

    def test_name3_m_equals_p_interior_matches(self, new_fn, ref_fn):
        """When m==p, Bug 1 is hidden: interior slices are the same range.
        But Bug 2 (wrong-axis boundary write) still corrupts y=0 plane.

        So interior matches ONLY at y>0. At y=0, Bug 2 overwrites values.
        """
        nt, nx, ny, nz = 2, 5, 8, 8  # ny == nz
        dz = 0.1
        u, _, _, z_coord = _make_random_4d(nt=nt, nx=nx, ny=ny, nz=nz)
        z_coord = (np.arange(nz) * dz).reshape(-1, 1)

        new_out = new_fn(u, z_coord, name=3)
        ref_out = ref_fn(u, z_coord, name=3)

        # Interior at y>0 should match when m==p (Bug 1 hidden, Bug 2
        # only affects y=0 plane)
        new_interior_y_gt_0 = new_out[:, :, 1:, 1:nz - 1]
        ref_interior_y_gt_0 = ref_out[:, :, 1:, 1:nz - 1]
        np.testing.assert_allclose(
            new_interior_y_gt_0, ref_interior_y_gt_0,
            rtol=1e-12, atol=1e-15,
            err_msg="Interior (y>0) should match when m==p (Bug 1 hidden)"
        )

        # z=nz-1 boundary should also match (uses correct 4D index in both)
        new_end = new_out[:, :, :, nz - 1]
        ref_end = ref_out[:, :, :, nz - 1]
        np.testing.assert_allclose(
            new_end, ref_end,
            rtol=1e-12, atol=1e-15,
            err_msg="z=end boundary should match (no bug there)"
        )

        # Interior at y=0 should DIFFER due to Bug 2 (wrong-axis boundary)
        new_y0_interior = new_out[:, :, 0, 1:nz - 1]
        ref_y0_interior = ref_out[:, :, 0, 1:nz - 1]
        # These should generally differ because REF's uxt[:,:,0] overwrites
        # the y=0 plane with boundary formula values
        diff = np.abs(new_y0_interior - ref_y0_interior)
        assert np.max(diff) > 1e-10, (
            "Interior at y=0 should differ due to Bug 2 (wrong-axis boundary)"
        )


# ---------------------------------------------------------------------------
# Part E: Quantify differences (diagnostic, always passes)
# ---------------------------------------------------------------------------

@pytest.mark.equivalence
class TestQuantifyDifferences:
    """Diagnostic test: quantify the numerical difference between NEW and
    REF for name==3. This test always passes but prints useful info."""

    def test_quantify_name3_differences(self, new_fn, ref_fn):
        """Report max absolute difference between NEW and REF for name==3."""
        u, _, _, z_coord = _make_random_4d(nt=3, nx=8, ny=6, nz=10)

        new_out = new_fn(u, z_coord, name=3)
        ref_out = ref_fn(u, z_coord, name=3)

        diff = np.abs(new_out - ref_out)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Count elements that differ
        n_differ = np.sum(diff > 1e-15)
        n_total = diff.size
        pct_differ = 100.0 * n_differ / n_total

        # This always passes -- it's for information
        print(f"\n--- Diff2_3 name==3 divergence report ---")
        print(f"  Shape: {new_out.shape}")
        print(f"  Max |NEW - REF|: {max_diff:.6e}")
        print(f"  Mean |NEW - REF|: {mean_diff:.6e}")
        print(f"  Elements differing (>1e-15): {n_differ}/{n_total} "
              f"({pct_differ:.1f}%)")
        print(f"  Source: Bug 1 (m vs p) + Bug 2 (3D vs 4D index)")
        print(f"--- end report ---\n")

        # Sanity: they SHOULD differ
        assert max_diff > 0, (
            "NEW and REF should differ for name==3 (ny=6 != nz=10)"
        )
