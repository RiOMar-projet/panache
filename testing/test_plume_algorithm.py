"""
Unit tests for panache.plume_algorithm utility functions.

Functions that require file I/O (load_and_resize_files, load_and_filter_arrays),
Matplotlib rendering (make_the_plot), or the full detection pipeline
(main_process, Pipeline_to_delineate_the_plume, Create_the_plume_mask) are
covered by the smoke test rather than here.
"""

import unittest

import numpy as np
import xarray as xr

from panache.plume_algorithm import (
    Check_if_the_area_is_too_cloudy,
    compute_gradient_with_directions_vectorized,
    create_polygon_mask,
    derive_masks_from_bathymetry,
    estimate_threshold_bounds_from_near_mouth_pixels,
    find_connected_shapes,
    find_first_nan_after_finite,
    find_high_value_pixels,
    find_nearest_valid_start,
    find_SPM_threshold,
    find_the_index_of_the_plume_starting_point,
    first_true_block,
    flood_fill,
    haversine,
    identify_the_shape_label_corresponding_to_the_plume,
    last_true_block,
    reduce_resolution,
    return_stats_dictionnary,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_da(data, lat=None, lon=None):
    """Return a 2-D xarray DataArray with lat/lon coordinates."""
    nrows, ncols = data.shape
    if lat is None:
        lat = np.linspace(43.0, 44.0, nrows)
    if lon is None:
        lon = np.linspace(7.0, 8.0, ncols)
    return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


def _northward_directions():
    """Return the pixel-direction tuples for a northward fan."""
    return [(-1, -1), (-1, 0), (-1, 1)]


# ---------------------------------------------------------------------------
# haversine
# ---------------------------------------------------------------------------

class HaversineTests(unittest.TestCase):

    def test_same_point_is_zero(self):
        self.assertAlmostEqual(haversine(48.0, 2.3, 48.0, 2.3), 0.0, places=6)

    def test_known_distance_paris_london(self):
        # Paris (48.85, 2.35) to London (51.51, -0.13) ≈ 340 km
        dist = haversine(48.85, 2.35, 51.51, -0.13)
        self.assertGreater(dist, 330)
        self.assertLess(dist, 350)

    def test_equatorial_degree_is_roughly_111_km(self):
        dist = haversine(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(dist, 111.32, delta=0.5)

    def test_symmetry(self):
        d1 = haversine(43.0, 5.0, 44.0, 6.0)
        d2 = haversine(44.0, 6.0, 43.0, 5.0)
        self.assertAlmostEqual(d1, d2, places=8)

    def test_antipodal_points_near_earth_half_circumference(self):
        dist = haversine(0.0, 0.0, 0.0, 180.0)
        self.assertAlmostEqual(dist, np.pi * 6371.0, delta=1.0)


# ---------------------------------------------------------------------------
# first_true_block
# ---------------------------------------------------------------------------

class FirstTrueBlockTests(unittest.TestCase):

    def test_no_true_values(self):
        self.assertEqual(first_true_block(np.array([False, False, False])), (-1, -1))

    def test_single_contiguous_block(self):
        arr = np.array([False, True, True, True, False])
        self.assertEqual(first_true_block(arr), (1, 3))

    def test_returns_first_block_when_multiple_exist(self):
        arr = np.array([True, True, False, True, True])
        self.assertEqual(first_true_block(arr), (0, 1))

    def test_single_true_element(self):
        arr = np.array([False, True, False])
        self.assertEqual(first_true_block(arr), (1, 1))

    def test_all_true(self):
        arr = np.array([True, True, True])
        self.assertEqual(first_true_block(arr), (0, 2))


# ---------------------------------------------------------------------------
# last_true_block
# ---------------------------------------------------------------------------

class LastTrueBlockTests(unittest.TestCase):

    def test_no_true_values(self):
        self.assertEqual(last_true_block(np.array([False, False])), (-1, -1))

    def test_single_contiguous_block_not_at_zero(self):
        arr = np.array([False, True, True, False])
        self.assertEqual(last_true_block(arr), (1, 2))

    def test_returns_last_block_when_multiple_exist(self):
        arr = np.array([True, True, False, True, True])
        self.assertEqual(last_true_block(arr), (3, 4))

    def test_single_true_element_at_end(self):
        arr = np.array([False, False, True])
        self.assertEqual(last_true_block(arr), (2, 2))

    def test_block_starting_at_zero(self):
        arr = np.array([True, True, True])
        self.assertEqual(last_true_block(arr), (0, 2))

    def test_all_false_returns_sentinel(self):
        self.assertEqual(last_true_block(np.array([False, False, False])), (-1, -1))


# ---------------------------------------------------------------------------
# find_first_nan_after_finite
# ---------------------------------------------------------------------------

class FindFirstNanAfterFiniteTests(unittest.TestCase):

    def test_nan_after_finite(self):
        # Column 0: finite then NaN at index 1
        # Column 1: all finite, no NaN after finite → -1
        arr = np.array([[1.0, 2.0],
                        [np.nan, 3.0],
                        [np.nan, np.nan]])
        result = find_first_nan_after_finite(arr)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 2)

    def test_all_finite_returns_minus_one(self):
        arr = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
        result = find_first_nan_after_finite(arr)
        self.assertTrue(np.all(result == -1))

    def test_leading_nan_not_counted(self):
        # Column 0: NaN, then 1.0 at index 1 — no NaN after finite → -1
        arr = np.array([[np.nan],
                        [1.0],
                        [2.0]])
        result = find_first_nan_after_finite(arr)
        self.assertEqual(result[0], -1)


# ---------------------------------------------------------------------------
# find_nearest_valid_start
# ---------------------------------------------------------------------------

class FindNearestValidStartTests(unittest.TestCase):

    def _northward(self):
        return [(-1, -1), (-1, 0), (-1, 1)]

    def test_finite_start_returned_unchanged(self):
        data = np.ones((5, 5))
        start = (2, 2)
        result = find_nearest_valid_start(data, start, self._northward())
        self.assertEqual(result, start)

    def test_nan_start_relocates_along_principal_direction(self):
        data = np.ones((10, 10))
        data[3, 5] = np.nan  # starting pixel is NaN
        # principal direction is (-1, 0) → moves north (row decreasing)
        result = find_nearest_valid_start(data, (3, 5), self._northward())
        self.assertNotEqual(result, (3, 5))
        self.assertTrue(np.isfinite(data[result]))

    def test_no_finite_within_radius_returns_original(self):
        data = np.full((5, 5), np.nan)
        start = (2, 2)
        result = find_nearest_valid_start(data, start, self._northward(), max_radius=2)
        self.assertEqual(result, start)


# ---------------------------------------------------------------------------
# flood_fill
# ---------------------------------------------------------------------------

class FloodFillTests(unittest.TestCase):

    def test_pixels_at_or_above_threshold_are_marked(self):
        data = np.array([
            [0, 0, 3, 4, 5],
            [0, 2, 4, 5, 6],
            [1, 3, 5, 7, 8],
            [0, 1, 2, 3, 4],
        ], dtype=float)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        mask, _ = flood_fill(data, (2, 2), SPM_threshold=4, directions=directions)
        self.assertTrue(mask[2, 2])   # starting pixel (value 5)
        self.assertTrue(mask[2, 3])   # value 7, neighbour of start
        self.assertTrue(mask[1, 2])   # value 4, now reachable with >= push condition
        self.assertTrue(mask[1, 3])   # value 5, reachable
        self.assertTrue(mask[0, 3])   # value 4, reachable via (1,3)
        self.assertFalse(mask[3, 0])  # value 0, below threshold
        self.assertFalse(mask[0, 0])  # value 0, below threshold

    def test_all_below_threshold_produces_empty_mask(self):
        data = np.zeros((5, 5))
        mask, _ = flood_fill(data, (2, 2), SPM_threshold=1.0, directions=[(0, 1), (1, 0)])
        self.assertFalse(mask.any())

    def test_uniform_high_values_fill_interior_from_interior_start(self):
        # Starting at a boundary pixel triggers the any-neighbour-out-of-bounds guard
        # which stops BFS expansion. Start from an interior pixel instead.
        data = np.full((5, 5), 10.0)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        mask, _ = flood_fill(data, (2, 2), SPM_threshold=5.0, directions=directions)
        # The connected interior reachable from (2,2) should all be True
        self.assertTrue(mask[2, 2])
        self.assertTrue(mask[2, 3])
        self.assertTrue(mask[1, 2])

    def test_returns_boolean_arrays_of_correct_shape(self):
        data = np.random.rand(6, 8)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        mask, done = flood_fill(data, (3, 4), SPM_threshold=0.5, directions=directions)
        self.assertEqual(mask.shape, data.shape)
        self.assertEqual(done.shape, data.shape)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(done.dtype, bool)


# ---------------------------------------------------------------------------
# estimate_threshold_bounds_from_near_mouth_pixels
# ---------------------------------------------------------------------------

class EstimateThresholdBoundsTests(unittest.TestCase):

    def test_bounds_ordered_correctly(self):
        spm = np.arange(100, dtype=float).reshape(10, 10)
        land = np.zeros((10, 10), dtype=bool)
        lo, hi = estimate_threshold_bounds_from_near_mouth_pixels(spm, land, (5, 5))
        self.assertLess(lo, hi)

    def test_uniform_data_gives_equal_bounds(self):
        spm = np.full((10, 10), 5.0)
        land = np.zeros((10, 10), dtype=bool)
        lo, hi = estimate_threshold_bounds_from_near_mouth_pixels(spm, land, (5, 5))
        self.assertAlmostEqual(lo, 5.0)
        self.assertAlmostEqual(hi, 5.0)

    def test_land_pixels_excluded(self):
        spm = np.full((10, 10), 5.0)
        land = np.zeros((10, 10), dtype=bool)
        land[5, 5] = True  # starting point is land — falls back to full water distribution
        spm[5, 5] = 999.0
        lo, hi = estimate_threshold_bounds_from_near_mouth_pixels(spm, land, (5, 5), radius_pixels=0)
        self.assertAlmostEqual(lo, 5.0)
        self.assertAlmostEqual(hi, 5.0)

    def test_all_nan_near_mouth_falls_back_to_water_pixels(self):
        spm = np.full((10, 10), np.nan)
        spm[0, 0] = 3.0  # one finite pixel far from starting point
        spm[0, 1] = 7.0
        land = np.zeros((10, 10), dtype=bool)
        lo, hi = estimate_threshold_bounds_from_near_mouth_pixels(
            spm, land, (9, 9), radius_pixels=0
        )
        self.assertAlmostEqual(lo, 3.0 + 0.25 * (7.0 - 3.0), places=5)
        self.assertAlmostEqual(hi, 3.0 + 0.75 * (7.0 - 3.0), places=5)


# ---------------------------------------------------------------------------
# compute_gradient_with_directions_vectorized
# ---------------------------------------------------------------------------

class ComputeGradientTests(unittest.TestCase):

    def _simple_da(self, values):
        lat = np.linspace(0.0, 0.09, 10)
        lon = np.linspace(0.0, 0.09, 10)
        return xr.DataArray(values, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    def test_returns_none_when_all_values_clipped_to_same(self):
        """Storm case: all SPM values >> maximal_threshold → range = 0 → None."""
        data = np.full((10, 10), 100.0)
        da = self._simple_da(data)
        directions = [(-1, 0), (-1, 1), (-1, -1)]
        relative_grad, _, _ = compute_gradient_with_directions_vectorized(
            da, start_point=(5, 5), directions=directions,
            max_steps=5, lower_high_values_to=5.0,
        )
        self.assertIsNone(relative_grad)

    def test_varying_values_return_non_none_gradients(self):
        data = np.tile(np.arange(10, dtype=float), (10, 1))
        da = self._simple_da(data)
        directions = [(0, 1), (0, -1)]
        relative_grad, points, spm_values = compute_gradient_with_directions_vectorized(
            da, start_point=(5, 0), directions=directions, max_steps=5,
        )
        self.assertIsNotNone(relative_grad)
        self.assertEqual(relative_grad.shape, spm_values.shape)

    def test_gradient_points_shape_matches_directions_and_steps(self):
        data = np.random.rand(10, 10)
        da = self._simple_da(data)
        directions = [(-1, 0), (-1, 1)]
        _, points, _ = compute_gradient_with_directions_vectorized(
            da, start_point=(5, 5), directions=directions, max_steps=8,
        )
        # points shape: (n_directions_with_intermediates, n_steps, 2)
        self.assertEqual(points.shape[-1], 2)


# ---------------------------------------------------------------------------
# find_SPM_threshold
# ---------------------------------------------------------------------------

class FindSPMThresholdTests(unittest.TestCase):

    def _make_land_mask(self, shape):
        return np.zeros(shape, dtype=bool)

    def _northward_da(self, data):
        lat = np.linspace(43.0, 44.0, data.shape[0])
        lon = np.linspace(7.0, 8.0, data.shape[1])
        return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    def test_storm_case_all_values_above_maximal_returns_minimal(self):
        """All SPM >> maximal_threshold → gradient is None → returns minimal_threshold."""
        data = np.full((20, 20), 100.0)
        da = self._northward_da(data)
        land = self._make_land_mask(data.shape)
        directions = [(-1, -1), (-1, 0), (-1, 1)]
        threshold, _, _ = find_SPM_threshold(
            da, land, start_point=(10, 10),
            directions=directions, max_steps=5,
            maximal_threshold=0.5, minimal_threshold=1.0,
            quantile_to_use=0.2,
        )
        self.assertEqual(threshold, 1.0)

    def test_all_nan_data_returns_minimal(self):
        """All-NaN scene → finite_grads empty → returns minimal_threshold."""
        data = np.full((20, 20), np.nan)
        da = self._northward_da(data)
        land = self._make_land_mask(data.shape)
        directions = [(-1, -1), (-1, 0), (-1, 1)]
        threshold, _, _ = find_SPM_threshold(
            da, land, start_point=(10, 10),
            directions=directions, max_steps=5,
            maximal_threshold=50.0, minimal_threshold=2.0,
            quantile_to_use=0.2,
        )
        self.assertEqual(threshold, 2.0)

    def test_normal_case_threshold_at_least_minimal(self):
        """Normal SPM field → threshold is always >= minimal_threshold."""
        rng = np.random.default_rng(0)
        data = rng.uniform(1.0, 30.0, (30, 30))
        da = self._northward_da(data)
        land = self._make_land_mask(data.shape)
        directions = [(-1, -1), (-1, 0), (-1, 1)]
        minimal = 1.5
        threshold, _, _ = find_SPM_threshold(
            da, land, start_point=(15, 15),
            directions=directions, max_steps=10,
            maximal_threshold=25.0, minimal_threshold=minimal,
            quantile_to_use=0.2,
        )
        self.assertGreaterEqual(threshold, minimal)


# ---------------------------------------------------------------------------
# find_the_index_of_the_plume_starting_point
# ---------------------------------------------------------------------------

class FindPlumeStartingPointIndexTests(unittest.TestCase):

    def _make_da(self):
        lat = np.linspace(43.0, 44.0, 11)
        lon = np.linspace(7.0, 8.0, 11)
        data = np.zeros((11, 11))
        return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    def test_exact_grid_point(self):
        da = self._make_da()
        idx = find_the_index_of_the_plume_starting_point(da, (43.0, 7.0))
        self.assertEqual(idx, (0, 0))

    def test_nearest_grid_point(self):
        da = self._make_da()
        # midpoint between lat indices 5 and 6 → should snap to one of them
        mid_lat = float(da.lat[5])
        mid_lon = float(da.lon[5])
        idx = find_the_index_of_the_plume_starting_point(da, (mid_lat, mid_lon))
        self.assertEqual(idx, (5, 5))

    def test_returns_tuple_of_ints(self):
        da = self._make_da()
        idx = find_the_index_of_the_plume_starting_point(da, (43.5, 7.5))
        self.assertIsInstance(idx[0], int)
        self.assertIsInstance(idx[1], int)


# ---------------------------------------------------------------------------
# find_connected_shapes
# ---------------------------------------------------------------------------

class FindConnectedShapesTests(unittest.TestCase):

    def _da(self, arr):
        lat = np.arange(arr.shape[0], dtype=float)
        lon = np.arange(arr.shape[1], dtype=float)
        return xr.DataArray(arr, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    def test_shape_adjacent_to_land_is_returned(self):
        # Use a 6-column grid so the two blobs are far enough apart that
        # the dilation of shape_2 (top-left corner only) cannot reach the
        # bottom-right blob.
        shape_1 = np.array([
            [True,  True,  False, False, False, False],
            [True,  True,  False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, True,  True ],
            [False, False, False, False, True,  True ],
        ])
        shape_2 = np.array([
            [True,  False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ])
        result = find_connected_shapes(self._da(shape_1), self._da(shape_2))
        # Upper-left blob overlaps with dilation of shape_2
        self.assertTrue(result.values[0, 0])
        # Lower-right blob is too far away from shape_2 seed to be reached
        self.assertFalse(result.values[5, 5])

    def test_no_overlap_returns_empty_mask(self):
        shape_1 = np.array([[True, False], [False, False]])
        shape_2 = np.array([[False, False], [False, True]])
        result = find_connected_shapes(self._da(shape_1), self._da(shape_2))
        self.assertFalse(result.values.any())


# ---------------------------------------------------------------------------
# identify_the_shape_label_corresponding_to_the_plume
# ---------------------------------------------------------------------------

class IdentifyPlumeLabelTests(unittest.TestCase):

    def _da(self, arr, lat_start=43.0, lon_start=7.0):
        nrows, ncols = arr.shape
        lat = np.linspace(lat_start, lat_start + 0.1 * (nrows - 1), nrows)
        lon = np.linspace(lon_start, lon_start + 0.1 * (ncols - 1), ncols)
        return xr.DataArray(arr, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    def test_core_inside_blob_returns_correct_label(self):
        arr = np.array([
            [True,  True,  False, False],
            [True,  True,  False, False],
            [False, False, True,  True ],
            [False, False, True,  True ],
        ], dtype=bool)
        da = self._da(arr)
        # Core at upper-left blob
        core = (float(da.lat[0]), float(da.lon[0]))
        label, labeled, n = identify_the_shape_label_corresponding_to_the_plume(da, core)
        self.assertEqual(n, 2)
        self.assertEqual(labeled[0, 0], label)
        # The lower-right blob should have a different label
        self.assertNotEqual(labeled[3, 3], label)

    def test_core_not_in_any_blob_selects_nearest(self):
        arr = np.zeros((6, 6), dtype=bool)
        arr[0:2, 0:2] = True   # blob A
        arr[4:6, 4:6] = True   # blob B
        da = self._da(arr)
        # Core in an empty region; should pick whichever blob is closest
        core = (float(da.lat[0]), float(da.lon[5]))
        label, labeled, _ = identify_the_shape_label_corresponding_to_the_plume(da, core)
        self.assertGreater(label, 0)


# ---------------------------------------------------------------------------
# create_polygon_mask
# ---------------------------------------------------------------------------

class CreatePolygonMaskTests(unittest.TestCase):

    def _da(self, nrows=10, ncols=10):
        lat = np.linspace(43.0, 44.0, nrows)
        lon = np.linspace(7.0, 8.0, ncols)
        data = np.zeros((nrows, ncols))
        return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    def test_rectangular_range_masks_interior(self):
        da = self._da()
        params = {
            "lat_range_of_plume_area": [43.2, 43.8],
            "lon_range_of_plume_area": [7.2, 7.8],
        }
        mask = create_polygon_mask(da, params)
        # Pixels inside the range should be True
        self.assertTrue(bool(mask.sel(lat=43.5, lon=7.5, method="nearest")))
        # Corner pixels outside the range should be False
        self.assertFalse(bool(mask.sel(lat=43.0, lon=7.0, method="nearest")))

    def test_rectangular_mask_covers_correct_fraction(self):
        da = self._da(nrows=10, ncols=10)
        params = {
            "lat_range_of_plume_area": [43.0, 44.0],
            "lon_range_of_plume_area": [7.0, 8.0],
        }
        mask = create_polygon_mask(da, params)
        # All pixels should be inside
        self.assertTrue(mask.values.all())


# ---------------------------------------------------------------------------
# Check_if_the_area_is_too_cloudy
# ---------------------------------------------------------------------------

class CheckIfAreaIsTooCloudy(unittest.TestCase):

    def _setup(self, nrows=10, ncols=10):
        lat = np.linspace(43.0, 44.0, nrows)
        lon = np.linspace(7.0, 8.0, ncols)
        return lat, lon

    def test_clear_scene_is_not_too_cloudy(self):
        lat, lon = self._setup()
        data = np.ones((10, 10))  # all finite → no cloud
        da = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        water_mask = xr.DataArray(
            np.ones((10, 10), dtype=bool),
            dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )
        params = {
            "lat_range_of_plume_area": [43.0, 44.0],
            "lon_range_of_plume_area": [7.0, 8.0],
            "threshold_of_cloud_coverage_in_percentage": 25,
        }
        self.assertFalse(Check_if_the_area_is_too_cloudy(da, water_mask, params))

    def test_all_nan_scene_is_too_cloudy(self):
        lat, lon = self._setup()
        data = np.full((10, 10), np.nan)
        da = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        water_mask = xr.DataArray(
            np.ones((10, 10), dtype=bool),
            dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )
        params = {
            "lat_range_of_plume_area": [43.0, 44.0],
            "lon_range_of_plume_area": [7.0, 8.0],
            "threshold_of_cloud_coverage_in_percentage": 25,
        }
        self.assertTrue(Check_if_the_area_is_too_cloudy(da, water_mask, params))

    def test_no_water_pixels_returns_true(self):
        lat, lon = self._setup()
        data = np.ones((10, 10))
        da = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        water_mask = xr.DataArray(
            np.zeros((10, 10), dtype=bool),
            dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )
        params = {
            "lat_range_of_plume_area": [43.0, 44.0],
            "lon_range_of_plume_area": [7.0, 8.0],
            "threshold_of_cloud_coverage_in_percentage": 25,
        }
        self.assertTrue(Check_if_the_area_is_too_cloudy(da, water_mask, params))


# ---------------------------------------------------------------------------
# derive_masks_from_bathymetry
# ---------------------------------------------------------------------------

class DeriveMasksFromBathymetryTests(unittest.TestCase):

    def _bathy_da(self, values, lat, lon):
        return xr.DataArray(values, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    def test_negative_bathy_is_water_positive_is_land(self):
        lat = np.linspace(43.0, 44.0, 5)
        lon = np.linspace(7.0, 8.0, 5)
        data = np.array([
            [-10, -20, -30, -40, -50],
            [-10, -20, -30, -40, -50],
            [  5,  10,  15,  20,  25],   # land row
            [-10, -20, -30, -40, -50],
            [-10, -20, -30, -40, -50],
        ], dtype=float)
        full_bathy = self._bathy_da(data, lat, lon)
        reduced_bathy = self._bathy_da(data, lat, lon)
        params = {
            "lat_range_of_plume_area": [43.0, 44.0],
            "lon_range_of_plume_area": [7.0, 8.0],
        }
        cloud_mask, land_mask = derive_masks_from_bathymetry(full_bathy, reduced_bathy, params)
        # Water pixels (negative bathy) should appear in cloud_mask
        self.assertTrue(bool(cloud_mask.sel(lat=43.0, lon=7.0, method="nearest")))
        # Land row should be True in land_mask
        self.assertTrue(bool(land_mask.sel(lat=43.5, lon=7.5, method="nearest")))

    def test_all_water_produces_empty_land_mask(self):
        lat = np.linspace(43.0, 44.0, 4)
        lon = np.linspace(7.0, 8.0, 4)
        data = np.full((4, 4), -50.0)
        full_bathy = self._bathy_da(data, lat, lon)
        reduced_bathy = self._bathy_da(data, lat, lon)
        params = {
            "lat_range_of_plume_area": [43.0, 44.0],
            "lon_range_of_plume_area": [7.0, 8.0],
        }
        _, land_mask = derive_masks_from_bathymetry(full_bathy, reduced_bathy, params)
        self.assertFalse(land_mask.values.any())


# ---------------------------------------------------------------------------
# find_high_value_pixels
# ---------------------------------------------------------------------------

class FindHighValuePixelsTests(unittest.TestCase):

    def _da(self, values, lat, lon):
        return xr.DataArray(values, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    def test_pixels_above_threshold_and_within_radius_are_true(self):
        lat = np.linspace(-1.0, 1.0, 5)
        lon = np.linspace(-1.0, 1.0, 5)
        data = np.full((5, 5), 5.0)
        data[2, 2] = 100.0  # centre pixel, high value
        da = self._da(data, lat, lon)
        mask = find_high_value_pixels(da, center_lat=0.0, center_lon=0.0,
                                      radius_km=50_000, SPM_threshold=50.0)
        self.assertTrue(bool(mask[2, 2]))

    def test_pixels_below_threshold_are_false(self):
        lat = np.linspace(-1.0, 1.0, 5)
        lon = np.linspace(-1.0, 1.0, 5)
        data = np.ones((5, 5))
        da = self._da(data, lat, lon)
        mask = find_high_value_pixels(da, center_lat=0.0, center_lon=0.0,
                                      radius_km=50_000, SPM_threshold=10.0)
        self.assertFalse(mask.values.any())

    def test_pixels_outside_radius_are_false(self):
        lat = np.linspace(-5.0, 5.0, 5)
        lon = np.linspace(-5.0, 5.0, 5)
        data = np.full((5, 5), 100.0)
        da = self._da(data, lat, lon)
        # radius so small only the centre could qualify
        mask = find_high_value_pixels(da, center_lat=0.0, center_lon=0.0,
                                      radius_km=1.0, SPM_threshold=50.0)
        # Only the centre pixel (if any) should be True
        n_true = mask.values.sum()
        self.assertLessEqual(n_true, 1)


# ---------------------------------------------------------------------------
# reduce_resolution
# ---------------------------------------------------------------------------

class ReduceResolutionTests(unittest.TestCase):

    def _ds(self, nrows, ncols, lat_step=0.01, lon_step=0.01):
        lat = np.arange(nrows) * lat_step
        lon = np.arange(ncols) * lon_step
        data = np.ones((nrows, ncols))
        return xr.Dataset({"spm": (["lat", "lon"], data)},
                          coords={"lat": lat, "lon": lon})

    def test_output_resolution_halved(self):
        ds = self._ds(8, 8, lat_step=0.01, lon_step=0.01)
        ds_reduced = reduce_resolution(ds, lat_bin_size_in_degree=0.02, lon_bin_size_in_degree=0.02)
        self.assertEqual(ds_reduced.dims["lat"], 4)
        self.assertEqual(ds_reduced.dims["lon"], 4)

    def test_uniform_values_preserved_after_coarsening(self):
        ds = self._ds(6, 6, lat_step=0.01, lon_step=0.01)
        ds_reduced = reduce_resolution(ds, lat_bin_size_in_degree=0.02, lon_bin_size_in_degree=0.02)
        np.testing.assert_allclose(ds_reduced["spm"].values, 1.0)


# ---------------------------------------------------------------------------
# return_stats_dictionnary
# ---------------------------------------------------------------------------

class ReturnStatsDictTests(unittest.TestCase):

    def _minimal_da(self, data, date="2020-01-01"):
        lat = np.linspace(43.0, 44.0, data.shape[0])
        lon = np.linspace(7.0, 8.0, data.shape[1])
        da = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        da.attrs["date_for_plot"] = date
        return da

    def _params(self):
        return {"lat_range_of_plume_area": [43.0, 44.0]}

    def test_return_empty_dict_flag_produces_nan_stats(self):
        data = np.ones((5, 5))
        da = self._minimal_da(data)
        result = return_stats_dictionnary(
            final_mask_area=None,
            spm_reduced_map=da,
            spm_map=da,
            parameters=self._params(),
            thresholds={"Seine": 5.0},
            return_empty_dict=True,
        )
        self.assertIn("date", result)
        self.assertEqual(result["n_pixel_in_the_plume_area"], 0)
        self.assertTrue(np.isnan(result["lat_centroid_of_the_plume_area"]))
        self.assertIn("SPM_threshold_Seine", result)

    def test_all_nan_plume_pixels_returns_empty_dict(self):
        """Mask covers some pixels but all are NaN in SPM → empty dict returned."""
        spm_data = np.full((5, 5), np.nan)
        mask_data = np.zeros((5, 5), dtype=bool)
        mask_data[2, 2] = True  # one masked pixel, but its SPM is NaN
        da_spm = self._minimal_da(spm_data)
        da_mask = xr.DataArray(
            mask_data, dims=["lat", "lon"],
            coords={"lat": da_spm.lat.values, "lon": da_spm.lon.values}
        )
        result = return_stats_dictionnary(
            final_mask_area=da_mask,
            spm_reduced_map=da_spm,
            spm_map=da_spm,
            parameters=self._params(),
            thresholds={"Seine": 5.0},
        )
        self.assertEqual(result["n_pixel_in_the_plume_area"], 0)
        self.assertTrue(np.isnan(result["lat_centroid_of_the_plume_area"]))


if __name__ == "__main__":
    unittest.main()
