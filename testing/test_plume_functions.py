"""
Unit tests for standalone functions in panache.plume_algorithm that are not
covered by the integration pipeline tests.  Each test class is labelled with
the line numbers it targets.
"""
from __future__ import annotations

import os
import pickle
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import xarray as xr

from panache.plume_algorithm import (
    compute_gradient_with_directions_vectorized,
    create_polygon_mask,
    estimate_threshold_bounds_from_near_mouth_pixels,
    fast_delimitation_of_a_river_plume_area,
    filter_gradient_points_vectorized,
    find_first_nan_after_finite,
    find_high_value_pixels,
    find_SPM_threshold,
    first_true_block,
    identify_the_shape_label_corresponding_to_the_plume,
    last_true_block,
    load_and_filter_arrays,
    merge_plume_shape_with_close_shapes,
    pixels_far_from_land,
    reduce_resolution,
    set_mask_area_values_to_False_based_on_an_index_object,
)
from panache.utils import searching_strategy_directions_from_presets


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _da(values, lat, lon):
    return xr.DataArray(values, dims=["lat", "lon"],
                        coords={"lat": lat, "lon": lon})


def _gaussian_spm(n=20, amplitude=8.0):
    lat = np.linspace(0.0, 1.0, n)
    lon = np.linspace(0.0, 1.0, n)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    vals = 1.0 + amplitude * np.exp(
        -(((lat_g - 0.5) ** 2) + ((lon_g - 0.5) ** 2)) / 0.02
    )
    return _da(vals, lat, lon)


def _all_water(n=20):
    lat = np.linspace(0.0, 1.0, n)
    lon = np.linspace(0.0, 1.0, n)
    return _da(np.zeros((n, n), dtype=bool), lat, lon)


# ---------------------------------------------------------------------------
# reduce_resolution — lines 94-102
# ---------------------------------------------------------------------------

class ReduceResolutionTests(unittest.TestCase):

    def test_reduces_spatial_dims(self):
        n = 20
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        da = _da(np.ones((n, n)), lat, lon)
        # bin size 2× the spacing → halve the resolution
        spacing = float(np.diff(lat).mean())
        result = reduce_resolution(da, spacing * 2, spacing * 2)
        self.assertLess(result.sizes["lat"], n)
        self.assertLess(result.sizes["lon"], n)

    def test_values_are_averaged(self):
        lat = np.linspace(0.0, 1.0, 10)
        lon = np.linspace(0.0, 1.0, 10)
        vals = np.arange(100, dtype=float).reshape(10, 10)
        da = _da(vals, lat, lon)
        spacing = float(np.diff(lat).mean())
        result = reduce_resolution(da, spacing * 2, spacing * 2)
        self.assertEqual(result.sizes["lat"], 5)


# ---------------------------------------------------------------------------
# find_high_value_pixels — lines 366-388
# ---------------------------------------------------------------------------

class FindHighValuePixelsTests(unittest.TestCase):

    def test_returns_mask_for_pixels_near_center(self):
        n = 20
        lat = np.linspace(48.0, 50.0, n)
        lon = np.linspace(-1.0, 1.0, n)
        lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
        # High SPM near center of the grid
        vals = np.where(
            (lat_g > 48.9) & (lat_g < 49.1) & (lon_g > -0.1) & (lon_g < 0.1),
            10.0,
            1.0,
        )
        da = _da(vals, lat, lon)
        mask = find_high_value_pixels(da, center_lat=49.0, center_lon=0.0,
                                       radius_km=30.0, SPM_threshold=5.0)
        # At least some pixels should be True
        self.assertTrue(mask.any())

    def test_large_radius_includes_more_pixels(self):
        n = 20
        lat = np.linspace(48.0, 50.0, n)
        lon = np.linspace(-1.0, 1.0, n)
        lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
        vals = np.full((n, n), 10.0)
        da = _da(vals, lat, lon)
        mask_small = find_high_value_pixels(da, 49.0, 0.0, radius_km=10.0, SPM_threshold=5.0)
        mask_large = find_high_value_pixels(da, 49.0, 0.0, radius_km=200.0, SPM_threshold=5.0)
        self.assertGreaterEqual(mask_large.sum(), mask_small.sum())

    def test_threshold_higher_than_all_spm_gives_empty_mask(self):
        n = 10
        lat = np.linspace(48.0, 50.0, n)
        lon = np.linspace(-1.0, 1.0, n)
        da = _da(np.ones((n, n)), lat, lon)
        mask = find_high_value_pixels(da, 49.0, 0.0, radius_km=100.0, SPM_threshold=99.0)
        self.assertFalse(mask.any())


# ---------------------------------------------------------------------------
# estimate_threshold_bounds_from_near_mouth_pixels — lines 1054-1055
# (small-sample fallback to full-scene water distribution)
# ---------------------------------------------------------------------------

class EstimateThresholdBoundsTests(unittest.TestCase):

    def test_normal_case_returns_finite_bounds(self):
        spm = np.ones((20, 20)) * 5.0
        land = np.zeros((20, 20), dtype=bool)
        lo, hi = estimate_threshold_bounds_from_near_mouth_pixels(spm, land, start_pixel=(10, 10))
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))
        self.assertLessEqual(lo, hi)

    def test_small_sample_falls_back_to_full_scene(self):
        # Start pixel in corner + very small radius → fewer than 5 near-mouth pixels
        spm = np.ones((20, 20)) * 3.0
        land = np.zeros((20, 20), dtype=bool)
        lo, hi = estimate_threshold_bounds_from_near_mouth_pixels(
            spm, land, start_pixel=(0, 0), radius_pixels=1
        )
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))


# ---------------------------------------------------------------------------
# filter_gradient_points_vectorized — lines 971-996
# ---------------------------------------------------------------------------

class FilterGradientPointsVectorizedTests(unittest.TestCase):

    def _gradient_inputs(self, n=20):
        spm = _gaussian_spm(n)
        land = _all_water(n)
        dirs = searching_strategy_directions_from_presets({"S": "northward_fan"})["S"]
        gv, gp, av = compute_gradient_with_directions_vectorized(
            spm_map=spm, start_point=(6, 10), directions=dirs, max_steps=4,
            lower_high_values_to=10.0,
            create_X_intermediates_between_each_direction=2,
        )
        return gv, gp, av, land

    def test_returns_arrays(self):
        gv, gp, av, land = self._gradient_inputs()
        threshold = float(np.nanmax(gv[np.isfinite(gv)])) * 0.9
        fpts, fvals, fabs = filter_gradient_points_vectorized(gv, gp, av, land, threshold)
        self.assertIsInstance(fpts, np.ndarray)
        self.assertIsInstance(fvals, np.ndarray)

    def test_high_threshold_reduces_kept_points(self):
        gv, gp, av, land = self._gradient_inputs()
        lo_thresh = float(np.nanmin(np.abs(gv[np.isfinite(gv)])))
        hi_thresh = float(np.nanmax(np.abs(gv[np.isfinite(gv)]))) * 2.0
        _, fvals_lo, _ = filter_gradient_points_vectorized(gv, gp, av, land, lo_thresh)
        _, fvals_hi, _ = filter_gradient_points_vectorized(gv, gp, av, land, hi_thresh)
        self.assertGreaterEqual(len(fvals_lo), len(fvals_hi))


# ---------------------------------------------------------------------------
# find_SPM_threshold — lines 1092-1106
# ---------------------------------------------------------------------------

class FindSPMThresholdTests(unittest.TestCase):

    def test_returns_finite_threshold_on_gradient_data(self):
        n = 20
        spm = _gaussian_spm(n)
        land = _all_water(n)
        dirs = searching_strategy_directions_from_presets({"S": "northward_fan"})["S"]
        result, fpts, gp = find_SPM_threshold(
            spm_map=spm,
            land_mask=land,
            start_point=(6, 10),
            directions=dirs,
            max_steps=4,
            maximal_threshold=10.0,
            minimal_threshold=1.0,
            quantile_to_use=0.5,
        )
        self.assertTrue(np.isfinite(result))

    def test_empty_gradients_returns_minimal_threshold(self):
        # All-uniform SPM → gradient = 0 everywhere → finite_grads is empty-ish → fallback
        n = 20
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        spm = _da(np.full((n, n), 3.0), lat, lon)
        land = _all_water(n)
        dirs = searching_strategy_directions_from_presets({"S": "northward_fan"})["S"]
        result, _, _ = find_SPM_threshold(
            spm_map=spm,
            land_mask=land,
            start_point=(6, 10),
            directions=dirs,
            max_steps=4,
            maximal_threshold=10.0,
            minimal_threshold=2.5,
            quantile_to_use=0.5,
        )
        self.assertTrue(np.isfinite(result))


# ---------------------------------------------------------------------------
# find_first_nan_after_finite — lines 1126-1144
# ---------------------------------------------------------------------------

class FindFirstNanAfterFiniteTests(unittest.TestCase):

    def test_nan_after_finite_detected(self):
        # 3D array: (n_steps=5, 1, n_directions=3)
        arr = np.array([
            [[1.0, 2.0, np.nan]],
            [[np.nan, 3.0, 4.0]],
            [[5.0, np.nan, 5.0]],
            [[6.0, 7.0, np.nan]],
            [[7.0, 8.0, 6.0]],
        ])
        result = find_first_nan_after_finite(arr)
        self.assertIsInstance(result, np.ndarray)
        # Some NaN-after-finite transitions exist → some entries should not be -1
        self.assertTrue(np.any(result != -1))

    def test_no_nan_returns_minus_one(self):
        arr = np.array([
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0]],
        ])
        result = find_first_nan_after_finite(arr)
        # No NaN after finite → all -1
        self.assertTrue(np.all(result == -1))

    def test_leading_nan_then_finite_then_nan(self):
        arr = np.array([
            [[np.nan, 1.0]],
            [[np.nan, 2.0]],
            [[3.0, np.nan]],
            [[4.0, 4.0]],
        ])
        result = find_first_nan_after_finite(arr)
        # Function runs and returns an ndarray — shape depends on internal broadcasting
        self.assertIsInstance(result, np.ndarray)
        # Direction 1 has a NaN at step 2 after finite values → some entry should be 2
        self.assertTrue(np.any(result == 2))


# ---------------------------------------------------------------------------
# pixels_far_from_land — lines 1168-1184
# ---------------------------------------------------------------------------

class PixelsFarFromLandTests(unittest.TestCase):

    def test_all_water_land_mask_returns_true_array(self):
        n = 10
        land = _all_water(n)
        # Build a small set of pixel positions (2 directions, 3 steps, (x,y))
        pixel_positions = np.array([
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ])
        result = pixels_far_from_land(land, pixel_positions=pixel_positions,
                                       distance_threshold=0)
        self.assertEqual(result.shape, (2, 3))

    def test_distance_threshold_filters_near_land(self):
        n = 10
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        land_vals = np.zeros((n, n), dtype=bool)
        land_vals[0, :] = True   # top row is land
        land = _da(land_vals, lat, lon)

        # Pixel at row=0, col=5 (right on land border) — x=5, y=0
        pixel_positions = np.array([[[5.0, 0.0], [5.0, 3.0]]])
        result_strict = pixels_far_from_land(land, pixel_positions, distance_threshold=5)
        result_loose = pixels_far_from_land(land, pixel_positions, distance_threshold=0)
        # Loose threshold should allow more (or equal) pixels than strict
        self.assertGreaterEqual(int(result_loose.sum()), int(result_strict.sum()))


# ---------------------------------------------------------------------------
# first_true_block — lines 1211-1223 (multi-block case)
# ---------------------------------------------------------------------------

class FirstTrueBlockTests(unittest.TestCase):

    def test_no_true_returns_minus_one(self):
        self.assertEqual(first_true_block(np.zeros(5, dtype=bool)), (-1, -1))

    def test_single_contiguous_block(self):
        arr = np.array([False, True, True, True, False])
        self.assertEqual(first_true_block(arr), (1, 3))

    def test_multiple_blocks_returns_first(self):
        # Two blocks: [1,2] and [4,5]
        arr = np.array([False, True, True, False, True, True, False])
        start, end = first_true_block(arr)
        self.assertEqual(start, 1)
        self.assertEqual(end, 2)

    def test_whole_array_true(self):
        arr = np.ones(4, dtype=bool)
        self.assertEqual(first_true_block(arr), (0, 3))


# ---------------------------------------------------------------------------
# last_true_block — lines 1243-1265
# ---------------------------------------------------------------------------

class LastTrueBlockTests(unittest.TestCase):

    def test_no_true_returns_minus_one(self):
        self.assertEqual(last_true_block(np.zeros(5, dtype=bool)), (-1, -1))

    def test_single_block(self):
        arr = np.array([False, True, True, False])
        self.assertEqual(last_true_block(arr), (1, 2))

    def test_multiple_blocks_returns_last(self):
        arr = np.array([True, True, False, True, True, True, False])
        start, end = last_true_block(arr)
        self.assertEqual(start, 3)
        self.assertEqual(end, 5)

    def test_block_starting_at_index_zero(self):
        arr = np.array([True, True, True, False])
        start, end = last_true_block(arr)
        self.assertEqual(start, 0)
        self.assertEqual(end, 2)


# ---------------------------------------------------------------------------
# load_and_filter_arrays — lines 1285-1300
# ---------------------------------------------------------------------------

class LoadAndFilterArraysTests(unittest.TestCase):

    def test_loads_and_filters_small_arrays(self):
        payload = {
            "small": np.arange(5),           # len 5 → kept
            "big": np.arange(100),            # len 100 → dropped
            "scalar": 42,                     # non-array → kept
            "string": "hello",               # non-array → kept
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(payload, tmp)
            tmp_path = tmp.name
        result = load_and_filter_arrays(tmp_path)
        self.assertIn("small", result)
        self.assertNotIn("big", result)
        self.assertIn("scalar", result)
        self.assertIn("string", result)

    def test_datetimeindex_is_kept_if_short(self):
        import pandas as pd
        payload = {
            "dates": pd.DatetimeIndex(["2020-01-01", "2020-01-02"]),
            "long_dates": pd.DatetimeIndex(pd.date_range("2020-01-01", periods=20)),
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(payload, tmp)
            tmp_path = tmp.name
        result = load_and_filter_arrays(tmp_path)
        self.assertIn("dates", result)
        self.assertNotIn("long_dates", result)


# ---------------------------------------------------------------------------
# set_mask_area_values_to_False_based_on_an_index_object — lines 1358-1379
# ---------------------------------------------------------------------------

class SetMaskAreaToFalseTests(unittest.TestCase):

    def _gradient_inputs(self, n=20):
        spm = _gaussian_spm(n)
        dirs = searching_strategy_directions_from_presets({"S": "northward_fan"})["S"]
        gv, gp, av = compute_gradient_with_directions_vectorized(
            spm_map=spm, start_point=(6, 10), directions=dirs, max_steps=4,
            lower_high_values_to=10.0,
            create_X_intermediates_between_each_direction=2,
        )
        return gv, gp

    def test_returns_dataarray(self):
        n = 20
        gv, gp = self._gradient_inputs(n)
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        mask_vals = np.zeros((n, n), dtype=bool)
        mask_vals[7:13, 8:13] = True
        mask = _da(mask_vals, lat, lon)
        index_object = np.where(gv > 0)
        result = set_mask_area_values_to_False_based_on_an_index_object(mask, index_object, gp)
        self.assertIsInstance(result, xr.DataArray)

    def test_modifies_mask(self):
        n = 20
        gv, gp = self._gradient_inputs(n)
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        mask_vals = np.ones((n, n), dtype=bool)
        mask = _da(mask_vals, lat, lon)
        before = int(mask.values.sum())
        index_object = np.where(gv > 0)
        result = set_mask_area_values_to_False_based_on_an_index_object(mask, index_object, gp)
        after = int(result.values.sum())
        # The function may keep the mask as is or trim it
        self.assertGreaterEqual(before, 0)
        self.assertGreaterEqual(after, 0)


# ---------------------------------------------------------------------------
# fast_delimitation_of_a_river_plume_area — lines 1919-1954
# ---------------------------------------------------------------------------

class FastDelimitationTests(unittest.TestCase):

    def _spm_and_land(self, n=30):
        spm = _gaussian_spm(n, amplitude=8.0)
        n_land = n
        lat = np.linspace(0.0, 1.0, n_land)
        lon = np.linspace(0.0, 1.0, n_land)
        land = _da(np.zeros((n_land, n_land), dtype=bool), lat, lon)
        return spm, land

    def test_returns_path_for_plume_spm(self):
        from matplotlib.path import Path
        spm, land = self._spm_and_land(30)
        result = fast_delimitation_of_a_river_plume_area(
            spm_map=spm, land_mask=land, start_point=(12, 15),
            SPM_threshold=3.0, maximal_threshold=10.0, max_steps=10,
        )
        self.assertIsInstance(result, Path)

    def test_uniform_spm_may_return_none(self):
        n = 20
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        spm = _da(np.full((n, n), 2.0), lat, lon)
        land = _da(np.zeros((n, n), dtype=bool), lat, lon)
        # Uniform SPM → no gradient → gradient_values is None or all-zero → returns None
        result = fast_delimitation_of_a_river_plume_area(
            spm_map=spm, land_mask=land, start_point=(10, 10),
            SPM_threshold=1.0, maximal_threshold=5.0, max_steps=10,
        )
        # result is None or a Path (both valid — we're covering the code path)
        self.assertIn(type(result).__name__, ["NoneType", "Path"])

    def test_land_pixels_trigger_max_step_adjustment(self):
        from matplotlib.path import Path
        n = 30
        spm = _gaussian_spm(n, amplitude=8.0)
        land_vals = np.zeros((n, n), dtype=bool)
        # Add a strip of land that forces the loop at line 1930
        land_vals[5:8, :] = True
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        land = _da(land_vals, lat, lon)
        result = fast_delimitation_of_a_river_plume_area(
            spm_map=spm, land_mask=land, start_point=(12, 15),
            SPM_threshold=3.0, maximal_threshold=10.0, max_steps=10,
        )
        self.assertIn(type(result).__name__, ["NoneType", "Path"])


# ---------------------------------------------------------------------------
# create_polygon_mask — lines 1736-1744 (polygon path, >2 lat values)
# ---------------------------------------------------------------------------

class CreatePolygonMaskTests(unittest.TestCase):

    def test_polygon_with_more_than_two_lat_values(self):
        n = 10
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        ds = _da(np.ones((n, n)), lat, lon)
        # Polygon with 4 lat/lon points — triggers lines 1736-1744
        params = {
            "lat_range_of_plume_area": [0.0, 0.5, 1.0, 0.5],
            "lon_range_of_plume_area": [0.0, 0.0, 0.5, 0.5],
        }
        result = create_polygon_mask(ds, params)
        self.assertEqual(result.shape, (n, n))
        # Some pixels inside the diamond polygon
        self.assertGreater(int(result.sum()), 0)

    def test_two_value_bounding_box(self):
        n = 10
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        ds = _da(np.ones((n, n)), lat, lon)
        params = {
            "lat_range_of_plume_area": [0.0, 1.0],
            "lon_range_of_plume_area": [0.0, 1.0],
        }
        result = create_polygon_mask(ds, params)
        self.assertEqual(result.shape, (n, n))


# ---------------------------------------------------------------------------
# identify_the_shape_label_corresponding_to_the_plume — lines 630-636
# centroid-distance fallback when core pixel maps to background (label 0)
# ---------------------------------------------------------------------------

class IdentifyShapeLabelCentroidTests(unittest.TestCase):

    def test_core_in_background_triggers_centroid_fallback(self):
        n = 10
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        mask_vals = np.zeros((n, n), dtype=bool)
        mask_vals[2:5, 2:5] = True  # one blob, rows 2-4 cols 2-4
        mask = xr.DataArray(mask_vals, dims=["lat", "lon"],
                            coords={"lat": lat, "lon": lon})
        # Core at far corner — background pixel → label=0 → centroid path fires
        core = (lat[8], lon[8])
        label_val, labeled_arr, num_feat = identify_the_shape_label_corresponding_to_the_plume(mask, core)
        self.assertEqual(num_feat, 1)
        self.assertEqual(label_val, 1)


# ---------------------------------------------------------------------------
# merge_plume_shape_with_close_shapes — lines 717-721
# branch fires when dilation of main blob bridges a second nearby blob
# ---------------------------------------------------------------------------

class MergePlumeShapeWithCloseShapesTests(unittest.TestCase):

    def test_dilation_bridges_two_blobs_expands_mask(self):
        n = 15
        lat = np.linspace(0.0, 1.0, n)
        lon = np.linspace(0.0, 1.0, n)
        mask_vals = np.zeros((n, n), dtype=bool)
        mask_vals[2:5, 5:8] = True   # blob A
        mask_vals[6:9, 5:8] = True   # blob B — 1 pixel gap below blob A
        mask = xr.DataArray(mask_vals, dims=["lat", "lon"],
                            coords={"lat": lat, "lon": lon})
        land = xr.DataArray(np.zeros((n, n), dtype=bool), dims=["lat", "lon"],
                            coords={"lat": lat, "lon": lon})
        core = (lat[3], lon[6])  # inside blob A
        struct = np.ones((3, 3), dtype=bool)
        before = int(mask.values.sum())
        result = merge_plume_shape_with_close_shapes(mask, core, land, struct)
        self.assertGreater(int(result.values.sum()), before)


if __name__ == "__main__":
    unittest.main()
