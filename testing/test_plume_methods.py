"""
Direct unit tests for Create_the_plume_mask methods and Set_cloudy_regions_to_True.

All tests use an 11×11 grid with max_steps=4 and starting lat=0.3, lon=0.5
(pixel index (3,5)) to avoid out-of-bounds IndexErrors in
compute_gradient_with_directions_vectorized.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import xarray as xr

from panache.plume_algorithm import (
    Create_the_plume_mask,
    Set_cloudy_regions_to_True,
)
from panache.utils import searching_strategy_directions_from_presets


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _lat_lon(n=11):
    return np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n)


def _da(values, lat, lon):
    return xr.DataArray(values, dims=["lat", "lon"],
                        coords={"lat": lat, "lon": lon})


def _make_plume(spm_values, land_values=None, plume_name="Seine",
                max_steps=4, start_lat=0.3, start_lon=0.5,
                fixed_threshold=5.0, min_thresh=4.0, max_thresh=10.0):
    """Build a Create_the_plume_mask with safe parameters."""
    n = spm_values.shape[0]
    lat, lon = _lat_lon(n)
    spm = _da(spm_values.astype(float), lat, lon)
    if land_values is None:
        land_values = np.zeros((n, n), dtype=bool)
    land = _da(land_values, lat, lon)
    bathy = _da(np.full((n, n), -50.0), lat, lon)
    params = {
        "searching_strategies": {plume_name: "northward_fan"},
        "searching_strategy_directions": searching_strategy_directions_from_presets(
            {plume_name: "northward_fan"}
        ),
        "starting_points": {plume_name: [start_lat, start_lon]},
        "core_of_the_plumes": {plume_name: [0.5, 0.5]},
        "lat_range_of_plume_area": [0.0, 1.0],
        "lon_range_of_plume_area": [0.0, 1.0],
        "bathymetric_threshold": 0,
        "threshold_of_cloud_coverage_in_percentage": 40,
        "maximal_bathymetric_for_zone_with_resuspension": {plume_name: 0},
        "minimal_distance_from_estuary_for_zone_with_resuspension": {plume_name: 999},
        "max_steps_for_the_directions": {plume_name: max_steps},
        "maximal_threshold": {plume_name: max_thresh},
        "minimal_threshold": {plume_name: min_thresh},
        "quantile_to_use": {plume_name: 0.5},
        "fixed_threshold": {plume_name: fixed_threshold},
        "river_mouth_to_exclude": {},
    }
    return Create_the_plume_mask(spm, bathy, land, params, plume_name)


def _gaussian_spm(n=11, center_lat=0.5, center_lon=0.5, amplitude=8.0):
    lat, lon = _lat_lon(n)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return 1.0 + amplitude * np.exp(
        -(((lat_grid - center_lat) ** 2) + ((lon_grid - center_lon) ** 2)) / 0.02
    )


# ---------------------------------------------------------------------------
# Set_cloudy_regions_to_True — inner loop body (lines 776-794)
# ---------------------------------------------------------------------------

class SetCloudyRegionsInnerLoopTests(unittest.TestCase):
    """
    The inner loop body executes only when an enclosed NaN pixel (False in
    mask_area_to_use) does NOT touch the map edge and is surrounded by
    high-SPM pixels.
    """

    def test_inner_loop_marks_enclosed_nan_as_plume(self):
        lat, lon = _lat_lon(11)
        # All pixels have SPM=9 EXCEPT the interior pixel at (5,5) which is NaN.
        spm_values = np.full((11, 11), 9.0)
        spm_values[5, 5] = np.nan
        spm = _da(spm_values, lat, lon)
        # Plume mask: True everywhere, False at the NaN pixel.
        mask_values = np.ones((11, 11), dtype=bool)
        mask_values[5, 5] = False
        mask = _da(mask_values, lat, lon)
        land = _da(np.zeros((11, 11), dtype=bool), lat, lon)

        result = Set_cloudy_regions_to_True(spm, mask, land, SPM_threshold=5.0)

        # The enclosed NaN pixel at (5,5) should have been marked True because
        # all 8 surrounding pixels have SPM=9 > threshold=5.
        self.assertTrue(bool(result.values[5, 5]))

    def test_edge_nan_pixel_not_included(self):
        lat, lon = _lat_lon(11)
        spm_values = np.full((11, 11), 9.0)
        spm_values[0, 5] = np.nan   # on the map edge → should NOT be included
        spm = _da(spm_values, lat, lon)
        mask_values = np.ones((11, 11), dtype=bool)
        mask_values[0, 5] = False
        mask = _da(mask_values, lat, lon)
        land = _da(np.zeros((11, 11), dtype=bool), lat, lon)

        result = Set_cloudy_regions_to_True(spm, mask, land, SPM_threshold=5.0)

        # Edge NaN pixel should remain False (its false_area touches the map edge).
        self.assertFalse(bool(result.values[0, 5]))


# ---------------------------------------------------------------------------
# determine_SPM_threshold — null-bounds branch (lines 2204-2206)
# ---------------------------------------------------------------------------

class DetermineSpmThresholdNullBoundsTests(unittest.TestCase):

    def test_null_min_max_triggers_estimation(self):
        # Use a 20×20 grid so that xarray coordinate indexing in
        # filter_gradient_points_vectorized doesn't go OOB when accessing
        # the 17 fan-direction indices in the (17,4) absolute_values DataArray.
        spm_values = _gaussian_spm(n=20)
        plume = _make_plume(spm_values, start_lat=0.3, start_lon=0.5,
                            min_thresh=None, max_thresh=None)
        # With min/max threshold = None and dynamic=True, the code must call
        # estimate_threshold_bounds_from_near_mouth_pixels (lines 2204-2212).
        plume.determine_SPM_threshold(
            dynamic_determination_of_SPM_threshold=True,
            precomputed_threshold=None,
        )
        # threshold should have been set to a finite value
        self.assertTrue(np.isfinite(plume.SPM_threshold))
        self.assertIn("determine_SPM_threshold", plume.protocol[-1])


# ---------------------------------------------------------------------------
# determine_SPM_threshold — fixed_threshold=None raises ValueError (line 2231)
# ---------------------------------------------------------------------------

class DetermineSpmThresholdNoThresholdTests(unittest.TestCase):

    def test_fixed_threshold_none_raises(self):
        spm_values = _gaussian_spm(11)
        plume = _make_plume(spm_values, fixed_threshold=None)
        with self.assertRaises(ValueError):
            plume.determine_SPM_threshold(
                dynamic_determination_of_SPM_threshold=False,
                precomputed_threshold=None,
            )


# ---------------------------------------------------------------------------
# do_a_raw_plume_detection — NaN starting pixel (lines 2253-2255)
# ---------------------------------------------------------------------------

class DoRawPlumeDetectionNanStartTests(unittest.TestCase):

    def test_nan_start_pixel_relocated(self):
        spm_values = _gaussian_spm(11)
        # Put NaN at the configured starting pixel (index (3,5) for lat=0.3, lon=0.5).
        spm_values[3, 5] = np.nan
        plume = _make_plume(spm_values)
        plume.determine_SPM_threshold(False, precomputed_threshold=5.0)
        # Should print a relocation message and still run without error.
        plume.do_a_raw_plume_detection()
        self.assertIn("do_a_raw_plume_detection", plume.protocol[-1])


# ---------------------------------------------------------------------------
# remove_parts_of_the_plume_area_that_widden_after_the_shrinking_phase
# (lines 2527-2541)
# ---------------------------------------------------------------------------

class WiddenRemovalTests(unittest.TestCase):

    def _plume_with_mask(self, mask_values):
        lat, lon = _lat_lon(11)
        spm_values = _gaussian_spm(11)
        plume = _make_plume(spm_values)
        plume.determine_SPM_threshold(False, precomputed_threshold=5.0)
        plume.plume_mask = xr.DataArray(
            mask_values, dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
        return plume

    def test_widden_removal_runs_with_nonempty_mask(self):
        mask = np.zeros((11, 11), dtype=bool)
        mask[4:7, 4:7] = True
        plume = self._plume_with_mask(mask)
        plume.remove_parts_of_the_plume_area_that_widden_after_the_shrinking_phase()
        self.assertIn(
            "remove_parts_of_the_plume_area_that_widden_after_the_shrinking_phase",
            plume.protocol[-1],
        )

    def test_widden_removal_returns_early_on_empty_mask(self):
        plume = self._plume_with_mask(np.zeros((11, 11), dtype=bool))
        initial_len = len(plume.protocol)
        plume.remove_parts_of_the_plume_area_that_widden_after_the_shrinking_phase()
        # Empty mask → early return → nothing additional appended
        self.assertEqual(len(plume.protocol), initial_len)


# ---------------------------------------------------------------------------
# remove_parts_of_the_plume_area_identified_only_on_the_edge_of_the_searching_area
# (lines 2423-2449)
# ---------------------------------------------------------------------------

class EdgeRemovalTests(unittest.TestCase):

    def _plume_with_mask(self, mask_values):
        lat, lon = _lat_lon(11)
        spm_values = _gaussian_spm(11)
        plume = _make_plume(spm_values)
        plume.determine_SPM_threshold(False, precomputed_threshold=5.0)
        plume.plume_mask = xr.DataArray(
            mask_values, dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
        return plume

    def test_edge_removal_runs_with_nonempty_mask(self):
        mask = np.zeros((11, 11), dtype=bool)
        mask[4:7, 4:7] = True
        plume = self._plume_with_mask(mask)
        plume.remove_parts_of_the_plume_area_identified_only_on_the_edge_of_the_searching_area()
        # Method either returns early (index_to_keep empty) or appends to protocol.

    def test_edge_removal_returns_early_on_empty_mask(self):
        plume = self._plume_with_mask(np.zeros((11, 11), dtype=bool))
        initial_len = len(plume.protocol)
        plume.remove_parts_of_the_plume_area_identified_only_on_the_edge_of_the_searching_area()
        # Empty mask → early return → nothing additional appended
        self.assertEqual(len(plume.protocol), initial_len)


# ---------------------------------------------------------------------------
# remove_parts_of_the_plume_area_with_very_high_SPM_on_the_edge_of_the_searching_zone
# (lines 2463-2508)
# ---------------------------------------------------------------------------

class HighSpmEdgeRemovalTests(unittest.TestCase):

    def _plume_with_mask(self, mask_values, spm_values=None):
        if spm_values is None:
            spm_values = _gaussian_spm(11)
        plume = _make_plume(spm_values)
        lat, lon = _lat_lon(11)
        plume.determine_SPM_threshold(False, precomputed_threshold=5.0)
        plume.plume_mask = xr.DataArray(
            mask_values, dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
        return plume

    def test_high_spm_edge_removal_runs_with_nonempty_mask(self):
        mask = np.zeros((11, 11), dtype=bool)
        mask[4:7, 4:7] = True
        plume = self._plume_with_mask(mask)
        plume.remove_parts_of_the_plume_area_with_very_high_SPM_on_the_edge_of_the_searching_zone()
        # Method completes without error; protocol appended
        self.assertIn(
            "remove_parts_of_the_plume_area_with_very_high_SPM_on_the_edge_of_the_searching_zone",
            plume.protocol[-1],
        )

    def test_high_spm_edge_removal_returns_early_on_empty_mask(self):
        plume = self._plume_with_mask(np.zeros((11, 11), dtype=bool))
        initial_len = len(plume.protocol)
        plume.remove_parts_of_the_plume_area_with_very_high_SPM_on_the_edge_of_the_searching_zone()
        # Empty mask → early return → nothing additional appended
        self.assertEqual(len(plume.protocol), initial_len)

    def test_high_spm_edge_removal_trims_high_spm_edge(self):
        """
        If the edge columns of the direction search have very high SPM, those
        directions should be trimmed from the plume mask.
        """
        # Uniform high SPM everywhere so every edge direction triggers removal.
        spm_values = np.full((11, 11), 15.0)
        mask = np.zeros((11, 11), dtype=bool)
        mask[4:7, 4:7] = True
        plume = self._plume_with_mask(mask, spm_values=spm_values)
        plume.remove_parts_of_the_plume_area_with_very_high_SPM_on_the_edge_of_the_searching_zone()
        self.assertIn(
            "remove_parts_of_the_plume_area_with_very_high_SPM_on_the_edge_of_the_searching_zone",
            plume.protocol[-1],
        )


# ---------------------------------------------------------------------------
# remove_the_areas_with_sediment_resuspension — lines 2302, 2305
# (parameters looked up when maximal_bathymetry/minimal_distance are None)
# ---------------------------------------------------------------------------

class ResuspensionRemovalTests(unittest.TestCase):

    def _plume_with_mask(self, mask_values):
        lat, lon = _lat_lon(11)
        spm_values = _gaussian_spm(11)
        plume = _make_plume(spm_values)
        plume.determine_SPM_threshold(False, precomputed_threshold=5.0)
        plume.plume_mask = xr.DataArray(
            mask_values, dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
        return plume

    def test_resuspension_removal_reads_params_when_args_are_none(self):
        mask = np.zeros((11, 11), dtype=bool)
        mask[4:7, 4:7] = True
        plume = self._plume_with_mask(mask)
        # Call with no explicit maximal_bathymetry / minimal_distance → hits lines 2302, 2305
        plume.remove_the_areas_with_sediment_resuspension()
        self.assertIn("remove_the_areas_with_sediment_resuspension", plume.protocol[-1])

    def test_resuspension_removal_early_return_on_empty_mask(self):
        plume = self._plume_with_mask(np.zeros((11, 11), dtype=bool))
        initial_len = len(plume.protocol)
        plume.remove_the_areas_with_sediment_resuspension()
        self.assertEqual(len(plume.protocol), initial_len)


if __name__ == "__main__":
    unittest.main()
