"""
Tests for panache.io — covers infer_primary_variable, _extract_date_for_plot,
normalize_map_data, ensure_valid_map_data, and load_map_data.
"""
from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from panache.io import (
    NoValidMapDataError,
    _extract_date_for_plot,
    ensure_valid_map_data,
    infer_primary_variable,
    load_map_data,
    normalize_map_data,
)


# ---------------------------------------------------------------------------
# infer_primary_variable
# ---------------------------------------------------------------------------

class InferPrimaryVariableTests(unittest.TestCase):

    def _ds(self, var_names, dims=("lat", "lon"), dtype=float):
        lat = np.linspace(0.0, 1.0, 5)
        lon = np.linspace(0.0, 1.0, 5)
        data = np.ones((5, 5))
        coords = {"lat": lat, "lon": lon}
        dvars = {name: (list(dims), data.astype(dtype)) for name in var_names}
        return xr.Dataset(dvars, coords=coords)

    def test_explicit_name_found(self):
        ds = self._ds(["myvar"])
        result = infer_primary_variable(ds, variable_name="myvar")
        self.assertEqual(result, "myvar")

    def test_explicit_name_not_found_raises(self):
        ds = self._ds(["myvar"])
        with self.assertRaises(KeyError):
            infer_primary_variable(ds, variable_name="missing")

    def test_candidate_name_matched(self):
        ds = self._ds(["analysed_spm"])
        result = infer_primary_variable(ds)
        self.assertEqual(result, "analysed_spm")

    def test_scored_fallback_picks_lat_lon_variable(self):
        # 'weird_name' has lat+lon dims → score=11; 'flat' has no spatial dims → score=1
        lat = np.linspace(0.0, 1.0, 3)
        lon = np.linspace(0.0, 1.0, 3)
        ds = xr.Dataset(
            {
                "weird_name": (["lat", "lon"], np.ones((3, 3))),
                "flat": (["x"], np.ones(3)),
            },
            coords={"lat": lat, "lon": lon, "x": np.arange(3)},
        )
        result = infer_primary_variable(ds)
        self.assertEqual(result, "weird_name")

    def test_no_suitable_variable_raises(self):
        # Object-dtype variable with no spatial dims → score=0 (not numeric, no lat/lon/time)
        ds = xr.Dataset(
            {"flag": (["x"], np.array(["a", "b", "c"], dtype=object))},
            coords={"x": np.arange(3)},
        )
        with self.assertRaises(ValueError):
            infer_primary_variable(ds)


# ---------------------------------------------------------------------------
# _extract_date_for_plot
# ---------------------------------------------------------------------------

class ExtractDateForPlotTests(unittest.TestCase):

    def _da_with_time(self):
        lat = np.linspace(0.0, 1.0, 3)
        lon = np.linspace(0.0, 1.0, 3)
        data = np.ones((1, 3, 3))
        t = np.datetime64("2021-06-15")
        return xr.DataArray(data, dims=["time", "lat", "lon"],
                            coords={"time": [t], "lat": lat, "lon": lon})

    def test_reads_time_coordinate(self):
        da = self._da_with_time()
        ts = _extract_date_for_plot(da, "file.nc")
        self.assertEqual(ts.year, 2021)
        self.assertEqual(ts.month, 6)
        self.assertEqual(ts.day, 15)

    def test_reads_start_date_attribute(self):
        lat = np.linspace(0.0, 1.0, 3)
        lon = np.linspace(0.0, 1.0, 3)
        da = xr.DataArray(np.ones((3, 3)), dims=["lat", "lon"],
                          coords={"lat": lat, "lon": lon})
        da.attrs["start_date"] = "2020-03-10 UTC"
        da.attrs["start_time"] = "12:00:00 UTC"
        ts = _extract_date_for_plot(da, "file.nc")
        self.assertEqual(ts.year, 2020)
        self.assertEqual(ts.month, 3)

    def test_reads_date_from_filename(self):
        lat = np.linspace(0.0, 1.0, 3)
        lon = np.linspace(0.0, 1.0, 3)
        da = xr.DataArray(np.ones((3, 3)), dims=["lat", "lon"],
                          coords={"lat": lat, "lon": lon})
        ts = _extract_date_for_plot(da, "/data/product_20190815_spm.nc")
        self.assertEqual(ts.year, 2019)
        self.assertEqual(ts.month, 8)
        self.assertEqual(ts.day, 15)

    def test_falls_back_to_mtime_when_no_date_cues(self):
        lat = np.linspace(0.0, 1.0, 3)
        lon = np.linspace(0.0, 1.0, 3)
        da = xr.DataArray(np.ones((3, 3)), dims=["lat", "lon"],
                          coords={"lat": lat, "lon": lon})
        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            ts = _extract_date_for_plot(da, tmp.name)
        self.assertIsInstance(ts, pd.Timestamp)


# ---------------------------------------------------------------------------
# normalize_map_data
# ---------------------------------------------------------------------------

class NormalizeMapDataTests(unittest.TestCase):

    def _da_with_time(self, date="2020-06-01"):
        lat = np.linspace(0.0, 1.0, 4)
        lon = np.linspace(0.0, 1.0, 4)
        data = np.arange(16, dtype=float).reshape(1, 4, 4) - 5  # some negatives
        return xr.DataArray(data, dims=["time", "lat", "lon"],
                            coords={"time": [np.datetime64(date)], "lat": lat, "lon": lon})

    def test_negatives_become_nan(self):
        da = self._da_with_time()
        result = normalize_map_data(da, "dummy.nc")
        self.assertTrue(np.isnan(result.values).any())

    def test_time_dimension_dropped(self):
        da = self._da_with_time()
        result = normalize_map_data(da, "dummy.nc")
        self.assertNotIn("time", result.dims)

    def test_date_for_plot_coordinate_added(self):
        da = self._da_with_time()
        result = normalize_map_data(da, "dummy.nc")
        self.assertIn("date_for_plot", result.coords)

    def test_no_time_dim_passes_through(self):
        lat = np.linspace(0.0, 1.0, 4)
        lon = np.linspace(0.0, 1.0, 4)
        da = xr.DataArray(np.ones((4, 4)), dims=["lat", "lon"],
                          coords={"lat": lat, "lon": lon})
        result = normalize_map_data(da, "product_20210101_spm.nc")
        self.assertEqual(result.shape, (4, 4))


# ---------------------------------------------------------------------------
# ensure_valid_map_data
# ---------------------------------------------------------------------------

class EnsureValidMapDataTests(unittest.TestCase):

    def _da(self, values):
        lat = np.linspace(0.0, 1.0, values.shape[0])
        lon = np.linspace(0.0, 1.0, values.shape[1])
        return xr.DataArray(values, dims=["lat", "lon"],
                            coords={"lat": lat, "lon": lon})

    def test_valid_data_passes_through(self):
        da = self._da(np.ones((4, 4)))
        result = ensure_valid_map_data(da, "file.nc")
        np.testing.assert_array_equal(result.values, da.values)

    def test_all_nan_raises(self):
        da = self._da(np.full((4, 4), np.nan))
        with self.assertRaises(NoValidMapDataError):
            ensure_valid_map_data(da, "file.nc")

    def test_empty_array_raises(self):
        da = xr.DataArray(np.empty((0, 0)), dims=["lat", "lon"],
                          coords={"lat": np.array([]), "lon": np.array([])})
        with self.assertRaises(NoValidMapDataError):
            ensure_valid_map_data(da, "file.nc")


# ---------------------------------------------------------------------------
# load_map_data
# ---------------------------------------------------------------------------

class LoadMapDataTests(unittest.TestCase):

    def _write_nc(self, tmpdir: Path, var_name="SPM", with_time=True) -> Path:
        lat = np.linspace(0.0, 1.0, 5)
        lon = np.linspace(0.0, 1.0, 5)
        data = np.arange(25, dtype=float).reshape(5, 5) + 1.0
        if with_time:
            ds = xr.Dataset(
                {var_name: (["time", "lat", "lon"], data[np.newaxis])},
                coords={"time": [np.datetime64("2020-01-01")], "lat": lat, "lon": lon},
            )
        else:
            ds = xr.Dataset(
                {var_name: (["lat", "lon"], data)},
                coords={"lat": lat, "lon": lon},
            )
        path = tmpdir / "test.nc"
        ds.to_netcdf(path)
        return path

    def test_load_nc_returns_data_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            nc_path = self._write_nc(Path(tmp))
            result = load_map_data(nc_path, lon_range=(0.0, 1.0), lat_range=(0.0, 1.0))
        self.assertIsInstance(result, xr.DataArray)

    def test_load_nc_explicit_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            nc_path = self._write_nc(Path(tmp), var_name="myvar")
            result = load_map_data(nc_path, lon_range=(0.0, 1.0), lat_range=(0.0, 1.0),
                                   variable_name="myvar")
        self.assertIsInstance(result, xr.DataArray)

    def test_load_nc_without_time_dim(self):
        with tempfile.TemporaryDirectory() as tmp:
            nc_path = self._write_nc(Path(tmp), with_time=False)
            result = load_map_data(nc_path, lon_range=(0.0, 1.0), lat_range=(0.0, 1.0))
        self.assertIsInstance(result, xr.DataArray)

    def test_unsupported_extension_raises(self):
        with self.assertRaises(ValueError):
            load_map_data(Path("data.txt"), lon_range=(0.0, 1.0), lat_range=(0.0, 1.0))

    def test_all_nan_nc_raises_no_valid_map_data_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            lat = np.linspace(0.0, 1.0, 5)
            lon = np.linspace(0.0, 1.0, 5)
            data = np.full((1, 5, 5), np.nan)
            ds = xr.Dataset(
                {"SPM": (["time", "lat", "lon"], data)},
                coords={"time": [np.datetime64("2020-01-01")], "lat": lat, "lon": lon},
            )
            path = Path(tmp) / "nan.nc"
            ds.to_netcdf(path)
            with self.assertRaises(NoValidMapDataError):
                load_map_data(path, lon_range=(0.0, 1.0), lat_range=(0.0, 1.0))

    def test_load_pkl_with_map_data_key(self):
        lat = np.linspace(0.0, 1.0, 5)
        lon = np.linspace(0.0, 1.0, 5)
        da = xr.DataArray(
            np.ones((5, 5)) * 3.0,
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
        da = da.assign_coords(date_for_plot="2020-01-01")
        payload = {"map_data": da}
        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = Path(tmp) / "data.pkl"
            with pkl_path.open("wb") as fh:
                pickle.dump(payload, fh)
            result = load_map_data(pkl_path, lon_range=(0.0, 1.0), lat_range=(0.0, 1.0))
        self.assertIsInstance(result, xr.DataArray)

    def test_load_pkl_with_basin_map_key(self):
        lat = np.linspace(0.0, 1.0, 5)
        lon = np.linspace(0.0, 1.0, 5)
        da = xr.DataArray(
            np.ones((5, 5)) * 2.0,
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
        da = da.assign_coords(date_for_plot="2020-01-01")
        payload = {"Basin_map": {"map_data": da}}
        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = Path(tmp) / "data.pkl"
            with pkl_path.open("wb") as fh:
                pickle.dump(payload, fh)
            result = load_map_data(pkl_path, lon_range=(0.0, 1.0), lat_range=(0.0, 1.0))
        self.assertIsInstance(result, xr.DataArray)

    def test_load_pkl_missing_map_data_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = Path(tmp) / "bad.pkl"
            with pkl_path.open("wb") as fh:
                pickle.dump({"other": 1}, fh)
            with self.assertRaises(KeyError):
                load_map_data(pkl_path, lon_range=(0.0, 1.0), lat_range=(0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
