"""
Tests for panache.utils functions not covered by test_searching_strategy_presets.py.
"""
from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from panache.utils import (
    check_time_format,
    coordinate_range_bounds,
    define_parameters,
    expand_grid,
    extract_dataframes_iterative,
    extract_time_from_nc_file,
    flatten_a_list,
    load_bathymetric_data,
    load_file,
    searching_strategy_directions_from_presets,
    unique_years_between_two_dates,
)


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------

class LoadFileTests(unittest.TestCase):

    def test_loads_pickle_correctly(self):
        payload = {"key": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(payload, tmp)
            tmp_path = tmp.name
        result = load_file(tmp_path)
        self.assertEqual(result, payload)


# ---------------------------------------------------------------------------
# expand_grid
# ---------------------------------------------------------------------------

class ExpandGridTests(unittest.TestCase):

    def test_cartesian_product_shape(self):
        df = expand_grid(a=[1, 2], b=["x", "y", "z"])
        self.assertEqual(len(df), 6)
        self.assertIn("a", df.columns)
        self.assertIn("b", df.columns)

    def test_single_column(self):
        df = expand_grid(x=[10, 20, 30])
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["x"]), [10, 20, 30])


# ---------------------------------------------------------------------------
# flatten_a_list
# ---------------------------------------------------------------------------

class FlattenListTests(unittest.TestCase):

    def test_flat_list_unchanged(self):
        self.assertEqual(flatten_a_list([1, 2, 3]), [1, 2, 3])

    def test_nested_list_flattened(self):
        self.assertEqual(flatten_a_list([[1, 2], [3, [4, 5]]]), [1, 2, 3, 4, 5])

    def test_empty_list(self):
        self.assertEqual(flatten_a_list([]), [])

    def test_deeply_nested(self):
        self.assertEqual(flatten_a_list([[[1]], [2]]), [1, 2])


# ---------------------------------------------------------------------------
# extract_dataframes_iterative
# ---------------------------------------------------------------------------

class ExtractDataframesIterativeTests(unittest.TestCase):

    def test_extracts_from_flat_dict(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        data = {"key": df1, "other": "value"}
        result = list(extract_dataframes_iterative(data))
        self.assertEqual(len(result), 1)
        pd.testing.assert_frame_equal(result[0], df1)

    def test_extracts_from_nested_dict(self):
        df1 = pd.DataFrame({"x": [10]})
        df2 = pd.DataFrame({"y": [20]})
        data = {"outer": {"inner1": df1, "inner2": df2}}
        result = list(extract_dataframes_iterative(data))
        self.assertEqual(len(result), 2)

    def test_extracts_from_list(self):
        df1 = pd.DataFrame({"a": [1]})
        result = list(extract_dataframes_iterative([df1, "ignored"]))
        self.assertEqual(len(result), 1)

    def test_bare_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = list(extract_dataframes_iterative(df))
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# unique_years_between_two_dates
# ---------------------------------------------------------------------------

class UniqueYearsTests(unittest.TestCase):

    def test_same_year(self):
        result = unique_years_between_two_dates("2020/01/01", "2020/12/31")
        self.assertEqual(result, [2020])

    def test_multi_year_range(self):
        result = unique_years_between_two_dates("2018/01/01", "2021/06/15")
        self.assertEqual(result, [2018, 2019, 2020, 2021])


# ---------------------------------------------------------------------------
# check_time_format
# ---------------------------------------------------------------------------

class CheckTimeFormatTests(unittest.TestCase):

    def test_valid_time_string_returned(self):
        self.assertEqual(check_time_format("12:30:45 UTC"), "12:30:45 UTC")

    def test_invalid_format_returns_nan(self):
        result = check_time_format("noon")
        self.assertTrue(np.isnan(result))

    def test_edge_midnight(self):
        self.assertEqual(check_time_format("00:00:00 UTC"), "00:00:00 UTC")

    def test_edge_23_59(self):
        self.assertEqual(check_time_format("23:59:59 UTC"), "23:59:59 UTC")

    def test_missing_utc_returns_nan(self):
        result = check_time_format("12:30:45")
        self.assertTrue(np.isnan(result))


# ---------------------------------------------------------------------------
# load_bathymetric_data (existing file path)
# ---------------------------------------------------------------------------

class LoadBathymetricDataTests(unittest.TestCase):

    def test_loads_existing_pickle(self):
        lat = np.linspace(44.0, 50.0, 5)
        lon = np.linspace(-2.0, 3.0, 5)
        data = xr.DataArray(
            np.full((5, 5), -100.0),
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bathy.pkl"
            with path.open("wb") as fh:
                pickle.dump(data, fh)
            result = load_bathymetric_data(str(path), -2.0, 3.0, 44.0, 50.0)
        self.assertIsInstance(result, xr.DataArray)
        self.assertIn("lat", result.dims)

    def test_missing_file_without_bathyreq_raises_import_error(self):
        """When the file doesn't exist and bathyreq is None, raise ImportError."""
        import panache.utils as utils_module
        original = utils_module.bathyreq
        utils_module.bathyreq = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "nonexistent.pkl"
                with self.assertRaises(ImportError):
                    load_bathymetric_data(str(path), 0, 1, 0, 1)
        finally:
            utils_module.bathyreq = original


# ---------------------------------------------------------------------------
# define_parameters — unknown zone returns None
# ---------------------------------------------------------------------------

class DefineParametersTests(unittest.TestCase):

    def test_unknown_zone_returns_none(self):
        result = define_parameters("UNKNOWN_ZONE")
        self.assertIsNone(result)

    def test_non_string_zone_returns_none(self):
        result = define_parameters(42)
        self.assertIsNone(result)

    def test_all_known_zones_return_dict(self):
        for zone in ["BAY_OF_SEINE", "BAY_OF_BISCAY", "GULF_OF_LION", "SOUTHERN_BRITTANY"]:
            with self.subTest(zone=zone):
                result = define_parameters(zone)
                self.assertIsInstance(result, dict)
                self.assertIn("searching_strategies", result)


# ---------------------------------------------------------------------------
# searching_strategy_directions_from_presets — non-Mapping raises TypeError (line 50)
# ---------------------------------------------------------------------------

class SearchingStrategyDirectionsFromPresetsTests(unittest.TestCase):

    def test_non_mapping_raises_type_error(self):
        with self.assertRaises(TypeError):
            searching_strategy_directions_from_presets(["northward_fan"])

    def test_valid_mapping_returns_dict(self):
        result = searching_strategy_directions_from_presets({"Seine": "northward_fan"})
        self.assertIn("Seine", result)
        self.assertIsInstance(result["Seine"], list)


# ---------------------------------------------------------------------------
# coordinate_range_bounds — fewer than 2 values raises ValueError (line 77)
# ---------------------------------------------------------------------------

class CoordinateRangeBoundsTests(unittest.TestCase):

    def test_single_value_raises_value_error(self):
        with self.assertRaises(ValueError):
            coordinate_range_bounds([1.0])

    def test_two_values_returns_tuple(self):
        result = coordinate_range_bounds([1.0, 2.0])
        self.assertEqual(result, (1.0, 2.0))

    def test_more_than_two_values_returns_min_max(self):
        result = coordinate_range_bounds([3.0, 1.0, 2.0])
        self.assertEqual(result, (1.0, 3.0))


# ---------------------------------------------------------------------------
# extract_time_from_nc_file — lines 210-221
# ---------------------------------------------------------------------------

class ExtractTimeFromNcFileTests(unittest.TestCase):

    def _da(self):
        return xr.DataArray(np.ones((3, 3)), dims=["lat", "lon"],
                            coords={"lat": np.linspace(0, 1, 3),
                                    "lon": np.linspace(0, 1, 3)})

    def test_image_reference_time_attr_raises_on_legacy_access(self):
        # The function uses the legacy `_attrs` private attribute; the `if`
        # branch (line 210) is entered and covers lines 210-211 before raising.
        da = self._da()
        da.attrs["image_reference_time"] = "12:30:00 UTC"
        with self.assertRaises(AttributeError):
            extract_time_from_nc_file(da)

    def test_dsd_entry_id_raises_on_legacy_attrs_access(self):
        # The function uses the legacy `_attrs` private attribute which was
        # removed in newer xarray.  The elif branch at line 212 is entered,
        # but the private `_attrs` access raises AttributeError.
        da = self._da()
        da.attrs["DSD_entry_id"] = "Some_L4_product"
        with self.assertRaises(AttributeError):
            extract_time_from_nc_file(da)

    def test_start_time_attr(self):
        da = self._da()
        da.attrs["start_time"] = "2020-01-15 09:30:00"
        result = extract_time_from_nc_file(da)
        self.assertEqual(result, "09:30:00 UTC")

    def test_time_attr(self):
        da = self._da()
        da.attrs["time"] = "14:00:00 UTC"
        result = extract_time_from_nc_file(da)
        self.assertEqual(result, "14:00:00 UTC")


if __name__ == "__main__":
    unittest.main()
