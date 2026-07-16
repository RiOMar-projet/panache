"""
Unit tests for the private helper functions in panache.runner that do not
require a full pipeline run.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from panache.runner import (
    _discover_input_files,
    _glob_root,
    _has_glob_pattern,
    _output_exists,
    _read_manifest,
    _resolve_output_stem,
    _skip_empty_file_message,
    _skip_existing_output_message,
    _write_manifest,
    compute_global_colour_limits,
    compute_global_threshold,
)


# ---------------------------------------------------------------------------
# _has_glob_pattern
# ---------------------------------------------------------------------------

class HasGlobPatternTests(unittest.TestCase):

    def test_plain_path_returns_false(self):
        self.assertFalse(_has_glob_pattern("/data/inputs/file.nc"))

    def test_star_returns_true(self):
        self.assertTrue(_has_glob_pattern("/data/inputs/*.nc"))

    def test_question_mark_returns_true(self):
        self.assertTrue(_has_glob_pattern("/data/inputs/file?.nc"))

    def test_bracket_returns_true(self):
        self.assertTrue(_has_glob_pattern("/data/inputs/[ab].nc"))


# ---------------------------------------------------------------------------
# _glob_root
# ---------------------------------------------------------------------------

class GlobRootTests(unittest.TestCase):

    def test_root_of_starred_path(self):
        root = _glob_root("/data/inputs/*.nc")
        self.assertEqual(root, Path("/data/inputs"))

    def test_deeply_nested_glob(self):
        root = _glob_root("/data/inputs/**/*.nc")
        self.assertEqual(root, Path("/data/inputs"))

    def test_glob_in_first_component(self):
        root = _glob_root("*.nc")
        self.assertEqual(root, Path("."))


# ---------------------------------------------------------------------------
# _discover_input_files
# ---------------------------------------------------------------------------

class DiscoverInputFilesTests(unittest.TestCase):

    def test_glob_pattern_finds_nc_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.nc").write_bytes(b"")
            (d / "b.nc").write_bytes(b"")
            (d / "c.txt").write_bytes(b"")
            files, base = _discover_input_files(str(d / "*.nc"))
            self.assertEqual(len(files), 2)
            self.assertTrue(all(f.suffix == ".nc" for f in files))

    def test_single_file_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "map.nc"
            f.write_bytes(b"")
            files, base = _discover_input_files(str(f))
            self.assertEqual(files, [f])
            self.assertEqual(base, f.parent)

    def test_directory_mode_recurses(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            nested = d / "sub"
            nested.mkdir()
            (nested / "x.nc").write_bytes(b"")
            (d / "y.nc").write_bytes(b"")
            files, base = _discover_input_files(str(d))
            self.assertEqual(len(files), 2)
            self.assertEqual(base, d)

    def test_empty_glob_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, _ = _discover_input_files(str(Path(tmp) / "*.nc"))
            self.assertEqual(files, [])


# ---------------------------------------------------------------------------
# _resolve_output_stem
# ---------------------------------------------------------------------------

class ResolveOutputStemTests(unittest.TestCase):

    def test_preserves_relative_structure(self):
        input_file = Path("/data/inputs/2020/map.nc")
        output_dir = Path("/results/out")
        input_base = Path("/data/inputs")
        stem = _resolve_output_stem(input_file, output_dir, input_base)
        self.assertEqual(stem, Path("/results/out/MAPS/2020/map"))

    def test_strips_extension(self):
        stem = _resolve_output_stem(
            Path("/a/b.nc"), Path("/out"), Path("/a")
        )
        self.assertFalse(stem.name.endswith(".nc"))


# ---------------------------------------------------------------------------
# _output_exists
# ---------------------------------------------------------------------------

class OutputExistsTests(unittest.TestCase):

    def test_missing_png_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "MAPS" / "map"
            self.assertFalse(_output_exists(stem))

    def test_existing_png_returns_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            maps = Path(tmp) / "MAPS"
            maps.mkdir()
            png = maps / "map.png"
            png.write_bytes(b"")
            self.assertTrue(_output_exists(maps / "map"))


# ---------------------------------------------------------------------------
# message helpers
# ---------------------------------------------------------------------------

class MessageHelperTests(unittest.TestCase):

    def test_skip_existing_contains_filename(self):
        msg = _skip_existing_output_message("/data/inputs/2020-01-01.nc")
        self.assertIn("2020-01-01.nc", msg)

    def test_skip_empty_contains_filename(self):
        msg = _skip_empty_file_message("/data/inputs/2020-01-01.nc")
        self.assertIn("2020-01-01.nc", msg)


# ---------------------------------------------------------------------------
# _read_manifest / _write_manifest
# ---------------------------------------------------------------------------

class ManifestTests(unittest.TestCase):

    def test_read_manifest_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _read_manifest(Path(tmp))
            self.assertEqual(result, set())

    def test_write_then_read_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            records = [
                {"input_file": "/a/b.nc", "status": "processed"},
                {"input_file": "/a/c.nc", "status": "skipped_existing"},
            ]
            _write_manifest(d, records)
            result = _read_manifest(d)
            self.assertIn("/a/b.nc", result)
            self.assertIn("/a/c.nc", result)

    def test_read_manifest_returns_set_of_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_manifest(d, [{"input_file": "/x.nc", "status": "processed"}])
            result = _read_manifest(d)
            self.assertIsInstance(result, set)
            self.assertIsInstance(next(iter(result)), str)

    def test_read_malformed_manifest_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "manifest.csv").write_text("not,a,valid,manifest\n")
            result = _read_manifest(d)
            # No 'input_file' column → empty set
            self.assertEqual(result, set())


# ---------------------------------------------------------------------------
# compute_global_colour_limits
# ---------------------------------------------------------------------------

class ComputeGlobalColourLimitsTests(unittest.TestCase):

    def _write_nc(self, tmpdir: Path, values: np.ndarray, date: str) -> Path:
        lat = np.linspace(0.0, 1.0, values.shape[0])
        lon = np.linspace(0.0, 1.0, values.shape[1])
        ds = xr.Dataset(
            {"SPM": (["time", "lat", "lon"], values[np.newaxis])},
            coords={"time": [np.datetime64(date)], "lat": lat, "lon": lon},
        )
        path = tmpdir / f"map_{date}.nc"
        ds.to_netcdf(path)
        return path

    def test_returns_positive_vmin_and_vmax(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            data = np.arange(1, 26, dtype=float).reshape(5, 5)
            path = self._write_nc(d, data, "2020-01-01")
            sample_ds = xr.DataArray(data, dims=["lat", "lon"],
                                     coords={"lat": np.linspace(0, 1, 5),
                                             "lon": np.linspace(0, 1, 5)})
            params = {
                "lat_range_of_plume_area": [0.0, 1.0],
                "lon_range_of_plume_area": [0.0, 1.0],
            }
            vmin, vmax = compute_global_colour_limits(
                [path], params, variable_name="SPM", sample_ds=sample_ds
            )
        self.assertGreater(vmin, 0.0)
        self.assertGreater(vmax, vmin)

    def test_all_invalid_files_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # Write a dataset with all-NaN values
            data = np.full((5, 5), np.nan)
            path = self._write_nc(d, data, "2020-01-01")
            sample_ds = xr.DataArray(np.ones((5, 5)), dims=["lat", "lon"],
                                     coords={"lat": np.linspace(0, 1, 5),
                                             "lon": np.linspace(0, 1, 5)})
            params = {
                "lat_range_of_plume_area": [0.0, 1.0],
                "lon_range_of_plume_area": [0.0, 1.0],
            }
            vmin, vmax = compute_global_colour_limits(
                [path], params, variable_name="SPM", sample_ds=sample_ds
            )
        self.assertAlmostEqual(vmin, 0.1)
        self.assertAlmostEqual(vmax, 1.0)


# ---------------------------------------------------------------------------
# _load_first_valid_map_data — lines 103-106 (skip) and 111 (all-fail raise)
# ---------------------------------------------------------------------------

class LoadFirstValidMapDataTests(unittest.TestCase):

    def _write_nc(self, tmpdir: Path, values: np.ndarray, date: str) -> Path:
        lat = np.linspace(0.0, 1.0, 5)
        lon = np.linspace(0.0, 1.0, 5)
        ds = xr.Dataset(
            {"SPM": (["time", "lat", "lon"], values[np.newaxis])},
            coords={"time": [np.datetime64(date)], "lat": lat, "lon": lon},
        )
        path = tmpdir / f"map_{date}.nc"
        ds.to_netcdf(path)
        return path

    def test_first_all_nan_file_is_skipped_and_second_returned(self):
        from panache.runner import _load_first_valid_map_data
        from panache.io import NoValidMapDataError
        params = {
            "lat_range_of_plume_area": [0.0, 1.0],
            "lon_range_of_plume_area": [0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            bad_path = self._write_nc(d, np.full((5, 5), np.nan), "2020-01-01")
            good_path = self._write_nc(d, np.ones((5, 5)), "2020-01-02")
            ds, valid = _load_first_valid_map_data([bad_path, good_path], params, variable_name="SPM")
        # First file skipped, second returned
        self.assertIsNotNone(ds)
        self.assertNotIn(bad_path, valid)
        self.assertIn(good_path, valid)

    def test_all_nan_files_raises_no_valid_map_data_error(self):
        from panache.runner import _load_first_valid_map_data
        from panache.io import NoValidMapDataError
        params = {
            "lat_range_of_plume_area": [0.0, 1.0],
            "lon_range_of_plume_area": [0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            bad1 = self._write_nc(d, np.full((5, 5), np.nan), "2020-01-01")
            bad2 = self._write_nc(d, np.full((5, 5), np.nan), "2020-01-02")
            with self.assertRaises(NoValidMapDataError):
                _load_first_valid_map_data([bad1, bad2], params, variable_name="SPM")


# ---------------------------------------------------------------------------
# _read_manifest — corrupt file triggers except branch (lines 371-372)
# ---------------------------------------------------------------------------

class ReadManifestCorruptFileTests(unittest.TestCase):

    def test_corrupt_binary_manifest_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            manifest = d / "manifest.csv"
            manifest.write_bytes(b"\x00\x01\x02\x03\xFF\xFE binary junk")
            result = _read_manifest(d)
        self.assertEqual(result, set())


# ---------------------------------------------------------------------------
# compute_global_threshold — lines 174-178, 183
# exception branches: NoValidMapDataError skipped, generic Exception skipped,
# and ValueError raised when all files fail
# ---------------------------------------------------------------------------

class ComputeGlobalThresholdExceptionTests(unittest.TestCase):

    _PARAMS = {
        "lat_range_of_plume_area": [0.0, 1.0],
        "lon_range_of_plume_area": [0.0, 1.0],
    }

    def _sample_ds(self):
        lat = np.linspace(0.0, 1.0, 5)
        lon = np.linspace(0.0, 1.0, 5)
        return xr.DataArray(np.ones((5, 5)), dims=["lat", "lon"],
                            coords={"lat": lat, "lon": lon})

    def test_no_valid_map_data_error_is_skipped_and_second_file_succeeds(self):
        """NoValidMapDataError on first file → continue (lines 174-175); second succeeds."""
        from panache.io import NoValidMapDataError
        from unittest.mock import patch
        import panache.runner as runner_mod

        good = self._sample_ds()
        calls = [0]

        def _side(*a, **kw):
            calls[0] += 1
            if calls[0] == 1:
                raise NoValidMapDataError("all NaN")
            return good

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p1 = d / "a.nc"; p1.touch()
            p2 = d / "b.nc"; p2.touch()
            with patch.object(runner_mod, "load_map_data", side_effect=_side):
                threshold, _ = compute_global_threshold(
                    [p1, p2], self._PARAMS, None, 0.95, self._sample_ds()
                )
        self.assertIsInstance(threshold, float)

    def test_generic_exception_is_skipped_and_second_file_succeeds(self):
        """Generic Exception on first file → warning + continue (lines 176-178)."""
        from unittest.mock import patch
        import panache.runner as runner_mod

        good = self._sample_ds()
        calls = [0]

        def _side(*a, **kw):
            calls[0] += 1
            if calls[0] == 1:
                raise OSError("disk error")
            return good

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p1 = d / "a.nc"; p1.touch()
            p2 = d / "b.nc"; p2.touch()
            with patch.object(runner_mod, "load_map_data", side_effect=_side):
                threshold, _ = compute_global_threshold(
                    [p1, p2], self._PARAMS, None, 0.95, self._sample_ds()
                )
        self.assertIsInstance(threshold, float)

    def test_all_files_failing_raises_value_error(self):
        """All files raise → ValueError (line 183)."""
        from unittest.mock import patch
        import panache.runner as runner_mod

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p1 = d / "a.nc"; p1.touch()
            with patch.object(runner_mod, "load_map_data", side_effect=OSError("bad")):
                with self.assertRaises(ValueError):
                    compute_global_threshold(
                        [p1], self._PARAMS, None, 0.95, self._sample_ds()
                    )


if __name__ == "__main__":
    unittest.main()
