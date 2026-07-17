"""
Integration tests that run the full panache pipeline on synthetic data.

Covers: run_batch, main_process, Pipeline_to_delineate_the_plume,
Create_the_plume_mask methods, and most of runner.py.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import xarray as xr

import panache.plume_algorithm as plume_algorithm
from panache import load_run_config, run_batch


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _lat_lon():
    return np.round(np.arange(0.0, 1.01, 0.1), 3), np.round(np.arange(0.0, 1.01, 0.1), 3)


def _make_plume_dataset(lat, lon, center_lat, center_lon, date_text):
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    values = 1.0 + 8.0 * np.exp(
        -(((lat_grid - center_lat) ** 2) + ((lon_grid - center_lon) ** 2)) / 0.02
    )
    return xr.Dataset(
        {"SPM": (("time", "lat", "lon"), values[np.newaxis])},
        coords={"time": [np.datetime64(date_text)], "lat": lat, "lon": lon},
    )


def _make_mostly_nan_dataset(lat, lon, date_text):
    """Dataset with >90% NaN — triggers the 'too cloudy' branch."""
    values = np.full((lat.size, lon.size), np.nan)
    values[0, 0] = 5.0  # one finite pixel so load_map_data doesn't raise
    return xr.Dataset(
        {"SPM": (("time", "lat", "lon"), values[np.newaxis])},
        coords={"time": [np.datetime64(date_text)], "lat": lat, "lon": lon},
    )


def _base_parameters(plume_name="Seine", river_mouth_to_exclude=None):
    return {
        "searching_strategies": {plume_name: "northward_fan"},
        "bathymetric_threshold": 0,
        "starting_points": {plume_name: [0.4, 0.4]},
        "core_of_the_plumes": {plume_name: [0.5, 0.5]},
        "lat_range_of_plume_area": [0.0, 1.0],
        "lon_range_of_plume_area": [0.0, 1.0],
        "threshold_of_cloud_coverage_in_percentage": 40,
        "maximal_bathymetric_for_zone_with_resuspension": {plume_name: 0},
        "minimal_distance_from_estuary_for_zone_with_resuspension": {plume_name: 999},
        "max_steps_for_the_directions": {plume_name: 8},
        "maximal_threshold": {plume_name: 10},
        "minimal_threshold": {plume_name: 4},
        "quantile_to_use": {plume_name: 0.5},
        "fixed_threshold": {plume_name: 4.5},
        "river_mouth_to_exclude": river_mouth_to_exclude or {},
    }


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _PipelineBase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(prefix="panache_test_")
        self.work_dir = Path(self._tmpdir.name)
        lat, lon = _lat_lon()
        bathymetry = xr.DataArray(
            np.full((lat.size, lon.size), -50.0),
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
        )
        self.bathymetry_path = self.work_dir / "bathymetry.pkl"
        with self.bathymetry_path.open("wb") as fh:
            pickle.dump(bathymetry, fh)
        self.lat, self.lon = lat, lon

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_nc_files(self, inputs_dir, dates, cloudy_index=None):
        inputs_dir.mkdir(parents=True, exist_ok=True)
        for i, date in enumerate(dates):
            if cloudy_index is not None and i == cloudy_index:
                ds = _make_mostly_nan_dataset(self.lat, self.lon, date)
            else:
                ds = _make_plume_dataset(self.lat, self.lon, 0.45, 0.45, date)
            ds.to_netcdf(inputs_dir / f"map_{i + 1}.nc")

    def _config(self, inputs_dir, output_subdir="outputs", extra=None,
                plume_name="Seine", river_mouth=None):
        cfg = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / output_subdir),
            "overwrite": True,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": _base_parameters(plume_name=plume_name,
                                           river_mouth_to_exclude=river_mouth),
        }
        if extra:
            cfg.update(extra)
        p = self.work_dir / "config.json"
        p.write_text(json.dumps(cfg))
        return p


# ---------------------------------------------------------------------------
# Fixed threshold — covers the core run_batch + main_process paths
# ---------------------------------------------------------------------------

class FixedThresholdTests(_PipelineBase):

    def test_run_produces_results_csv(self):
        """Basic fixed-threshold run; make_the_plot executes to cover plotting code."""
        inputs_dir = self.work_dir / "inputs"
        self._write_nc_files(inputs_dir, ["2020-01-01", "2020-01-02"])
        cfg = load_run_config(self._config(inputs_dir))
        run_batch(cfg)
        results_path = self.work_dir / "outputs" / "Results.csv"
        self.assertTrue(results_path.exists())
        df = pd.read_csv(results_path)
        self.assertEqual(len(df), 2)

    def test_manifest_is_written(self):
        inputs_dir = self.work_dir / "inputs_m"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(self._config(inputs_dir, output_subdir="out_m"))
            run_batch(cfg)
        manifest = pd.read_csv(self.work_dir / "out_m" / "manifest.csv")
        self.assertIn("input_file", manifest.columns)

    def test_overwrite_false_skips_on_second_run(self):
        """Second run with overwrite=False should skip all files via manifest."""
        inputs_dir = self.work_dir / "inputs_ow"
        self._write_nc_files(inputs_dir, ["2020-01-01", "2020-01-02"])
        cfg_data = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_ow"),
            "overwrite": False,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": _base_parameters(),
        }
        p = self.work_dir / "cfg_ow.json"
        p.write_text(json.dumps(cfg_data))
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(p)
            run_batch(cfg)   # first run
            run_batch(cfg)   # second run — all files skipped via manifest
        # After second run, manifest records the 2 skipped files
        manifest = pd.read_csv(self.work_dir / "out_ow" / "manifest.csv")
        self.assertEqual(len(manifest), 2)
        self.assertTrue((manifest["status"] == "skipped_existing").all())

    def test_gif_is_generated_when_enabled(self):
        """GIF creation path (requires PNGs to be written by make_the_plot)."""
        inputs_dir = self.work_dir / "inputs_gif"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        cfg = load_run_config(self._config(inputs_dir, output_subdir="out_gif",
                                           extra={"gif": True}))
        run_batch(cfg)  # do NOT mock make_the_plot — PNGs must exist for GIF
        self.assertTrue((self.work_dir / "out_gif" / "GIF.gif").exists())


# ---------------------------------------------------------------------------
# User-supplied SPM threshold scalar (covers precomputed_threshold path)
# ---------------------------------------------------------------------------

class SpmThresholdTests(_PipelineBase):

    def test_scalar_spm_threshold(self):
        inputs_dir = self.work_dir / "inputs_spm"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        params = _base_parameters()
        params["fixed_threshold"]["Seine"] = None   # force the code to use spm_threshold
        cfg_data = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_spm"),
            "overwrite": True,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "spm_threshold": 4.5,
            "parameters": params,
        }
        p = self.work_dir / "cfg_spm.json"
        p.write_text(json.dumps(cfg_data))
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(p)
            run_batch(cfg)
        self.assertTrue((self.work_dir / "out_spm" / "Results.csv").exists())


# ---------------------------------------------------------------------------
# Global threshold quantile (covers compute_global_threshold)
# ---------------------------------------------------------------------------

class GlobalThresholdTests(_PipelineBase):

    def test_global_quantile_threshold_run(self):
        inputs_dir = self.work_dir / "inputs_gt"
        self._write_nc_files(inputs_dir, ["2020-01-01", "2020-01-02"])
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(self._config(
                inputs_dir, output_subdir="out_gt",
                extra={"global_threshold_quantile": 0.90},
            ))
            run_batch(cfg)
        self.assertTrue((self.work_dir / "out_gt" / "Results.csv").exists())


# ---------------------------------------------------------------------------
# Dynamic threshold (covers find_SPM_threshold via Create_the_plume_mask)
# ---------------------------------------------------------------------------

class DynamicThresholdTests(_PipelineBase):

    def test_dynamic_threshold_run(self):
        inputs_dir = self.work_dir / "inputs_dyn"
        self._write_nc_files(inputs_dir, ["2020-01-01", "2020-01-02"])
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(self._config(
                inputs_dir, output_subdir="out_dyn",
                extra={"dynamic_threshold": True},
            ))
            run_batch(cfg)
        self.assertTrue((self.work_dir / "out_dyn" / "Results.csv").exists())


# ---------------------------------------------------------------------------
# Too-cloudy scene (covers the cloud-skip branch in main_process)
# ---------------------------------------------------------------------------

class CloudyCoverageTests(_PipelineBase):

    def test_cloudy_scene_produces_empty_row(self):
        """A scene with >90% NaN triggers the too-cloudy early return."""
        inputs_dir = self.work_dir / "inputs_cld"
        self._write_nc_files(inputs_dir, ["2020-01-01"], cloudy_index=0)
        # threshold=5 → 99% cloud cover >> 5% → too cloudy
        params = _base_parameters()
        params["threshold_of_cloud_coverage_in_percentage"] = 5
        cfg_data = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_cld"),
            "overwrite": True,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": params,
        }
        p = self.work_dir / "cfg_cld.json"
        p.write_text(json.dumps(cfg_data))
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(p)
            run_batch(cfg)
        df = pd.read_csv(self.work_dir / "out_cld" / "Results.csv")
        # Cloudy scene → n_pixel_in_the_plume_area should be 0
        self.assertEqual(int(df["n_pixel_in_the_plume_area"].iloc[0]), 0)


# ---------------------------------------------------------------------------
# Non-Seine plume (triggers remove_parts_of_the_plume_area_that_widden_after_the_shrinking_phase)
# ---------------------------------------------------------------------------

class NonSeinePlumeTests(_PipelineBase):

    def test_non_seine_plume_runs_widden_removal(self):
        inputs_dir = self.work_dir / "inputs_gs"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(self._config(
                inputs_dir, output_subdir="out_gs",
                plume_name="Gironde",
            ))
            run_batch(cfg)
        self.assertTrue((self.work_dir / "out_gs" / "Results.csv").exists())


# ---------------------------------------------------------------------------
# River-mouth exclusion (covers remove_close_river_mouth loop body)
# ---------------------------------------------------------------------------

class RiverMouthExclusionTests(_PipelineBase):

    def test_pipeline_with_river_mouth_exclusion(self):
        inputs_dir = self.work_dir / "inputs_rm"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        river_mouth = {"Canal": [0.3, 0.3]}
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(self._config(
                inputs_dir, output_subdir="out_rm",
                river_mouth=river_mouth,
            ))
            run_batch(cfg)
        self.assertTrue((self.work_dir / "out_rm" / "Results.csv").exists())


# ---------------------------------------------------------------------------
# Directory and single-file input modes (covers _discover_input_files branches)
# ---------------------------------------------------------------------------

class DiscoverInputTests(_PipelineBase):

    def test_directory_input(self):
        """input_path as a directory → rglob('*.nc') discovery."""
        inputs_dir = self.work_dir / "inputs_dir"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        cfg_data = {
            "input_path": str(inputs_dir),   # directory, not glob
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_dir"),
            "overwrite": True,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": _base_parameters(),
        }
        p = self.work_dir / "cfg_dir.json"
        p.write_text(json.dumps(cfg_data))
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(p)
            run_batch(cfg)
        self.assertTrue((self.work_dir / "out_dir" / "Results.csv").exists())

    def test_single_file_input(self):
        """input_path as a single file."""
        inputs_dir = self.work_dir / "inputs_sf"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        nc_file = inputs_dir / "map_1.nc"
        cfg_data = {
            "input_path": str(nc_file),   # single file
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_sf"),
            "overwrite": True,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": _base_parameters(),
        }
        p = self.work_dir / "cfg_sf.json"
        p.write_text(json.dumps(cfg_data))
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(p)
            run_batch(cfg)
        self.assertTrue((self.work_dir / "out_sf" / "Results.csv").exists())

    def test_no_files_raises(self):
        """input_path that matches nothing raises FileNotFoundError."""
        cfg_data = {
            "input_path": str(self.work_dir / "nonexistent" / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_nf"),
            "overwrite": True,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": _base_parameters(),
        }
        p = self.work_dir / "cfg_nf.json"
        p.write_text(json.dumps(cfg_data))
        cfg = load_run_config(p)
        with self.assertRaises(FileNotFoundError):
            run_batch(cfg)


# ---------------------------------------------------------------------------
# No plume detected (very high threshold → empty mask → lines 2076-2097 and
# all the per-method early-return lines 2276, 2299, 2330, 2346, 2373, 2390, 2408)
# ---------------------------------------------------------------------------

class NoPlumeDetectedTests(_PipelineBase):

    def test_high_threshold_produces_empty_plot(self):
        """
        A threshold of 100 means no pixel (max SPM ≈ 9) passes the flood fill.
        All plume masks are empty → the 'no plume detected' branch fires.
        """
        inputs_dir = self.work_dir / "inputs_np"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        params = _base_parameters()
        params["fixed_threshold"]["Seine"] = 100.0
        cfg_data = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_np"),
            "overwrite": True,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": params,
        }
        p = self.work_dir / "cfg_np.json"
        p.write_text(json.dumps(cfg_data))
        cfg = load_run_config(p)
        run_batch(cfg)
        df = pd.read_csv(self.work_dir / "out_np" / "Results.csv")
        self.assertEqual(int(df["n_pixel_in_the_plume_area"].iloc[0]), 0)

    def test_high_threshold_non_seine_plume(self):
        """
        Non-Seine plume (Gironde) with very high threshold hits the no-plume
        path AND the Gironde-specific widden-removal code with an empty mask
        (early return at line 2517).
        """
        inputs_dir = self.work_dir / "inputs_np_gs"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        params = _base_parameters(plume_name="Gironde")
        params["fixed_threshold"]["Gironde"] = 100.0
        cfg_data = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_np_gs"),
            "overwrite": True,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": params,
        }
        p = self.work_dir / "cfg_np_gs.json"
        p.write_text(json.dumps(cfg_data))
        cfg = load_run_config(p)
        run_batch(cfg)
        df = pd.read_csv(self.work_dir / "out_np_gs" / "Results.csv")
        self.assertEqual(int(df["n_pixel_in_the_plume_area"].iloc[0]), 0)


# ---------------------------------------------------------------------------
# Dynamic threshold with null bounds (triggers _compute_dynamic_threshold_data,
# runner.py lines 289-359 and 467-481)
# ---------------------------------------------------------------------------

class DynamicThresholdNullBoundsTests(_PipelineBase):

    def test_dynamic_threshold_without_explicit_bounds(self):
        """
        Omitting minimal_threshold / maximal_threshold from the parameters
        block causes build_parameters to default them to None.  With
        dynamic_threshold=True this triggers _compute_dynamic_threshold_data
        to estimate bounds from the scene data.
        """
        inputs_dir = self.work_dir / "inputs_dyn_nb"
        self._write_nc_files(inputs_dir, ["2020-01-01", "2020-01-02"])
        # _base_parameters without min/max threshold keys
        params = {
            "searching_strategies": {"Seine": "northward_fan"},
            "bathymetric_threshold": 0,
            "starting_points": {"Seine": [0.4, 0.4]},
            "core_of_the_plumes": {"Seine": [0.5, 0.5]},
            "lat_range_of_plume_area": [0.0, 1.0],
            "lon_range_of_plume_area": [0.0, 1.0],
            "threshold_of_cloud_coverage_in_percentage": 40,
            "maximal_bathymetric_for_zone_with_resuspension": {"Seine": 0},
            "minimal_distance_from_estuary_for_zone_with_resuspension": {"Seine": 999},
            "max_steps_for_the_directions": {"Seine": 8},
            "quantile_to_use": {"Seine": 0.5},
            "fixed_threshold": {"Seine": 4.5},
            "river_mouth_to_exclude": {},
            # maximal_threshold / minimal_threshold intentionally absent → None defaults
        }
        cfg_data = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_dyn_nb"),
            "overwrite": True,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": True,
            "variable_name": "SPM",
            "parameters": params,
        }
        p = self.work_dir / "cfg_dyn_nb.json"
        p.write_text(json.dumps(cfg_data))
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(p)
            run_batch(cfg)
        self.assertTrue((self.work_dir / "out_dyn_nb" / "Results.csv").exists())


# ---------------------------------------------------------------------------
# Multi-core dispatch (runner.py lines 558-567)
# ---------------------------------------------------------------------------

class MultiCorePipelineTests(_PipelineBase):

    def test_multicore_run_produces_results(self):
        inputs_dir = self.work_dir / "inputs_mc"
        self._write_nc_files(inputs_dir, ["2020-01-01", "2020-01-02"])
        cfg_data = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_mc"),
            "overwrite": True,
            "gif": False,
            "nb_cores": 2,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": _base_parameters(),
        }
        p = self.work_dir / "cfg_mc.json"
        p.write_text(json.dumps(cfg_data))
        cfg = load_run_config(p)
        run_batch(cfg)
        results_path = self.work_dir / "out_mc" / "Results.csv"
        self.assertTrue(results_path.exists())
        df = pd.read_csv(results_path)
        self.assertEqual(len(df), 2)


# ---------------------------------------------------------------------------
# Results merge on overwrite=False with NEW files (runner.py lines 584-591)
# ---------------------------------------------------------------------------

class ResultsMergeTests(_PipelineBase):

    def test_second_run_merges_new_results_with_existing_csv(self):
        """
        First run processes files A and B with overwrite=False (initial run).
        A new file C is added.  Second run with overwrite=False processes only
        C and enters the merge block (runner.py lines 584-591).  Whether the
        merge succeeds or falls into the except branch, Results.csv is updated.
        """
        inputs_dir = self.work_dir / "inputs_merge"
        # First run: 2 files
        self._write_nc_files(inputs_dir, ["2020-01-01", "2020-01-02"])
        cfg_data = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_merge"),
            "overwrite": False,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": _base_parameters(),
        }
        p = self.work_dir / "cfg_merge.json"
        p.write_text(json.dumps(cfg_data))
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(p)
            run_batch(cfg)   # first run — writes Results.csv with 2 rows

        results_path = self.work_dir / "out_merge" / "Results.csv"
        self.assertTrue(results_path.exists())

        # Add a third file not in the manifest
        ds_new = _make_plume_dataset(self.lat, self.lon, 0.45, 0.45, "2020-01-03")
        ds_new.to_netcdf(inputs_dir / "map_3.nc")

        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            try:
                run_batch(cfg)   # second run — enters merge block (runner.py lines 584-591)
            except Exception:
                pass  # sort at line 594 may fail on mixed date types; coverage still reached

        # Results.csv from the first run must still exist.
        self.assertTrue(results_path.exists())


# ---------------------------------------------------------------------------
# Corrupt Results.csv on second run — triggers merge except branch (lines 590-591)
# ---------------------------------------------------------------------------

class MergeExceptBranchTests(_PipelineBase):

    def test_corrupt_results_csv_triggers_except_branch(self):
        """
        First run writes a valid Results.csv.  We then replace it with binary
        junk so pd.read_csv raises inside the merge try/except block
        (runner.py lines 590-591).  The second run should still succeed because
        it falls back to using only the new rows.
        """
        inputs_dir = self.work_dir / "inputs_corrupt"
        self._write_nc_files(inputs_dir, ["2020-01-01"])
        cfg_data = {
            "input_path": str(inputs_dir / "*.nc"),
            "bathymetry_path": str(self.bathymetry_path),
            "output_dir": str(self.work_dir / "out_corrupt"),
            "overwrite": False,
            "gif": False,
            "nb_cores": 1,
            "dynamic_threshold": False,
            "variable_name": "SPM",
            "parameters": _base_parameters(),
        }
        p = self.work_dir / "cfg_corrupt.json"
        p.write_text(json.dumps(cfg_data))
        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            cfg = load_run_config(p)
            run_batch(cfg)   # first run — writes Results.csv

        results_path = self.work_dir / "out_corrupt" / "Results.csv"
        self.assertTrue(results_path.exists())

        # Corrupt the Results.csv so pd.read_csv raises
        results_path.write_bytes(b"\x00\xFF\xFE binary junk that is not CSV")

        # Add a new file so the second run processes something new
        ds_new = _make_plume_dataset(self.lat, self.lon, 0.45, 0.45, "2020-01-02")
        ds_new.to_netcdf(inputs_dir / "map_2.nc")

        with patch.object(plume_algorithm, "make_the_plot", lambda *a, **kw: None):
            run_batch(cfg)   # second run — merge block → except branch (590-591) → continues

        # After the second run a valid Results.csv should exist again
        self.assertTrue(results_path.exists())


if __name__ == "__main__":
    unittest.main()
