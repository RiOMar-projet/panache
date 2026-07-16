"""
Additional tests for panache.config — covers branches not exercised by the
existing test_searching_strategy_presets.py.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from panache.config import (
    RunConfig,
    _required_bool,
    build_parameters,
    load_run_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_raw_params(plume_name="Test"):
    return {
        "lon_new_resolution": 0.015,
        "lat_new_resolution": 0.015,
        "searching_strategies": {plume_name: "northward_fan"},
        "bathymetric_threshold": 0,
        "starting_points": {plume_name: [1.0, 2.0]},
        "core_of_the_plumes": {plume_name: [1.0, 2.0]},
        "lat_range_of_plume_area": [0.0, 1.0],
        "lon_range_of_plume_area": [0.0, 1.0],
        "threshold_of_cloud_coverage_in_percentage": 25,
        "maximal_bathymetric_for_zone_with_resuspension": {plume_name: 30},
        "minimal_distance_from_estuary_for_zone_with_resuspension": {plume_name: 30},
        "max_steps_for_the_directions": {plume_name: 10},
    }


# ---------------------------------------------------------------------------
# _required_bool
# ---------------------------------------------------------------------------

class RequiredBoolTests(unittest.TestCase):

    def test_true_value_passes(self):
        self.assertTrue(_required_bool({"key": True}, "key"))

    def test_false_value_passes(self):
        self.assertFalse(_required_bool({"key": False}, "key"))

    def test_non_bool_raises_type_error(self):
        with self.assertRaises(TypeError):
            _required_bool({"key": 1}, "key")

    def test_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            _required_bool({"key": "true"}, "key")


# ---------------------------------------------------------------------------
# build_parameters — optional default injection
# ---------------------------------------------------------------------------

class BuildParametersDefaultsTests(unittest.TestCase):

    def test_missing_maximal_threshold_defaults_to_none(self):
        raw = _minimal_raw_params()
        params = build_parameters(raw)
        self.assertIn("maximal_threshold", params)
        self.assertIsNone(params["maximal_threshold"]["Test"])

    def test_missing_minimal_threshold_defaults_to_none(self):
        raw = _minimal_raw_params()
        params = build_parameters(raw)
        self.assertIsNone(params["minimal_threshold"]["Test"])

    def test_missing_quantile_defaults_to_0_2(self):
        raw = _minimal_raw_params()
        params = build_parameters(raw)
        self.assertAlmostEqual(params["quantile_to_use"]["Test"], 0.2)

    def test_missing_fixed_threshold_defaults_to_none(self):
        raw = _minimal_raw_params()
        params = build_parameters(raw)
        self.assertIsNone(params["fixed_threshold"]["Test"])

    def test_missing_river_mouth_to_exclude_defaults_to_empty(self):
        raw = _minimal_raw_params()
        params = build_parameters(raw)
        self.assertEqual(params["river_mouth_to_exclude"], {})

    def test_river_mouth_to_exclude_normalised_to_tuples(self):
        raw = _minimal_raw_params()
        raw["river_mouth_to_exclude"] = {"Canal": [1.0, 2.0]}
        params = build_parameters(raw)
        self.assertIsInstance(params["river_mouth_to_exclude"]["Canal"], tuple)


class BuildParametersMissingKeyTests(unittest.TestCase):

    def test_missing_required_key_raises_value_error(self):
        raw = _minimal_raw_params()
        del raw["searching_strategies"]
        with self.assertRaises(ValueError):
            build_parameters(raw)

    def test_all_required_keys_missing_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_parameters({})


# ---------------------------------------------------------------------------
# load_run_config
# ---------------------------------------------------------------------------

class LoadRunConfigTests(unittest.TestCase):

    def _write_config(self, data: dict, tmpdir: Path) -> Path:
        p = tmpdir / "config.json"
        p.write_text(json.dumps(data))
        return p

    def _minimal_config_data(self, tmpdir: Path) -> dict:
        bathy = tmpdir / "bathy.pkl"
        bathy.write_bytes(b"")   # existence matters, not content for these tests
        return {
            "input_path": str(tmpdir / "*.nc"),
            "bathymetry_path": str(bathy),
            "output_dir": str(tmpdir / "out"),
            "overwrite": True,
            "gif": False,
            "zone": "BAY_OF_SEINE",
        }

    def test_zone_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data = self._minimal_config_data(tmpdir)
            p = self._write_config(data, tmpdir)
            cfg = load_run_config(p)
            self.assertIsInstance(cfg, RunConfig)
            self.assertEqual(cfg.zone, "BAY_OF_SEINE")

    def test_parameters_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bathy = tmpdir / "bathy.pkl"
            bathy.write_bytes(b"")
            data = {
                "input_path": str(tmpdir / "*.nc"),
                "bathymetry_path": str(bathy),
                "output_dir": str(tmpdir / "out"),
                "overwrite": False,
                "gif": False,
                "parameters": _minimal_raw_params(),
            }
            p = self._write_config(data, tmpdir)
            cfg = load_run_config(p)
            self.assertIsNotNone(cfg.parameters)

    def test_both_zone_and_parameters_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bathy = tmpdir / "bathy.pkl"
            bathy.write_bytes(b"")
            data = {
                "input_path": str(tmpdir / "*.nc"),
                "bathymetry_path": str(bathy),
                "output_dir": str(tmpdir / "out"),
                "overwrite": True,
                "gif": False,
                "zone": "BAY_OF_SEINE",
                "parameters": _minimal_raw_params(),
            }
            p = self._write_config(data, tmpdir)
            with self.assertRaises(ValueError):
                load_run_config(p)

    def test_neither_zone_nor_parameters_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bathy = tmpdir / "bathy.pkl"
            bathy.write_bytes(b"")
            data = {
                "input_path": str(tmpdir / "*.nc"),
                "bathymetry_path": str(bathy),
                "output_dir": str(tmpdir / "out"),
                "overwrite": True,
                "gif": False,
            }
            p = self._write_config(data, tmpdir)
            with self.assertRaises(ValueError):
                load_run_config(p)

    def test_both_spm_threshold_and_quantile_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bathy = tmpdir / "bathy.pkl"
            bathy.write_bytes(b"")
            data = {
                "input_path": str(tmpdir / "*.nc"),
                "bathymetry_path": str(bathy),
                "output_dir": str(tmpdir / "out"),
                "overwrite": True,
                "gif": False,
                "zone": "BAY_OF_SEINE",
                "spm_threshold": 5.0,
                "global_threshold_quantile": 0.9,
            }
            p = self._write_config(data, tmpdir)
            with self.assertRaises(ValueError):
                load_run_config(p)

    def test_quantile_out_of_range_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bathy = tmpdir / "bathy.pkl"
            bathy.write_bytes(b"")
            data = {
                "input_path": str(tmpdir / "*.nc"),
                "bathymetry_path": str(bathy),
                "output_dir": str(tmpdir / "out"),
                "overwrite": True,
                "gif": False,
                "zone": "BAY_OF_SEINE",
                "global_threshold_quantile": 1.5,   # out of (0, 1)
            }
            p = self._write_config(data, tmpdir)
            with self.assertRaises(ValueError):
                load_run_config(p)

    def test_optional_fields_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data = self._minimal_config_data(tmpdir)
            p = self._write_config(data, tmpdir)
            cfg = load_run_config(p)
            self.assertEqual(cfg.nb_cores, 1)
            self.assertFalse(cfg.dynamic_threshold)
            self.assertIsNone(cfg.spm_threshold)
            self.assertIsNone(cfg.annual_map_path)
            self.assertAlmostEqual(cfg.near_mouth_lower_quantile, 0.25)
            self.assertAlmostEqual(cfg.near_mouth_upper_quantile, 0.75)

    def test_optional_fields_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bathy = tmpdir / "bathy.pkl"
            bathy.write_bytes(b"")
            data = {
                "input_path": str(tmpdir / "*.nc"),
                "bathymetry_path": str(bathy),
                "output_dir": str(tmpdir / "out"),
                "overwrite": True,
                "gif": True,
                "nb_cores": 4,
                "dynamic_threshold": True,
                "near_mouth_lower_quantile": 0.1,
                "near_mouth_upper_quantile": 0.9,
                "lat_new_resolution": 0.05,
                "lon_new_resolution": 0.05,
                "variable_name": "SPM",
                "zone": "BAY_OF_SEINE",
            }
            p = self._write_config(data, tmpdir)
            cfg = load_run_config(p)
            self.assertEqual(cfg.nb_cores, 4)
            self.assertTrue(cfg.dynamic_threshold)
            self.assertAlmostEqual(cfg.near_mouth_lower_quantile, 0.1)
            self.assertAlmostEqual(cfg.lat_new_resolution, 0.05)


if __name__ == "__main__":
    unittest.main()
