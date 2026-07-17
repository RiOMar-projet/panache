#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import xarray as xr


EXPECTED_DATES = ["2020-01-01", "2020-01-02"]


def _import_panache():
    try:
        from panache import load_run_config, run_batch
        import panache.plume_algorithm as plume_algorithm

        return load_run_config, run_batch, plume_algorithm, "installed"
    except ImportError:
        repo_src = Path(__file__).resolve().parents[1] / "src"
        if not repo_src.exists():
            raise

        sys.path.insert(0, str(repo_src))
        from panache import load_run_config, run_batch
        import panache.plume_algorithm as plume_algorithm

        return load_run_config, run_batch, plume_algorithm, f"local:{repo_src}"


def _make_dataset(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    peak: float,
    width: float,
    date_text: str,
) -> xr.Dataset:
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    plume = 1.0 + peak * np.exp(-(((lat_grid - center_lat) ** 2) + ((lon_grid - center_lon) ** 2)) / width)
    plume[(lat_grid < 0.15) & (lon_grid < 0.15)] = np.nan
    noise = np.full_like(plume, 0.25)

    return xr.Dataset(
        {
            "SPM": (("time", "lat", "lon"), plume[np.newaxis, :, :]),
            "noise": (("time", "lat", "lon"), noise[np.newaxis, :, :]),
        },
        coords={"time": [np.datetime64(date_text)], "lat": lat, "lon": lon},
    )


def _build_parameters() -> dict:
    return {
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
        "maximal_threshold": {"Seine": 10},
        "minimal_threshold": {"Seine": 4},
        "quantile_to_use": {"Seine": 0.5},
        "fixed_threshold": {"Seine": 4.5},
        "river_mouth_to_exclude": {},
    }


def _prepare_synthetic_project(work_dir: Path, with_plots: bool = False) -> Path:
    inputs_dir = work_dir / "inputs" / "nested"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    lat = np.round(np.arange(0.0, 1.01, 0.1), 3)
    lon = np.round(np.arange(0.0, 1.01, 0.1), 3)

    bathymetry = xr.DataArray(
        np.full((lat.size, lon.size), -50.0),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="bathymetry",
    )
    bathymetry_path = work_dir / "bathymetry.pkl"
    with bathymetry_path.open("wb") as handle:
        pickle.dump(bathymetry, handle)

    plume_positions = [(0.45, 0.45), (0.50, 0.50)]
    for index, ((center_lat, center_lon), date_text) in enumerate(zip(plume_positions, EXPECTED_DATES), start=1):
        dataset = _make_dataset(lat, lon, center_lat, center_lon, 8.0, 0.02, date_text)
        dataset.to_netcdf(inputs_dir / f"map_{index}.nc")

    config = {
        "input_path": str(work_dir / "inputs" / "**" / "*.nc"),
        "bathymetry_path": str(bathymetry_path),
        "output_dir": str(work_dir / "outputs"),
        "overwrite": False,
        "gif": with_plots,
        "nb_cores": 1,
        "dynamic_threshold": False,
        "variable_name": "SPM",
        "parameters": _build_parameters(),
    }
    config_path = work_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def _prepare_global_threshold_project(work_dir: Path) -> Path:
    """Config that points at the real January 1998 data and uses global_threshold_quantile."""
    inputs_dir = work_dir / "inputs" / "nested"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    lat = np.round(np.arange(0.0, 1.01, 0.1), 3)
    lon = np.round(np.arange(0.0, 1.01, 0.1), 3)

    bathymetry = xr.DataArray(
        np.full((lat.size, lon.size), -50.0),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="bathymetry",
    )
    bathymetry_path = work_dir / "bathymetry.pkl"
    with bathymetry_path.open("wb") as handle:
        pickle.dump(bathymetry, handle)

    plume_positions = [(0.45, 0.45), (0.50, 0.50)]
    for index, ((center_lat, center_lon), date_text) in enumerate(zip(plume_positions, EXPECTED_DATES), start=1):
        dataset = _make_dataset(lat, lon, center_lat, center_lon, 8.0, 0.02, date_text)
        dataset.to_netcdf(inputs_dir / f"map_{index}.nc")

    params = _build_parameters()
    # Use global_threshold_quantile instead of fixed_threshold
    config = {
        "input_path": str(work_dir / "inputs" / "**" / "*.nc"),
        "bathymetry_path": str(bathymetry_path),
        "output_dir": str(work_dir / "outputs_global"),
        "overwrite": True,
        "gif": False,
        "nb_cores": 1,
        "dynamic_threshold": False,
        "variable_name": "SPM",
        "global_threshold_quantile": 0.90,
        "parameters": params,
    }
    config_path = work_dir / "config_global.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def _assert_results(results_path: Path, output_dir: Path, with_plots: bool) -> None:
    if not results_path.exists():
        raise AssertionError(f"Missing results file: {results_path}")

    results = pd.read_csv(results_path)
    if len(results) != 2:
        raise AssertionError(f"Expected 2 result rows, found {len(results)}")

    parsed_dates = pd.to_datetime(results["date"]).dt.strftime("%Y-%m-%d").tolist()
    if parsed_dates != EXPECTED_DATES:
        raise AssertionError(f"Unexpected dates: {parsed_dates}")

    if not (results["SPM_threshold_Seine"] == 4.5).all():
        raise AssertionError(f"Unexpected thresholds: {results['SPM_threshold_Seine'].tolist()}")

    if not (results["mean_SPM_in_the_plume_area"] > 4.0).all():
        raise AssertionError("The configured SPM variable was not used to build plume statistics.")

    # Verify legacy per-day CSVs are no longer written
    stats_csvs = sorted((output_dir / "MAPS").rglob("*_statistics.csv"))
    if stats_csvs:
        raise AssertionError(
            f"Legacy per-day _statistics.csv files should not exist but found: {stats_csvs}"
        )
    plume_mask_csvs = sorted((output_dir / "MAPS").rglob("*_plume_mask.csv"))
    if plume_mask_csvs:
        raise AssertionError(
            f"Legacy per-day _plume_mask.csv files should not exist but found: {plume_mask_csvs}"
        )

    map_pngs = sorted((output_dir / "MAPS").rglob("*.png"))
    gif_path = output_dir / "GIF.gif"

    if with_plots:
        if len(map_pngs) != 2:
            raise AssertionError(f"Expected 2 PNG maps, found {len(map_pngs)}")
        if not gif_path.exists():
            raise AssertionError(f"Missing GIF output: {gif_path}")
    else:
        if map_pngs or gif_path.exists():
            raise AssertionError("Plot outputs were created even though plotting was disabled for the smoke test.")


def _assert_global_threshold_results(results_path: Path, output_dir: Path) -> None:
    if not results_path.exists():
        raise AssertionError(f"Missing results file: {results_path}")

    results = pd.read_csv(results_path)
    if len(results) != 2:
        raise AssertionError(f"Expected 2 result rows, found {len(results)}")

    # All plumes should share the same threshold (the global quantile value)
    thresholds = results["SPM_threshold_Seine"].dropna().unique()
    if len(thresholds) != 1:
        raise AssertionError(
            f"Expected a single global threshold shared across all files, got: {thresholds}"
        )

    threshold = thresholds[0]
    if not np.isfinite(threshold) or threshold <= 0:
        raise AssertionError(f"Global threshold is not a finite positive value: {threshold}")

    # Manifest should be written
    manifest_path = output_dir / "manifest.csv"
    if not manifest_path.exists():
        raise AssertionError(f"Missing manifest file: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if "input_file" not in manifest.columns:
        raise AssertionError("Manifest is missing 'input_file' column.")

    # No legacy per-day CSVs
    stats_csvs = sorted((output_dir / "MAPS").rglob("*_statistics.csv"))
    if stats_csvs:
        raise AssertionError(f"Legacy _statistics.csv files found: {stats_csvs}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the standalone plume package with synthetic data.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory where synthetic inputs and outputs should be written. Defaults to a new temporary directory.",
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="Run the full plotting path as well. Disabled by default for headless CI environments.",
    )
    parser.add_argument(
        "--test-global-threshold",
        action="store_true",
        help="Also run the global_threshold_quantile test path using synthetic data.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_run_config, run_batch, plume_algorithm, import_source = _import_panache()

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="panache_smoke_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Standard fixed-threshold smoke test ---
    config_path = _prepare_synthetic_project(work_dir, with_plots=args.with_plots)

    if not args.with_plots:
        plume_algorithm.make_the_plot = lambda *args, **kwargs: None

    config = load_run_config(config_path)
    run_batch(config)
    results_path = work_dir / "outputs" / "Results.csv"
    _assert_results(results_path, work_dir / "outputs", with_plots=args.with_plots)

    print(f"import_source={import_source}")
    print(f"work_dir={work_dir}")
    print(f"results={results_path}")
    print(f"plotting={'enabled' if args.with_plots else 'disabled'}")
    print("status=ok [fixed_threshold]")

    # --- Optional global threshold test using synthetic data ---
    if args.test_global_threshold:
        print("\nRunning global_threshold_quantile test...")
        plume_algorithm.make_the_plot = lambda *args, **kwargs: None
        config_global_path = _prepare_global_threshold_project(work_dir)
        config_global = load_run_config(config_global_path)
        run_batch(config_global)
        results_global_path = work_dir / "outputs_global" / "Results.csv"
        _assert_global_threshold_results(results_global_path, work_dir / "outputs_global")
        print(f"results_global={results_global_path}")
        print("status=ok [global_threshold_quantile]")


if __name__ == "__main__":
    main()
