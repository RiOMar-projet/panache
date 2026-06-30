from __future__ import annotations

import glob
from pathlib import Path

import imageio.v2 as imageio
import pandas as pd

try:
    import geopandas as gpd
except ImportError:  # Optional when no boundary shapefile overlay is requested.
    gpd = None

try:
    import multiprocess
except ImportError:  # Fall back to the stdlib module for simple batch runs.
    import multiprocessing as multiprocess

from .config import RunConfig
from .io import load_map_data
from .utils import align_bathymetry_to_resolution, coordinate_range_bounds
from .plume_algorithm import (
    create_polygon_mask,
    derive_masks_from_bathymetry,
    main_process,
    reduce_resolution,
)

_GLOB_CHARS = frozenset("*?[")


def _has_glob_pattern(path: str) -> bool:
    return any(char in path for char in _GLOB_CHARS)


def _glob_root(pattern: str) -> Path:
    parts = []
    for part in Path(pattern).parts:
        if _has_glob_pattern(part):
            break
        parts.append(part)
    return Path(*parts) if parts else Path(".")


def _discover_input_files(input_path: str) -> tuple[list[Path], Path]:
    path = Path(input_path)

    if _has_glob_pattern(input_path):
        input_files = sorted(
            Path(match)
            for match in glob.glob(input_path, recursive=True)
            if Path(match).is_file()
        )
        return input_files, _glob_root(input_path)

    if path.is_file():
        return [path], path.parent

    input_files = sorted(path.rglob("*.nc"))
    return input_files, path


def _resolve_output_stem(input_file: Path, output_dir: Path, input_base: Path) -> Path:
    relative = input_file.relative_to(input_base)
    return output_dir / "MAPS" / relative.with_suffix("")


def _load_boundary(boundary_path: Path | None):
    if boundary_path is None:
        return None
    if gpd is None:
        raise ImportError("geopandas is required when 'coast_shapefile' is provided.")
    return gpd.read_file(boundary_path)


def _run_task(task):
    input_file = task[0]
    result = main_process(*task)
    return input_file, result


def run_batch(config: RunConfig) -> Path:
    input_files, input_base = _discover_input_files(config.input_path)
    if not input_files:
        raise FileNotFoundError(f"No input files matched: {config.input_path}")

    print(f"Found {len(input_files)} input files.")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    coast_boundary = _load_boundary(config.coast_shapefile)
    parameters = config.parameters

    ds = load_map_data(
        input_files[0], 
        lon_range=coordinate_range_bounds(parameters['lon_range_of_plume_area']),
        lat_range=coordinate_range_bounds(parameters['lat_range_of_plume_area']),
        variable_name=config.variable_name)
    ds_reduced = (
        reduce_resolution(ds, parameters["lat_new_resolution"], parameters["lon_new_resolution"])
        if parameters["lat_new_resolution"] is not None
        else ds
    )

    bathymetry = align_bathymetry_to_resolution(ds_reduced, str(config.bathymetry_path))
    input_bathymetry = align_bathymetry_to_resolution(ds, str(config.bathymetry_path))
    inside_polygon_mask = create_polygon_mask(ds_reduced, parameters)
    cloud_check_water_mask, land_mask = derive_masks_from_bathymetry(
        input_bathymetry,
        bathymetry,
        parameters,
    )

    print("Project structure complete. Starting batch processing...")
    print(f"Running batch with {config.nb_cores} cores...")

    tasks = [
        (
            str(input_file),
            parameters,
            bathymetry,
            cloud_check_water_mask,
            land_mask,
            inside_polygon_mask,
            str(_resolve_output_stem(input_file, output_dir, input_base)),
            config.dynamic_threshold,
            coast_boundary,
            config.variable_name,
        )
        for input_file in input_files
    ]

    if config.nb_cores > 1:
        results = []
        total = len(tasks)

        with multiprocess.Pool(config.nb_cores) as pool:
            for completed, (input_file, result) in enumerate(pool.imap_unordered(_run_task, tasks), 1):
                print(f"[{completed}/{total}] Completed {Path(input_file).name}", flush=True)
                results.append(result)
    else:
        results = []
        total = len(tasks)

        for completed, task in enumerate(tasks, 1):
            input_file, result = _run_task(task)
            print(f"[{completed}/{total}] Completed {Path(input_file).name}", flush=True)
            results.append(result)

    print("Batch processing complete. Saving results...")

    statistics = pd.DataFrame([result for result in results if result is not None])
    if not statistics.empty:
        statistics = statistics.sort_values("date").reset_index(drop=True)
    results_path = output_dir / "Results.csv"
    statistics.to_csv(results_path, index=False)

    saved_maps = sorted((output_dir / "MAPS").rglob("*.png"))
    if saved_maps:
        with imageio.get_writer(output_dir / "GIF.gif", mode="I", fps=1) as writer:
            for figure_file in saved_maps:
                writer.append_data(imageio.imread(figure_file))

    return results_path
