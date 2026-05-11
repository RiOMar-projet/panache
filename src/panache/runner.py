from __future__ import annotations

import glob
import os
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
from .utils import align_bathymetry_to_resolution
from .plume_algorithm import (
    create_polygon_mask,
    derive_masks_from_bathymetry,
    main_process,
    reduce_resolution,
)


def _resolve_output_stem(input_file: Path, output_dir: Path, input_root: Path) -> Path:
    relative = input_file.relative_to(input_root)
    return output_dir / "MAPS" / relative.with_suffix("")


def _load_boundary(boundary_path: Path | None):
    if boundary_path is None:
        return None
    if gpd is None:
        raise ImportError("geopandas is required when 'coast_shapefile' is provided.")
    return gpd.read_file(boundary_path)


def run_batch(config: RunConfig) -> Path:
    input_files = sorted(Path(path) for path in glob.glob(config.input_glob, recursive=True))
    if not input_files:
        raise FileNotFoundError(f"No input files matched: {config.input_glob}")

    input_root = config.input_root or Path(os.path.commonpath([str(path) for path in input_files]))
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    coast_boundary = _load_boundary(config.coast_shapefile)
    parameters = config.parameters

    ds = load_map_data(input_files[0], variable_name=config.variable_name)
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

    tasks = [
        (
            str(input_file),
            parameters,
            bathymetry,
            coast_boundary,
            cloud_check_water_mask,
            land_mask,
            inside_polygon_mask,
            str(_resolve_output_stem(input_file, output_dir, input_root)),
            config.dynamic_threshold,
            config.variable_name,
        )
        for input_file in input_files
    ]

    if config.nb_cores > 1:
        with multiprocess.Pool(config.nb_cores) as pool:
            results = pool.starmap(main_process, tasks)
    else:
        results = [main_process(*task) for task in tasks]

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
