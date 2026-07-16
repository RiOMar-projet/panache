from __future__ import annotations

import glob
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import xarray as xr

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import geopandas as gpd
except ImportError:  # Optional when no boundary shapefile overlay is requested.
    gpd = None

try:
    import multiprocess
except ImportError:  # Fall back to the stdlib module for simple batch runs.
    import multiprocessing as multiprocess

from .config import RunConfig
from .io import NoValidMapDataError, load_map_data
from .utils import align_bathymetry_to_resolution, coordinate_range_bounds
from .plume_algorithm import (
    create_polygon_mask,
    derive_masks_from_bathymetry,
    find_the_index_of_the_plume_starting_point,
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


def _output_exists(output_stem: Path) -> bool:
    return Path(f"{output_stem}.png").exists()


def _skip_existing_output_message(input_file: str | Path) -> str:
    return f"Skipping {Path(input_file).name}: output PNG already exists."


def _skip_empty_file_message(input_file: str | Path) -> str:
    return f"Skipping {Path(input_file).name}: file contains no finite data values."


def _load_first_valid_map_data(
    input_files: list[Path],
    parameters: dict,
    variable_name: str | None,
) -> tuple[xr.DataArray, list[Path]]:
    skipped_files = []
    for input_file in input_files:
        try:
            dataset = load_map_data(
                input_file,
                lon_range=coordinate_range_bounds(parameters["lon_range_of_plume_area"]),
                lat_range=coordinate_range_bounds(parameters["lat_range_of_plume_area"]),
                variable_name=variable_name,
            )
        except NoValidMapDataError:
            print(_skip_empty_file_message(input_file), flush=True)
            skipped_files.append(input_file)
            continue

        valid_input_files = [path for path in input_files if path not in skipped_files]
        return dataset, valid_input_files

    raise NoValidMapDataError("No input files contained finite data values.")


def _load_boundary(boundary_path: Path | None):
    if boundary_path is None:
        return None
    if gpd is None:
        raise ImportError("geopandas is required when 'coast_shapefile' is provided.")
    return gpd.read_file(boundary_path)


def compute_global_threshold(
    input_files: list[Path],
    parameters: dict,
    variable_name: str | None,
    quantile: float,
    sample_ds: xr.DataArray,
) -> tuple[float, tuple[float, float]]:
    """Compute the SPM quantile threshold and colourbar limits from the full input dataset.

    Loads every file in the bbox crop once, concatenates all finite values, and
    returns ``(threshold, (vmin, vmax))``.  Before loading, prints an estimate of
    the RAM required and, when psutil is available, checks that sufficient memory
    is free.
    """
    n = len(input_files)
    n_pixels = sample_ds.size
    bytes_needed = n * n_pixels * 8  # float64 worst case
    gb_needed = bytes_needed / 1e9

    print(f"\nComputing global SPM threshold ({quantile:.0%}-ile) and colourbar limits from {n} files.")
    print(f"  Estimated peak RAM required: {gb_needed:.2f} GB", flush=True)

    if _HAS_PSUTIL:
        available_gb = psutil.virtual_memory().available / 1e9
        print(f"  Available RAM:              {available_gb:.2f} GB", flush=True)
        if bytes_needed > psutil.virtual_memory().available:
            raise MemoryError(
                f"Insufficient RAM to compute global threshold: need ~{gb_needed:.1f} GB "
                f"but only {available_gb:.1f} GB is available. "
                f"Set 'spm_threshold' in your config to provide a fixed value and skip "
                f"pre-computation."
            )
    else:
        print(
            "  (Install psutil to enable automatic RAM availability check.)",
            flush=True,
        )

    lon_range = coordinate_range_bounds(parameters["lon_range_of_plume_area"])
    lat_range = coordinate_range_bounds(parameters["lat_range_of_plume_area"])
    n_digits = len(str(n))
    all_values: list[np.ndarray] = []

    for i, path in enumerate(input_files, 1):
        print(
            f"\r  Loading [{i:{n_digits}d}/{n}] ({100 * i // n:3d}%)  {path.name:<50}",
            end="",
            flush=True,
        )
        try:
            da = load_map_data(path, lon_range=lon_range, lat_range=lat_range, variable_name=variable_name)
            all_values.append(da.values.ravel())
        except NoValidMapDataError:
            continue
        except Exception as exc:
            print(f"\n  Warning: could not load {path.name}: {exc}", flush=True)
            continue

    print(flush=True)  # newline after progress line

    if not all_values:
        raise ValueError(
            "No valid data found across all input files — cannot compute global threshold."
        )

    combined = np.concatenate(all_values)
    threshold = float(np.nanquantile(combined, quantile))
    finite = combined[np.isfinite(combined) & (combined > 0)]
    if finite.size > 0:
        vmin = float(max(np.nanmin(finite), 0.1))
        vmax = float(max(np.nanquantile(finite, 0.95), 1.0))
    else:
        vmin, vmax = 0.1, 1.0
    print(f"  Global SPM threshold ({quantile:.0%}-ile): {threshold:.4f}", flush=True)
    print(f"  Global colourbar: vmin={vmin:.4f}, vmax={vmax:.4f}\n", flush=True)
    return threshold, (vmin, vmax)


def compute_global_colour_limits(
    input_files: list[Path],
    parameters: dict,
    variable_name: str | None,
    sample_ds: xr.DataArray,
) -> tuple[float, float]:
    """Compute vmin/vmax for the colourbar from the full input dataset.

    Returns (vmin, vmax) where vmin >= 0.1 and vmax >= 1.0, derived from the
    5th and 95th percentiles of all finite values across the bbox and all time steps.
    """
    n = len(input_files)
    n_digits = len(str(n))
    n_pixels = sample_ds.size
    bytes_needed = n * n_pixels * 8
    gb_needed = bytes_needed / 1e9

    print(f"\nComputing global colourbar limits from {n} files.", flush=True)

    if _HAS_PSUTIL:
        available_gb = psutil.virtual_memory().available / 1e9
        if bytes_needed > psutil.virtual_memory().available:
            print(
                f"  Warning: insufficient RAM ({gb_needed:.1f} GB needed, "
                f"{available_gb:.1f} GB available). Falling back to first-file colour limits.",
                flush=True,
            )
            vals = sample_ds.values.ravel()
            finite = vals[np.isfinite(vals) & (vals > 0)]
            if finite.size == 0:
                return 0.1, 1.0
            return float(max(np.nanmin(finite), 0.1)), float(max(np.nanquantile(finite, 0.95), 1.0))

    lon_range = coordinate_range_bounds(parameters["lon_range_of_plume_area"])
    lat_range = coordinate_range_bounds(parameters["lat_range_of_plume_area"])
    all_values: list[np.ndarray] = []

    for i, path in enumerate(input_files, 1):
        print(
            f"\r  [{i:{n_digits}d}/{n}] {path.name:<60}",
            end="",
            flush=True,
        )
        try:
            da = load_map_data(path, lon_range=lon_range, lat_range=lat_range, variable_name=variable_name)
            vals = da.values.ravel()
            all_values.append(vals[np.isfinite(vals) & (vals > 0)])
        except NoValidMapDataError:
            continue
        except Exception as exc:
            print(f"\n  Warning: could not load {path.name}: {exc}", flush=True)
            continue

    print(flush=True)

    if not all_values:
        return 0.1, 1.0

    combined = np.concatenate(all_values)
    vmin = float(max(np.nanmin(combined), 0.1))
    vmax = float(max(np.nanquantile(combined, 0.95), 1.0))
    print(f"  Global colourbar: vmin={vmin:.4f}, vmax={vmax:.4f}\n", flush=True)
    return vmin, vmax


def _compute_dynamic_threshold_data(
    input_files: list[Path],
    parameters: dict,
    variable_name: str | None,
    pixel_starts: dict[str, tuple[int, int]],
    land_mask_values: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
    lat_new_resolution: float | None,
    lon_new_resolution: float | None,
    radius_pixels: int = 15,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    """Load the full file stack to compute near-mouth threshold bounds and colour limits.

    A single pass over all input files accumulates near-mouth SPM samples for each
    plume in ``pixel_starts`` and colour-limit values from the full bbox.  Each file
    is reduced to the run resolution (if configured) so that the pixel indices in
    ``pixel_starts`` remain valid.

    Returns
    -------
    bounds : dict mapping plume name → (minimal_threshold, maximal_threshold)
    colour_limits : (vmin, vmax)
    """
    n = len(input_files)
    n_digits = len(str(n))
    lon_range = coordinate_range_bounds(parameters["lon_range_of_plume_area"])
    lat_range = coordinate_range_bounds(parameters["lat_range_of_plume_area"])

    plume_samples: dict[str, list[np.ndarray]] = {name: [] for name in pixel_starts}
    colour_values: list[np.ndarray] = []

    print(
        f"\nLoading {n} files to estimate near-mouth threshold bounds and colour limits...",
        flush=True,
    )

    for i, path in enumerate(input_files, 1):
        print(
            f"\r  [{i:{n_digits}d}/{n}] {path.name:<60}",
            end="",
            flush=True,
        )
        try:
            da = load_map_data(path, lon_range=lon_range, lat_range=lat_range, variable_name=variable_name)
        except NoValidMapDataError:
            continue
        except Exception as exc:
            print(f"\n  Warning: could not load {path.name}: {exc}", flush=True)
            continue

        if lat_new_resolution is not None or lon_new_resolution is not None:
            da = reduce_resolution(da, lat_new_resolution, lon_new_resolution)

        vals = da.values
        rows, cols = vals.shape

        finite_pos = vals[np.isfinite(vals) & (vals > 0)]
        if finite_pos.size > 0:
            colour_values.append(finite_pos)

        for plume_name, (r0, c0) in pixel_starts.items():
            r_range = np.arange(max(0, r0 - radius_pixels), min(rows, r0 + radius_pixels + 1))
            c_range = np.arange(max(0, c0 - radius_pixels), min(cols, c0 + radius_pixels + 1))
            rr, cc = np.meshgrid(r_range, c_range, indexing='ij')
            within = np.sqrt((rr - r0) ** 2 + (cc - c0) ** 2) <= radius_pixels
            near_vals = vals[rr[within], cc[within]]
            valid_mask = np.isfinite(near_vals) & ~land_mask_values[rr[within], cc[within]]
            if valid_mask.any():
                plume_samples[plume_name].append(near_vals[valid_mask])

    print(flush=True)

    bounds: dict[str, tuple[float, float]] = {}
    for plume_name, chunks in plume_samples.items():
        sample = np.concatenate(chunks) if chunks else np.array([1.0, 10.0])
        lo = float(np.nanquantile(sample, lower_quantile))
        hi = float(np.nanquantile(sample, upper_quantile))
        bounds[plume_name] = (lo, hi)
        print(
            f"  '{plume_name}': "
            f"p{lower_quantile * 100:.0f}={lo:.2f}, "
            f"p{upper_quantile * 100:.0f}={hi:.2f} g m⁻³",
            flush=True,
        )

    if colour_values:
        combined = np.concatenate(colour_values)
        vmin = float(max(np.nanmin(combined), 0.1))
        vmax = float(max(np.nanquantile(combined, 0.95), 1.0))
    else:
        vmin, vmax = 0.1, 1.0
    print(f"  Global colourbar: vmin={vmin:.4f}, vmax={vmax:.4f}\n", flush=True)

    return bounds, (vmin, vmax)


def _read_manifest(output_dir: Path) -> set[str]:
    """Return the set of input_file paths recorded in the manifest from a prior run."""
    manifest_path = output_dir / "manifest.csv"
    if not manifest_path.exists():
        return set()
    try:
        df = pd.read_csv(manifest_path)
        if "input_file" in df.columns:
            return set(df["input_file"].dropna().astype(str))
    except Exception:
        pass
    return set()


def _write_manifest(output_dir: Path, records: list[dict]) -> None:
    manifest_path = output_dir / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)


def _run_task(task):
    input_file = task[0]
    try:
        result = main_process(*task)
    except Exception as exc:
        print(f"\nError processing {Path(input_file).name}: {exc}", flush=True)
        result = None
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

    ds, input_files = _load_first_valid_map_data(input_files, parameters, config.variable_name)

    native_lat_res = float(np.diff(ds.lat.values).mean())
    native_lon_res = float(np.diff(ds.lon.values).mean())
    print(f"Native data resolution: {native_lat_res:.5f}° lat × {native_lon_res:.5f}° lon", flush=True)

    if config.lat_new_resolution is not None or config.lon_new_resolution is not None:
        target_lat = config.lat_new_resolution or native_lat_res
        target_lon = config.lon_new_resolution or native_lon_res
        lat_factor = round(target_lat / native_lat_res)
        lon_factor = round(target_lon / native_lon_res)
        print(
            f"Regridding to {target_lat:.5f}° lat × {target_lon:.5f}° lon "
            f"(every {lat_factor} lat pixel(s) and {lon_factor} lon pixel(s) averaged into one).",
            flush=True,
        )

    ds_reduced = (
        reduce_resolution(ds, config.lat_new_resolution, config.lon_new_resolution)
        if config.lat_new_resolution is not None
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

    # Inject run-level quantile settings so determine_SPM_threshold can read them
    # via self.parameters without threading them through every function signature.
    parameters['near_mouth_lower_quantile'] = config.near_mouth_lower_quantile
    parameters['near_mouth_upper_quantile'] = config.near_mouth_upper_quantile

    # --- Pre-compute near-mouth threshold bounds (once per batch) ---
    # When dynamic_threshold is active and a plume's minimal/maximal_threshold is
    # None, estimate it from the full file stack so values are stable across the
    # entire batch rather than re-derived per scene.
    precomputed_threshold: float | None = None
    global_colour_limits: tuple[float, float] | None = None

    if config.dynamic_threshold:
        lower_q = config.near_mouth_lower_quantile
        upper_q = config.near_mouth_upper_quantile
        pixel_starts = {
            plume_name: find_the_index_of_the_plume_starting_point(ds_reduced, starting_point)
            for plume_name, starting_point in parameters['starting_points'].items()
            if (parameters['minimal_threshold'][plume_name] is None
                or parameters['maximal_threshold'][plume_name] is None)
        }
        if pixel_starts:
            bounds, global_colour_limits = _compute_dynamic_threshold_data(
                input_files,
                parameters,
                config.variable_name,
                pixel_starts,
                land_mask.values,
                lower_q,
                upper_q,
                config.lat_new_resolution,
                config.lon_new_resolution,
            )
            for plume_name, (minimal, maximal) in bounds.items():
                parameters['minimal_threshold'][plume_name] = minimal
                parameters['maximal_threshold'][plume_name] = maximal

    # --- Resolve SPM threshold ---
    # Priority: dynamic_threshold (per-scene gradient) > spm_threshold /
    # global_threshold_quantile (pre-computed scalars) > fixed_threshold in
    # parameters (zone preset values).
    # When dynamic_threshold is True, precomputed_threshold stays None so the
    # per-scene gradient path in plume_algorithm runs unconditionally.
    if not config.dynamic_threshold:
        if config.spm_threshold is not None:
            precomputed_threshold = config.spm_threshold
            print(f"Using user-supplied SPM threshold: {precomputed_threshold}")

        elif config.global_threshold_quantile is not None:
            try:
                precomputed_threshold, global_colour_limits = compute_global_threshold(
                    input_files,
                    parameters,
                    config.variable_name,
                    config.global_threshold_quantile,
                    ds,
                )
            except MemoryError as exc:
                raise SystemExit(f"[panache] {exc}") from exc
            except ValueError as exc:
                raise SystemExit(f"[panache] {exc}") from exc

    # --- Compute global colourbar limits (skipped when already derived above) ---
    if global_colour_limits is None:
        global_colour_limits = compute_global_colour_limits(
            input_files, parameters, config.variable_name, ds
        )

    # --- Build task list, honouring overwrite=False via manifest + PNG check ---
    prior_processed = _read_manifest(output_dir) if not config.overwrite else set()

    print("Project structure complete. Starting batch processing...")
    print(f"Running batch with {config.nb_cores} cores...")

    tasks = []
    skipped_from_manifest: list[str] = []

    for input_file in input_files:
        output_stem = _resolve_output_stem(input_file, output_dir, input_base)

        already_done = (
            str(input_file) in prior_processed
            or (not config.overwrite and _output_exists(output_stem))
        )
        if already_done:
            print(_skip_existing_output_message(input_file), flush=True)
            skipped_from_manifest.append(str(input_file))
            continue

        tasks.append(
            (
                str(input_file),
                parameters,
                bathymetry,
                cloud_check_water_mask,
                land_mask,
                inside_polygon_mask,
                str(output_stem),
                config.dynamic_threshold,
                coast_boundary,
                config.variable_name,
                precomputed_threshold,
                global_colour_limits,
                config.lat_new_resolution,
                config.lon_new_resolution,
            )
        )

    # --- Run tasks ---
    new_stats: list[dict] = []
    total = len(tasks)

    if config.nb_cores > 1:
        multiprocess.set_start_method('spawn', force=True)
        with multiprocess.Pool(config.nb_cores) as pool:
            for completed, (input_file, result) in enumerate(
                pool.imap_unordered(_run_task, tasks), 1
            ):
                status = "Skipped" if result is None else "Completed"
                print(f"[{completed}/{total}] {status} {Path(input_file).name}", flush=True)
                if result is not None:
                    new_stats.append(result)
    else:
        for completed, task in enumerate(tasks, 1):
            input_file, result = _run_task(task)
            status = "Skipped" if result is None else "Completed"
            print(f"[{completed}/{total}] {status} {Path(input_file).name}", flush=True)
            if result is not None:
                new_stats.append(result)

    # --- Assemble Results.csv ---
    # When overwrite=False we merge new results with any rows already in Results.csv
    # from a prior run, avoiding duplicates by date.
    print("Batch processing complete. Saving results...")

    results_path = output_dir / "Results.csv"
    new_df = pd.DataFrame(new_stats)

    if not config.overwrite and results_path.exists() and not new_df.empty:
        try:
            old_df = pd.read_csv(results_path)
            if "date" in new_df.columns and "date" in old_df.columns:
                old_df = old_df[~old_df["date"].isin(new_df["date"])]
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception as exc:
            print(f"  Warning: could not merge existing Results.csv: {exc}", flush=True)

    if "date" in new_df.columns:
        new_df = new_df.sort_values("date").reset_index(drop=True)

    new_df.to_csv(results_path, index=False)
    print(f"{results_path}")

    # --- Write manifest ---
    manifest_records = [
        {"input_file": str(f), "status": "skipped_existing"}
        for f in skipped_from_manifest
    ] + [
        {"input_file": task[0], "status": "processed"}
        for task in tasks
    ]
    _write_manifest(output_dir, manifest_records)

    # --- Optional GIF ---
    if config.gif:
        saved_maps = sorted((output_dir / "MAPS").rglob("*.png"))
        if saved_maps:
            print("GIF processing started. Multiple years of data may take a long time to process.")
            with imageio.get_writer(output_dir / "GIF.gif", mode="I", fps=1) as writer:
                for figure_file in saved_maps:
                    writer.append_data(imageio.imread(figure_file))
            print("GIF processing complete.")

    return "All processes complete."
