# What's New

## v0.4.1 — 17 July 2026

### Performance: each input file is now read from disk exactly once

Previously, a full batch run could read every input file from disk two or
three times: once while pre-computing the SPM threshold and colourbar limits
from the full dataset, and again inside `main_process` for each scene's
detection pass. `runner.py` now loads every valid input file into memory a
single time (`_load_all_scenes`), before the batch starts, and reuses that
same in-memory `xr.DataArray` for threshold/colourbar pre-computation and for
per-scene plume detection — including under `nb_cores > 1`, where the
already-loaded array is handed to the worker process instead of a file path.

### Breaking change: removed `lat_new_resolution` / `lon_new_resolution`

The optional resolution-downsampling feature (`lat_new_resolution` and
`lon_new_resolution` config keys, and the `reduce_resolution` coarsening step)
has been removed. `panache` now always operates at the native resolution of
the input product. Configs that still set these keys can simply drop them;
they are no longer read.

### Bug fix: `PlumeMasks.nc` assembly crash on large batches

Assembling `PlumeMasks.nc` at the end of a run could raise
`ValueError: 'date_for_plot' not present in all datasets and coords='different'`
on large batches (observed on a 10,000+ file run). Each scene's mask carried a
`date_for_plot` scalar coordinate left over from the source SPM map, which is
redundant with the `time` coordinate assigned to the mask itself; xarray's
`xr.concat` only complains when that coordinate has differing values on some
scenes and is absent on others, which is data-dependent and did not surface on
small test batches. `_mask_with_time` now drops `date_for_plot` from every
scene's mask before it is returned, so the coordinate can never reach the
final `xr.concat`.

### Per-scene log messages now report the scene date, not the input filename

Batch progress lines (`[n/N] Completed ...`) and the "too cloudy" / "no plume
detected" skip messages now print the scene's date (e.g. `2024-01-01`) rather
than the input file's name.

---

## v0.4.0 — 17 July 2026

### New: PlumeMasks.nc — aggregated daily pixel masks

Each `panache` run now writes a `PlumeMasks.nc` file alongside `Results.csv`
in the output directory. The file stores the per-pixel plume mask for every
processed day as a three-dimensional `(time, lat, lon)` array of int8 values
(0 = no plume, 1 = plume). Days on which the area was too cloudy or no plume
was detected are represented as all-zero slices, so the time axis covers every
input file regardless of detection outcome.

When `overwrite: false` is set and a `PlumeMasks.nc` file already exists from
a prior run, the new batch results are merged into it by date, mirroring the
existing `Results.csv` resume logic.

This output makes it straightforward to use the detected plume footprint as a
spatial mask on any co-registered gridded product — for example, to extract
daily in-plume SPM concentration distributions from the original Sextant files
without re-running the detection algorithm. See the new **Using PlumeMasks.nc**
vignette on the documentation site for a worked example.

---

## v0.3.5 — 17 July 2026

### Bug fix: static threshold pipeline accessing non-existent dynamic arguments

When `dynamic_threshold: false`, the pipeline in `runner.py` was still attempting
to read per-plume near-mouth threshold bounds that are only computed during a
dynamic-threshold run, raising a `KeyError` on any static-threshold batch.
The call has been removed from the static path and the corresponding dead branch
in `plume_algorithm.py` has been pruned.

### Bug fix: date parsed from filename before internal NetCDF metadata

`io.py` now extracts the scene date from the filename first and falls back to
the `time` coordinate inside the NetCDF only when no date is recoverable from
the name. This fixes processing of the full 2005 Sextant SPM archive, where an
upstream data error caused every file's internal `time` coordinate to reference
an incorrect date while the filename remained correct.

### Documentation site: workflow vignette

A new Workflow Vignette page has been added to the documentation site, showing
real L4 SPM plume-detection outputs for 1 January 2024 (winter) and 1 July 2024
(summer) across all four built-in zones (Bay of Seine, Bay of Biscay, Gulf of
Lion, Southern Brittany). Each zone is illustrated under both the dynamic
gradient-based threshold and the global p95 threshold, placed side by side using
two-column card layouts so the sensitivity difference between the two methods is
immediately visible.

### Documentation site: theme switcher, README-driven index, codecov badge

The documentation site has been updated with a light/dark theme toggle in the
top navigation bar. The main landing page now mirrors the project README exactly,
so any update to the README is automatically reflected on the site. The Codecov
coverage badge is displayed on both the README and the documentation landing
page.

---

## v0.3.x — July 2026

### Stable release: four zones tested end-to-end

The detection algorithm has been validated across all four built-in zone presets
(`BAY_OF_SEINE`, `BAY_OF_BISCAY`, `GULF_OF_LION`, `SOUTHERN_BRITTANY`) and the
Var estuary as an additional edge case. Bathymetry alignment, cloud screening,
threshold selection, and plume delineation all produce physically reasonable
output for these regions.

### Codecov integration and 90% test coverage

Continuous integration now uploads coverage reports to
[Codecov](https://codecov.io/gh/RiOMar-projet/panache) on every push to `main`.
The test suite covers 90% of the source code, spanning unit tests for all major
pipeline components and a synthetic smoke test that runs the full pipeline
without real satellite data.

### Documentation website

A Sphinx documentation site (this site) is now deployed automatically from the
`main` branch via GitHub Actions. It includes installation instructions, a
quick-start guide, configuration reference, and a full API reference generated
from docstrings.

### Bug fix: empty-slice warnings in `find_SPM_threshold`

Three early-return guards were added to prevent `np.nanquantile` from receiving
an empty array. These edge cases arise when every transect pixel is clipped to
the same `maximal_threshold` (storm events with saturated SPM), when heavy cloud
cover breaks all gradient transects, or when the 90% steepness filter eliminates
every gradient point. In all three cases the function now returns
`minimal_threshold` immediately.

### Bug fix: NaN start point in flood fill

`find_nearest_valid_start` now relocates the BFS seed pixel to the nearest
finite neighbour when the configured `starting_points` coordinate falls on a
cloud-masked (NaN) pixel. Applies to both the dynamic-threshold transect origin
and the flood-fill entry point.

### Bug fix: empty-mask statistics

`return_stats_dictionnary` checks for finite SPM pixels within the plume mask
before calling `np.nanmean` / `np.nanstd`. When the detected area is entirely
cloud-covered, the function returns an empty-stats dict via `make_an_empty_dict`
rather than raising a warning.

### New: river mouth labels and plume core markers on daily PNG maps

Each PNG now shows a grey `×` at each `core_of_the_plumes` coordinate and a
labelled black `+` at each `starting_points` coordinate, with the river-mouth
name in a white annotation box. A subtitle legend at the lower-left explains
the two marker types.

### New: global bounding-box threshold

When `global_threshold_quantile` is set in the config, `panache` now loads the
full spatial bounding box for each input file to compute the quantile threshold,
rather than reading only the plume-detection sub-region. This produces a more
representative ambient-background threshold for the whole scene.

### New: inverted-fan second flood fill

`Pipeline_to_delineate_the_plume` performs a second BFS pass in the opposite
fan direction (e.g. `southward_fan` → `northward_fan`) and OR-merges the two
masks. This removes linear artifacts caused by the directional constraint of the
first pass.

### New: five threshold parameters are now optional

`REQUIRED_PARAMETER_KEYS` has been reduced from 15 to 10 keys. The following
five keys are now optional in a custom `parameters` block and are filled with
sensible defaults by `build_parameters` when absent: `maximal_threshold`,
`minimal_threshold`, `quantile_to_use`, `fixed_threshold`, and
`river_mouth_to_exclude`.

---

## v0.2.x — June–July 2026

### `input_path` replaces `input_glob` / `input_root`

The two previous input arguments `input_glob` and `input_root` have been merged
into a single `input_path` key. Pass a glob string (`/data/spm/*.nc`), a
directory path, or a single file path — `panache` figures out which it is.

### Results compiled from all `_statistics.csv` files

After processing, `run_batch` now searches all subdirectories of `output_dir`
for `_statistics.csv` files written in previous runs and merges them with the
current batch. This means interrupted runs can be resumed with `overwrite: false`
and the final `Results.csv` will always contain the full time series.

### GIF is now optional

Set `"gif": false` in the config to skip animated GIF generation. Useful for
large batches where GIF assembly adds significant wall-clock time.

### Error handling for empty or malformed files

Files that cannot be opened or that contain no valid data are now caught and
logged as `Failed` in `manifest.csv` rather than raising an unhandled exception
and aborting the batch.

### Flood fill seed point improvements

The BFS starting pixel can now be configured more intuitively. The algorithm
accepts any water pixel near the river mouth and handles edge cases where the
initial pixel sits at the boundary of the bathymetry mask.

---

## v0.1.x — May 2026

### Initial release

`panache` is a Python package for detecting river plumes in gridded satellite
data (SPM in NetCDF format). It implements a BFS flood-fill algorithm with
dynamic or fixed SPM thresholds, bathymetry-based land and cloud masking, and
directional fan constraints.

**Key capabilities in the initial release:**

- Reads NetCDF files directly via `xarray`; common SPM variable names are
  detected automatically.
- Two config modes: `zone` (built-in presets) and `parameters` (fully custom).
- Four built-in zone presets: `BAY_OF_SEINE`, `BAY_OF_BISCAY`, `GULF_OF_LION`,
  `SOUTHERN_BRITTANY`.
- Four fan directions: `northward_fan`, `southward_fan`, `eastward_fan`,
  `westward_fan`.
- Three SPM threshold modes: dynamic (per-scene gradient), global quantile
  (dataset-wide), and fixed (per-plume constant).
- Single-core and multi-core batch processing via `multiprocess`.
- Outputs: `Results.csv`, `manifest.csv`, per-file PNG maps, optional GIF.
- Automatic bathymetry download via `bathyreq` when the pickle is absent.
- Installable from PyPI as `panache-riomar`; CLI entry point is `panache`.
