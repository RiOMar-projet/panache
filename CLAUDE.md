# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

`panache` is a Python package for detecting river plumes in gridded geophysical satellite data (primarily suspended particulate matter, SPM, in NetCDF format). It takes NetCDF inputs, a bathymetry mask, and a JSON config to produce per-timestep PNG maps, a `Results.csv` summary, an optional manifest, and an optional animated GIF.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+. Key dependencies: `xarray`, `numpy`, `pandas`, `geopandas`, `scipy`, `scikit-image`, `matplotlib`, `imageio`, `multiprocess`, `bathyreq`, `psutil`.

## Running

```bash
panache <config.json>
```

The entrypoint is `panache.cli:main`. It calls `load_run_config` then `run_batch`. See `example_zone_config.json` and `example_parameter_config.json` for the two config modes.

## Tests

The unit tests use standard `unittest`:

```bash
python -m unittest testing/test_searching_strategy_presets.py
```

The smoke test runs the full pipeline on synthetic data (no real NetCDF files needed):

```bash
python testing/smoke_test_module.py
python testing/smoke_test_module.py --with-plots   # also generates PNG/GIF outputs
```

The integration scripts `testing/test_single_file_plume.py` and `testing/test_batch_plume.py` require real SPM NetCDF data from a local `pCloudDrive` path and are not intended for CI.

## Publishing to PyPI

The distribution is named `panache-riomar` on PyPI (the name `panache` was already taken by an unrelated package); the importable module and `panache` CLI command are unaffected.

Releases are built and published automatically by `.github/workflows/publish.yml` whenever a GitHub Release is published, using [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no stored API token). Before the first release, register this repo/workflow as a trusted publisher for the `panache-riomar` project in PyPI's project settings.

To release a new version:

1. Bump `version` in `pyproject.toml`.
2. Tag and publish a GitHub Release from that commit.
3. The `publish` workflow builds the sdist/wheel and uploads them to PyPI.

To build and check a release locally first:

```bash
pip install build twine
python -m build
twine check dist/*
```

Test the process against [TestPyPI](https://test.pypi.org/) before a real release if making packaging changes.

## Architecture

```
src/panache/
├── cli.py            — argparse entry point; calls load_run_config + run_batch
├── config.py         — RunConfig dataclass; load_run_config + build_parameters
├── io.py             — load_map_data, infer_primary_variable, normalize_map_data
├── runner.py         — run_batch: file discovery, mask setup, single/multi-core dispatch
├── plume_algorithm.py — all detection logic: flood_fill, main_process, make_the_plot, ...
└── utils.py          — define_parameters (zone presets), align_bathymetry_to_resolution,
                        SEARCHING_STRATEGY_PRESETS, searching_strategy_directions_from_presets
```

### Data flow

1. `load_run_config` parses JSON → `RunConfig`. A config must specify either `zone` (resolved via `define_parameters` in `utils.py`) or explicit `parameters`. Both paths produce the same parameter dict structure.
2. `run_batch` discovers input `.nc` files (glob, single file, or directory recursion), loads the first valid file to derive grid geometry, aligns bathymetry to that grid, and builds shared masks.
3. Each file is dispatched to `main_process` in `plume_algorithm.py`, either sequentially or via `multiprocess.Pool`. `main_process` writes a `*_plume_mask.png` per file and returns a stats dict in memory.
4. After all files are processed, `run_batch` assembles all returned dicts and writes a single `Results.csv`. A `manifest.csv` listing each input file and its processing status is also written.

### Config modes

- **Zone preset** (`zone` key): resolved by `define_parameters` in `utils.py`. Currently defined zones: `BAY_OF_SEINE`, `BAY_OF_BISCAY`, `GULF_OF_LION`, `SOUTHERN_BRITTANY`.
- **Custom parameters** (`parameters` key): the 10 keys in `REQUIRED_PARAMETER_KEYS` (`config.py:11`) must be present. Five additional keys are optional and filled with defaults by `build_parameters` when absent: `maximal_threshold`, `minimal_threshold`, `quantile_to_use`, `fixed_threshold`, and `river_mouth_to_exclude`. `searching_strategies` values must be one of the four named presets: `northward_fan`, `southward_fan`, `eastward_fan`, `westward_fan`. These are resolved to pixel-direction tuples by `searching_strategy_directions_from_presets`.

### Key algorithm concepts in `plume_algorithm.py`

- `flood_fill`: BFS-based connected-region growth from estuary starting point, respecting SPM threshold and directional constraints.
- `find_SPM_threshold` / `compute_gradient_with_directions_vectorized`: dynamic threshold detection using SPM gradients along directional transects from the river mouth. `compute_gradient_with_directions_vectorized` returns `None` (not an array) when all transect pixels are clipped to the same `maximal_threshold` (range = 0, normalisation undefined). `find_SPM_threshold` guards against this and two other empty-array edge cases, returning `minimal_threshold` as the fallback in all three.
- `Pipeline_to_delineate_the_plume`: orchestration function that runs two BFS flood fills per plume — first in the configured fan direction, then in the opposite direction (via `_OPPOSITE_FAN` and `SEARCHING_STRATEGY_PRESETS`) — and OR-merges the two masks. The second fill removes linear artifacts caused by the directional constraint of the first pass.
- `_OPPOSITE_FAN`: module-level dict in `plume_algorithm.py` mapping each fan preset to its opposite (`northward_fan` ↔ `southward_fan`, `eastward_fan` ↔ `westward_fan`). Used exclusively inside `Pipeline_to_delineate_the_plume`.
- `make_the_plot`: accepts `core_of_the_plumes` (keyword, default `None`). When provided, renders a grey `×` at each core coordinate and a labelled black `+` at each starting point on the right-hand panel of the daily PNG, with a subtitle legend. All three call sites in `main_process` pass this argument.
- `derive_masks_from_bathymetry`: produces `cloud_check_water_mask` (open-water pixels for cloud coverage assessment) and `land_mask` from the bathymetry pickle.
- `create_polygon_mask`: restricts detection to the `lat_range_of_plume_area` / `lon_range_of_plume_area` polygon defined in the parameters.
- `remove_coastal_areas_with_sediment_resuspension`: removes shallow-water, near-estuary pixels likely driven by resuspension rather than plume signal.
- `river_mouth_to_exclude`: optional parameter (defaults to `{}`) used to mask pixels near secondary tidal channels that would otherwise contaminate the plume mask (e.g. Canal de Caen à la mer in `BAY_OF_SEINE`). It is not dead code.

### `starting_points` vs `core_of_the_plumes`

These parameters are distinct and serve different roles at different stages of the pipeline.

**`starting_points`** is the algorithmic seed for each plume. It is the pixel where:
1. `flood_fill` begins its BFS expansion.
2. Directional gradient transects originate in `find_SPM_threshold` (dynamic threshold mode).
3. Cross-sectional scanning begins in the plume-trimming steps (`remove_parts_of_the_plume_area_...`).

The pixel must be a valid water pixel within or immediately adjacent to the high-SPM signal at the river mouth. If it falls on a NaN (cloud gap), `find_nearest_valid_start` relocates it to the nearest finite neighbour before flood fill begins.

**`core_of_the_plumes`** is a trusted reference coordinate placed inside the expected plume body. It is used after the raw mask exists, for three distinct purposes:
1. **Shape selection** (`identify_the_main_plume_shape_based_on_the_plume_core_location`): after flood fill, the mask may contain several disconnected blobs. The algorithm keeps only the connected component that contains or is nearest to the core coordinate.
2. **Resuspension filter** (`remove_coastal_areas_with_sediment_resuspension`): Haversine distance from the core is computed for each pixel; shallow pixels beyond the `minimal_distance_from_estuary` threshold are candidates for removal.
3. **Shape merging** (`dilate_the_main_plume_area_to_merge_close_plume_areas`): also calls shape-selection internally using the core to locate the right blob after dilation.

The two coordinates are often near-identical, but they differ when the river mouth is very close to the coastline. In that case `starting_points` is placed right at the mouth (a coastal water pixel), while `core_of_the_plumes` is offset slightly offshore so that shape-selection reliably targets the correct blob. Example: `BAY_OF_SEINE` places the Seine starting point at the coast (0.145°E) and its core 0.15° offshore (0.0°E).

### Bathymetry

Stored as a pickled `xr.DataArray`. If the pickle does not exist at the configured path, `bathyreq` is used to download it automatically and save it. The pickle must expose `lat` and `lon` coordinates.

### Output structure

```
<output_dir>/
├── Results.csv          — one row per input file, assembled in memory at run end
├── manifest.csv         — input_file + status for every file seen in this run
├── GIF.gif              (optional, when gif: true)
└── MAPS/
    └── <stem>.png       — per-timestep comparison map
```

`overwrite: false` skips any file whose `<stem>.png` already exists or whose path appears in `manifest.csv` from a prior run. On resume, prior rows in `Results.csv` are merged with new rows by date to produce the final output.

### Threshold resolution order

For each run, the SPM threshold is resolved in this priority order:

1. **`dynamic_threshold: true`** — per-scene, per-plume gradient-based threshold computed by `find_SPM_threshold`. Overrides all other options; `precomputed_threshold` is always `None` on this path (enforced in `runner.py`).
2. **`spm_threshold`** (float in config) — user-supplied scalar applied uniformly to all river mouths; fastest path, no data pre-loading. Only evaluated when `dynamic_threshold: false`.
3. **`global_threshold_quantile`** (float 0–1 in config) — computed from the full dataset before daily processing begins. Only evaluated when `dynamic_threshold: false`. Recommended scientific approach; see Gangloff et al. (2017, doi:10.1016/j.csr.2017.06.024). Use 0.95 for the ambient background boundary. `spm_threshold` and `global_threshold_quantile` are mutually exclusive.
4. **Per-plume `fixed_threshold`** values in the zone preset or `parameters` block — fallback when none of the above are set.