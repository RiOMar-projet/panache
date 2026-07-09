# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

`panache` is a Python package for detecting river plumes in gridded geophysical satellite data (primarily suspended particulate matter, SPM, in NetCDF format). It takes NetCDF inputs, a bathymetry mask, and a JSON config to produce plume masks, per-timestep PNG maps, CSV statistics, and an optional animated GIF.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+. Key dependencies: `xarray`, `numpy`, `pandas`, `geopandas`, `scipy`, `scikit-image`, `matplotlib`, `imageio`, `multiprocess`, `bathyreq`.

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
3. Each file is dispatched to `main_process` in `plume_algorithm.py`, either sequentially or via `multiprocess.Pool`. `main_process` writes `*_statistics.csv`, `*_plume_mask.csv`, and `*_plume_mask.png` per file.
4. After all files are processed, `_load_statistics` collects all `*_statistics.csv` files and writes `Results.csv`.

### Config modes

- **Zone preset** (`zone` key): resolved by `define_parameters` in `utils.py`. Currently defined zones: `BAY_OF_SEINE`, `BAY_OF_BISCAY`, `GULF_OF_LION`, `SOUTHERN_BRITTANY`.
- **Custom parameters** (`parameters` key): all keys in `REQUIRED_PARAMETER_KEYS` (`config.py:11`) must be present. `searching_strategies` values must be one of the four named presets: `northward_fan`, `southward_fan`, `eastward_fan`, `westward_fan`. These are resolved to pixel-direction tuples by `searching_strategy_directions_from_presets`.

### Key algorithm concepts in `plume_algorithm.py`

- `flood_fill`: BFS-based connected-region growth from estuary starting point, respecting SPM threshold and directional constraints.
- `find_SPM_threshold` / `compute_gradient_with_directions_vectorized`: dynamic threshold detection using SPM gradients along directional transects from the river mouth.
- `derive_masks_from_bathymetry`: produces `cloud_check_water_mask` (open-water pixels for cloud coverage assessment) and `land_mask` from the bathymetry pickle.
- `create_polygon_mask`: restricts detection to the `lat_range_of_plume_area` / `lon_range_of_plume_area` polygon defined in the parameters.
- `remove_coastal_areas_with_sediment_resuspension`: removes shallow-water, near-estuary pixels likely driven by resuspension rather than plume signal.

### Bathymetry

Stored as a pickled `xr.DataArray`. If the pickle does not exist at the configured path, `bathyreq` is used to download it automatically and save it. The pickle must expose `lat` and `lon` coordinates.

### Output structure

```
<output_dir>/
├── Results.csv
├── GIF.gif              (optional)
└── MAPS/
    ├── <stem>_plume_mask.png
    ├── <stem>_plume_mask.csv
    └── <stem>_statistics.csv
```

`overwrite: false` skips files whose `*_statistics.csv` already exists. `Results.csv` is always rebuilt from all `*_statistics.csv` files found under `MAPS/`.