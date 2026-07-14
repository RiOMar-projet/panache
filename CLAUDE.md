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
├── Results.csv          — one row per input file, assembled in memory at run end
├── manifest.csv         — input_file + status for every file seen in this run
├── GIF.gif              (optional, when gif: true)
└── MAPS/
    └── <stem>.png       — per-timestep comparison map
```

`overwrite: false` skips any file whose `<stem>.png` already exists or whose path appears in `manifest.csv` from a prior run. On resume, prior rows in `Results.csv` are merged with new rows by date to produce the final output.

### Threshold resolution order

For each run, the SPM threshold is resolved in this priority order:

1. `spm_threshold` (float in config) — user-supplied scalar, applied to all river mouths; fastest path, no data pre-loading.
2. `global_threshold_quantile` (float 0–1 in config) — computed from the full dataset before any daily processing. Recommended scientific approach; see Gangloff et al. (2017, doi:10.1016/j.csr.2017.06.024). Use 0.95 for the ambient background boundary.
3. Per-plume `fixed_threshold` values in the zone preset or `parameters` block — legacy fallback.