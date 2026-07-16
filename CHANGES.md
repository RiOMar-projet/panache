# Change log

## 2026-07-14

### Bug fix: `RuntimeWarning: Mean of empty slice` in `find_SPM_threshold`

Three early-return guards were added to `find_SPM_threshold` in `plume_algorithm.py` to handle edge
cases that previously triggered `np.nanquantile` on an empty array:

1. **`None` gradient** — `compute_gradient_with_directions_vectorized` returns `None` when every
   non-NaN transect pixel is clipped to the same `maximal_threshold` value (range = 0, normalisation
   undefined). The new first guard catches this and immediately returns `minimal_threshold`.
2. **No finite gradients** — if all gradient values are NaN (e.g. transects composed entirely of
   cloud pixels), the second guard returns `minimal_threshold` before calling `nanquantile`.
3. **No candidate edge pixels** — if the 90 % steepness filter eliminates every gradient point, the
   third guard returns `minimal_threshold` before the quantile call.

These cases most commonly arise during storm events (saturated SPM >> `maximal_threshold` on every
transect) or when heavy cloud cover breaks all gradient transects.

---

### Bug fix: `RuntimeWarning: Mean of empty slice` in `return_stats_dictionnary`

`return_stats_dictionnary` in `plume_algorithm.py` now checks whether any finite SPM pixels lie
within the final plume mask before calling `np.nanmean` / `np.nanstd`. When the entire detected
plume area is cloud-covered (all NaN), the function returns an empty-stats dict immediately via
`make_an_empty_dict()` rather than passing an empty array to `nanmean`.

---

### New: river mouth labels and plume core markers on daily PNG maps

`make_the_plot` in `plume_algorithm.py` now accepts a `core_of_the_plumes` keyword argument.
When provided, the following are rendered on the right-hand (detected plume) panel of each daily
PNG:

- A grey `×` at each `core_of_the_plumes` coordinate.
- A labelled black `+` at each `starting_points` coordinate, with the river mouth name in a
  white-background annotation box.
- A subtitle legend at the lower-left explaining the two marker types.

All three `make_the_plot` call sites in `main_process` pass `core_of_the_plumes` from the
parameters dict. No new dependencies were added.

---

### New: dynamic threshold parameter guide

`testing/dynamic_threshold_parameter_guide.md` documents the interactions between
`maximal_threshold`, `minimal_threshold`, and `quantile_to_use` in three ways: a step-by-step
trace through a single transect, an isolated description of each parameter's role, and a
diagnostic table of what goes wrong when each is mis-set. All examples use literal values from
the Gironde plume (BAY_OF_BISCAY, August 2000). The file is symlinked to
`~/RiOMar/manuscript/dynamic_threshold_parameter_guide.md`.

---

### New: inverted-fan second flood fill in `Pipeline_to_delineate_the_plume`

After the initial BFS flood fill, `Pipeline_to_delineate_the_plume` now performs a second
`do_a_raw_plume_detection()` in the exact opposite fan direction (e.g. `southward_fan` →
`northward_fan`), then OR-merges the two masks. This removes linear artifacts that arise when the
fan direction enforces a one-sided BFS expansion from the river mouth starting point.

The mapping between fan presets and their opposites is stored in the module-level dict
`_OPPOSITE_FAN` in `plume_algorithm.py`. The `SEARCHING_STRATEGY_PRESETS` import from `utils.py`
is used to look up the pixel-direction tuples for the opposite preset. The original
`searching_strategy_directions` are restored after the second fill so the rest of the pipeline is
unaffected.

---

### New: optional threshold parameters in custom parameter configs

`REQUIRED_PARAMETER_KEYS` in `config.py` has been reduced from 15 to 10 keys. The following five
keys are now optional in the `parameters` block of a custom config JSON:

| Key | Default when absent |
|---|---|
| `maximal_threshold` | `{plume_name: None}` for each plume — triggers automatic estimation from near-mouth pixels |
| `minimal_threshold` | `{plume_name: None}` for each plume — triggers automatic estimation from near-mouth pixels |
| `quantile_to_use` | `{plume_name: 0.2}` for each plume |
| `fixed_threshold` | `{plume_name: None}` for each plume |
| `river_mouth_to_exclude` | `{}` (no exclusions) |

`build_parameters` in `config.py` fills in these defaults after the required-key check.
`river_mouth_to_exclude` was already optional in zone presets and is now also optional in custom
parameter blocks; it is used to mask pixels near secondary tidal channels that would otherwise
contaminate the plume mask (e.g. Canal de Caen à la mer in `BAY_OF_SEINE`).

---

## 2026-07-09

### Updated smoke test to match current API

`testing/smoke_test_module.py` was written against an older version of the parameter and config API. It failed immediately on `load_run_config` with `ValueError: Missing required parameter keys`. Four issues were fixed:

- `_build_parameters()`: replaced old `searching_strategies` dict format (with `grid`/`coordinates_of_center` keys) with named preset strings (`{"Seine": "northward_fan"}`); replaced the old separate cloud-check, map-plot, and search-area range keys with the current `lat_range_of_plume_area` / `lon_range_of_plume_area`
- `_prepare_synthetic_project()`: replaced `input_glob`/`input_root` pair with `input_path` (glob string); added required `overwrite` and `gif` booleans; added `with_plots` parameter so `gif` is set correctly
- `_assert_results()`: narrowed the CSV glob from `*.csv` to `*_plume_mask.csv` (the MAPS directory also contains `_statistics.csv` files); corrected the no-plots check to not require zero CSVs (CSVs are always written, regardless of the plot flag)
- `main()`: `run_batch` returns a status string, not a path — construct the results path directly as `work_dir / "outputs" / "Results.csv"` instead

---

### Removed commented-out and unused code

Cleaned up all Python code that was commented out or otherwise unused across the source files. No behaviour was changed; the unit tests continue to pass.

**[src/panache/utils.py](src/panache/utils.py)**
- Removed unused `geopandas` import block (`gpd` was never referenced in this module)
- Removed unused `proj_dir` module-level variable
- Removed commented-out `# bathymetric_data.plot()` debug line

**[src/panache/plume_algorithm.py](src/panache/plume_algorithm.py)**
- Removed four commented-out import lines (`glob`, `imageio`, `geopandas`, `PathPatch`) and three commented-out entries inside the `from .utils import` block
- Removed `all_river_mouth_to_remove` variable and the block that populated it (its only consumer was already commented out)
- Removed commented-out alternative colourbar calculations in `make_the_plot`
- Removed commented-out CSV-saving block in `make_the_plot`
- Removed commented-out alternative normalization approach in `compute_gradient_with_directions_vectorized`
- Removed three commented-out visualization blocks and several commented-out alternative implementations in `filter_gradient_points_vectorized`
- Removed commented-out lines in `find_SPM_threshold` (unused variable, alternative threshold, three visualization blocks, alternative SPM calculation)
- Removed commented-out assert in `find_first_nan_after_finite`
- Removed three commented-out visualization blocks and two commented-out alternative implementations in `set_mask_area_values_to_False_based_on_an_index_object`
- Removed two commented-out visualization blocks in `find_index_and_values_of_multiple_directions_in_the_plume_area`
- Removed trailing dead code block after `return test` in `Check_if_the_area_is_too_cloudy`
- Removed commented-out alternatives and two visualization blocks in `fast_delimitation_of_a_river_plume_area`
- Removed commented-out `final_close_river_mouth_area` line and superseded CSV-saving block in `main_process`
- Removed eight commented-out variable assignments in `determine_SPM_threshold`
- Removed four commented-out variable assignments in `do_a_raw_plume_detection`
- Removed six commented-out variable assignments and a visualization block in `remove_close_river_mouth`
- Removed one commented-out alternative inside `remove_parts_of_the_plume_area_with_very_high_SPM_on_the_edge_of_the_searching_zone`
- Removed debug lines (`SPM_threshold = 10`, `plume_mask.plot()`, `close_river_mouth_mask.plot()`, `protocol`) and two commented-out pipeline-step calls in `Pipeline_to_delineate_the_plume`

---

### Removed EASTERN_CHANNEL zone references

`EASTERN_CHANNEL` was listed as an available zone in the error message of `define_parameters` and in the unit test suite, but was never implemented. It has been removed from three locations:

- [src/panache/utils.py](src/panache/utils.py) — removed from the unavailable-zone error message
- [testing/test_searching_strategy_presets.py](testing/test_searching_strategy_presets.py) — removed from the list of zones exercised by `test_built_in_zones_use_named_presets`
- [CLAUDE.md](CLAUDE.md) — removed from the list of currently defined zone presets