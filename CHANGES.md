# Change log

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