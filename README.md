# panache 🌊

**`panache`** is a standalone Python module for detecting river plumes in gridded geophysical data.

📚 Full documentation and API reference: https://RiOMar-projet.github.io/panache/

As input it takes a list of NetCDF files, a bathymetry mask, and a JSON run configuration in order to create plume masks, illustrate per-timestep maps, write summary statistics, and build a quick animated overview of the processed run.

## ✨ Highlights

- 🛰️ Reads NetCDF products directly
- 🌊 Creates a bathymetry mask matching the study area and data resolution
- 🗺️ Uses bathymetry mask for both land masking and cloud-coverage screening
- ⚙️ Runs from one explicit JSON config with no hidden folder assumptions
- 🚀 Supports single-core or multi-core batch processing
- 📊 Produces CSV statistics, plume masks, PNG maps, and an animated GIF
- 🧭 Works with built-in zone presets or fully custom plume parameters

## 📦 Installation

**`panache`** requires Python 3.10 or newer.

NB: It is advised to install the module into a virtual environment rather than into the base Python installation for your machine.

Here is an example of how to create a new virtual environment named 'panache_env':

```bash
# Create and activate a Python virtual environment
python -m venv panache_env
source panache_env/bin/activate  # On Windows: panache_env\Scripts\activate

# Upgrade pip and install the package
pip install --upgrade pip
pip install panache_env
```

Once the desired virtual environment has been activate (or not), the **`panache`** module may be installed with:

```bash
git clone https://github.com/RiOMar-project/panache.git
cd panache
pip install -e .
```

## 🚀 Quick Start

Create a config file, then point the `panache` command at it:

```bash
panache example_zone_config.json
```

The command prints the path to the generated `Results.csv` file when the run completes.

## 🧪 Example Config

```json
{
  "zone": "GULF_OF_LION",
  "input_path": "/data/SEXTANT/SPM/*.nc",
  "bathymetry_path": "/data/bathymetry/Bathy_data.pkl",
  "output_dir": "/data/panache-output",
  "overwrite": false,
  "gif": false,
  "nb_cores": 4,
  "dynamic_threshold": false,
  "variable_name": "analysed_spim",
  "coast_shapefile": "/data/boundaries/gadm41_FRA_0.shp"
}
```

Minimal required fields are:

- `zone` or `parameters`
- `input_path`
- `bathymetry_path`
- `output_dir`
- `overwrite`
- `gif`

## 🗂️ Outputs

Each run writes its results into the configured `output_dir`:

```text
panache-output/
├── Results.csv
├── GIF.gif
└── MAPS/
    ├── [base_file_name]_plume_mask.png
    └── [base_file_name]_plume_mask.csv
    └── [base_file_name]_statistics.csv
```

| Output | Description |
| --- | --- |
| `Results.csv` | Batch-level plume statistics sorted by date. |
| `GIF.gif` | Animated preview assembled from generated plume maps. |
| `MAPS/*.png` | One rendered plume map per processed input file. |
| `MAPS/*_statistics.csv` | Per-scene summary statistics used to rebuild `Results.csv`. |
| `MAPS/*_plume_mask.csv` | Per-scene plume mask data for detected plume outputs. |

## ⚙️ Configuration Modes

### 🧭 Use a Zone Preset

Set `zone` to a known plume-detection preset, such as:

```json
{
  "zone": "GULF_OF_LION"
}
```

### 🛠️ Bring Your Own Parameters

Set `parameters` instead of `zone` when you need a custom detection area, search strategy, bathymetric threshold, plume core, or thresholding behavior.

Search strategies are named presets rather than boolean pixel grids. Set each plume to one of `northward_fan`, `southward_fan`, `eastward_fan`, or `westward_fan`:

```json
{
  "searching_strategies": {
    "Grand Rhone": "southward_fan",
    "Petit Rhone": "southward_fan"
  }
}
```

**`panache`** resolves those preset names into the pixel directions used internally by the plume algorithm.

To see a full example of the rquired arguments, see the example JSON file [here](https://github.com/RiOMar-projet/panache/blob/main/example_parameter_config.json).

Use `lat_range_of_plume_area` and `lon_range_of_plume_area` to define the plume domain used for input subsetting, cloud checks, map extents, and plume masking. Each can be a two-value min/max range, or matching polygon-coordinate lists for polygon plume domains.

### `starting_points` and `core_of_the_plumes`

These two parameters are placed near the same location but serve distinct roles at different stages of the algorithm.

`starting_points` is the **algorithmic seed** for each plume: the pixel where the flood-fill BFS begins and from which all directional gradient transects originate when computing the dynamic threshold. It must be a water pixel situated within or immediately adjacent to the high-SPM signal at the river mouth, so that the flood fill propagates into the plume body rather than into the open ocean.

`core_of_the_plumes` is a **trusted reference coordinate inside the plume body**, used after the raw mask has been created. It serves three purposes: (1) identifying which connected blob in the flood-fill result belongs to the river plume, (2) acting as the estuary centre when computing distances for the resuspension-removal filter, and (3) locating the main shape during the dilation-and-merge step.

In practice the two coordinates are often near-identical, but they may differ when the river mouth is very close to the coastline. In that case `starting_points` is placed right at the mouth (a coastal water pixel), while `core_of_the_plumes` is placed slightly further offshore, within the expected plume footprint, to ensure the shape-selection step reliably identifies the correct blob. See the `BAY_OF_SEINE` preset in `utils.py` for an example: the Seine starting point sits at the coast (0.145°E) while its core is offset 0.15° offshore (0.0°E).

## 🛰️ Input Expectations

- NetCDF inputs should expose `lat` and `lon` coordinates
- Common variable names for SPM are detected automatically
- If a file contains multiple plausible geophysical variables, set `variable_name`
- `coast_shapefile` is optional and only serves to add coastal shapes to the maps and GIF