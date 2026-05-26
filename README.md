# panache 🌊

**`panache`** is a standalone Python module for detecting river plumes in gridded geophysical data.

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
  "input_glob": "/data/SEXTANT/SPM/*.nc",
  "input_root": "/data/SEXTANT/SPM",
  "bathymetry_path": "/data/bathymetry/Bathy_data.pkl",
  "output_dir": "/data/panache-output",
  "nb_cores": 4,
  "dynamic_threshold": false,
  "variable_name": "analysed_spim",
  "coast_shapefile": "/data/boundaries/gadm41_FRA_0.shp"
}
```

Minimal required fields are:

- `zone` or `parameters`
- `input_glob`
- `bathymetry_path`
- `output_dir`

## 🗂️ Outputs

Each run writes its results into the configured `output_dir`:

```text
panache-output/
├── Results.csv
├── GIF.gif
└── MAPS/
    ├── 19980101-EUR-L4-SPIM-ATL-v01-fv01-OI_plume_mask.png
    └── 19980101-EUR-L4-SPIM-ATL-v01-fv01-OI_plume_mask.csv
```

| Output | Description |
| --- | --- |
| `Results.csv` | Batch-level plume statistics sorted by date. |
| `GIF.gif` | Animated preview assembled from generated plume maps. |
| `MAPS/*.png` | One rendered plume map per processed input file. |
| `MAPS/*.csv` | Per-scene plume mask data for detected plume outputs. |

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

Panache resolves those preset names into the pixel directions used internally by the plume algorithm.

Use `lat_range_of_plume_area` and `lon_range_of_plume_area` to define the plume domain used for input subsetting, cloud checks, map extents, and plume masking. Each can be a two-value min/max range, or matching polygon-coordinate lists for polygon plume domains.

## 🛰️ Input Expectations

- NetCDF inputs should expose `lat` and `lon` coordinates
- SEXTANT-style `analysed_spim` variables are detected automatically
- If a file contains multiple plausible geophysical variables, set `variable_name`
- `coast_shapefile` is optional and only serves to add coastal shapes to the maps and GIF