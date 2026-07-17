# Workflow Vignette

This page shows what a complete `panache` run looks like from config file to
output, using one month of real L4 SPM satellite data over the Gulf of Lion as
the primary example. Outputs from three additional zones are shown at the end
to illustrate the range of plume geometries the algorithm handles.

---

## About vignette images and PyPI

Documentation vignettes for Python packages **do not need to compile on the
user's system.** The images and statistics shown here are pre-rendered on the
authors' machines and committed to the repository as static assets. When
Sphinx builds the documentation site, it simply embeds them. Users can read
the vignette without having satellite data or a bathymetry file available
locally.

This is standard practice for scientific Python packages whose inputs are
large, proprietary, or geographically specific datasets (e.g. xarray, cartopy,
ESMValTool). The recommended workflow for adding your own outputs to a vignette
is described in the [Adding your own outputs](#adding-your-own-outputs) section
below.

---

## 1. Prepare a config file

Create a JSON config that points `panache` at your data. The simplest approach
is to use one of the four built-in zone presets:

```json
{
  "zone": "GULF_OF_LION",
  "input_path": "/data/SEXTANT/SPM/",
  "bathymetry_path": "/data/bathymetry/Bathy_data.pkl",
  "output_dir": "/data/panache-output/GULF_OF_LION",
  "overwrite": false,
  "gif": true,
  "nb_cores": 4,
  "dynamic_threshold": false,
  "variable_name": "analysed_spim",
  "coast_shapefile": "/data/boundaries/gadm41_FRA_0.shp"
}
```

Key choices:

| Key | What to set |
|---|---|
| `zone` | One of `GULF_OF_LION`, `BAY_OF_SEINE`, `BAY_OF_BISCAY`, `SOUTHERN_BRITTANY` |
| `input_path` | Path to a folder of `.nc` files, a glob string, or a single file |
| `bathymetry_path` | Path to the bathymetry pickle. If it does not exist, `panache` downloads and creates it automatically. |
| `dynamic_threshold` | `true` for per-scene gradient-based threshold; `false` to use `fixed_threshold` from the zone preset |
| `gif` | `true` to assemble an animated GIF at the end of the run |

---

## 2. Run the pipeline

```bash
panache gulf_of_lion_config.json
```

`panache` prints a progress line for each processed file and reports the path
to `Results.csv` when the run completes. Multi-core processing (`"nb_cores": 4`)
is used by default and speeds up large batches substantially.

Example terminal output:

```
Processing: 20191001-EUR-L4-SPIM-ATL-v01-fv01-OI.nc  [1/31]
Processing: 20191002-EUR-L4-SPIM-ATL-v01-fv01-OI.nc  [2/31]
...
Processing: 20191031-EUR-L4-SPIM-ATL-v01-fv01-OI.nc  [31/31]
Results written to: /data/panache-output/GULF_OF_LION/Results.csv
```

---

## 3. Examine the outputs

After the run, the output directory contains:

```
GULF_OF_LION/
├── Results.csv       ← batch statistics, one row per day
├── manifest.csv      ← status of every input file (Completed / Skipped / Failed)
├── GIF.gif           ← animated overview of the full month
└── MAPS/
    ├── 01/20191001-EUR-L4-SPIM-ATL-v01-fv01-OI.png
    ├── 02/20191002-EUR-L4-SPIM-ATL-v01-fv01-OI.png
    └── ...
```

### Daily plume map

Each PNG shows two panels side by side: the raw SPM field on the left and the
detected plume mask (red overlay) on the right. Black `+` markers indicate the
user-defined river-mouth starting points; grey `×` markers indicate the plume
core coordinates used for blob selection and the resuspension filter.

**Gulf of Lion — 22 October 2019**
(Grand Rhône and Petit Rhône plumes; fixed threshold mode)

```{image} _static/images/gulf_of_lion_example.png
:alt: Gulf of Lion plume detection — 22 October 2019
:width: 100%
```

The left panel shows the raw SPM field in g m⁻³ on a log scale. The right
panel shows the detected plume area in red, masked to pixels shallower than
20 m depth and above the SPM threshold, with land and very-shallow coastal
pixels excluded by the bathymetry mask.

### `Results.csv` — batch statistics

`Results.csv` contains one row per successfully processed file. The columns
reported for the Gulf of Lion run are shown below (first three days):

| date | area km² | mean SPM g m⁻³ | cloud cover % | threshold Grand Rhône | threshold Petit Rhône |
|---|---|---|---|---|---|
| 2019-10-01 | 196 | 2.34 | 0 | 1.48 | 3.54 |
| 2019-10-02 | 294 | 2.53 | 0 | 1.53 | 1.63 |
| 2019-10-03 | 402 | 2.96 | 0 | 1.49 | 1.68 |

All columns: `date`, `n_pixel_in_the_plume_area`, `area_of_the_plume_mask_in_km2`,
`mean_SPM_in_the_plume_area`, `sd_SPM_in_the_plume_area`,
`mass_SPM_in_the_plume_area_in_g_m`, `lat_centroid_of_the_plume_area`,
`lon_centroid_of_the_plume_area`, `lat_weighted_centroid_of_the_plume_area`,
`lon_weighted_centroid_of_the_plume_area`, `confidence_index_in_perc`,
and one `SPM_threshold_<name>` column per river mouth.

---

## 4. Outputs from other zones

### Bay of Seine — December 2000

The Seine plume is detected along the Normandy coast. The bathymetric mask
excludes the very-shallow tidal flats near the Canal de Caen, which would
otherwise dominate the signal.

```{image} _static/images/bay_of_seine_example.png
:alt: Bay of Seine plume detection — 9 December 2000
:width: 100%
```

### Bay of Biscay — October 2019

Three river mouths are tracked simultaneously: the Gironde, the Charente,
and the Sèvre Niortaise. The large Gironde plume dominates the northern
shelf but the algorithm correctly separates the three contributions.

```{image} _static/images/bay_of_biscay_example.png
:alt: Bay of Biscay plume detection — 15 October 2019
:width: 100%
```

### Var estuary — October 2019

The Var is a small, steep-gradient Mediterranean river. Its plume footprint
is much smaller than Atlantic estuaries, and the detection window is
correspondingly compact. The Paillon estuary is tracked as a second river
mouth within the same zone.

```{image} _static/images/var_example.png
:alt: Var plume detection — 9 October 2019
:width: 100%
```

---

## Adding your own outputs

To add pre-rendered outputs from your own runs to the documentation:

1. Run `panache` on your data and locate the output PNGs in `<output_dir>/MAPS/`.
2. Copy representative images to `docs/_static/images/`.
3. Reference them in `docs/vignette.md` with:
   ````
   ```{image} _static/images/my_example.png
   :alt: Description of the image
   :width: 100%
   ```
   ````
4. Rebuild the docs locally with `sphinx-build -b html docs docs/_build/html`
   and verify the page looks right.
5. Commit the images and the updated `vignette.md` and push to `main`. The
   GitHub Actions workflow will redeploy the site automatically.
