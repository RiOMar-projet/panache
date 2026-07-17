# Workflow Vignette

This page shows what a complete `panache` run looks like from config file to
output, and demonstrates how plume detection varies across seasons and
threshold methods. All images were produced from real L4 SPM satellite data
(EUR-L4-SPIM-ATL-v01-fv01-OI) for **1 January 2024** (winter) and
**1 July 2024** (summer). Each zone is shown under both the **dynamic
gradient-based threshold** and the **global 95th-percentile (p95) threshold**,
so the sensitivity difference between the two methods is immediately visible.

---

## About vignette images and PyPI

Documentation vignettes for Python packages **do not need to compile on the
user's system.** The images shown here are pre-rendered on the authors'
machines and committed to the repository as static assets. When Sphinx builds
the documentation site, it simply embeds them. Users can read the vignette
without having satellite data or a bathymetry file available locally.

This is standard practice for scientific Python packages whose inputs are
large or geographically specific datasets (e.g. xarray, cartopy, ESMValTool).
The [Adding your own outputs](#adding-your-own-outputs) section below describes
how to add your own pre-rendered maps to this page.

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
  "coast_shapefile": "/data/boundaries/gadm41_FRA_0.shp",
  "overwrite": false,
  "gif": true,
  "nb_cores": 4,
  "dynamic_threshold": false,
  "global_threshold_quantile": 0.95,
  "variable_name": "analysed_spim"
}
```

Key choices:

| Key | What to set |
|---|---|
| `zone` | One of `GULF_OF_LION`, `BAY_OF_SEINE`, `BAY_OF_BISCAY`, `SOUTHERN_BRITTANY` |
| `input_path` | Path to a folder of `.nc` files, a glob string, or a single file |
| `bathymetry_path` | Path to the bathymetry pickle; downloaded automatically if absent |
| `dynamic_threshold` | `true` for per-scene gradient-based threshold; `false` to use `global_threshold_quantile` or zone `fixed_threshold` |
| `global_threshold_quantile` | Compute this percentile from the full scene and use it as the flood-fill threshold |
| `gif` | `true` to assemble an animated GIF at the end of the run |

---

## 2. Run the pipeline

```bash
panache gulf_of_lion_config.json
```

`panache` prints a progress line per file and reports the path to `Results.csv`
when the run completes.

---

## 3. Examine the outputs

```
GULF_OF_LION/
├── Results.csv       ← batch statistics, one row per day
├── manifest.csv      ← status of every input file
├── GIF.gif           ← animated overview (when gif: true)
└── MAPS/
    └── 20240101-EUR-L4-SPIM-ATL-v01-fv01-OI.png
```

Each PNG shows two panels: the raw SPM field (left) and the detected plume
mask in red (right). Black `+` markers mark the river-mouth starting points;
grey `×` markers show the plume core coordinates.

---

## 4. Zone outputs — winter vs summer, dynamic vs p95

The four panels for each zone share the same colour scale. The plume mask
threshold (shown in the right-panel title) differs between the dynamic and p95
runs, illustrating how threshold choice affects the detected plume area.

---

### Bay of Seine

The Seine estuary opens onto the eastern English Channel. Winter brings high
river discharge and elevated SPM; summer discharge is lower and the plume
footprint contracts. The bathymetric mask excludes very-shallow tidal flats
near the Canal de Caen.

**1 January 2024 — Winter**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/bay_of_seine_20240101_dynamic.png
:alt: Bay of Seine — 1 Jan 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/bay_of_seine_20240101_p95.png
:alt: Bay of Seine — 1 Jan 2024 — p95 threshold
:width: 100%
```
:::
::::

**1 July 2024 — Summer**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/bay_of_seine_20240701_dynamic.png
:alt: Bay of Seine — 1 Jul 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/bay_of_seine_20240701_p95.png
:alt: Bay of Seine — 1 Jul 2024 — p95 threshold
:width: 100%
```
:::
::::

---

### Bay of Biscay

Three river mouths are tracked simultaneously: the Gironde, the Charente, and
the Sèvre Niortaise. The large Gironde plume typically dominates the northern
shelf; winter cloud cover (shown as white patches in the left panel) can
partially obscure the signal.

**1 January 2024 — Winter**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/bay_of_biscay_20240101_dynamic.png
:alt: Bay of Biscay — 1 Jan 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/bay_of_biscay_20240101_p95.png
:alt: Bay of Biscay — 1 Jan 2024 — p95 threshold
:width: 100%
```
:::
::::

**1 July 2024 — Summer**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/bay_of_biscay_20240701_dynamic.png
:alt: Bay of Biscay — 1 Jul 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/bay_of_biscay_20240701_p95.png
:alt: Bay of Biscay — 1 Jul 2024 — p95 threshold
:width: 100%
```
:::
::::

---

### Gulf of Lion

The Grand Rhône and Petit Rhône are tracked as two separate river mouths. The
Mediterranean receives less riverine input in summer, and the plume footprint
is typically smaller and more confined to the immediate coastal zone. The
dynamic threshold adapts per plume per scene; the p95 threshold is a single
value computed from the full bounding box.

**1 January 2024 — Winter**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/gulf_of_lion_20240101_dynamic.png
:alt: Gulf of Lion — 1 Jan 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/gulf_of_lion_20240101_p95.png
:alt: Gulf of Lion — 1 Jan 2024 — p95 threshold
:width: 100%
```
:::
::::

**1 July 2024 — Summer**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/gulf_of_lion_20240701_dynamic.png
:alt: Gulf of Lion — 1 Jul 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/gulf_of_lion_20240701_p95.png
:alt: Gulf of Lion — 1 Jul 2024 — p95 threshold
:width: 100%
```
:::
::::

---

### Southern Brittany

The Loire and Vilaine are the two tracked mouths. The Loire is one of France's
largest rivers by discharge; the p95 threshold computed from summer data can
sometimes exceed ambient SPM concentrations in low-flow conditions, resulting
in no plume detected — a physically meaningful result rather than an error.

**1 January 2024 — Winter**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/southern_brittany_20240101_dynamic.png
:alt: Southern Brittany — 1 Jan 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/southern_brittany_20240101_p95.png
:alt: Southern Brittany — 1 Jan 2024 — p95 threshold
:width: 100%
```
:::
::::

**1 July 2024 — Summer**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/southern_brittany_20240701_dynamic.png
:alt: Southern Brittany — 1 Jul 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/southern_brittany_20240701_p95.png
:alt: Southern Brittany — 1 Jul 2024 — p95 threshold
:width: 100%
```
:::
::::

---

### Var

The Var is a small, steep-gradient Mediterranean river with a compact plume
window. The Paillon estuary is tracked as a second river mouth in the same
zone using custom `parameters` (rather than a `zone` preset). Because the L4
product at 1 km resolution places only a handful of pixels within the detection
domain, threshold sensitivity is especially pronounced here.

In winter the dynamic threshold can fall very low (≈0.3 g m⁻³), at which point
the entire near-coastal domain exceeds the threshold and the mask floods outward
to the domain boundary. Conversely, the p95 threshold (≈1.3 g m⁻³) may exceed
all pixels near the river mouth, giving no detection. Neither result represents
a detection failure in the software sense — both reflect genuine ambiguity in
the SPM signal on this particular day. In summer the p95 threshold yields a
compact, well-constrained plume shape.

**1 January 2024 — Winter**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/var_20240101_dynamic.png
:alt: Var — 1 Jan 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/var_20240101_p95.png
:alt: Var — 1 Jan 2024 — p95 threshold
:width: 100%
```
:::
::::

**1 July 2024 — Summer**

::::{grid} 2
:::{grid-item-card} Dynamic threshold
```{image} _static/images/vignette/var_20240701_dynamic.png
:alt: Var — 1 Jul 2024 — dynamic threshold
:width: 100%
```
:::
:::{grid-item-card} p95 threshold
```{image} _static/images/vignette/var_20240701_p95.png
:alt: Var — 1 Jul 2024 — p95 threshold
:width: 100%
```
:::
::::

---

## Adding your own outputs

To add pre-rendered outputs from your own runs:

1. Run `panache` on your data and locate the output PNGs in `<output_dir>/MAPS/`.
2. Copy representative images to `docs/_static/images/vignette/`.
3. Reference them in this file using:
   ````
   ```{image} _static/images/vignette/my_example.png
   :alt: Description
   :width: 100%
   ```
   ````
4. Rebuild locally: `sphinx-build -b html docs docs/_build/html`
5. Commit the images and updated markdown, then push to `main`. The GitHub
   Actions workflow redeploys the site automatically.
