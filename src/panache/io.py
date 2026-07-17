from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_VARIABLE_CANDIDATES = (
    "analysed_spim",
    "analysed_spm",
    "suspended_particulate_matter",
    "SPM",
    "spm",
)


class NoValidMapDataError(ValueError):
    """Raised when an input map contains no finite data values."""


def infer_primary_variable(dataset: xr.Dataset, variable_name: str | None = None) -> str:
    if variable_name:
        if variable_name not in dataset.data_vars:
            raise KeyError(f"Variable '{variable_name}' not found in dataset.")
        return variable_name

    for candidate in DEFAULT_VARIABLE_CANDIDATES:
        if candidate in dataset.data_vars:
            return candidate

    ranked = []
    for name, data in dataset.data_vars.items():
        score = 0
        if {"lat", "lon"}.issubset(data.dims):
            score += 10
        if "time" in data.dims:
            score += 3
        if np.issubdtype(data.dtype, np.number):
            score += 1
        ranked.append((score, name))

    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] <= 0:
        raise ValueError("Could not infer the primary geophysical variable from the NetCDF file.")
    return ranked[0][1]


def _extract_date_for_plot(data_array: xr.DataArray, source_path: str) -> pd.Timestamp:
    # Filename date (YYYYMMDD) is tried first: it is always authoritative for
    # SEXTANT and similar products whose NetCDF time-unit reference epoch can
    # be corrupt (e.g. SEXTANT 2005 files carry "seconds since 1981-01-01"
    # instead of "seconds since 1998-01-01", which shifts decoded dates by
    # 17 years).
    match = re.search(r"(\d{8})", Path(source_path).name)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d")

    if "time" in data_array.coords and data_array.coords["time"].size:
        return pd.to_datetime(data_array.coords["time"].values[0])

    attrs = data_array.attrs
    if "start_date" in attrs:
        date_text = str(attrs["start_date"]).replace(" UTC", "")
        time_text = str(attrs.get("start_time", "00:00:00 UTC")).replace(" UTC", "")
        return pd.to_datetime(f"{date_text} {time_text}", errors="coerce")

    return pd.Timestamp(Path(source_path).stat().st_mtime, unit="s")


def normalize_map_data(data_array: xr.DataArray, source_path: str) -> xr.DataArray:
    plot_date = _extract_date_for_plot(data_array, source_path)

    if "time" in data_array.dims:
        data_array = data_array.isel(time=0, drop=True)

    data_array = data_array.squeeze(drop=True).sortby("lat").sortby("lon")
    data_array = data_array.where(data_array >= 0, np.nan)
    data_array = data_array.assign_coords(date_for_plot=plot_date)
    return data_array


def ensure_valid_map_data(data_array: xr.DataArray, source_path: str | Path) -> xr.DataArray:
    if data_array.size == 0 or not np.isfinite(data_array.values).any():
        raise NoValidMapDataError(f"{source_path} contains no finite data values.")
    return data_array


def load_map_data(path: str | Path, lon_range: tuple[float, float], lat_range: tuple[float, float], variable_name: str | None = None, ) -> xr.DataArray:
    path = Path(path)

    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            data = pickle.load(handle)

        if "Basin_map" in data:
            data = data["Basin_map"]
        if "map_data" not in data:
            raise KeyError(f"Pickle file {path} does not contain a 'map_data' entry.")
        return ensure_valid_map_data(data["map_data"], path)

    if path.suffix not in {".nc", ".nc4", ".cdf"}:
        raise ValueError(f"Unsupported file type for {path}")

    with xr.open_dataset(path) as dataset:
        variable = infer_primary_variable(dataset, variable_name=variable_name)
        data_array = dataset[variable].sel(lon=slice(lon_range[0], lon_range[1]), lat=slice(lat_range[0], lat_range[1])).load()

    return ensure_valid_map_data(normalize_map_data(data_array, str(path)), path)
