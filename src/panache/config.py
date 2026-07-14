from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .utils import define_parameters, searching_strategy_directions_from_presets


REQUIRED_PARAMETER_KEYS = {
    "lon_new_resolution",
    "lat_new_resolution",
    "searching_strategies",
    "bathymetric_threshold",
    "starting_points",
    "core_of_the_plumes",
    "lat_range_of_plume_area",
    "lon_range_of_plume_area",
    "threshold_of_cloud_coverage_in_percentage",
    "maximal_bathymetric_for_zone_with_resuspension",
    "minimal_distance_from_estuary_for_zone_with_resuspension",
    "max_steps_for_the_directions",
    "maximal_threshold",
    "minimal_threshold",
    "quantile_to_use",
    "fixed_threshold",
    "river_mouth_to_exclude",
}


@dataclass
class RunConfig:
    input_path: str
    bathymetry_path: Path
    output_dir: Path
    overwrite: bool
    gif: bool
    nb_cores: int = 1
    dynamic_threshold: bool = False
    annual_map_path: Path | None = None
    coast_shapefile: Path | None = None
    variable_name: str | None = None
    zone: str | None = None
    parameters: dict | None = None
    spm_threshold: float | None = None
    global_threshold_quantile: float | None = None


def _required_bool(data: dict, key: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(f"'{key}' must be a JSON boolean: true or false.")
    return value


def _normalize_searching_strategies(searching_strategies: dict) -> dict:
    if not isinstance(searching_strategies, Mapping):
        raise TypeError(
            "searching_strategies must be a mapping from plume name to preset name."
        )
    return dict(searching_strategies)


def _normalize_coordinate_map(values: dict) -> dict:
    return {key: tuple(value) for key, value in values.items()}


def build_parameters(raw_parameters: dict) -> dict:
    missing = REQUIRED_PARAMETER_KEYS - set(raw_parameters)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required parameter keys: {missing_list}")

    parameters = dict(raw_parameters)
    parameters["searching_strategies"] = _normalize_searching_strategies(parameters["searching_strategies"])
    parameters["starting_points"] = _normalize_coordinate_map(parameters["starting_points"])
    parameters["core_of_the_plumes"] = _normalize_coordinate_map(parameters["core_of_the_plumes"])
    parameters["river_mouth_to_exclude"] = _normalize_coordinate_map(parameters["river_mouth_to_exclude"])
    parameters["searching_strategy_directions"] = searching_strategy_directions_from_presets(
        parameters["searching_strategies"]
    )
    return parameters


def load_run_config(config_path: str | Path) -> RunConfig:
    config_path = Path(config_path)
    data = json.loads(config_path.read_text())

    zone = data.get("zone")
    raw_parameters = data.get("parameters")
    if zone and raw_parameters:
        raise ValueError("Use either 'zone' or 'parameters', not both.")
    if not zone and not raw_parameters:
        raise ValueError("A config must define either 'zone' or 'parameters'.")

    parameters = define_parameters(zone) if zone else build_parameters(raw_parameters)
    if parameters is None:
        raise ValueError(f"Unknown zone preset: {zone}")

    raw_spm_threshold = data.get("spm_threshold")
    raw_quantile = data.get("global_threshold_quantile")
    if raw_spm_threshold is not None and raw_quantile is not None:
        raise ValueError("Use either 'spm_threshold' or 'global_threshold_quantile', not both.")
    if raw_quantile is not None and not (0.0 < raw_quantile < 1.0):
        raise ValueError("'global_threshold_quantile' must be a float strictly between 0 and 1.")

    return RunConfig(
        input_path=data["input_path"],
        bathymetry_path=Path(data["bathymetry_path"]),
        output_dir=Path(data["output_dir"]),
        overwrite=_required_bool(data, "overwrite"),
        gif=_required_bool(data, "gif"),
        nb_cores=int(data.get("nb_cores", 1)),
        dynamic_threshold=bool(data.get("dynamic_threshold", False)),
        annual_map_path=Path(data["annual_map_path"]) if data.get("annual_map_path") else None,
        coast_shapefile=Path(data["coast_shapefile"]) if data.get("coast_shapefile") else None,
        variable_name=data.get("variable_name"),
        zone=zone,
        parameters=parameters,
        spm_threshold=float(raw_spm_threshold) if raw_spm_threshold is not None else None,
        global_threshold_quantile=float(raw_quantile) if raw_quantile is not None else None,
    )
