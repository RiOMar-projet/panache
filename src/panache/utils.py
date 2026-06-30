#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import pickle
import datetime
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
from itertools import product
from collections.abc import Mapping, Iterable

try:
    import bathyreq
except ImportError:  # Only needed when bathymetry must be downloaded on the fly.
    bathyreq = None

try:
    import geopandas as gpd
except ImportError:  # Only needed when shapefile data is requested.
    gpd = None

proj_dir = os.path.dirname( os.path.abspath('__file__') )


# Direction tuples are (lat_index_delta, lon_index_delta). Loaded maps are
# sorted by increasing latitude and longitude, so positive deltas point north/east.
SEARCHING_STRATEGY_PRESETS = {
    "northward_fan": [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1)],
    "southward_fan": [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)],
    "eastward_fan": [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)],
    "westward_fan": [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)],
}


def searching_strategy_directions_from_presets(searching_strategies):
    
    """
    Resolve named search strategy presets into relative pixel directions.

    Parameters
    ----------
    searching_strategies : Mapping[str, str]
        A mapping from plume name to one of the available preset names.

    Returns
    -------
    dict
        A dictionary where each plume name maps to the relative pixel directions
        used by the plume algorithm.
    """
    
    if not isinstance(searching_strategies, Mapping):
        raise TypeError(
            "searching_strategies must be a mapping from plume name to preset name."
        )
    
    available_presets = ", ".join(sorted(SEARCHING_STRATEGY_PRESETS))
    directions = {}
    
    for plume_name, preset_name in searching_strategies.items():
        if not isinstance(preset_name, str):
            raise TypeError(
                f"Search strategy for {plume_name!r} must be one of: {available_presets}."
            )
        
        if preset_name not in SEARCHING_STRATEGY_PRESETS:
            raise ValueError(
                f"Unknown search strategy preset {preset_name!r} for {plume_name!r}. "
                f"Available presets: {available_presets}."
            )
        
        directions[plume_name] = list(SEARCHING_STRATEGY_PRESETS[preset_name])
    
    return directions


def coordinate_range_bounds(coordinate_range):
    values = list(coordinate_range)
    if len(values) < 2:
        raise ValueError("Coordinate ranges must contain at least two values.")
    if len(values) > 2:
        return (min(values), max(values))
    return tuple(values)


def load_file(file_name):
    
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def expand_grid(**kwargs):
    
    """
    Create a DataFrame from the Cartesian product of input arrays.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments where keys are column names and values are arrays.

    Returns
    -------
    pandas.DataFrame
        DataFrame representing the Cartesian product of input arrays.
    """
    
    # Compute the Cartesian product of input values.
    rows = product(*kwargs.values())
    return pd.DataFrame(rows, columns=kwargs.keys())


def load_bathymetric_data(path_to_bathy_data, min_lon, max_lon, min_lat, max_lat) : 
    
    # If the bathymetric data doesn't exist, request it and save it
    if not os.path.exists( path_to_bathy_data ) : 
        if bathyreq is None:
            raise ImportError(
                "bathyreq is required to download bathymetry automatically; "
                "provide an existing bathymetry pickle instead."
            )
        
        req = bathyreq.BathyRequest() # Create a bathymetric data request
        data, lonvec, latvec = req.get_area(longitude=[min_lon, max_lon], 
                                            latitude=[min_lat, max_lat])
        bathymetric_data = xr.DataArray(data, coords=[latvec[::-1], lonvec], dims=['lat', 'lon']) # Create a data array for bathymetry
        
        # Save the bathymetric data for future use
        bathy_path = Path(path_to_bathy_data)
        bathy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bathy_path, 'wb') as f:
            pickle.dump(bathymetric_data, f)
        
        # bathymetric_data.plot()
    else : 
        
        # Load the pre-saved bathymetric data
        with open(path_to_bathy_data, 'rb') as f:
            bathymetric_data = pickle.load(f)
            
    return bathymetric_data


def align_bathymetry_to_resolution(dataset, path_to_bathy_data) : 
               
    """
    Align bathymetric data to the resolution of the input dataset.

    Parameters
    ----------
    dataset : xarray.DataArray
        The input dataset to which the bathymetry should be aligned.
    parameters : dict
        Configuration parameters for plume detection.
    path_to_bathy_data : str
        Path to the raw bathymetric map of the Zone (e.g. f'{work_dir}/RESULTS/{Zone}/Bathy_data.pkl').
        

    Returns
    -------
    xarray.DataArray
        The bathymetric data aligned to the input dataset's resolution.
    """
        
    bathymetric_data = load_bathymetric_data(path_to_bathy_data, 
                                             min_lon = np.min(dataset.lon)-1, max_lon = np.max(dataset.lon)+1, 
                                             min_lat = np.min(dataset.lat)-1, max_lat = np.max(dataset.lat)+1)    
               
    # Align the bathymetric data to the reduced resolution dataset
    bathymetry_data_aligned_to_reduced_map = bathymetric_data.interp(lat = dataset.lat, lon = dataset.lon)
    
    return bathymetry_data_aligned_to_reduced_map


def check_time_format(time_str):

    # Regular expression pattern for HH:MM:SS format
    time_pattern = r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9] UTC$'
    
    # Check if the time string matches the pattern
    if re.match(time_pattern, time_str):
        return time_str
    else:
        return np.nan


def flatten_a_list(lst):
    
    """
    Flatten a nested list into a single list.

    Parameters
    ----------
    lst : list
        A potentially nested list of elements.

    Returns
    -------
    list
        A flattened list containing all elements from the nested list.
    """
    
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_a_list(item))  # Recursive call for nested lists
        else:
            flat_list.append(item)
    return flat_list


def extract_time_from_nc_file(map_data) : 
        
    if 'image_reference_time' in map_data.attrs : # For SEXTANT products
        time = map_data._attrs['image_reference_time']
    elif 'DSD_entry_id' in map_data.attrs and 'L4' in map_data._attrs['DSD_entry_id'] : # For SEXTANT merged products
        time = ""
    elif 'start_time' in map_data.attrs :  # For ODATIS products
        time = pd.to_datetime(map_data.attrs['start_time']).strftime('%H:%M:%S UTC')
    elif 'time' in map_data.attrs :  # For ODATIS products    
        time = map_data.attrs['time']
        
    time = check_time_format(time)
    
    return time


def extract_dataframes_iterative(data):
    """Efficiently extract all DataFrames from a nested dictionary using an iterative approach."""
    stack = [data]  # Use a stack to avoid deep recursion

    while stack:
        current = stack.pop()

        if isinstance(current, pd.DataFrame):
            yield current  # Yield instead of appending to a list (memory-efficient)
        elif isinstance(current, Mapping):  # Check if it's a dictionary
            stack.extend(current.values())  # Add dictionary values to the stack
        elif isinstance(current, Iterable) and not isinstance(current, (str, bytes)):  
            stack.extend(current)  # Add list/tuple elements to the stack


def unique_years_between_two_dates(start_date: str, end_date: str):
    start_year = datetime.datetime.strptime(start_date, "%Y/%m/%d").year
    end_year = datetime.datetime.strptime(end_date, "%Y/%m/%d").year
    return list(range(start_year, end_year + 1))


def define_parameters(Zone) : 
    
    """
    Define region-specific parameters based on the selected zone.

    Parameters
    ----------
    Zone : str
        Name of the geographic zone (e.g., "BAY_OF_SEINE", "BAY_OF_BISCAY").

    Returns
    -------
    dict
        A dictionary containing the parameters for the specified zone.
    """

    if not isinstance(Zone, str):
        print("Zone must be a string.")
        return None

    if Zone == 'BAY_OF_SEINE' :        
        lon_new_resolution = 0.015
        lat_new_resolution = 0.015
        searching_strategies = {'Seine': 'westward_fan'}
        bathymetric_threshold = 0
        starting_points = {'Seine' : (49.43, 0.145)}
        core_of_the_plumes = {'Seine' : (49.43, 0)}
        lat_range_of_plume_area = [49.25, 50.25]
        lon_range_of_plume_area = [-1.5, 0.5]
        threshold_of_cloud_coverage_in_percentage = 25
        maximal_bathymetric_for_zone_with_resuspension = {'Seine' : 30}
        minimal_distance_from_estuary_for_zone_with_resuspension = {'Seine' : 30}
        max_steps_for_the_directions = {'Seine' : 40}
        maximal_threshold = {'Seine' : 11} # 15
        minimal_threshold = {'Seine' : 7} # 4
        quantile_to_use = {'Seine' : 0.10}
        fixed_threshold = {'Seine' : 9.5}
        river_mouth_to_exclude = {'Canal de Caen à la mer' : [49.296, -0.245]}
        
    elif Zone == 'BAY_OF_BISCAY' :        
        lon_new_resolution = 0.015
        lat_new_resolution = 0.015
        searching_strategies = {
            'Gironde': 'westward_fan',
            'Charente': 'westward_fan',
            'Sevre': 'westward_fan',
        }
        bathymetric_threshold = 0
        starting_points = {'Gironde' : (45.59, -1.05),
                          'Charente' : (45.96, -1.01),
                          'Sevre' : (46.30, -1.13)}
        core_of_the_plumes = {'Gironde' : (45.59, -1.05),
                              # 'Gironde' : (45.65, -1.33),
                              'Charente' : (45.98, -1.17),
                              'Sevre' : (46.24, -1.24)}
        lat_range_of_plume_area = [44.5, 46.5]
        lon_range_of_plume_area = [-4, -0.5]
        threshold_of_cloud_coverage_in_percentage = 25
        maximal_bathymetric_for_zone_with_resuspension = {'Gironde' : 20, 'Charente' : 20, 'Sevre' : 20}
        minimal_distance_from_estuary_for_zone_with_resuspension = {'Gironde' : 30, 'Charente' : 20, 'Sevre' : 20}
        max_steps_for_the_directions = {'Gironde' : 100, 'Charente' : 50, 'Sevre' : 50}
        maximal_threshold = {'Gironde' : 8, 'Charente' : 10, 'Sevre' : 8} 
        minimal_threshold = {'Gironde' : 4, 'Charente' : 6, 'Sevre' : 4} 
        quantile_to_use = {'Gironde' : 0.2, 'Charente' : 0.2, 'Sevre' : 0.2} 
        fixed_threshold = {'Gironde' : 4.7, 'Charente' : 7.8, 'Sevre' : 5.2} 
        river_mouth_to_exclude = {}
    
    elif Zone == 'GULF_OF_LION' :        
        lon_new_resolution = 0.015
        lat_new_resolution = 0.015
        searching_strategies = {
            'Grand Rhone': 'southward_fan',
            'Petit Rhone': 'southward_fan',
        }
        bathymetric_threshold = 20
        starting_points = {'Grand Rhone' : (43.41, 4.83),
                           'Petit Rhone' : (43.47, 4.39)}
        core_of_the_plumes = {'Grand Rhone' : (43.32, 4.85),
                              'Petit Rhone' : (43.43, 4.39)}
        lat_range_of_plume_area = [41, 44]
        lon_range_of_plume_area = [3, 6]
        threshold_of_cloud_coverage_in_percentage = 25
        maximal_bathymetric_for_zone_with_resuspension = {'Grand Rhone' : 30, 'Petit Rhone' : 30}
        minimal_distance_from_estuary_for_zone_with_resuspension = {'Grand Rhone' : 30, 'Petit Rhone' : 30}
        max_steps_for_the_directions = {'Grand Rhone' : 35, 'Petit Rhone' : 35}
        maximal_threshold = {'Grand Rhone' : 3, 'Petit Rhone' : 3} # 3
        minimal_threshold = {'Grand Rhone' : 0.75, 'Petit Rhone' : 1} # 0.75
        quantile_to_use = {'Grand Rhone' : 0.2, 'Petit Rhone' : 0.2}
        fixed_threshold = {'Grand Rhone' : 2.1, 'Petit Rhone' : 2.1} 
        river_mouth_to_exclude = {}
      
    elif Zone == 'SOUTHERN_BRITTANY':         
        lon_new_resolution = 0.015
        lat_new_resolution = 0.015
        searching_strategies = {
            'Loire': 'westward_fan',
            'Vilaine': 'westward_fan',
        }
        bathymetric_threshold = 0
        starting_points = {'Loire' : (47.29, -2.10),
                           'Vilaine' : (47.50, -2.46)}
        core_of_the_plumes = {'Loire' : (47.19, -2.36),
                              'Vilaine' : (47.47, -2.59)}
        lat_range_of_plume_area = [46.5, 48]
        lon_range_of_plume_area = [-5, -1.5]
        threshold_of_cloud_coverage_in_percentage = 25
        maximal_bathymetric_for_zone_with_resuspension = {'Loire' : 20, 'Vilaine' : 20}
        minimal_distance_from_estuary_for_zone_with_resuspension = {'Loire' : 20, 'Vilaine' : 20}
        max_steps_for_the_directions = { 'Loire' : 200, 'Vilaine' : 50}
        maximal_threshold = { 'Loire' : 8, 'Vilaine' : 8} # 12
        minimal_threshold = { 'Loire' : 4, 'Vilaine' : 4} # 3
        quantile_to_use = { 'Loire' : 0.2, 'Vilaine' : 0.2}
        fixed_threshold = {'Loire' : 5.4, 'Vilaine' : 5.0} 
        river_mouth_to_exclude = {}

    else :
        print(f"The zone {Zone} is not available. Please select one of the following zones : 'BAY_OF_SEINE', 'BAY_OF_BISCAY', 'GULF_OF_LION', 'EASTERN_CHANNEL', 'SOUTHERN_BRITTANY'.")
        return None
    
    searching_strategy_directions = searching_strategy_directions_from_presets(searching_strategies)
    
    return {
        'lon_new_resolution' : lon_new_resolution, 
        'lat_new_resolution' : lat_new_resolution, 
        'searching_strategies' : searching_strategies, 
        'bathymetric_threshold' : bathymetric_threshold, 
        'starting_points' : starting_points, 
        'core_of_the_plumes' : core_of_the_plumes,
        'lat_range_of_plume_area' : lat_range_of_plume_area, 
        'lon_range_of_plume_area' : lon_range_of_plume_area, 
        'threshold_of_cloud_coverage_in_percentage' : threshold_of_cloud_coverage_in_percentage,
        'maximal_bathymetric_for_zone_with_resuspension' : maximal_bathymetric_for_zone_with_resuspension,
        'minimal_distance_from_estuary_for_zone_with_resuspension' : minimal_distance_from_estuary_for_zone_with_resuspension,
        'max_steps_for_the_directions' : max_steps_for_the_directions,
        'maximal_threshold' : maximal_threshold,
        'minimal_threshold' : minimal_threshold,
        'quantile_to_use' : quantile_to_use,
        'fixed_threshold' : fixed_threshold,
        'river_mouth_to_exclude' : river_mouth_to_exclude,
        'searching_strategy_directions' : searching_strategy_directions
    }
