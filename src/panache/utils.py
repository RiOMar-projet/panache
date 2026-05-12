#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
#### Modules
# =============================================================================


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

try:
    from concave_hull import concave_hull
except ImportError:  # Preserve the raw point ordering when the helper is unavailable.
    def concave_hull(points):
        return list(points)

proj_dir = os.path.dirname( os.path.abspath('__file__') )


# =============================================================================
#### Utility functions
# =============================================================================
    

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
        searching_strategies = {'Seine' : {'grid' : np.array([    [False, False, False, False, False],
                                                                  [False, True,  True,  True,  False],
                                                                  [False, True,  True,  False,  False],
                                                                  [False, True,  True,  False,  False],
                                                                  [False, False, False, False, False],
                                                                ]),
                                      'coordinates_of_center' : (2,2)}}
        bathymetric_threshold = 0
        starting_points = {'Seine' : (49.43, 0.145)}
        core_of_the_plumes = {'Seine' : (49.43, 0)}
        lat_range_of_the_area_to_check_for_clouds = [49.25, 49.75]
        lon_range_of_the_area_to_check_for_clouds = [-0.3, 0.3]
        threshold_of_cloud_coverage_in_percentage = 25
        lat_range_of_the_map_to_plot = [49, 50.5] # [49.20, 51.25]
        lon_range_of_the_map_to_plot = [-1.5, 2] # [-1.5, 2.5]
        lat_range_to_search_plume_area = [49.25, 50.25]
        lon_range_to_search_plume_area = [-1.5, 0.5]
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
        searching_strategies = {'Gironde' : {'grid' : np.array([  [False, False, False, False, False],
                                                                  [False, True,  True, True, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, True,  False,  False, False],
                                                                  [False, False, False, False, False],
                                                                ]),
                                      'coordinates_of_center' : (2,2)},
                              
                                  'Charente' : {'grid' : np.array([ [False, False, False, False, False],
                                                                    [False, True,  True, False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, False, False, False, False],
                                                                  ]),
                                                                'coordinates_of_center' : (2,2)},
                                  'Sevre' : {'grid' : np.array([  [False, False, False, False, False],
                                                                  [False, True,  True, False, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, False, False, False, False],
                                                                ]),
                                                                'coordinates_of_center' : (2,2)}}
        bathymetric_threshold = 0
        starting_points = {'Gironde' : (45.59, -1.05),
                          'Charente' : (45.96, -1.01),
                          'Sevre' : (46.30, -1.13)}
        core_of_the_plumes = {'Gironde' : (45.59, -1.05),
                              # 'Gironde' : (45.65, -1.33),
                              'Charente' : (45.98, -1.17),
                              'Sevre' : (46.24, -1.24)}
        lat_range_of_the_area_to_check_for_clouds = [45.5, 46.35]
        lon_range_of_the_area_to_check_for_clouds = [-1.8, -1.2]
        threshold_of_cloud_coverage_in_percentage = 25
        lat_range_of_the_map_to_plot = [45, 46.75] # [44.75, 46.75]
        lon_range_of_the_map_to_plot = [-4, -0.5] # [-4.5, -1]
        lat_range_to_search_plume_area = [44.5, 46.5]
        lon_range_to_search_plume_area = [-4, -0.5]
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
        searching_strategies = {'Grand Rhone' : {'grid' : np.array([  [False, False, False, False, False, False, False],
                                                                      [False, False, False, False, False, False, False],
                                                                      [False, False, False,  True, False, False, False],
                                                                      [True,  True,  True,  True,  True, True, True],
                                                                      [False, False, False, False, False, False, False],
                                                                    ]),
                                      'coordinates_of_center' : (2,3)},
                                
                                'Petit Rhone' : {'grid' : np.array([    [False, False, False, False, False],
                                                                        [False, False, False, False, False],
                                                                        [False, False, True,  False, False],
                                                                        [True,  True,  True,  True,  True],
                                                                        [False, False, False, False, False],
                                                                      ]),
                                            'coordinates_of_center' : (2,2)}}
        bathymetric_threshold = 25
        starting_points = {'Grand Rhone' : (43.41, 4.83),
                           'Petit Rhone' : (43.47, 4.39)}
        core_of_the_plumes = {'Grand Rhone' : (43.32, 4.85),
                              'Petit Rhone' : (43.43, 4.39)}
        lat_range_of_the_area_to_check_for_clouds = [43, 43.4]
        lon_range_of_the_area_to_check_for_clouds = [4.5, 5]
        threshold_of_cloud_coverage_in_percentage = 25
        lat_range_of_the_map_to_plot = [42.25, 44] # [42, 43.7]
        lon_range_of_the_map_to_plot = [3, 6] # [2.75, 6.55]
        lat_range_to_search_plume_area = [41, 44]
        lon_range_to_search_plume_area = [3, 6]
        maximal_bathymetric_for_zone_with_resuspension = {'Grand Rhone' : 30, 'Petit Rhone' : 30}
        minimal_distance_from_estuary_for_zone_with_resuspension = {'Grand Rhone' : 30, 'Petit Rhone' : 30}
        max_steps_for_the_directions = {'Grand Rhone' : 35, 'Petit Rhone' : 35}
        maximal_threshold = {'Grand Rhone' : 3, 'Petit Rhone' : 3} # 3
        minimal_threshold = {'Grand Rhone' : 0.75, 'Petit Rhone' : 1} # 0.75
        quantile_to_use = {'Grand Rhone' : 0.2, 'Petit Rhone' : 0.2}
        fixed_threshold = {'Grand Rhone' : 1.2, 'Petit Rhone' : 1.9} 
        river_mouth_to_exclude = {}
        
    elif Zone == 'EASTERN_CHANNEL' :        
        lon_new_resolution = 0.015
        lat_new_resolution = 0.015
        searching_strategies = {'Arques' : {'grid' : np.array([   [False, False, False, False, False],
                                                                  [False, True,  True,  True, False],
                                                                  [False, True,  True,  True, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, False, False, False, False],
                                                                ]),
                                      'coordinates_of_center' : (2,2)},
                                'Bresle' : {'grid' : np.array([   [False, False, False, False, False],
                                                                  [False, True,  True,  True, False],
                                                                  [False, True,  True,  True, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, False, False, False, False],
                                                                ]),
                                      'coordinates_of_center' : (2,2)},
                                'Somme' : {'grid' : np.array([    [False, False, False, False, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, False, False, False, False],
                                                                ]),
                                      'coordinates_of_center' : (2,2)},
                                'Authie' : {'grid' : np.array([     [False, False, False, False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, False, False, False, False],
                                                                  ]),
                                                              'coordinates_of_center' : (2,2)},
                                'Canche' : {'grid' : np.array([     [False, False, False, False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, False, False, False, False],
                                                                  ]),
                                                              'coordinates_of_center' : (2,2)},
                                'Liane' : {'grid' : np.array([      [False, False, False, False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, False, False, False, False],
                                                                  ]),
                                                              'coordinates_of_center' : (2,2)}}
          
        bathymetric_threshold = 0
        starting_points = { 'Arques' : (49.94, 1.08),
                            'Bresle' : (50.06, 1.37),
                            'Somme' : (50.23, 1.58),
                            'Authie' : (50.37, 1.58),
                            'Canche' : (50.55, 1.60),
                            'Liane' : (50.73, 1.59)}
        core_of_the_plumes = {'Arques' : (49.95, 1.07),
                              'Bresle' : (50.08, 1.37),
                              'Somme' : (50.25, 1.45),
                              'Authie' : (50.38, 1.52),
                              'Canche' : (50.56, 1.54),
                              'Liane' : (50.75, 1.56)}
        lat_range_of_the_area_to_check_for_clouds = [49.75, 50.85]
        lon_range_of_the_area_to_check_for_clouds = [0.75, 1.75]
        threshold_of_cloud_coverage_in_percentage = 25
        lat_range_of_the_map_to_plot = [49.20, 51.5]
        lon_range_of_the_map_to_plot = [-1.5, 3]
        lat_range_to_search_plume_area = [49.75, 49.75, 51.15, 50.4]
        lon_range_to_search_plume_area = [0.5, 1.75, 1.75, 0.5]
        max_steps_for_the_directions = {'Arques' : None, 'Bresle' : None, 'Somme' : None,
                                        'Authie' : None, 'Canche' : None, 'Liane' : None}
        river_mouth_to_exclude = {}
      
    elif Zone == 'SOUTHERN_BRITTANY':         
        lon_new_resolution = 0.015
        lat_new_resolution = 0.015
        searching_strategies = {'Loire' : {'grid' : np.array([    [False, False, False, False, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, True,  True,  False, False],
                                                                  [False, True,  True,  True, False],
                                                                  [False, False, False, False, False],
                                                                ]), 'coordinates_of_center' : (2,2)},
                                'Vilaine' : {'grid' : np.array([    [False, False, False, False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, True,  True,  False, False],
                                                                    [False, False, False, False, False],
                                                                  ]), 'coordinates_of_center' : (2,2)}
                                                                  }
        bathymetric_threshold = 0
        starting_points = {'Loire' : (47.29, -2.10),
                           'Vilaine' : (47.50, -2.46)}
        core_of_the_plumes = {'Loire' : (47.19, -2.36),
                              'Vilaine' : (47.47, -2.59)}
        lat_range_of_the_area_to_check_for_clouds = [46.87, 47.55]
        lon_range_of_the_area_to_check_for_clouds = [-3, -2.01]
        threshold_of_cloud_coverage_in_percentage = 25
        lat_range_of_the_map_to_plot = [46, 48] # [46, 48.5]
        lon_range_of_the_map_to_plot = [-5, -1] # [-5, -1.5]
        lat_range_to_search_plume_area = [46.5, 48]
        lon_range_to_search_plume_area = [-5, -1.5]
        maximal_bathymetric_for_zone_with_resuspension = {'Loire' : 20, 'Vilaine' : 20}
        minimal_distance_from_estuary_for_zone_with_resuspension = {'Loire' : 20, 'Vilaine' : 20}
        max_steps_for_the_directions = { 'Loire' : 100, 'Vilaine' : 50}
        maximal_threshold = { 'Loire' : 8, 'Vilaine' : 8} # 12
        minimal_threshold = { 'Loire' : 4, 'Vilaine' : 4} # 3
        quantile_to_use = { 'Loire' : 0.2, 'Vilaine' : 0.2}
        fixed_threshold = {'Loire' : 5.4, 'Vilaine' : 5.0} 
        river_mouth_to_exclude = {}

    else :
        print(f"The zone {Zone} is not available. Please select one of the following zones : 'BAY_OF_SEINE', 'BAY_OF_BISCAY', 'GULF_OF_LION', 'EASTERN_CHANNEL', 'SOUTHERN_BRITTANY'.")
        return None
    
    # TODO: Investigate why this is causing errors
    searching_strategy_directions = coordinates_of_pixels_to_inspect(searching_strategies)
    
    return {
        'lon_new_resolution' : lon_new_resolution, 
        'lat_new_resolution' : lat_new_resolution, 
        'searching_strategies' : searching_strategies, 
        'bathymetric_threshold' : bathymetric_threshold, 
        'starting_points' : starting_points, 
        'core_of_the_plumes' : core_of_the_plumes,
        'lat_range_of_the_area_to_check_for_clouds' : lat_range_of_the_area_to_check_for_clouds, 
        'lon_range_of_the_area_to_check_for_clouds' : lon_range_of_the_area_to_check_for_clouds, 
        'threshold_of_cloud_coverage_in_percentage' : threshold_of_cloud_coverage_in_percentage,
        'lat_range_of_the_map_to_plot' : lat_range_of_the_map_to_plot, 
        'lon_range_of_the_map_to_plot' : lon_range_of_the_map_to_plot, 
        'lat_range_to_search_plume_area' : lat_range_to_search_plume_area, 
        'lon_range_to_search_plume_area' : lon_range_to_search_plume_area,
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


def coordinates_of_pixels_to_inspect(searching_strategies) : 
     
    """
   Computes the relative distances from a center pixel to all "True" pixels 
   in a given grid for each search strategy.

   Parameters
   ----------
   searching_strategies : dict
       A dictionary where each key corresponds to a search strategy. Each value 
       is another dictionary with:
           - 'grid' : 2D boolean numpy array
               A boolean grid where "True" indicates pixels of interest.
           - 'coordinates_of_center' : tuple of int
               Coordinates (row, column) of the center pixel in the grid.

   Returns
   -------
   to_return : dict
       A dictionary where each key corresponds to a search strategy and each value 
       is a list of tuples representing the relative distances of "True" pixels 
       from the center pixel.
   """

    to_return = {} # Initialize an empty dictionary to store results
       
    # Iterate through each search strategy in the input dictionary
    for index, searching_strategy in searching_strategies.items() : 
    
        # Initialize a list to store the distances (as tuples) for this strategy
        distance_list = []
        
        # Extract the boolean grid (a 2D array) and center pixel coordinates
        boolean_array = searching_strategy['grid']
        coordinate_of_the_center = searching_strategy['coordinates_of_center']
        
        # Loop through each pixel in the grid
        for i in range(boolean_array.shape[0]): # Iterate over rows
        
            for j in range(boolean_array.shape[1]):  # Iterate over columns
            
                # Check if the pixel is "True"
                if boolean_array[i, j]:
                    
                    # Calculate the horizontal distance (x-axis) from the center
                    distance_x = coordinate_of_the_center[0] - i
                    
                    # Calculate the vertical distance (y-axis) and invert sign for standard image coordinates
                    # We multiply by -1 to account for typical image coordinate systems where y-coordinates increase downwards
                    distance_y = (coordinate_of_the_center[1] - j) * -1
                    
                    # Append the distance (as a tuple) to the list
                    distance_list.append((distance_x, distance_y))
             
        # Exclude the center pixel itself (distance of (0, 0))
        distance_list = concave_hull( [x for x in distance_list if x != (0,0)] )
        
        # Ensure that distances are ordered sequentially (no large jumps)
        distance_list_in_good_order = abs( np.array( np.diff( [ np.sum(x) for x in distance_list ] ) ) ) <= 1
        
        # If distances are not in a good order, reorder them
        if any( distance_list_in_good_order == False ) : 
            index_start_element = np.where( distance_list_in_good_order == False )[0] +1
            # Reorder the distance list by concatenating segments
            distance_list = [distance_list[index_start_element[0]], 
                           distance_list[index_start_element[0]:],
                           distance_list[:index_start_element[0]]]
            distance_list = flatten_a_list(distance_list)  # Flatten the reordered list
            distance_list = list(dict.fromkeys(distance_list))  # Remove duplicates while preserving order
        
        # Store the computed list of distances in the dictionary
        to_return[f'{index}'] = distance_list
           
    # Return the dictionary containing the distances for all search strategies 
    return to_return