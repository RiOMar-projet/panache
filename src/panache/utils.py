#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
#### Modules
# =============================================================================


import os, sys, subprocess, pickle, glob, datetime, importlib.resources, tempfile, shutil, re
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
from itertools import product, chain
from functools import reduce
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


def exit_program():
    print("Exiting the program...")
    sys.exit(0)
    

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


def degrees_to_km(lat_deg, lon_deg, latitude):
    """
    Convert distances in degrees to kilometers.
    
    Parameters:
    - lat_deg: Distance in degrees of latitude
    - lon_deg: Distance in degrees of longitude
    - latitude: Latitude where the conversion is needed
    
    Returns:
    - lat_km: Distance in kilometers for latitude
    - lon_km: Distance in kilometers for longitude
    """
    # Conversion factor for latitude
    lat_km = lat_deg * 111.32
    
    # Conversion factor for longitude, adjusted by latitude
    lon_km = lon_deg * 111.32 * np.cos(np.radians(latitude))
    
    return lat_km, lon_km


def km_to_degrees(lat_km, lon_km, latitude):
    """
    Convert a distance in meters to degrees of latitude and longitude.

    Args:
        meters (float): Distance in meters.
        latitude (float): Latitude in degrees where the conversion is applied.

    Returns:
        (float, float): (Latitude degrees, Longitude degrees)
    """
    # 1 degree latitude ≈ 111.32 km (constant)
    lat_deg = lat_km / 111.320  

    # 1 degree longitude ≈ 111.32 * cos(latitude) km
    lon_deg = lon_km / (111.320 * np.cos(np.radians(latitude)))

    return lat_deg, lon_deg


# def generic_variable_names() : 
    
#     return ['CHLA', 'SPM', 'SST']


# def find_sat_data_files(info, path_to_sat_data) : 

#     if isinstance(path_to_sat_data, str) : 
#         path_to_sat_data = [path_to_sat_data]

#     if ('Year' in info) and isinstance(info['Year'], str) and (info['Year'] == 'MULTIYEAR') : 
#         file_pattern = '/*.pkl'
#     elif np.isin(info.Satellite_variable, generic_variable_names()) : 
#         file_pattern = '/*.nc'
#     else : 
#         file_pattern = f'/*{info.Satellite_variable}*.nc'    

#     map_files = []
    
#     if '[YEAR]' in path_to_sat_data[0] : 
        
#         for Year in info['Year'] :    
    
#             # path_to_files = (path_to_sat_data 
#             #                      .replace('[YEAR]', str(Year))
#             #                      .replace('[MONTH]', '*')
#             #                      .replace('[DAY]', '*'))
                              
#             files = glob.glob(path_to_sat_data + file_pattern)
            
#             map_files.extend( files )
            
#     else : 
        
#         map_files = list(chain.from_iterable(glob.glob(path + file_pattern) for path in path_to_sat_data))
        
#     return map_files


def store_arguments(arguments, locally = False, globally = False, return_arguments = False):
    
    arguments_to_return = []
    for key, value in arguments.items():
        if locally : 
            locals()[key] = value
            continue
        if globally : 
            globals()[key] = value
            continue
        if return_arguments : 
            arguments_to_return.append(value)
        
    if return_arguments : 
        return arguments_to_return
            
    # print(Data_sources)  # Works inside this function


# def path_to_fill_to_where_to_save_satellite_files(where_to_save_files) : 
    
#     path = f'{where_to_save_files}/[DATA_SOURCE]/[PARAMETER]/[SENSOR]/[ATMOSPHERIC_CORRECTION]/[TIME_FREQUENCY]/[YEAR]/[MONTH]/[DAY]'

#     return path


# def fill_the_sat_paths(info, path_to_fill, local_path = False, dates = []) : 
    
#     if len(dates) > 0 and isinstance(dates[0], str) : 
#         dates = pd.to_datetime(dates)
    
#     path_to_fill = ( path_to_fill  
#                         .replace('[DATA_SOURCE]', info.Data_source)
#                         .replace('[ATMOSPHERIC_CORRECTION]', info.atmospheric_correction)
#                         .replace('[SENSOR]', info.sensor_name if 'sensor_name' in info else info.Satellite_sensor) )
                      
#     if 'Temporal_resolution' in info.keys() or 'Temporal_resolution' in info.keys() : 
        
#         Temporal_resolution = info.Temporal_resolution if 'Temporal_resolution' in info.keys() else info.Temporal_resolution
#         if local_path == False : 
#             Temporal_resolution = (Temporal_resolution
#                                     .replace('DAILY', 'day')
#                                     .replace('MONTHLY', 'month')
#                                     .replace('WEEKLY', '8-day'))        
            
#     elif isinstance(info.Year, str) and (info.Year == 'MULTIYEAR') :
        
#         Temporal_resolution = 'ANNUAL'
        
#     else :
        
#         Temporal_resolution = 'DAILY'
            
#     path_to_fill = path_to_fill.replace('[TIME_FREQUENCY]', Temporal_resolution)
        
#     if local_path : 
        
#         Folder_name_for_the_variable = ('CHLA' if 'CHL' in info.Satellite_variable.upper()
#                                         else 'SPM' if 'SPM' in info.Satellite_variable.upper()
#                                         else 'SST' if 'SST' in info.Satellite_variable.upper()
#                                         else info.Satellite_variable)
        
#         path_to_fill = (path_to_fill.replace('[PARAMETER]', Folder_name_for_the_variable))
        
#     else : 
        
#         path_to_fill = (path_to_fill.replace('[PARAMETER]', info.Satellite_variable_name_on_remote_folder))

#     if len(dates) > 0 : 
        
#         paths_to_sat_files = [ ( path_to_fill
#                                       .replace('[YEAR]', str(date.year))
#                                       .replace('[MONTH]', str(date.month).zfill(2))
#                                       .replace('[DAY]', str(date.day).zfill(2))
#                                       .replace("[DOY]", date.strftime("%j")) ) 
#                                 for date in dates ]
        
#     else : 
        
#         paths_to_sat_files = (path_to_fill
#                                 .replace('[YEAR]', '*')
#                                 .replace('[MONTH]', '*')
#                                 .replace('[DAY]', '*')
#                                 .replace("[DOY]", '*')) 
    
#     return paths_to_sat_files


# def get_all_cases_to_process(core_arguments) : 
        
#     cases_to_process = expand_grid(Data_source = core_arguments['Data_sources'], 
#                                   sensor_name = core_arguments['Sensor_names'], 
#                                   atmospheric_correction = core_arguments['Atmospheric_corrections'],
#                                   Satellite_variable = core_arguments['Satellite_variables'],
#                                   Temporal_resolution = core_arguments['Temporal_resolution'] if 'Temporal_resolution' in core_arguments else ['DAILY'])
    
#     cases_to_process['atmospheric_correction'] = cases_to_process.apply(lambda row: 'Standard' 
#                                                                         if np.isin(row['Data_source'], ['SEXTANT', 'EUMETSAT']) 
#                                                                         else row['atmospheric_correction'], axis=1)
#     cases_to_process['Satellite_variable'] = cases_to_process.apply(lambda row: 'SPM' 
#                                                                         if row['Data_source'] == 'SEXTANT' and 'SPM' in row['Satellite_variable'] 
#                                                                         else row['Satellite_variable'], axis=1)
    
#     cases_to_process = cases_to_process.drop_duplicates().reset_index(drop = True)  
    
#     return cases_to_process


# def get_all_cases_to_process_for_regional_maps_or_plumes_or_X11(core_arguments) : 

#     all_possibilities = expand_grid(Zone = core_arguments['Zones'],
#                                     Data_source = core_arguments['Data_sources'], 
#                                     sensor_name = core_arguments['Sensor_names'], 
#                                     atmospheric_correction = core_arguments['Atmospheric_corrections'],
#                                     Year = core_arguments['Years'],
#                                     Satellite_variable = core_arguments['Satellite_variables'],
#                                     Temporal_resolution = (core_arguments['Temporal_resolution'] if 'Temporal_resolution' in core_arguments 
#                                                        else core_arguments['Temporal_resolution'] if 'Temporal_resolution' in core_arguments 
#                                                        else '') )
#     all_possibilities['atmospheric_correction'] = all_possibilities.apply(lambda row: 'Standard' 
#                                                                         if row['Data_source'] == 'SEXTANT' 
#                                                                         else row['atmospheric_correction'], axis=1)
#     all_possibilities = all_possibilities.drop_duplicates()
    
#     return all_possibilities


# def create_arborescence(paths):
#     arborescence = {}
#     for path in paths:
#         keys = path.split("/")
#         current_level = arborescence
#         for key in keys:
#             if key not in current_level:
#                 current_level[key] = {}
#             current_level = current_level[key]  # Move deeper
#     return arborescence


# def return_the_parameter_name_based_on_file_name(file_name) : 
                        
#     regular_expression = r'(?:(SPM-[G|R]|SPIM|suspended_matters|TSM_NN|CHL|CHL1|CHL-OC5|CHL-GONS|chlorophyll_a|POC|NRRS[0-9]*|RRS[0-9]*|DOC|CDOM|BBP|T-FNU|SST(?:-NIGHT|)))'
#     match = re.search(regular_expression, file_name)
    
#     if match : 
        
#         return match.group(0) 
            
#     else : 
        
#         print('!!! Impossible to find the parameter name from the file name (see the function return_the_parameter_name_based_on_file_name) !!!')                        


# def add_array_to_dict(dictionary, path, array):
#     """
#     Adds an array to a specific position in a nested dictionary.
    
#     Parameters:
#         dictionary (dict): The main dictionary to update.
#         path (str): The path in the form "A/B/C/D/filename.nc".
#         array (numpy.ndarray): The array to be stored.
    
#     Returns:
#         None (modifies the dictionary in place).
#     """
#     keys = path.split("/")
#     filename = keys[-1]  # Extract the filename
    
#     # Extract parameter name (e.g., SPM-G)
#     param = return_the_parameter_name_based_on_file_name(filename)

#     # Navigate the dictionary hierarchy
#     current_level = dictionary
#     for key in keys[:-1]:  # Exclude filename from navigation
#         if key not in current_level:
#             current_level[key] = {}
#         current_level = current_level[key]

#     # Add the array under the parameter name
#     current_level[param] = array


# def access_item_in_a_dictionnary(dictionary, path):
    
#     keys = path.split("/")

#     item = reduce(lambda dictionary, key: dictionary[key], keys[1:], dictionary)
    
#     return item
        
    
# def merge_dicts(dicts):
#     """
#     Merges multiple nested dictionaries into a single dictionary.
    
#     Parameters:
#         dicts (list): List of dictionaries to merge.
    
#     Returns:
#         dict: A merged dictionary.
#     """
#     def recursive_merge(d1, d2):
#         """Recursively merges two dictionaries."""
#         for key, value in d2.items():
#             if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
#                 recursive_merge(d1[key], value)  # Merge sub-dictionaries
#             else:
#                 d1[key] = value  # Overwrite or add new keys
#         return d1

#     merged_dict = {}
#     for d in dicts:
#         merged_dict = recursive_merge(merged_dict, d)

#     return merged_dict


def get_empty_paths(dictionary, prefix=""):
    
    paths = []
    
    if isinstance(dictionary, dict) and len(dictionary) > 0:  # If it's a non-empty dictionary
        for key, val in dictionary.items():
            paths.extend(get_empty_paths(val, f"{prefix}/{key}"))  # Recursive call
    elif len(dictionary) == 0:  # Exclude NaN values
        paths.append(prefix)  # Add the path if it's a valid value
        # paths.append( [f'{prefix}/{x}' for x in list(dictionary.keys())] )  # Add the path if it's a valid value
    
    return paths


def get_non_empty_paths(dictionary, prefix=""):
    
    paths = []
    
    if isinstance(dictionary, dict) and len(dictionary) > 0 :  # If it's a non-empty dictionary
        for key, val in dictionary.items():
            paths.extend(get_non_empty_paths(val, prefix = f"{prefix}/{key}"))  # Recursive call

    elif isinstance(dictionary, dict) == False:  # Exclude NaN values
        paths.append(prefix)  # Add the path if it's a valid value
    else :  # Exclude NaN values
        for key, val in dictionary.items():
            paths.append(f"{prefix}")  # Recursive call
            break
    
    # paths = np.unique( [x.replace('Sat_values', '').replace('Time', '') for x in paths] )
    
    return paths


def check_time_format(time_str):
    
    import re
    import numpy as np

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


def extract_the_time_from_the_satellite_file(map_data) : 
        
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


def load_shapefile_data() : 
    
    france_shapefile = load_csv_files_in_the_package_folder(FRANCE_shapefile = True)
    
    return france_shapefile 
            
    # try : 
    #     france_shapefile = pygadm.Items(name="FRANCE", content_level=0)
    #     return france_shapefile
    # except Exception as e :
    #     print(f"The France shapefile can't be accessed through pygadm : {e}")
    #     print("The France shapefiles can be manually downloaded for free : e.g. https://gadm.org/download_country.html ")
 
    
def extract_and_format_date_from_path(path):
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', path)
    return ''.join(match.groups()) if match else None   


def load_csv_files_in_the_package_folder(SOMLIT = False, REPHY = False, FRANCE_shapefile = False, 
                                         RIVER_FLOW = False, Zone_of_river_flow = None, 
                                         RIVER_FLOW_time_resolution = ''):
    
    if SOMLIT : 
        SOMLIT_dir = os.path.join( proj_dir, 'data', 'INSITU_data', 'SOMLIT' )
        SOMLIT_data = os.path.join( SOMLIT_dir, 'Somlit.csv' )
        return (pd.read_csv(SOMLIT_data, sep = ";", header = 2).iloc[1:]
                                .rename(columns = {'gpsLat*':'LATITUDE', 
                                                   'gpsLong*':'LONGITUDE',
                                                   'nomSite*':"Site"}))
        
    if REPHY : 
        REPHY_dir = os.path.join( proj_dir, 'data', 'INSITU_data', 'REPHY' )
        REPHY_data = os.path.join( REPHY_dir, 'Table1_REPHY_hydro_RIOMAR.csv.gz' )
        return pd.read_csv(REPHY_data, sep = ";", header = 0, encoding = "ISO-8859-1", compression = {'method' : 'gzip'})
        
    if FRANCE_shapefile : 
        if gpd is None:
            raise ImportError("geopandas is required to load the packaged France shapefile.")
        
        shp_folder = os.path.join( proj_dir, 'data', 'FRANCE_shapefile' )  # Directly get the package folder path
    
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Extract all necessary shapefile components
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                # shp_file = shp_folder / f'gadm41_FRA_0{ext}'
                shp_file = os.path.join(shp_folder, f'gadm41_FRA_0{ext}')
                if os.path.exists(shp_file): # Ensure the file exists before copying
                    shutil.copy(shp_file, os.path.join(tmp_dir, f'gadm41_FRA_0{ext}'))
    
            # Read the shapefile from the temporary directory
            shapefile_path = os.path.join(tmp_dir, 'gadm41_FRA_0.shp')
            return gpd.read_file(shapefile_path)

    if RIVER_FLOW : 
        
        where_are_river_data = os.path.join( proj_dir, 'data', 'RIVER_FLOW', Zone_of_river_flow)
        
        files_to_load = glob.glob(os.path.join(where_are_river_data, '*'))
        
        # Convert file paths to Path objects for consistent handling
        files_to_read = [Path(f) for f in files_to_load if Path(f).suffix in ('.txt', '.dat', '.csv', '.ascii')]

        # Load each file into a dictionary of DataFrames
        data_dict = {}
        for file in files_to_read:
            ext = file.suffix.lower()
            if ext == '.csv':
                df = pd.read_csv(file)
            if ext == '.ascii' : 
                df = pd.DataFrame( np.loadtxt(file) )
            else:
                df = pd.read_table(file, sep=None, engine='python', header = None)  # Auto-detect separator
            data_dict[file.name] = df  # Store in dictionary
                    
        for key, the_df in data_dict.items() : 
            
            if Zone_of_river_flow == 'GULF_OF_LION' :
                
                the_df.columns = ['Flow', 'Year', 'Month', 'Day']
                the_df['Date'] = pd.to_datetime(the_df['Year'].astype(int).astype(str) + '-' + 
                                                the_df['Month'].apply(lambda x: str(int(x)).zfill(2)) + '-' + 
                                                the_df['Day'].apply(lambda x: str(int(x)).zfill(2)) ) 
                
            if np.isin( Zone_of_river_flow, ['BAY_OF_BISCAY', 'SOUTHERN_BRITTANY', 'BAY_OF_SEINE']) :
                
                the_df.columns = ['Date', 'Time', 'Flow']
                the_df['Date'] = pd.to_datetime(the_df['Date'], format = "%d/%m/%Y" ) 

            data_dict[key] = the_df.loc[:,['Flow', 'Date']]
    
        final_df = pd.concat(data_dict.values()).groupby("Date", as_index=False).agg(Values=('Flow', 'sum'), n_rivers=('Flow', 'count'))
        final_df = final_df[ final_df.n_rivers == len(files_to_read) ]
        
        bin_centers = [4, 12, 20, 28]
        def assign_bin(day):
            return min(bin_centers, key=lambda x: abs(x - day))
        
        # Apply the function to create a 'bin' column
        final_df['bin'] = final_df['Date'].dt.day.apply(assign_bin)
        final_df_reduced = final_df.loc[:,['Date', 'bin', 'Values']]
        final_df_reduced['Values'] = pd.to_numeric(final_df_reduced['Values'])
        
        # TODO: There is no 'DAILY' logic gate so I added one and closed this chain of logic gates
        if RIVER_FLOW_time_resolution == 'DAILY' :
            final_df = final_df_reduced

        elif RIVER_FLOW_time_resolution == 'WEEKLY' : 
            final_df_binned = final_df_reduced.groupby([final_df_reduced['Date'].dt.to_period('M'), 'bin']).agg({'Values': 'mean'}).reset_index()
            final_df_binned['Date'] = pd.to_datetime( final_df_binned['Date'].astype(str) + "-" + final_df_binned['bin'].astype(str), format = "%Y-%m-%d" )
            final_df = final_df_binned
            
        elif RIVER_FLOW_time_resolution == 'MONTHLY' :     
            final_df_binned = final_df_reduced.groupby([final_df_reduced['Date'].dt.to_period('M')]).agg({'Values': 'mean'}).reset_index()
            final_df_binned['Date'] = pd.to_datetime( final_df_binned['Date'].astype(str) + "-" + "15", format = "%Y-%m-%d" )
            final_df = final_df_binned

        elif RIVER_FLOW_time_resolution == 'ANNUAL' :     
            final_df_binned = final_df_reduced.groupby([final_df_reduced['Date'].dt.to_period('Y')]).agg({'Values': 'mean'}).reset_index()
            final_df_binned['Date'] = pd.to_datetime( final_df_binned['Date'].astype(str) + "-06-30", format = "%Y-%m-%d" )
            final_df = final_df_binned
            
        else : 
            print("The RIVER_FLOW_time_resolution must be one of the following: 'DAILY', 'WEEKLY', 'MONTHLY', 'ANNUAL'")
            exit_program()

        return final_df
         
        
def get_the_values_from_a_list_comprehension(lst_comprehension, return_the_unique_values) : 
    
    the_values = list(chain(*lst_comprehension))
    
    if return_the_unique_values : 
        the_values = np.unique(the_values)
    
    return the_values


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


def daily_integral(file_dir, overwrite=False):
    
    """
    Creates daily integral NetCDF files from their hourly versions.
    Remember that this requires the first hour of the next day to compute the daily integral.
    file_dir : str
        Directory where the input files are located.
    overwrite : bool, optional
        Whether to overwrite existing output files. Default is False.    
    """

    # Get list of files in the directory
    dir_files = os.listdir(file_dir)

    # Split out files with 'daily' in their names
    daily_files = [f for f in dir_files if 'daily' in f]
    hourly_files = [f for f in dir_files if 'daily' not in f]

    # Remove all_files with the existing daily files if overwrite is False
    if not overwrite :
        hourly_files_check = [f for f in hourly_files if f.split('_')[0] + '_daily_' + f.split('_')[1] + '_' + f.split('_')[2] not in daily_files]
    else : 
        hourly_files_check = hourly_files
        # Remove any existing daily files if overwrite is True
        for f in daily_files:
            os.remove(os.path.join(file_dir, f))

    # Stop if no hourly files need to be processed
    if len(hourly_files_check) == 0:
        print("No new hourly files to process.")
        return

    # Process each file to create daily integral versions
    for hourly_file in hourly_files_check:
            
        daily_file = hourly_file.split('_')[0] + '_daily_' + hourly_file.split('_')[1] + '_' + hourly_file.split('_')[2]

        # Calculate daily means
        subprocess.run(['cdo', 'daymean', os.path.join(file_dir, hourly_file), 
                        os.path.join(file_dir, daily_file)], check=True)
