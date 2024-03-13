"""
This module contains functions for loading satellite data from a directory of
tiles.
"""
from pathlib import Path
from typing import Tuple, List, Set
import os
from itertools import groupby
import re
from dataclasses import dataclass
import tifffile
import numpy as np

@dataclass
class Metadata:
    """
    A class to store metadata about a satellite file.
    """
    satellite_type: str
    file_name: List[str]
    tile_id: str
    bands: List[str]
    time: str


def print_single_metadata(metadata):
    
    print("Satellite Type:", metadata.satellite_type)
    print("File Names:", ', '.join(metadata.file_name))
    print("Tile ID:", metadata.tile_id)
    print("Bands:", ', '.join(metadata.bands))
    print("Time:", metadata.time)
    print()

def print_list_metadata(metadata_list):
    
    for i, metadata in enumerate(metadata_list):
        print(f"Metadata {i}:")
        print_single_metadata(metadata)
        
#helper function to get filetype and return the sat type
def satellite_type_from_filename(filename: str) -> str:
    '''
    patterns = {
            "viirs": 'DNB_VNP46A1_*',
            "sentinel1": 'S1A_IW_GRDH_*',
            "sentinel2": 'L2A_*',
            "landsat": 'LC08_L1TP_*',
            "gt": "groundTruth.tif"
        }
    '''

    """Determine the satellite type from the filename."""
    if re.match(r"DNB_VNP46A1_.*", filename):
        return "viirs"
    elif re.match(r"S1A_IW_GRDH_*", filename):
        return "sentinel1"
    elif re.match(r"L2A_*", filename):
        return "sentinel2"
    elif re.match(r"LC08_L1TP_*", filename):
        return "landsat"
    elif re.match(r"groundTruth.tif", filename):  
        return "gt"
    else:
        return "unknown"
    
def process_viirs_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a VIIRS file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: DNB_VNP46A1_A2020221.tif
    Example output: ("2020221", "0")

    Parameters
    ----------
    filename : str
        The filename of the VIIRS file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    parts = filename.split('_')
    date = parts[2][1:-4]

    
    band = "0"

    return date, band

#DONE
def process_s1_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-1 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: S1A_IW_GRDH_20200804_VV.tif
    Example output: ("20200804", "VV")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-1 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    
    parts = filename.split('_')
    date = parts[3]  
    band = parts[4].split('.')[0]  

    return date, band

#DONE
def process_s2_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-2 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: L2A_20200816_B01.tif
    Example output: ("20200804", "B01")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-2 file.

    Returns
    -------
    Tuple[str, str]
    """

    parts = filename.split('_')
    date = parts[1]  
    band = parts[2].split('.')[0]
    band = band[1:]  
    return date, band

#DONE
def process_landsat_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band)

    Example input: LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-08-30", "B9")

    Parameters
    ----------
    filename : str
        The filename of the Landsat file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    parts = filename.split('_')
    date = parts[2]  
    band = parts[3].split('.')[0]  
    band = band[1:]
    return date, band

#DONE
def process_ground_truth_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of the ground truth file and returns
    ("0", "0"), as there is only one ground truth file.

    Example input: groundTruth.tif
    Example output: ("0", "0")

    Parameters
    ----------
    filename: str
        The filename of the ground truth file though we will ignore it.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    return "0","0"

#DONE
def get_satellite_files(tile_dir: Path, satellite_type: str) -> List[Path]:
    #virrs      DNB_VNP46A1_A2020221.tif
    #s1         S1A_IW_GRDH_20200804_VV.tif
    #s2         L2A_20200816_B01.tif
    #landsat    LC08_L1TP_2020-08-30_B9.tif
    #gt         groundTruth.tif
    
    """
    
    Retrieve all satellite files matching the satellite type pattern.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file.
    """
    processed_files = []

    for file_path in tile_dir.glob("*"):
        if file_path.is_file():
            if satellite_type_from_filename(file_path.name) == satellite_type:
                processed_files.append(file_path)
    
    return processed_files

#DONE
def get_filename_pattern(satellite_type: str) -> str:
    """
    Return the filename pattern for the given satellite type.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    str
        The filename pattern for the given satellite type.


        patterns = {
            "viirs": 'DNB_VNP46A1_*',
            "sentinel1": 'S1A_IW_GRDH_*',
            "sentinel2": 'L2A_*',
            "landsat": 'LC08_L1TP_*',
            "gt": "groundTruth.tif"
        }
    """
    if satellite_type == "viirs":
        return "DNB_VNP46A1_*"

    elif satellite_type == "sentinel1":
        return "S1A_IW_GRDH_*"

    elif satellite_type == "sentinel2":
        return "L2A_*"

    elif satellite_type == "landsat":
        return "LC08_L1TP_*"  

    elif satellite_type == "gt":
        return "groundTruth.tif"  

    else:
        raise ValueError(f"Unsupported satellite type: {satellite_type}")

#DONE       
def read_satellite_files(sat_files: List[Path]) -> List[np.ndarray]:
    """
    Read satellite files into a list of numpy arrays.

    Parameters
    ----------
    sat_files : List[Path]
        A list of Path objects for each satellite file.

    Returns
    -------
    List[np.ndarray]

    """

    # Associate each array with a satellite type using a dictionary
    
    arr_list = []

    for file_path in sat_files:
        if file_path.is_file():
            # Read the .tif file into a numpy array using tifffile
            img_array = tifffile.imread(str(file_path))
            arr_list.append(img_array)

    return arr_list

def stack_satellite_data(
        sat_data: List[np.ndarray],
        file_names: List[str],
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Stack satellite data into a single array and collect filenames.
    Parameters
    ----------
    sat_data : List[np.ndarray]
        A list containing the image data for all bands with respect to
        a single satellite (sentinel-1, sentinel-2, landsat-8, or viirs)
        at a specific timestamp.
    file_names : List[str]
        A list of filenames corresponding to the satellite and timestamp.
    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with dimensions
        (time, band, height, width) and a list of the filenames.
    """

    #convert filepath to string
    if not isinstance(file_names[0], str):
        fn = [str(path.name) for path in file_names]   
        file_names = fn

    # get grouping function
    process_func = get_grouping_function(satellite_type)

    dates = []
    bands = []
    metadata_list = []

    for file in file_names:
        date,band = process_func(file)
        dates.append(date)
        bands.append(band)
    
    height,width, = sat_data[0].shape
    shape = (len(np.unique(dates)),len(np.unique(bands)),height,width)
    stacked = np.zeros(shape)


    info_list = []
    for i in range(len(bands)):
        info_list.append((dates[i],bands[i],file_names[i],sat_data[i]))
    
    date_grouping = lambda x: x[0]
    band_grouping = lambda x: x[1]
    

    sorted_data = []
    info_list.sort(key=date_grouping)

    for date,group_by_date in groupby(info_list,key = date_grouping):
        group_by_date = list(group_by_date)
        group_by_date.sort(key=band_grouping)

        band_list = []
        file_list = []

        for item in group_by_date:
            band = item[1]
            file = item[2]
            data = item[3]
            if band not in band_list:
                band_list.append(band)
            file_list.append(file)
            sorted_data.append(data)
        
        metadata = Metadata(satellite_type,file_list,'',band_list,date)
        metadata_list.append(metadata)
    for t in range(shape[0]):
        for b in range(shape[1]):
            stacked[t][b] = sorted_data.pop(0)
   
    
    return stacked, metadata_list

#DONE
def get_grouping_function(satellite_type: str):
    """
    Return the function to group satellite files by date and band.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    function
        The function to group satellite files by date and band.
    """
    
    if satellite_type == "viirs":
        return process_viirs_filename
    elif satellite_type == "sentinel1":
        return process_s1_filename
    elif satellite_type == "sentinel2":
        return process_s2_filename
    elif satellite_type == "landsat":
        return process_landsat_filename
    elif satellite_type == "gt":
        return process_ground_truth_filename
    else:
        raise ValueError(f"Unsupported satellite type: {satellite_type}")

#DONE
def get_unique_dates_and_bands(
        metadata_keys: Set[Tuple[str, str]]
        ) -> Tuple[Set[str], Set[str]]:
    """
    Extract unique dates and bands from satellite metadata keys.

    Parameters
    ----------
    metadata_keys : Set[Tuple[str, str]]
        A set of tuples containing the date and band for each satellite file.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        A tuple containing the unique dates and bands.
    """
    unique_dates = set()
    unique_bands = set()

    for date, band in metadata_keys:
        unique_dates.add(date)
        unique_bands.add(band)

    return unique_dates, unique_bands
#todo
def load_satellite(
        tile_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Load all bands for a given satellite type from a directory of tile files.

    Parameters
    ----------
    tile_dir : str or os.PathLike
        The Tile directory containing the satellite tiff files.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with
        dimensions (time, band, height, width) and a list of the metadata.
    """    

    #read all the files in to an x var
    #read_satelite_files(x) to get the np array
    #then use stack satelite data

    # Retrieve satellite files based on the satellite type
    satellite_files = get_satellite_files(Path(tile_dir), satellite_type)

    # Extract file names from the satellite files
    image_array = []

    file_names = [file.name for file in satellite_files]
    for file_path in satellite_files:
        img = tifffile.imread(str(file_path))
        image_array.append(img)

    arr,meta_data = stack_satellite_data(image_array,file_names,satellite_type)
    
    for item in meta_data:
        item.tile_id = tile_dir#.name
    
    return arr,meta_data
    

    
def get_subdirectories(path_to_directory: str | os.PathLike) -> List[str]:
    # Convert the input to a Path object
    directory_path = Path(path_to_directory)

    # List all subdirectories using the glob method
    subdirectories = [d for d in directory_path.glob('*/') if d.is_dir()]

    return subdirectories


def load_satellite_dir(
        data_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
                
    """
    Load all bands for a given satellite type from a directory of multiple
    tile files.

    Parameters
    ----------
    data_dir : str or os.PathLike
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with
        dimensions (tile_dir, time, band, height, width) and a list of the
        Metadata objects.
    """
    sub_dirs = get_subdirectories(data_dir)
    four_D_arr_list = []
    metL = []
    
    for tile in sub_dirs:
        four_D_arr,met = load_satellite(Path(tile),satellite_type)
        metL.append(met)
        four_D_arr_list.append(four_D_arr)
    combined_array = np.stack(four_D_arr_list, axis=0)

    return combined_array,metL
