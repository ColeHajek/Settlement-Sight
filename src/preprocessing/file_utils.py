"""
This module contains functions for loading satellite data from a directory of
tiles. It includes processing filename patterns for various satellite types
and loading satellite data into xarray DataArrays for analysis.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import tifffile
import xarray as xr
import re

# local modules
sys.path.append(".")
from src.utilities import SatelliteType


def process_viirs_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function processes the filename of a VIIRS file and extracts the date and band information.

    Example input: C:/users/foo/data/DNB_VNP46A1_A2020221.tif
    Example output: ("2020-08-08", "0")

    The date format is {year}{day}, where day is the day of the year (Julian day).

    Parameters
    ----------
    file_path : Path
        The Path of the VIIRS file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date (YYYY-MM-DD) and the band.
    
    Raises
    ------
    ValueError
        If the date pattern cannot be found in the filename.
    """
    pathStr = file_path.as_posix()
    pattern = r"\d{7}"  # Regex pattern for 7 consecutive digits (Julian date)
    match = re.search(pattern, pathStr)

    if not match:
        raise ValueError(f"No date found in the file path: {pathStr}")

    dateStr = match.group()
    dateStrFormat = datetime.strptime(dateStr, "%Y%j").strftime("%Y-%m-%d")

    endInd = pathStr.find(dateStr) + 7
    if pathStr[endInd] != ".":
        return (dateStrFormat, pathStr[endInd + 1 : pathStr.rfind(".")])
    else:
        return (dateStrFormat, "0")


def process_s1_filename(file_path: Path) -> Tuple[str, str]:
    """
    Processes the filename of a Sentinel-1 file and extracts the date and band information.

    Example input: C:/users/foo/data/S1A_IW_GRDH_20200804_VV.tif
    Example output: ("2020-08-04", "VV")

    The date format is {year}{month}{day}.

    Parameters
    ----------
    file_path : Path
        The Path of the Sentinel-1 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date (YYYY-MM-DD) and the band.
    
    Raises
    ------
    ValueError
        If the date pattern cannot be found in the filename.
    """
    pathStr = file_path.as_posix()
    pattern = r"\d{8}"  # Regex pattern for 8 consecutive digits (YYYYMMDD)
    match = re.search(pattern, pathStr)

    if not match:
        raise ValueError(f"No date found in the file path: {pathStr}")

    dateStr = match.group()
    dateStrFormat = datetime.strptime(dateStr, "%Y%m%d").strftime("%Y-%m-%d")

    endInd = pathStr.find(dateStr) + 8
    if pathStr[endInd] != ".":
        return (dateStrFormat, pathStr[endInd + 1 : pathStr.rfind(".")])
    else:
        return (dateStrFormat, "0")


def process_s2_filename(file_path: Path) -> Tuple[str, str]:
    """
    Processes the filename of a Sentinel-2 file and extracts the date and band information.

    Example input: C:/users/foo/data/L2A_20200816_B01.tif
    Example output: ("2020-08-16", "01")

    The date format is {year}{month}{day}.

    Parameters
    ----------
    file_path : Path
        The Path of the Sentinel-2 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date (YYYY-MM-DD) and the band.
    
    Raises
    ------
    ValueError
        If the date pattern cannot be found in the filename.
    """
    pathStr = file_path.as_posix()
    pattern = r"\d{8}"  # Regex pattern for 8 consecutive digits (YYYYMMDD)
    match = re.search(pattern, pathStr)

    if not match:
        raise ValueError(f"No date found in the file path: {pathStr}")

    dateStr = match.group()
    dateStrFormat = datetime.strptime(dateStr, "%Y%m%d").strftime("%Y-%m-%d")

    endInd = pathStr.find(dateStr) + 8
    if pathStr[endInd] != ".":
        return (dateStrFormat, pathStr[endInd + 2 : pathStr.rfind(".")])
    else:
        return (dateStrFormat, "0")


def process_landsat_filename(file_path: Path) -> Tuple[str, str]:
    """
    Processes the filename of a Landsat file and extracts the date and band information.

    Example input: C:/users/foo/data/LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-08-30", "9")

    The date format is {year}-{month}-{day}.

    Parameters
    ----------
    file_path : Path
        The Path of the Landsat file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date (YYYY-MM-DD) and the band.
    
    Raises
    ------
    ValueError
        If the date pattern cannot be found in the filename.
    """
    pathStr = file_path.as_posix()
    pattern = r"\d{4}"  # Regex pattern for 4 consecutive digits
    match = re.search(pattern, pathStr)

    if not match:
        raise ValueError(f"No date found in the file path: {pathStr}")

    dateInd = pathStr.find(match.group())
    dateStr = pathStr[dateInd : dateInd + 10]

    endInd = dateInd + 10
    if pathStr[endInd] != ".":
        return (dateStr, pathStr[endInd + 2 : pathStr.rfind(".")])
    else:
        return (dateStr, "0")


def process_ground_truth_filename(file_path: Path) -> Tuple[str, str]:
    """
    Processes the filename of the ground truth file. Since there's only one ground truth file, 
    it returns ("0001-01-01", "0").

    Parameters
    ----------
    file_path: Path
        The Path of the ground truth file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing ("0001-01-01", "0").
    """
    return "0001-01-01", "0"


def get_filename_pattern(satellite_type: SatelliteType) -> str:
    """
    Returns the filename pattern for the given satellite type.
    """

    patterns = {
        "viirs": "DNB_VNP46A1_",
        "sentinel1": "S1A_IW_GRDH_",
        "sentinel2": "L2A_",
        "landsat": "LC08_L1TP_",
        "gt": "groundTruth.tif",
    }
    return patterns[satellite_type.value]


def get_satellite_files(tile_dir: Path, satellite_type: SatelliteType) -> List[Path]:
    """
    Retrieves all satellite files in the directory matching the satellite type pattern.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : SatelliteType
        The type of satellite.

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file in the directory.
    """
    tile_sat_files = []
    pattern = get_filename_pattern(satellite_type)
    for file in tile_dir.iterdir():
        if file.is_file() and pattern in file.name and file.suffix == ".tif":
            tile_sat_files.append(file)
    return tile_sat_files


def get_grouping_function(satellite_type: SatelliteType) -> Callable:
    """
    Returns the function to group satellite files by date and band for the given satellite type.

    Parameters
    ----------
    satellite_type : SatelliteType
        The type of satellite.

    Returns
    -------
    Callable
        The function to group satellite files by date and band.
    """
    patterns = {
        "viirs": process_viirs_filename,
        "sentinel1": process_s1_filename,
        "sentinel2": process_s2_filename,
        "landsat": process_landsat_filename,
        "gt": process_ground_truth_filename,
    }
    return patterns[satellite_type.value]


def get_unique_dates_and_bands(
    tile_dir: Path, satellite_type: SatelliteType
) -> Tuple[List[str], List[str]]:
    """
    Extracts unique dates and bands from the tile directory.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite files.
    satellite_type : SatelliteType
        The type of satellite.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists: unique dates and unique bands.
    """
    satellite_files = get_satellite_files(tile_dir, satellite_type)
    grouping_function = get_grouping_function(satellite_type)

    dates = set()
    satType = set()
    for sFile in satellite_files:
        dateNband = grouping_function(sFile)
        dates.add(dateNband[0])
        satType.add(dateNband[1])

    return sorted(dates), sorted(satType)


def get_parent_tile_id(tile_dir: Path) -> str:
    """
    Returns the name of the tile directory (parent_tile_id).

    Parameters
    ----------
    tile_dir : Path
        The tile directory.

    Returns
    -------
    str
        The parent tile ID.
    """
    return tile_dir.name


def read_satellite_file(satellite_file: Path) -> np.ndarray:
    """
    Reads the satellite file into a np.ndarray using tifffile.

    Parameters
    ----------
    satellite_file : Path
        The satellite file path.

    Returns
    -------
    np.ndarray
        The satellite data as a 2D numpy array.
    
    Raises
    ------
    ValueError
        If the file cannot be read.
    """
    try:
        image_data = tifffile.imread(str(satellite_file)).astype(np.float32)
        return image_data
    except Exception as e:
        raise ValueError(f"Error reading file {satellite_file}: {e}")


def load_satellite(tile_dir: Path, satellite_type: SatelliteType) -> xr.DataArray:
    """
    Loads all bands for a given satellite type from a directory of tile files.

    Parameters
    ----------
    tile_dir : Path
        The tile directory containing the satellite files.
    satellite_type : SatelliteType
        The type of satellite.

    Returns
    -------
    xr.DataArray
        A DataArray containing the satellite data with dimensions (date, band, height, width).
    """
    file_names = get_satellite_files(tile_dir, satellite_type)
    grouping_function = get_grouping_function(satellite_type)
    parent_tile_id = get_parent_tile_id(tile_dir)
    dates, bands = get_unique_dates_and_bands(tile_dir, satellite_type)

    dataDates = []
    for date in dates:
        dataBands = []
        for band in bands:
            for fileName in file_names:
                retDate, retBand = grouping_function(fileName)
                if date == retDate and band == retBand:
                    dataBands.append(read_satellite_file(fileName))
                    break  # Exit loop once the file is found for the given date and band
        dataDates.append(np.stack(dataBands))
    finData = np.stack(dataDates)

    height, width = finData.shape[-2:]
    coords = {"date": dates, "band": bands, "height": range(height), "width": range(width)}

    xrArr = xr.DataArray(finData, dims=("date", "band", "height", "width"), coords=coords)
    xrArr.attrs["satellite_type"] = satellite_type.value
    xrArr.attrs["tile_dir"] = str(tile_dir)
    xrArr.attrs["parent_tile_id"] = parent_tile_id

    return xrArr


def load_satellite_list(
    tile_dir: Path, satellite_type_list: List[SatelliteType]
) -> List[xr.DataArray]:
    """
    Loads all the satellites from the tile directory based on the satellite type list.

    Parameters
    ----------
    tile_dir : Path
        The tile directory.
    satellite_type_list : List[SatelliteType]
        A list of satellite types.

    Returns
    -------
    List[xr.DataArray]
        A list of DataArrays for each satellite type.
    """
    return [load_satellite(tile_dir, satType) for satType in satellite_type_list]


def load_satellite_dir(
    data_dir: Path, satellite_type_list: List[SatelliteType]
) -> List[List[xr.DataArray]]:
    """
    Loads all bands for a given satellite type from a directory of multiple tile files.

    Parameters
    ----------
    data_dir : Path
        The directory containing all the satellite tiles.
    satellite_type_list : List[SatelliteType]
        A list of satellite types.

    Returns
    -------
    List[List[xr.DataArray]]
        A list of tiles, where each element contains a list of DataArrays for each satellite type.
    """
    return [
        load_satellite_list(tile_dir, [satellite_type_list])
        for tile_dir in sorted(data_dir.iterdir())
        if tile_dir.is_dir()
    ]


def create_satellite_dataset_list(
    list_of_data_array_list: List[List[xr.DataArray]],
    satellite_type_list: List[SatelliteType],
    list_of_preprocess_func_list: List[List[Callable]] = None,
) -> List[xr.Dataset]:
    """
    Creates a dataset for each satellite type from the loaded satellite data.

    Parameters
    ----------
    list_of_data_array_list : List[List[xr.DataArray]]
        A list of tiles, where each element contains a list of DataArrays for each satellite type.
    satellite_type_list : List[SatelliteType]
        A list of satellite types.
    list_of_preprocess_func_list : List[List[Callable]], optional
        A list of preprocessing functions to apply to the DataArrays, by satellite type.

    Returns
    -------
    List[xr.Dataset]
        A list of Datasets for each satellite type.
    
    Raises
    ------
    TypeError
        If a preprocessing function does not return an xr.DataArray.
    """
    data_dict_list = [dict() for _ in satellite_type_list]
    
    for satellite_list in list_of_data_array_list:
        for index, data_array in enumerate(satellite_list):
            if list_of_preprocess_func_list is not None:
                if list_of_preprocess_func_list[index] is not None:
                    for func in list_of_preprocess_func_list[index]:
                        data_array = func(data_array)
                        if not isinstance(data_array, xr.DataArray):
                            raise TypeError("Preprocessing function did not return an xr.DataArray")
            data_dict_list[index][data_array.attrs["parent_tile_id"]] = data_array

    data_set_list = []
    for index, data_dict in enumerate(data_dict_list):
        dataset = xr.Dataset(data_dict)
        dataset.attrs["satellite_type"] = satellite_type_list[index].value
        data_set_list.append(dataset)

    return data_set_list
