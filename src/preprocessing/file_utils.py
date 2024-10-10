"""
Module for loading and processing satellite data from a directory of tiles.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple, Dict

import numpy as np
import tifffile
import xarray as xr
import re

sys.path.append(".")
from src.utilities import SatelliteType


def parse_filename_date_and_band(file_path: Path, date_format: str, date_length: int) -> Tuple[str, str]:
    """
    Parses the filename of a satellite file to extract the date and band.

    Parameters
    ----------
    file_path : Path
        The path of the satellite file.
    date_format : str
        The format string for parsing the date.
    date_length : int
        The length of the date substring to extract.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the formatted date and band.
    """
    file_name = file_path.as_posix()
    date_match = re.search(r"\d{" + str(date_length) + r"}", file_name)

    if not date_match:
        raise ValueError(f"Date not found in filename: {file_path}")

    date_str = date_match.group()
    formatted_date = datetime.strptime(date_str, date_format).strftime("%Y-%m-%d")

    # Extract the band information from the file name
    band_start_index = file_name.find(date_str) + date_length
    band = file_name[band_start_index + 1 : file_name.rfind(".")] if file_name[band_start_index] != "." else "0"

    return formatted_date, band


def process_viirs_filename(file_path: Path) -> Tuple[str, str]:
    """Processes the filename of a VIIRS satellite file to extract date and band."""
    return parse_filename_date_and_band(file_path, date_format="%Y%j", date_length=7)


def process_s1_filename(file_path: Path) -> Tuple[str, str]:
    """Processes the filename of a Sentinel-1 satellite file to extract date and band."""
    return parse_filename_date_and_band(file_path, date_format="%Y%m%d", date_length=8)


def process_s2_filename(file_path: Path) -> Tuple[str, str]:
    """Processes the filename of a Sentinel-2 satellite file to extract date and band."""
    return parse_filename_date_and_band(file_path, date_format="%Y%m%d", date_length=8)


def process_landsat_filename(file_path: Path) -> Tuple[str, str]:
    """Processes the filename of a Landsat satellite file to extract date and band."""
    return parse_filename_date_and_band(file_path, date_format="%Y-%m-%d", date_length=10)


def process_ground_truth_filename(file_path: Path) -> Tuple[str, str]:
    """Returns a fixed date and band for ground truth files."""
    return "0001-01-01", "0"


def get_filename_pattern(satellite_type: SatelliteType) -> str:
    """Returns the filename pattern for a given satellite type."""
    patterns = {
        SatelliteType.VIIRS: "DNB_VNP46A1_",
        SatelliteType.S1: "S1A_IW_GRDH_",
        SatelliteType.S2: "L2A_",
        SatelliteType.LANDSAT: "LC08_L1TP_",
        SatelliteType.GT: "groundTruth.tif",
    }
    return patterns[satellite_type]


def get_satellite_files(tile_dir: Path, satellite_type: SatelliteType) -> List[Path]:
    """Retrieves all satellite files in the tile directory matching the satellite type pattern."""
    pattern = get_filename_pattern(satellite_type)
    return [file for file in tile_dir.iterdir() if file.is_file() and pattern in file.name and file.suffix == ".tif"]


def get_grouping_function(satellite_type: SatelliteType) -> Callable:
    """Returns the function to group satellite files by date and band based on satellite type."""
    grouping_functions = {
        SatelliteType.VIIRS: process_viirs_filename,
        SatelliteType.S1: process_s1_filename,
        SatelliteType.S2: process_s2_filename,
        SatelliteType.LANDSAT: process_landsat_filename,
        SatelliteType.GT: process_ground_truth_filename,
    }
    return grouping_functions[satellite_type]


def get_unique_dates_and_bands(tile_dir: Path, satellite_type: SatelliteType) -> Tuple[List[str], List[str]]:
    """Extracts unique dates and bands from satellite files in the given directory."""
    satellite_files = get_satellite_files(tile_dir, satellite_type)
    grouping_function = get_grouping_function(satellite_type)

    dates, bands = set(), set()
    for file in satellite_files:
        date, band = grouping_function(file)
        dates.add(date)
        bands.add(band)

    return sorted(dates), sorted(bands)


def get_parent_tile_id(tile_dir: Path) -> str:
    """Returns the name (parent_tile_id) of the tile directory as a string."""
    return tile_dir.name


def read_satellite_file(satellite_file: Path) -> np.ndarray:
    """Reads a satellite file into a numpy array of type float32."""
    try:
        return tifffile.imread(str(satellite_file)).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error reading file {satellite_file}: {e}")


def load_satellite(tile_dir: Path, satellite_type: SatelliteType) -> xr.DataArray:
    """Loads satellite data for a given satellite type from a directory of tile files."""
    file_names = get_satellite_files(tile_dir, satellite_type)
    grouping_function = get_grouping_function(satellite_type)
    parent_tile_id = get_parent_tile_id(tile_dir)
    dates, bands = get_unique_dates_and_bands(tile_dir, satellite_type)

    data = [
        np.stack([read_satellite_file(file) for file in file_names if grouping_function(file) == (date, band)])
        for date in dates for band in bands
    ]
    stacked_data = np.stack(data, axis=0).reshape((len(dates), len(bands), *data[0].shape[-2:]))

    coords = {"date": dates, "band": bands, "height": range(stacked_data.shape[-2]), "width": range(stacked_data.shape[-1])}
    data_array = xr.DataArray(stacked_data, dims=("date", "band", "height", "width"), coords=coords)

    data_array.attrs["satellite_type"] = satellite_type.value
    data_array.attrs["tile_dir"] = str(tile_dir)
    data_array.attrs["parent_tile_id"] = parent_tile_id

    return data_array


def load_satellite_list(tile_dir: Path, satellite_type_list: List[SatelliteType]) -> List[xr.DataArray]:
    """Loads satellite data for multiple satellite types from a directory of tile files."""
    return [load_satellite(tile_dir, satellite_type) for satellite_type in satellite_type_list]


def load_satellite_dir(data_dir: Path, satellite_type_list: List[SatelliteType]) -> List[List[xr.DataArray]]:
    """Loads satellite data for multiple tiles and satellite types from a directory of tile files."""
    return [load_satellite_list(tile_dir, satellite_type_list) for tile_dir in data_dir.iterdir() if tile_dir.is_dir()]


def create_satellite_dataset_list(
    list_of_data_array_list: List[List[xr.DataArray]],
    satellite_type_list: List[SatelliteType],
    list_of_preprocess_func_list: List[List[Callable]] = None,
) -> List[xr.Dataset]:
    """
    Combines data arrays from multiple tiles into a single dataset for each satellite type.

    Parameters
    ----------
    list_of_data_array_list : List[List[xr.DataArray]]
        A list of tiles, each element containing a list of satellite data arrays.
    satellite_type_list : List[SatelliteType]
        A list of satellite types to include in the dataset.
    list_of_preprocess_func_list : List[List[Callable]], optional
        List of preprocessing functions to apply to the data arrays.

    Returns
    -------
    List[xr.Dataset]
        List of xarray datasets, each containing data for a single satellite type over all tiles.
    """
    data_dict_list = [{} for _ in satellite_type_list]

    for tile_data_list in list_of_data_array_list:
        for i, data_array in enumerate(tile_data_list):
            if list_of_preprocess_func_list and list_of_preprocess_func_list[i]:
                for func in list_of_preprocess_func_list[i]:
                    data_array = func(data_array)
            data_dict_list[i][data_array.attrs["parent_tile_id"]] = data_array

    return [xr.Dataset(data_dict).assign_attrs({"satellite_type": satellite_type.value})
            for data_dict, satellite_type in zip(data_dict_list, satellite_type_list)]
