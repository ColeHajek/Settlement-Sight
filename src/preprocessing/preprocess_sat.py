"""
Module for applying preprocessing functions on the IEEE GRSS ESD satellite data.
"""

import sys
from typing import Callable
import numpy as np
import xarray as xr
import scipy.ndimage

sys.path.append(".")
from src.utilities import SatelliteType


def apply_per_image_operation(data_array: xr.DataArray, operation: Callable, **kwargs) -> xr.DataArray:
    """
    Applies the given operation on each (date, band) pair of the data array.

    Parameters
    ----------
    data_array : xr.DataArray
        The input data array with shape (date, band, height, width).
    operation : Callable
        The function to apply on each (date, band) pair.
    **kwargs
        Additional keyword arguments to pass to the operation function.

    Returns
    -------
    xr.DataArray
        The data array after applying the operation on each (date, band) pair.
    """
    for date in range(data_array.shape[0]):
        for band in range(data_array.shape[1]):
            data_array[date, band] = operation(data_array[date, band], **kwargs)
    return data_array


def gaussian_filter(data_array: xr.DataArray, sigma: float = 1) -> xr.DataArray:
    """
    Applies a Gaussian filter to each (height, width) image in the data array.

    Parameters
    ----------
    data_array : xr.DataArray
        The input data array with shape (date, band, height, width).
    sigma : float, optional
        The standard deviation for the Gaussian kernel.

    Returns
    -------
    xr.DataArray
        The data array after applying the Gaussian filter.
    """
    return apply_per_image_operation(data_array, scipy.ndimage.gaussian_filter, sigma=sigma)


def quantile_clip(data_array: xr.DataArray, clip_quantile: float, group_by_time: bool = True) -> xr.DataArray:
    """
    Clips the outliers in the data array based on a given quantile.

    Parameters
    ----------
    data_array : xr.DataArray
        The input data array with shape (date, band, height, width).
    clip_quantile : float
        The quantile value for clipping outliers (between 0 and 0.5).
    group_by_time : bool, optional
        If True, quantile limits are shared along the time dimension.

    Returns
    -------
    xr.DataArray
        The clipped data array.
    """
    if group_by_time:
        for band_index in range(data_array.shape[1]):
            q_low = np.quantile(data_array[:, band_index].values, q=clip_quantile, method="lower")
            q_high = np.quantile(data_array[:, band_index].values, q=1 - clip_quantile, method="higher")
            data_array[:, band_index] = np.clip(data_array[:, band_index].values, q_low, q_high)
    else:
        def clip_func(array, q_low, q_high):
            return np.clip(array, q_low, q_high)

        return apply_per_image_operation(data_array, clip_func, q_low=clip_quantile, q_high=1 - clip_quantile)

    return data_array


def minmax_scale(data_array: xr.DataArray, group_by_time: bool = True) -> xr.DataArray:
    """
    Scales the values in the data array to the range [0, 1].

    Parameters
    ----------
    data_array : xr.DataArray
        The input data array with shape (date, band, height, width).
    group_by_time : bool, optional
        If True, min-max scaling is shared along the time dimension.

    Returns
    -------
    xr.DataArray
        The min-max scaled data array.
    """
    def scale(array, min_val, max_val):
        return (array - min_val) / (max_val - min_val) if min_val != max_val else np.ones_like(array)

    if group_by_time:
        for band_index in range(data_array.shape[1]):
            min_value, max_value = data_array[:, band_index].min().values, data_array[:, band_index].max().values
            data_array[:, band_index] = scale(data_array[:, band_index], min_value, max_value)
    else:
        return apply_per_image_operation(data_array, scale)

    return data_array


def brighten(data_array: xr.DataArray, alpha: float = 0.13, beta: float = 0) -> xr.DataArray:
    """
    Brightens the image using the formula: `alpha * pixel_value + beta`.

    Parameters
    ----------
    data_array : xr.DataArray
        The input data array with shape (date, band, height, width).
    alpha : float, optional
        The multiplicative factor for brightness adjustment.
    beta : float, optional
        The additive factor for brightness adjustment.

    Returns
    -------
    xr.DataArray
        The brightened data array.
    """
    def brighten_func(array, alpha, beta):
        return alpha * array + beta

    return apply_per_image_operation(data_array, brighten_func, alpha=alpha, beta=beta)


def gammacorr(data_array: xr.DataArray, gamma: float = 2) -> xr.DataArray:
    """
    Applies gamma correction to the image using the formula: `pixel_value ** (1/gamma)`.

    Parameters
    ----------
    data_array : xr.DataArray
        The input data array with shape (date, band, height, width).
    gamma : float, optional
        The gamma correction parameter.

    Returns
    -------
    xr.DataArray
        The gamma corrected data array.
    """
    def gamma_func(array, gamma):
        return np.power(array, 1 / gamma)

    return apply_per_image_operation(data_array, gamma_func, gamma=gamma)


def convert_data_to_db(data_array: xr.DataArray) -> xr.DataArray:
    """
    Converts raw Sentinel-1 SAR data to decibel (dB) format using logarithmic transformation.

    Parameters
    ----------
    data_array : xr.DataArray
        The input data array with shape (date, band, height, width).

    Returns
    -------
    xr.DataArray
        The converted data array in dB format.
    """
    return apply_per_image_operation(data_array, lambda x: 10 * np.log10(np.where(x != 0, x, np.nan)))


def preprocess_sentinel1(data_array: xr.DataArray, clip_quantile: float = 0.01, sigma: float = 1) -> xr.DataArray:
    """
    Preprocesses Sentinel-1 data by converting to dB, clipping outliers, applying Gaussian filter, and min-max scaling.

    Parameters
    ----------
    data_array : xr.DataArray
        The Sentinel-1 data array with shape (date, band, height, width).
    clip_quantile : float, optional
        The quantile value for clipping outliers.
    sigma : float, optional
        The standard deviation for Gaussian kernel.

    Returns
    -------
    xr.DataArray
        The preprocessed Sentinel-1 data array.
    """
    data_array = convert_data_to_db(data_array)
    data_array = quantile_clip(data_array, clip_quantile)
    data_array = gaussian_filter(data_array, sigma)
    return minmax_scale(data_array)

def preprocess_sentinel2(
    sentinel2_data_array: xr.DataArray, clip_quantile: float = 0.05, gamma: float = 2.2
) -> xr.DataArray:
    """
    In this function we will preprocess sentinel-2. The steps for
    preprocessing are the following:
        - Clip higher and lower quantile outliers
        - Apply a gamma correction
        - Minmax scale

    Parameters
    ----------
    sentinel2_data_array : xr.DataArray
        The sentinel2_data_array to be preprocessed. The shape of the array is (date, band, height, width).
    Returns
    -------
    xr.DataArray
        The processed sentinel2_data_array. The shape of the array is (date, band, height, width).
    """
    data_array = quantile_clip(sentinel2_data_array, clip_quantile)
    data_array = gammacorr(data_array, gamma)
    return minmax_scale(data_array)

def preprocess_landsat(
    landsat_data_array: xr.DataArray, clip_quantile: float = 0.01, gamma: float = 2.2
) -> xr.DataArray:
    """
    In this function we will preprocess landsat. The steps for preprocessing
    are the following:
        - Clip higher and lower quantile outliers
        - Apply a gamma correction
        - Minmax scale

    Parameters
    ----------
    landsat_data_array : xr.DataArray
        The landsat_data_array to be preprocessed. The shape of the array is (date, band, height, width).
    Returns
    -------
    xr.DataArray
        The processed landsat_data_array. The shape of the array is (date, band, height, width).
    """
    data_array = quantile_clip(landsat_data_array, clip_quantile)
    data_array = gammacorr(data_array, gamma)
    return minmax_scale(data_array)

def preprocess_viirs(
    viirs_data_array: xr.DataArray, clip_quantile: float = 0.05
) -> xr.DataArray:
    """
    In this function we will preprocess viirs. The steps for preprocessing are
    the following:
        - Clip higher and lower quantile outliers
        - Minmax scale

    Parameters
    ----------
    viirs_data_array : xr.DataArray
        The viirs_data_array to be preprocessed. The shape of the array is (date, band, height, width).
    Returns
    -------
    xr.DataArray
        The processed viirs_data_array. The shape of the array is (date, band, height, width).
    """
    data_array = quantile_clip(viirs_data_array, clip_quantile)
    return minmax_scale(data_array)

def maxprojection_viirs(data_array: xr.DataArray) -> xr.DataArray:
    """
    This function takes a VIIRS data_array and returns a single image that is the max projection of the images
    to identify areas with the highest levels of nighttime lights or electricity usage.

    The value of any pixel is the maximum value over all time steps, like shown below

       Date 1               Date 2                      Output
    -------------       -------------               -------------
    | 0 | 1 | 0 |       | 2 | 0 | 0 |               | 2 | 1 | 0 |
    -------------       -------------   ======>     -------------
    | 0 | 0 | 3 |       | 0 | 4 | 0 |   ======>     | 0 | 4 | 3 |
    -------------       -------------   ======>     -------------
    | 9 | 6 | 0 |       | 0 | 8 | 7 |               | 9 | 8 | 7 |
    -------------       -------------               -------------

    Parameters
    ----------
    data_array : xr.DataArray
        The data_array to be brightened. The shape of the array is (date, band, height, width).
    Returns
    -------
    xr.DataArray
        Max projection of the VIIRS data_array. The shape of the array is (date, band, height, width)
    """
    # set the band index to 0 (VIIRS only has 1 band)
    bandInd = 0
    # set the maximum to the first image (date, band) from the data_array
    maxImg = data_array[0][bandInd]

    # iterate by date
    for date in range(data_array.shape[0]):
        # set the maximum to be the max of (maximum, current image (date, band)),
        # this can be done numerous ways, here are some suggestions:
        # https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        maxImg = np.maximum(maxImg, data_array[date, bandInd].values)
       

    # create a new data array (max_viirs_array) with shape (1, 1, 800, 800) and the
    # relevant dims and coords. You can use np.reshape to transform the maximum (you
    # just calculated it above) to have shape (1, 1, 800, 800)
    max_array = xr.DataArray(np.reshape(maxImg.values, (1, 1, 800, 800)),
                                    dims=("date", "band", "height", "width"),
                                    coords={"date": [date], "band": [0]})

    # set the attributes of the max_viirs_array. The satellite_type should be the
    # SatelliteType.VIIRS_MAX_PROJ, while the other 2 attributes can be the same as the
    # original data_array
    max_array.attrs["satellite_type"] = SatelliteType.VIIRS_MAX_PROJ.value
    max_array.attrs["tile_dir"] = data_array.attrs["tile_dir"]
    max_array.attrs["parent_tile_id"] = data_array.attrs["parent_tile_id"]
   
    return max_array