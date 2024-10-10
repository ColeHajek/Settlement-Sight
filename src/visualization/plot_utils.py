""" This module contains functions for plotting satellite images. """
import sys
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# Local modules
sys.path.append(".")
from src.utilities import get_satellite_dataset_size


def flatten_dataset(data_set: xr.DataArray) -> np.ndarray:
    """
    Flatten dataset into 1D array with shape (tile * time * band * height * width).
    """
    return xr.concat(list(data_set.data_vars.values()), dim='date').stack(z=data_set.dims).values


def flatten_dataset_by_band(data_set: xr.DataArray) -> np.ndarray:
    """
    Flatten dataset by band into shape (band, tile * time * height * width).
    """
    return np.array([flatten_dataset(data_set.sel(band=band)) for band in data_set.band.values])


def _create_plot(data_array: np.ndarray, ax, n_bins=100, scale='log', **kwargs):
    """Helper function to create a histogram plot on the given axis."""
    ax.set_yscale(scale)
    ax.hist(data_array, bins=n_bins)
    ax.tick_params(axis='both', which='major', labelsize=kwargs.get("labelsize", 12))
    ax.set_title(kwargs.get("title", ""), fontsize=kwargs.get("fontsize", 9))


def _save_or_show(fig, image_dir, image_name):
    """Helper function to save or display the figure."""
    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        path = image_dir / f"{image_name}.png"
        path.touch(exist_ok=True)
        fig.savefig(path, format="png")
        plt.close()
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()


def plot_histogram(data_set: xr.Dataset, image_dir: Path, n_bins=20, satellite_name="", title_template=""):
    """
    General function to plot histograms for any satellite dataset.
    """
    num_tiles, num_dates, num_bands = len(data_set), *get_satellite_dataset_size(data_set)
    flattened_dataset_list = flatten_dataset_by_band(data_set)

    fig, ax = plt.subplots(
        num_bands if num_bands > 1 else 1, 1, figsize=(4, 4), squeeze=False, tight_layout=True
    )

    for band, band_list in enumerate(flattened_dataset_list):
        _create_plot(
            band_list,
            ax[band, 0],
            n_bins=n_bins,
            title=f"{satellite_name} Histogram (log scaled)\n{num_tiles} tiles, {num_dates} dates\nBand {data_set.band.values[band]}",
            fontsize=12 if satellite_name == "Sentinel 1" else 9,
            labelsize=15 if satellite_name == "Sentinel 1" else 9
        )

    _save_or_show(fig, image_dir, f"{satellite_name.lower()}_histogram")


def plot_viirs_histogram(data_set: xr.Dataset, image_dir: Path = None, n_bins=100) -> None:
    plot_histogram(data_set, image_dir, n_bins=n_bins, satellite_name="VIIRS")


def plot_sentinel1_histogram(data_set: xr.Dataset, image_dir: Path = None, n_bins=20) -> None:
    plot_histogram(data_set, image_dir, n_bins=n_bins, satellite_name="Sentinel 1")


def plot_sentinel2_histogram(data_set: xr.Dataset, image_dir: Path = None, n_bins=20) -> None:
    plot_histogram(data_set, image_dir, n_bins=n_bins, satellite_name="Sentinel 2")


def plot_landsat_histogram(data_set: xr.Dataset, image_dir: Path = None, n_bins=20) -> None:
    plot_histogram(data_set, image_dir, n_bins=n_bins, satellite_name="Landsat 8")


def plot_gt_histogram(data_set: xr.Dataset, image_dir: Path = None, n_bins=4) -> None:
    plot_histogram(data_set, image_dir, n_bins=n_bins, satellite_name="Ground Truth", title_template="GT Histogram (log scaled), {num_tiles} tiles")


def plot_image(data_array: xr.DataArray, image_dir: Path, image_name: str, title: str):
    """Helper function to plot a single image."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), squeeze=False, tight_layout=True)
    ax[0, 0].imshow(data_array[0][0])
    ax[0, 0].set_title(title, fontsize=9)
    _save_or_show(fig, image_dir, image_name)


def plot_gt(data_array: xr.DataArray, image_dir: Path = None) -> None:
    plot_image(data_array, image_dir, "ground_truth", f"Ground Truth\n{data_array.attrs['parent_tile_id']}")


def plot_max_projection_viirs(data_array: xr.DataArray, image_dir: Path = None) -> None:
    plot_image(data_array, image_dir, "viirs_max_projection", f"Max Projection VIIRS\n{data_array.attrs['parent_tile_id']}")


def plot_viirs(data_array: xr.DataArray, image_dir: Path = None) -> None:
    num_dates = data_array.shape[0]
    fig, ax = plt.subplots(3, 3, figsize=(4, 4), squeeze=False, tight_layout=True)

    for i, (row, col) in enumerate([(i // 3, i % 3) for i in range(9)]):
        if i < num_dates:
            ax[row, col].imshow(data_array[i, 0])
            ax[row, col].set_title(f"VIIRS Image\n{data_array['date'][i].values}", fontsize=9)
            ax[row, col].axis("off")
        else:
            ax[row, col].set_visible(False)

    _save_or_show(fig, image_dir, "viirs_plot_by_date")


def create_rgb_composite_s1(data_array: xr.DataArray, image_dir: Path = None) -> None:
    num_dates = data_array.shape[0]
    fig, ax = plt.subplots(2, int(np.floor(np.divide(num_dates, 2))), figsize=(4, 4), squeeze=False, tight_layout=True)

    for i, (row, col) in enumerate([(i // 2, i % 2) for i in range(num_dates)]):
        red, green = data_array[i].sel(band="VH"), data_array[i].sel(band="VV")
        blue = np.clip(red - green, 0, 1)
        rgb_image = np.stack((red, green, blue), axis=2)

        ax[row, col].imshow(rgb_image)
        ax[row, col].set_title(f"Sentinel 1 RGB composite\n{data_array[i]['date'].values}", fontsize=9)
        ax[row, col].axis("off")

    _save_or_show(fig, image_dir, "plot_sentinel1")


def plot_satellite_by_bands(data_array: xr.DataArray, bands_to_plot: List[str], image_dir: Path = None):
    """Plots satellite images by the specified bands."""
    num_dates = data_array.shape[0]
    fig, ax = plt.subplots(num_dates, len(bands_to_plot), figsize=(4, 4), squeeze=False, tight_layout=True)

    for band_ind, bands in enumerate(bands_to_plot):
        for date_ind, date in enumerate(data_array.date):
            img_stack = np.stack([data_array.sel(date=date, band=band).values for band in bands], axis=2)
            ax[date_ind, band_ind].imshow(img_stack.transpose(2, 1, 0), cmap="gray")
            ax[date_ind, band_ind].axis("off")
            ax[date_ind, band_ind].set_title(f"{data_array.attrs['satellite_type']} {' '.join(bands)}\n{data_array['date'][date_ind].values}", fontsize=9)

    _save_or_show(fig, image_dir, f"plot_{data_array.attrs['satellite_type']}")
