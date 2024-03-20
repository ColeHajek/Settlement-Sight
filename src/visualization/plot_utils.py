""" This module contains functions for plotting satellite images. """
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..preprocessing.file_utils import Metadata
from ..preprocessing.preprocess_sat import quantile_clip
from ..preprocessing.preprocess_sat import minmax_scale
from ..preprocessing.preprocess_sat import (
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
    preprocess_viirs
)

import sys
import pyprojroot
from tqdm import tqdm
from typing import Tuple, Dict, List

from copy import deepcopy
root = pyprojroot.here()
sys.path.append(str(root))

import pyprojroot
root = pyprojroot.here()
sys.path.append(root)
from src.esd_data.dataset import DSE
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor
)
from torchvision import transforms
from src.preprocessing.subtile_esd import restitch

def plot_viirs_histogram(
        viirs_stack: np.ndarray,
        image_dir: None | str | os.PathLike = None,
        n_bins=100
        ) -> None:
    
    """
    This function plots the histogram over all VIIRS values.
    note: viirs_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    viirs_stack = preprocess_data(viirs_stack,"viirs")
    flattened_data = viirs_stack.flatten()

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_data, bins=n_bins, range=[0,np.max(flattened_data)], color='blue', alpha=0.7)
    plt.title('VIIRS_histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "VIIRS_histogram.png")
        plt.close()
    return

def plot_sentinel1_histogram(
        sentinel1_stack: np.ndarray,
        metadata: List[Metadata],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    
    """
    This function plots the Sentinel-1 histogram over all Sentinel-1 values.
    note: sentinel1_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    sentinel1_stack : np.ndarray
        The Sentinel-1 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """

    sentinel1_stack = preprocess_data(sentinel1_stack,"sentinel1")
    flattened_data = sentinel1_stack.flatten()
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_data, bins=n_bins, range=[0,np.max(flattened_data)], color='blue', alpha=0.7)
    plt.title('Sentinel1 Histogram')
    plt.xlabel(np.max(flattened_data))
    plt.ylabel('Frequency')
    plt.grid(True)

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel1_histogram.png")
        plt.close()
    return

def plot_sentinel2_histogram(
        sentinel2_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20) -> None:
    """
    This function plots the Sentinel-2 histogram over all Sentinel-2 values.

    Parameters
    ----------
    sentinel2_stack : np.ndarray
        The Sentinel-2 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-2 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """

    flattened_data = sentinel2_stack.flatten()

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_data, bins=n_bins, range=[0,np.max(flattened_data)], color='blue', alpha=0.7)
    plt.title('Sentinel2 Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel2_histogram.png")
        plt.close()
    return

def plot_landsat_histogram(
        landsat_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the landsat histogram over all landsat values over all
    tiles present in the landsat_stack.

    Parameters
    ----------
    landsat_stack : np.ndarray
        The landsat image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the landsat image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    landsat_stack = preprocess_data(landsat_stack,"landsat")
    landsat_stack = minmax_scale(landsat_stack,False)
    flattened_data = landsat_stack.flatten()

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_data, bins=n_bins, range=[0, 256], color='blue', alpha=0.7)
    plt.title('Histogram of Pixel Values')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "landsat_histogram.png")
        plt.close()
    return

def plot_gt_counts(ground_truth: np.ndarray,
                   image_dir: None | str | os.PathLike = None
                   ) -> None:
    """
    This function plots the ground truth histogram over all ground truth
    values over all tiles present in the groundTruth_stack.

    Parameters
    ----------
    groundTruth : np.ndarray
        The ground truth image stack volume.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    
    flattened_data = ground_truth.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(flattened_data, range=[1,5])
    plt.title('Histogram of GT Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth_histogram.png")
        plt.close()
    return

def plot_viirs(
        viirs: np.ndarray, plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """ This function plots the VIIRS image.

    Parameters
    ----------
    viirs : np.ndarray
        The VIIRS image.
    plot_title : str
        The title of the plot.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    viirs = preprocess_data(viirs,"viirs")
    plt.figure(figsize=(6, 6))

    plt.imshow(viirs)  # Use grayscale color map for 2D images
    plt.title(plot_title)
    
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_max_projection.png")
        plt.close()
    return

def plot_viirs_by_date(
        viirs_stack: np.array,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None) -> None:
    """
    This function plots the VIIRS image by band in subplots.

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the VIIRS image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    
    viirs_stack = preprocess_data(viirs_stack,"viirs")
    fig, ax = plt.subplots(nrows=len(metadata[0].bands), ncols=len(metadata), figsize=(15, 10))  # Adjust the figsize as needed
    fig.suptitle('VIIRS Plot by Date')

    for i, data in enumerate(metadata):
        for j, band in enumerate(data.bands):
            if len(metadata[0].bands) == 1:
                ax[i].set_xlabel(f'Time: {data.time}')
                ax[i].imshow(viirs_stack[i, j, :])
            else:
                ax[j, i].set_xlabel(f'Time: {data.time}')
                ax[j, i].imshow(viirs_stack[i, j])
            ax[j].set_ylabel(f'Band: {band}')

    plt.tight_layout(pad=3.0)  # Adjust padding as needed
    fig.subplots_adjust(top=0.9)  # Adjust top spacing to fit the suptitle
    if image_dir is None:
        plt.show()
    else:
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(image_dir) / "viirs_plot_by_date.png")
        plt.close()
    return
    

def preprocess_data(
        satellite_stack: np.ndarray,
        satellite_type: str
        ) -> np.ndarray:
    
    """
    This function preprocesses the satellite data based on the satellite type.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    np.ndarray
    """
    if satellite_type == "viirs":
        satellite_stack = preprocess_viirs(satellite_stack)
    elif satellite_type == "sentinel1":
        satellite_stack = preprocess_sentinel1(satellite_stack)
    elif satellite_type == "sentinel2":
        satellite_stack = preprocess_sentinel2(satellite_stack)
    elif satellite_type == "landsat":
        satellite_stack = preprocess_landsat(satellite_stack)
    else:
        print("ERROR no such sat type exists")

    return satellite_stack


def create_rgb_composite_s1(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function creates an RGB composite for Sentinel-1.
    This function needs to extract the band identifiers from the metadata
    and then create the RGB composite. For the VV-VH composite, after
    the subtraction, you need to minmax scale the image.

    Parameters
    ----------
    processed_stack : np.ndarray
        The Sentinel-1 image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot. Cannot accept more than 3 bands.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    
    if len(bands_to_plot) > 3:
        raise ValueError("Cannot plot more than 3 bands.")
    
    #1. get the rgb np array to plot
    #r=VV, g=VH, b=VV-VH

    #init rgb array
    width = processed_stack.shape[-1]  # Length of the last dimension
    height = processed_stack.shape[-2]  # Length of the second last dimension
    rgb = np.zeros((len(metadata),height,width,3))
    newband = np.zeros((len(metadata),3,height,width))
    

    #fill rgb array with data from processed_stack
    for i,date in enumerate(metadata):
        vv = np.zeros((height,width))
        vh = np.zeros((height,width))
        vvvh = np.zeros((height,width))

        for j,band in enumerate(date.bands):
            if band=="VH":
                vh = processed_stack[i][j]
            if band=="VV":
                vv = processed_stack[i][j]
        
        vvvh = vv-vh
        print(vvvh.shape)
        vvvh = minmax_scale(vvvh,True)
        
        
        rgb[i,:,:,0] = vv
        rgb[i,:,:,1] = vh
        rgb[i,:,:,2] = vvvh
        

    #2. plot the arrays by date
    fig, ax = plt.subplots(nrows=len(metadata), figsize=(40, 10 * len(metadata)))
    fig.suptitle('Sentinel1 RGB Composite')

    for i,date in enumerate(metadata):
        ax[i].set_xlabel(f'Time: {date.time}')
        ax[i].imshow(rgb[i])
        
    
    plt.tight_layout(pad=3.0, h_pad=5.0)  # Adjust horizontal padding between plots
    fig.subplots_adjust(top=0.95)  # Adjust top spacing to fit the suptitle

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "plot_sentinel1.png")
        plt.close()

    return

def validate_band_identifiers(
          bands_to_plot: List[List[str]],
          band_mapping: dict) -> None:
    """
    This function validates the band identifiers.

    Parameters
    ----------
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.

    Returns
    -------
    None
    """
    for band_group in bands_to_plot:
        for band in band_group:
            if band not in band_mapping:
                raise ValueError(f"Invalid band identifier '{band}' found. It's not present in the band mapping.")
    return


def plot_images(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        band_mapping: dict,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ):
    """
    This function plots the satellite images.

    Parameters
    ----------
    processed_stack : np.ndarray
        The satellite image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    
    '''print_list_metadata(metadata)
    print("stack shape:",processed_stack.shape)
    print("bands to plot:",bands_to_plot)
    print("bandmapping:",band_mapping)'''

    width = processed_stack.shape[-1]  # Length of the last dimension
    height = processed_stack.shape[-2]  # Length of the second last dimension

    nrows = len(bands_to_plot)
    ncols = processed_stack.shape[0]
    fig,ax = plt.subplots(nrows,ncols,figsize=(20 * ncols, 20 * nrows))
    
    fig.supxlabel('Time')
    for w, band_list in enumerate(bands_to_plot):
        for i, time_step in enumerate(processed_stack):
            data = metadata[i]
            combined_bands = np.zeros((height,width,len(band_list)))
            for j, band in enumerate(band_list):
                combined_bands[:,:,j] = time_step[band_mapping[band]]
                

            if len(bands_to_plot) == 1:
                ax[i].imshow(combined_bands, cmap='gray')
                ax[i].set_xlabel(data.time)
            else:
                ax[w,i].imshow(combined_bands)
                ax[w,i].set_xlabel(data.time)

    if image_dir is None:
        plt.show()
    elif metadata[0].satellite_type== "sentinel2":
        plt.savefig(Path(image_dir) / "plot_sentinel2.png")
        plt.close()
    else:
        plt.savefig(Path(image_dir) / "landsat.png")
        plt.close()
    return
    


def plot_satellite_by_bands(
        satellite_stack: np.ndarray,
        metadata: List[Metadata],
        bands_to_plot: List[List[str]],
        satellite_type: str,
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the satellite image by band in subplots.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    bands_to_plot : List[List[str]]
        The bands to plot.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    None
    """
    
    processed_stack = preprocess_data(satellite_stack, satellite_type)

    if satellite_type == "sentinel1":
        create_rgb_composite_s1(processed_stack, bands_to_plot, metadata, image_dir=image_dir)
    else:
        band_ids_per_timestamp = extract_band_ids(metadata)
        all_band_ids = [band_id for timestamp in band_ids_per_timestamp for
                        band_id in timestamp]
        
        unique_band_ids = sorted(list(set(all_band_ids)))
        band_mapping = {band_id: idx for
                        idx, band_id in enumerate(unique_band_ids)}
        validate_band_identifiers(bands_to_plot, band_mapping)
        
        plot_images(processed_stack, bands_to_plot, band_mapping, metadata, image_dir)
        
    return


def extract_band_ids(metadata: List[Metadata]) -> List[List[str]]:
    """
    Extract the band identifiers from file names for each timestamp based on
    satellite type.

    Parameters
    ----------
    file_names : List[List[str]]
        A list of file names.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    List[List[str]]
        A list of band identifiers.
    """
    band_ids = [meta.bands for meta in metadata]
    return band_ids


def plot_ground_truth(
        ground_truth: np.array,
        plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the groundTruth image.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.
    """
    '''
    1	Human settlements without electricity (Region of Interest)	    ff0000
    2	No human settlements without electricity	    0000ff
    3	Human settlements with electricity	    ffff00
    4	No human settlements with electricity	    b266ff
'''
        
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)

    
    plt.figure(figsize=(6, 6))
    plt.imshow(ground_truth[0][0],cmap=cmap)  
    plt.title(plot_title)
    
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth.png")
        plt.close()
    return

def plot_restitched_from_tiles(subtile_dir: str | os.PathLike, 
                               satellite_type: str, 
                               tile_id: str, 
                               x_range: Tuple[int, int], 
                               y_range: Tuple[int, int], 
                               bands: List[str] = ["04", "03", "02"],
                               image_dir: str | os.PathLike | None = None):
    stitched, metadata = restitch(subtile_dir, satellite_type, tile_id, x_range, y_range)
    satellite_bands = metadata[0][0].satellites[satellite_type].bands
    bands_index = []
    for b in bands:
        bands_index.append(satellite_bands.index(b))
    
    plt.title(f"{tile_id} restitched")
    plt.imshow(np.transpose(stitched[0,bands_index,:,:], axes=(1,2,0)))

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "restitched_tile.png")
        plt.close()

def plot_transforms(subtile_folder: str | os.PathLike, 
                    data_idx: int,
                    selected_bands: Dict[str, List[str]] = {"sentinel2": ["04", "03", "02"]},
                    image_dir: str | os.PathLike | None = None):
    transforms_to_apply = [
                AddNoise(0, 0.5),
                Blur(20),
                RandomHFlip(p=1.0),
                RandomVFlip(p=1.0)
            ]

    names = ["Noise", "Blur", "HFlip", "VFlip"]

    fig, axs = plt.subplots(len(transforms_to_apply), 5)

    for i, transform in enumerate(transforms_to_apply):
        dataset = DSE(subtile_folder, 
                    selected_bands=selected_bands,
                    transform = transform)
        X, y, metadata = dataset[data_idx]

        X = X.reshape(4,3,200,200)

        plt.suptitle(f"{metadata.parent_tile_id}, subtile ({metadata.x_gt}, {metadata.y_gt})")

        for j in range(X.shape[0]):
            axs[i, j].set_title(f"t = {j}, tr = {names[i]}")
            axs[i, j].imshow(np.dstack([X[j,0],X[j,1],X[j,2]]))
        axs[i, -1].set_title("Ground Truth")
        axs[i, -1].imshow(y[0])

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "restitched_tile.png")
        plt.close()

def plot_2D_scatter_plot(X_dim_red, y_flat, projection_name, image_dir: str | os.PathLike | None = None):
    # Create a list of colors
    colors = np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff'])
    labels = ['Human Settlements Without Electricity', 'No Human Settlement Without Electricity', 'Human Settlements With Electricity', 'No Human Settlements with Electricity']
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors[y_flat[:, 0].astype(int)-1])
    for i in range(len(colors)):
        plt.scatter(X_dim_red[y_flat[:, 0].astype(int) == i+1, 0], X_dim_red[y_flat[:, 0].astype(int) == i+1, 1], c=colors[i], label=labels[i])
    plt.xlabel(f"{projection_name} 1")
    plt.ylabel(f"{projection_name} 2")
    plt.title(f"{projection_name} projection of tiles")
    plt.legend()

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / f"{projection_name}_scatterplot.png")
        plt.close()

if __name__ == "__main__":
    plot_restitched_from_tiles(root/"data/processed/Train/subtiles", "sentinel2", "Tile1", (0,4), (0,4))

    plot_transforms(root / 'data'/'processed'/'Train'/'subtiles', 0)