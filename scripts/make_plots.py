#Non functional work in progress

'''import argparse
from argparse import ArgumentParser

import sys

from pathlib import Path
from typing import List
import xarray as xr
#import numpy as np
#import matplotlib.pyplot as plt

sys.path.append(".")
# Local imports (assuming they're in your project directory)
from src.utilities import get_satellite_dataset_size
from src.visualization.plot_utils import (
    plot_viirs_histogram,
    plot_sentinel1_histogram,
    plot_sentinel2_histogram,
    plot_landsat_histogram,
    plot_gt_histogram,
    plot_satellite_by_bands,
)


class EDAOptions:
    """Class to store options for the EDA process."""
    def __init__(self,
                 data_dir: Path = "./data/processed/Train/subtiles/Tile1/0_0",
                 satellite_type: str = "VIIRS",
                 plot_type: str = "histogram",
                 tile_id: str = "Tile57",
                 bands: List[str] = ["B01", "B02", "B03"],
                 bins: int = 100,
                 output_dir: Path = "./plots"
                 ):
        
        self.data_dir = data_dir
        self.satellite_type = satellite_type
        self.plot_type = plot_type
        self.tile_id = tile_id
        self.bands = bands
        self.bins = bins
        self.output_dir = output_dir

#todo fix to load all items instead of just one fragment of a tile
def load_dataset(data_dir: Path, satellite_type: str) -> xr.Dataset:
    """Loads the satellite dataset from the given directory."""
    file_path = data_dir / f"{satellite_type}.nc"
    return xr.open_dataset(file_path)

#todo fix selection option
def plot_histograms(options: EDAOptions):
    """Plots histograms based on satellite type."""

    data_set = load_dataset(options.data_dir, options.satellite_type)
    
    plot_viirs_histogram(data_set, options.output_dir, options.bins)
    plot_sentinel1_histogram(data_set, options.output_dir, options.bins)
    plot_sentinel2_histogram(data_set, options.output_dir, options.bins)
    plot_landsat_histogram(data_set, options.output_dir, options.bins)
    plot_gt_histogram(data_set, options.output_dir, 4)
    

def plot_tile_by_bands(options: EDAOptions):
    """Plots specific tile by combining the selected bands."""
    if options.tile_id is None or options.bands is None:
        raise ValueError("Both tile_id and bands are required for tile plotting.")
    
    data_set = load_dataset(options.data_dir, options.satellite_type)
    tile_data = data_set.sel(tile_id=options.tile_id)
    plot_satellite_by_bands(tile_data, options.bands, options.output_dir)


def main(options):
    """Main function for EDA."""
    if options.plot_type == "histogram":
        plot_histograms(options)
    elif options.plot_type == "tile":
        plot_tile_by_bands(options)
    else:
        raise ValueError(f"Invalid input")


if __name__ == "__main__":
    # Check if running with command-line arguments
    parser = ArgumentParser()
    options = EDAOptions()
    
    parser.add_argument("--data-dir", type=Path, help="Directory containing satellite datasets.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save plots.")
    parser.add_argument("--satellite-type", type=str, choices=["VIIRS", "Sentinel1", "Sentinel2", "Landsat8", "GT"], help="Satellite type to analyze.")
    parser.add_argument("--plot-type", type=str, choices=["histogram", "tile"], help="Type of plot to generate: 'histogram' or 'tile'.")
    parser.add_argument("--tile-id", type=str, default=None, help="ID of the tile to plot.")
    parser.add_argument("--bands", nargs="+", default=None, help="Bands to combine and plot (only used for tile plots).")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for histograms (only used for histogram plots).")
    #default=config.raw_dir
    parse_args = parser.parse_args()

    config = EDAOptions(**parse_args.__dict__)
    main(options)
    '''