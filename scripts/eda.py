#Revised version not yet functional with new implementation

'''import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import pyprojroot
import matplotlib.pyplot as plt

# Get the root directory for the project
root = pyprojroot.here()

# Local imports
sys.path.append(".")
from src.utilities import ESDConfig, load_config
from src.preprocessing.data_loader import load_and_preprocess_satellite_data
from src.visualization.plot_utils import (
    plot_satellite_by_bands,
    plot_viirs_histogram,
    plot_gt,
    plot_gt_histogram,
    plot_sentinel2_histogram,
    plot_landsat_histogram,
    plot_sentinel1_histogram,
    plot_viirs
)


def main(options):
    """
    Main function for visualizing the dataset using the brown rice project modules.
    """
    # Define tile numbers and directories for the tiles
    tile_nums = options.tile_nums
    tile_dirs = [Path(options.raw_dir) / f"Tile{tile_num}" for tile_num in tile_nums]

    # Define directory for saving plots
    save_plots_dir = Path(root) / 'visualize'
    save_plots_dir.mkdir(parents=True, exist_ok=True)

    # Define test flags for visualization
    test_flags = {
        "viirs": True, #options.test_viirs,
        "histograms": True, #options.test_histograms,
        "sentinel1": True, #options.test_s1,
        "landsat": True, #options.test_landsat,
        "sentinel2": True, #options.test_s2,
        "ground_truth": True, #options.test_gt,
    }

    # Iterate over each tile directory and visualize based on test flags
    for tile_dir in tile_dirs:
        print(f"Processing tile: {tile_dir.name}")

        if test_flags["viirs"]:
            viirs_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.VIIRS)
            max_proj_viirs = viirs_stack.max(dim="date", skipna=True)  # Assuming the function has been updated
            plot_viirs(max_proj_viirs, image_dir=save_plots_dir)
            print("VIIRS max projection plot complete.")

        if test_flags["histograms"]:
            plot_histograms(tile_dir, save_plots_dir)

        if test_flags["sentinel1"]:
            sentinel1_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.SENTINEL1)
            plot_satellite_by_bands(sentinel1_stack, ['VV', 'VH', 'VV-VH'], image_dir=save_plots_dir)
            print("Sentinel-1 bands plot complete.")

        if test_flags["sentinel2"]:
            sentinel2_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.SENTINEL2)
            bands = ['04', '03', '02']
            plot_satellite_by_bands(sentinel2_stack, bands, image_dir=save_plots_dir)
            print("Sentinel-2 bands plot complete.")

        if test_flags["landsat"]:
            landsat_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.LANDSAT)
            bands = ['4', '3', '2']
            plot_satellite_by_bands(landsat_stack, bands, image_dir=save_plots_dir)
            print("Landsat bands plot complete.")

        if test_flags["ground_truth"]:
            gt_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.GT)
            plot_gt(gt_stack, image_dir=save_plots_dir)
            print("Ground truth plot complete.")


def plot_histograms(tile_dir: Path, save_plots_dir: Path):
    """
    Function to plot histograms for various satellite types.

    Parameters:
        tile_dir (Path): The directory containing tile data.
        save_plots_dir (Path): Directory to save the histograms.
    """
    viirs_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.VIIRS)
    plot_viirs_histogram(viirs_stack, image_dir=save_plots_dir)
    print("VIIRS histogram plot complete.")

    s1_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.SENTINEL1)
    plot_sentinel1_histogram(s1_stack, image_dir=save_plots_dir)
    print("Sentinel-1 histogram plot complete.")

    s2_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.SENTINEL2)
    plot_sentinel2_histogram(s2_stack, image_dir=save_plots_dir)
    print("Sentinel-2 histogram plot complete.")

    landsat_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.LANDSAT)
    plot_landsat_histogram(landsat_stack, image_dir=save_plots_dir)
    print("Landsat histogram plot complete.")

    gt_stack = load_and_preprocess_satellite_data(tile_dir, SatelliteType.GT)
    plot_gt_histogram(gt_stack, image_dir=save_plots_dir)
    print("Ground truth histogram plot complete.")


if __name__ == "__main__":
    config = load_config()  # Assuming `load_config()` reads configurations for the brown rice project.

    parser = ArgumentParser(description="Dataset visualization script for the brown rice project.")
    parser.add_argument("--tile_nums", type=int, nargs="+", default=[4], help="List of tile numbers to process.")
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory.")
    parser.add_argument("--processed_dir", type=str, default=config.processed_dir, help="Path to processed directory.")
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Results directory for saving plots.")
    
    # Optional flags for different types of plots
    parser.add_argument("--test_viirs", action="store_true", help="Plot VIIRS data.")
    parser.add_argument("--test_histograms", action="store_true", help="Plot histograms for all data types.")
    parser.add_argument("--test_s1", action="store_true", help="Plot Sentinel-1 data.")
    parser.add_argument("--test_s2", action="store_true", help="Plot Sentinel-2 data.")
    parser.add_argument("--test_landsat", action="store_true", help="Plot Landsat data.")
    parser.add_argument("--test_gt", action="store_true", help="Plot ground truth data.")

    args = parser.parse_args()
    main(args)'''