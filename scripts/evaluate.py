import pyprojroot
import sys
import os
import re
import tifffile as tiff
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)
from scripts.evaluate_config import EvalConfig
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import (
    restitch_eval,
    restitch_and_plot
)
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import tifffile

@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    results_dir: str | os.PathLike = root / 'data/predictions' / "UNet"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "models" / "UNet" / "last-v3.ckpt"


def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    # Load datamodule
    datamodule = ESDDataModule(
        processed_dir = options.processed_dir,
        raw_dir = options.raw_dir,
        selected_bands = {'sentinel1': ['VV', 'VH']},
        tile_size_gt = options.tile_size_gt,
        batch_size = options.batch_size,
        seed = options.seed
    )
    tiles = get_parent_tiles(str(options.processed_dir / "Val"/"subtiles"))

    datamodule.setup()

    # load model from checkpoint at options.model_path
    print("options.model.path: ", options)
    model = ESDSegmentation.load_from_checkpoint(checkpoint_path = str(options.model_path))

    # set the model to evaluation mode (model.eval())
    model.eval()

    # instantiate pytorch lightning trainer
    trainer = pl.Trainer()

    # run the validation loop with trainer.validate
    trainer.validate(model, datamodule.val_dataloader())
    
    # run restitch_and_plot
        
    # run restitch_and_plot 

    # for every subtile in options.processed_dir/Val/subtiles
    # run restitch_eval on that tile followed by picking the best scoring class


    # save the file as a tiff using tifffile
    # save the file as a png using matplotlib
    
    range_x = (0,16//options.tile_size_gt)
    range_y = (0,16//options.tile_size_gt)
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
    for parent_tile_id in tiles:
        restitch_and_plot(
            options = options,
            datamodule = datamodule, 
            model = model, 
            parent_tile_id = parent_tile_id,
            image_dir = options.results_dir
            )
        satellite_subtile, gt_subtile, y_pred = restitch_eval(options.processed_dir,"sentinel2",parent_tile_id,range_x,range_y,datamodule,model)
        
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(y_pred, vmin=-0.5, vmax=3.5,cmap=cmap)
        plt.savefig(options.results_dir / f"{parent_tile_id}.png")
        tiff.imwrite(str(options.results_dir / f"{parent_tile_id}.tiff"), y_pred)


def get_parent_tiles(directory: str) -> list:
    """
    Extract unique tile types from the file names in the given directory.

    Args:
    directory (str): The directory containing the tile files.

    Returns:
    list: A list of unique tile types in the format "Tilen".
    """
    # Set to store unique tile types
    tile_types = set()

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        # Split the filename into parts based on the underscore
        parts = filename.split('_')
        # Check if the filename structure matches expected pattern
        if len(parts) >= 3 and parts[0].startswith("Tile"):
            # Add the tile type to the set (e.g., "Tilen")
            tile_types.add(parts[0])
    # Convert the set to a list and return
    tile_types = list(tile_types)
    sorted_tiles = sorted(tile_types, key=lambda x: int(re.search(r'\d+', x).group()))
    return sorted_tiles


if __name__ == '__main__':
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Model path.", default=config.model_path)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir, help=".")
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Results dir")
    main(EvalConfig(**parser.parse_args().__dict__))