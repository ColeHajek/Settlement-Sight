import os
import sys
import pyprojroot
import torch
root = pyprojroot.here()
sys.path.append(str(root))
from pathlib import Path
from typing import List, Tuple
from PIL import Image

from src.preprocessing.subtile_esd_hw02 import restitch, grid_slice
from scripts.evaluate_config import EvalConfig
from scripts.evaluate_config import ESDDataModule
from scripts.evaluate_config import ESDSegmentation


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.preprocessing.subtile_esd_hw02 import TileMetadata, Subtile


def get_subtile_preds(subtile_path,datamodule,model):
    tile_name = subtile_path.name            
    # find the tile in the datamodule 
    idx = -1
    for i in range(len(datamodule.val_dataset.tiles)):
        if datamodule.val_dataset.tiles[i].name == tile_name:
            idx = i

    if idx==-1:
        raise ValueError(f"Tile {tile_name} not found in dataset.")
    
    X, y, subtile_metadata = datamodule.val_dataset.__getitem__(idx) #FIXME better if we don't hardcode val_dataset but works for now


    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)

    # evaluate the tile with the model
    # You need to add a dimension of size 1 at dim 0 so that
    # some CNN layers work
    # i.e., (batch_size, channels, width, height) with batch_size = 1
    X_tensor = X_tensor.unsqueeze(0)

    # make sure that the tile is in GPU memory, i.e., X = X.cuda()
    if torch.cuda.is_available():
        X_tensor = X_tensor.cuda()
        y_tensor = y_tensor.cuda()

    # convert y to numpy array
    y = y_tensor.cpu().numpy()

    # detach predictions from the gradient, move to cpu and convert to numpy
    with torch.no_grad():
        predictions = model.forward(X_tensor)
        predictions = torch.argmax(predictions,dim=1)
        predictions = predictions.squeeze(0).cpu().numpy()      #might cause problems

    return predictions, y 


def restitch_and_plot(options, datamodule, model, parent_tile_id, satellite_type="sentinel2", rgb_bands=[3,2,1], image_dir: None | str | os.PathLike = None):
    """
    Plots the 1) rgb satellite image 2) ground truth 3) model prediction in one row.

    Args:
        options: EvalConfig
        datamodule: ESDDataModule
        model: ESDSegmentation
        parent_tile_id: str
        satellite_type: str
        rgb_bands: List[int]
    """
    #raise NotImplementedError # Complete this function using the code snippets below. Do not forget to remove this line.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
    fig, axs = plt.subplots(nrows=1, ncols=3)
    
        # 1. Get sat imaging data
    sat_path = Path(options.processed_dir/"Val"/"subtiles")
    restitch_len = 16//options.tile_size_gt
    stitched_sat, _ = restitch(sat_path,satellite_type,parent_tile_id,(0,restitch_len),(0,restitch_len))
    
    # Stack bands to get RGB image
    stitched_RGB = np.dstack([stitched_sat[0,rgb_bands[0],:,:],stitched_sat[0,rgb_bands[1],:,:],stitched_sat[0,rgb_bands[2],:,:]])
    
    axs[0].set_title("RGB"+satellite_type)
    axs[0].imshow(stitched_RGB)


        # 2. Get the ground truth data
    gt_path = Path(options.raw_dir/ parent_tile_id/'groundTruth.tif')
    gt_arr = np.array(Image.open(gt_path))

    #subtract one from the original values to correspond to predicted value 0-3 indexing
    gt_arr = np.subtract(gt_arr,1)
    axs[1].set_title("Ground Truth")
    axs[1].imshow(gt_arr,cmap=cmap, vmin=-0.5, vmax=3.5)
    im = axs[1].imshow(gt_arr,cmap=cmap, vmin=-0.5, vmax=3.5)

        # 3. Get prediction data
    #axs[2] instructions
    full_tile_preds = np.zeros((16,16))
    tile_size = options.tile_size_gt
    for x in range(restitch_len):
        for y in range(restitch_len):
            st_path = Path(sat_path / f"{parent_tile_id}_{x}_{y}.npz")
            subtile_pred, _ = get_subtile_preds(st_path,datamodule,model)
            full_tile_preds[x*tile_size:(x+1)*tile_size,y*tile_size:(y+1)*tile_size] = subtile_pred


    #make predictions = np.zeros of dimension 16x16
    #for col in gt
    #for row in col
    
    #get predictions[col,row]
    #axs[2].set_title("Predictions")
    #axs[2].imshow(predictions,cmap=cmap, vmin = -0.5, vmax = 3.5) 

    axs[2].set_title("Model Predictions")
    axs[2].imshow(full_tile_preds,cmap=cmap, vmin=-0.5, vmax=3.5)

    # make sure to use cmap=cmap, vmin=-0.5 and vmax=3.5 when running
    # axs[i].imshow on the 1d images in order to have the correct 
    # colormap for the images.
    # On one of the 1d images' axs[i].imshow, make sure to save its output as 
    # `im`, i.e, im = axs[i].imshow
    
    # The following lines sets up the colorbar to the right of the images    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['Sttlmnts Wo Elec', 'No Sttlmnts Wo Elec', 'Sttlmnts W Elec', 'No Sttlmnts W Elec'])
    
    if image_dir is None:
        plt.show()
    else:
        if not Path(image_dir).exists():
            Path(image_dir).mkdir(parents=True,exist_ok = True)
        plt.savefig(Path(image_dir) / f"{parent_tile_id}_restitched_visible_gt_predction.png")
        plt.close()

def restitch_eval(dir: str | os.PathLike, satellite_type: str, tile_id: str, range_x: Tuple[int, int], range_y: Tuple[int, int], datamodule, model) -> np.ndarray:
    
    """
    Given a directory of processed subtiles, a tile_id and a satellite_type, 
    this function will retrieve the tiles between (range_x[0],range_y[0])
    and (range_x[1],range_y[1]) in order to stitch them together to their
    original image. It will also get the tiles from the datamodule, evaluate
    it with model, and stitch the ground truth and predictions together.

    Input:
        dir: str | os.PathLike
            Directory where the subtiles are saved
        satellite_type: str
            Satellite type that will be stitched
        tile_id: str
            Tile id that will be stitched
        range_x: Tuple[int, int]
            Range of tiles that will be stitched on width dimension [0,5)]
        range_y: Tuple[int, int]
            Range of tiles that will be stitched on height dimension
        datamodule: pytorch_lightning.LightningDataModule
            Datamodule with the dataset
        model: pytorch_lightning.LightningModule
            LightningModule that will be evaluated
    
    Output:
        stitched_image: np.ndarray
            Stitched image, of shape (time, bands, width, height)
        stitched_ground_truth: np.ndarray
            Stitched ground truth of shape (width, height)
        stitched_prediction_subtile: np.ndarray
            Stitched predictions of shape (width, height)
    """
    
    dir = Path(dir)
    satellite_subtile = []
    ground_truth_subtile = []
    predictions_subtile = []
    satellite_metadata_from_subtile = []
   
    for i in range(*range_x):
        satellite_subtile_row = []
        ground_truth_subtile_row = []
        predictions_subtile_row = []
        satellite_metadata_from_subtile_row = []
        for j in range(*range_y):
            subtile = Subtile().load(dir / 'Val' / 'subtiles' / f"{tile_id}_{i}_{j}.npz")
            
            st_path = Path(dir/'Val' / 'subtiles'/ f"{tile_id}_{i}_{j}.npz")
            predictions, y = get_subtile_preds(st_path,datamodule,model)            
            
            ground_truth_subtile_row.append(y)
            predictions_subtile_row.append(predictions)
            satellite_subtile_row.append(subtile.satellite_stack[satellite_type])
            satellite_metadata_from_subtile_row.append(subtile.tile_metadata)

        ground_truth_subtile.append(np.concatenate(ground_truth_subtile_row, axis=-1))
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
        satellite_subtile.append(np.concatenate(satellite_subtile_row, axis=-1))
        satellite_metadata_from_subtile.append(satellite_metadata_from_subtile_row)
    return np.concatenate(satellite_subtile, axis=-2), np.concatenate(ground_truth_subtile, axis=-2), np.concatenate(predictions_subtile, axis=-2)

def get_subtiles_by_tile_id(dataset,tile_id):
    subtiles = []
    for subtile in dataset:
        if subtile.metadata['tile_id'] == tile_id:
            subtiles.append(subtile)
    sorted_subtiles = sorted(subtiles, key=lambda sub: (sub['x_gt'],sub['y_gt']))
    return sorted_subtiles

