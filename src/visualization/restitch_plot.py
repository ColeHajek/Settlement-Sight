from pathlib import Path
from typing import List
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.subtile import Subtile
from src.utilities import SatelliteType


def restitch_and_plot(
    options,
    datamodule,
    model,
    parent_tile_id: str,
    accelerator: str,
    satellite_type: SatelliteType = SatelliteType.SENTINEL_2,
    selected_bands: List = ["04", "03", "02"],
    results_dir: Path = None,
):
    """
    Plots the RGB satellite image, ground truth, and model predictions side-by-side.

    Parameters
    ----------
    options : Namespace
        Configuration options for the data.
    datamodule : DataModule
        DataModule used to load the dataset.
    model : nn.Module
        Trained model used for prediction.
    parent_tile_id : str
        Identifier of the parent tile being processed.
    accelerator : str
        Either 'cpu' or 'gpu' to specify device type.
    satellite_type : SatelliteType, optional
        Satellite type to use for the RGB image (default is Sentinel-2).
    selected_bands : List[str], optional
        List of bands to use for RGB plotting (default is ["04", "03", "02"]).
    results_dir : Path, optional
        Directory to save the plot. If None, the plot is displayed.

    Raises
    ------
    KeyError
        If the specified satellite data is not available in the subtile.
    """
    subtile, prediction = restitch_eval(
        processed_dir=options.processed_dir / "Val",
        parent_tile_id=parent_tile_id,
        accelerator=accelerator,
        datamodule=datamodule,
        model=model,
    )

    y_pred = prediction[0].argmax(axis=0)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "Settlements", np.array(["#ff0000", "#0000ff", "#ffff00", "#b266ff"]), N=4
    )

    # Create subplots
    fig, axs = plt.subplots(nrows=1, ncols=3)

    # Select satellite data to plot
    satellite_data = None
    for sat_data in subtile.satellite_list:
        if sat_data.satellite_type == satellite_type.value:
            satellite_data = sat_data

    if satellite_data is None:
        raise KeyError("Missing satellite from subtile")
    
    # Convert satellite data to RGB image
    rgb_image = satellite_data.sel(band=selected_bands).to_numpy()[0].transpose(1, 2, 0)
    
    # Plot RGB image, ground truth, and prediction
    axs[0].set_title("RGB image")
    axs[0].imshow(rgb_image)

    axs[1].set_title("Ground truth")
    axs[1].imshow(subtile.ground_truth.values[0][0]-1, cmap=cmap, vmin=-0.5, vmax=3.5)
    
    axs[2].set_title("Prediction")
    im = axs[2].imshow(y_pred, cmap=cmap, vmin=-0.5, vmax=3.5)

    # Add color bar to the figure
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["Sttlmnts Wo Elec","No Sttlmnts Wo Elec","Sttlmnts W Elec","No Sttlmnts W Elec",])
    
    if results_dir is None:
        plt.show()
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / f"{parent_tile_id}.png", format="png")
        plt.close()

def restitch_eval(
    processed_dir: Path, parent_tile_id: str, accelerator: str, datamodule, model
) -> np.ndarray:
    """
    Restitches subtiles into a full image and predicts on the restitched image.

    Parameters
    ----------
    processed_dir : Path
        Directory containing the processed subtiles.
    parent_tile_id : str
        Identifier for the parent tile.
    accelerator : str
        Either 'cpu' or 'gpu' to specify the computation device.
    datamodule : DataModule
        DataModule to handle the dataset.
    model : nn.Module
        Trained model used for prediction.

    Returns
    -------
    subtile : Subtile
        Subtile object containing the restitched image.
    predictions_subtile : np.ndarray
        Array of predictions from the model for each subtile.
    """

    #Initialize the Subtile object
    slice_size = datamodule.slice_size
    subtile = Subtile(satellite_list=[], ground_truth=[], slice_size=slice_size, parent_tile_id=parent_tile_id)
    subtile.restitch(processed_dir, datamodule.satellite_type_list)

    # Set device based on accelerator
    device = torch.device("cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Make predictions for each subtile
    predictions_subtile = []
    for i in range(slice_size[0]):
        predictions_subtile_row = []
        for j in range(slice_size[1]):
            X, _ = retrieve_subtile_file(i, j, processed_dir, parent_tile_id, datamodule)
            # You need to add a dimension of size 1 at dim 0 so that
            # some CNN layers work
            # i.e., (batch_size, channels, width, height) with batch_size = 1
            X = X.reshape((1, X.shape[-3], X.shape[-2], X.shape[-1]))

            predictions = model(X.float().to(device)).detach().cpu().numpy()

            predictions_subtile_row.append(predictions)
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
    
    return subtile, np.concatenate(predictions_subtile, axis=-2)

def retrieve_subtile_file(
    i: int, j: int, processed_dir: str, parent_tile_id: str, datamodule
):
    """
    Retrieve a specific subtile file from the directory.

    Parameters
    ----------
    i : int
        X-coordinate of the subtile.
    j : int
        Y-coordinate of the subtile.
    processed_dir : str
        Directory containing the processed subtiles.
    parent_tile_id : str
        Identifier of the parent tile.
    datamodule : DataModule
        DataModule to handle the dataset.

    Returns
    -------
    X : np.ndarray
        The subtile data.
    y : np.ndarray
        The corresponding label.
    """
    subtile_file = processed_dir / "subtiles" / parent_tile_id / f"{i}_{j}"
    if subtile_file in datamodule.train_dataset.subtile_dirs:
        index = datamodule.train_dataset.subtile_dirs.index(subtile_file)
        X, y = datamodule.train_dataset[index]
    else:
        index = datamodule.val_dataset.subtile_dirs.index(subtile_file)
        X, y = datamodule.val_dataset[index]

    return X, y

'''def restitch_eval_csv(
    processed_dir: Path, parent_tile_id: str, accelerator: str, datamodule, model
) -> np.ndarray:
    """ """
    slice_size = datamodule.slice_size
    subtile = Subtile(
        satellite_list=[],
        ground_truth=[],
        slice_size=slice_size,
        parent_tile_id=parent_tile_id,
    )
    subtile.restitch(processed_dir, datamodule.satellite_type_list)

    predictions_subtile = []
    for i in range(slice_size[0]):
        predictions_subtile_row = []
        for j in range(slice_size[1]):
            X, _ = retrieve_subtile_file_csv(
                i, j, processed_dir, parent_tile_id, datamodule
            )
            # You need to add a dimension of size 1 at dim 0 so that
            # some CNN layers work
            # i.e., (batch_size, channels, width, height) with batch_size = 1
            X = X.reshape((1, X.shape[-3], X.shape[-2], X.shape[-1]))

            predictions = None
            if accelerator == "cpu":
                predictions = model(X.float())
            elif accelerator == "gpu":
                predictions = model(X.float().cuda())
            assert (
                predictions != None
            ), "accelerator passing not configured for restich_eval"

            predictions = predictions.detach().cpu().numpy()

            predictions_subtile_row.append(predictions)
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
    return subtile, np.concatenate(predictions_subtile, axis=-2)'''

def retrieve_subtile_file_csv(
    x: int, y: int, processed_dir: str, parent_tile_id: str, datamodule
):
    """
    Retrieve a specific subtile file for CSV processing.

    Parameters
    ----------
    i : int
        X-coordinate of the subtile.
    j : int
        Y-coordinate of the subtile.
    processed_dir : str
        Directory containing the processed subtiles.
    parent_tile_id : str
        Identifier of the parent tile.
    datamodule : DataModule
        DataModule to handle the dataset.

    Returns
    -------
    X : np.ndarray
        The subtile data.
    y : np.ndarray
        The corresponding label.
    """
    subtile_file = processed_dir / "subtiles" / parent_tile_id / f"{x}_{y}" 
    index = datamodule.test_dataset.subtile_dirs.index(subtile_file)
    X, y = datamodule.test_dataset[index]
    return X, y

def plot_tile(
    subtile: Subtile,
    selected_bands: List[str] = ["04", "03", "02"],
    satellite_type: SatelliteType = SatelliteType.SENTINEL_2,
    results_dir: Path = None,
):
    """
    Plots the RGB satellite image based on selected bands.

    Parameters
    ----------
    subtile : Subtile
        Subtile object containing the satellite data.
    selected_bands : List[str], optional
        List of bands to use for RGB plotting (default is ["04", "03", "02"]).
    satellite_type : SatelliteType, optional
        Satellite type to use for the RGB image (default is Sentinel-2).
    results_dir : Path, optional
        Directory to save the plot. If None, the plot is displayed.

    Raises
    ------
    KeyError
        If the specified satellite data is not available in the subtile.
    """
    # Select satellite data to plot
    satellite_data = None
    for sat_data in subtile.satellite_list:
        if sat_data.satellite_type == satellite_type.value:
            satellite_data = sat_data

    if satellite_data is None:
        raise KeyError("Missing satellite data from subtile")

    # Convert satellite band data to image
    tile_image = satellite_data.sel(band=selected_bands).to_numpy()[0].transpose(1, 2, 0)

    # Create a plot
    fig, ax = plt.subplots()
    ax.set_title(f"Image (Bands {selected_bands})")
    ax.imshow(tile_image)

    if results_dir is None:
        plt.show()
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "rgb_image.png", format="png")
        plt.close()
