import os
import sys
import pyprojroot
from tqdm import tqdm
from typing import Tuple, Dict, List
import numpy as np
from copy import deepcopy
root = pyprojroot.here()
import matplotlib.pyplot as plt
sys.path.append(str(root))

import pyprojroot
root = pyprojroot.here()
import matplotlib.pyplot as plt
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
from pathlib import Path

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