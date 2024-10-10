"""Legacy code from hw03 datamodule.py winter 2024"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import xarray as xr
from sklearn.model_selection import train_test_split
from torchvision import transforms as torchvision_transforms
from tqdm import tqdm

sys.path.append(".")
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor,
)
from src.esd_data.dataset import ESDDataset
from src.preprocessing.file_utils import load_satellite
from src.preprocessing.preprocess_sat import (
    maxprojection_viirs,
    preprocess_landsat,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_viirs,
)
from src.preprocessing.subtile import Subtile
from src.utilities import SatelliteType


def collate_fn(batch):
    """
    Custom collate function for combining samples into a batch.
    
    Parameters:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): List of tuples containing image and label.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch of stacked images and labels.
    """
    Xs, ys = zip(*batch)  # Unpack image and label pairs
    Xs = torch.stack(Xs)  # Stack images into a single tensor
    ys = torch.stack(ys)  # Stack labels into a single tensor
    return Xs, ys

class ESDDataModule(pl.LightningDataModule):
    """
    The ESDDataModule class is designed to encapsulate data loading and processing logic for a model within the PyTorch Lightning framework.

    Attributes:
        processed_dir: Path to the directory containing processed data.
        raw_dir: Path to the directory containing raw data.
        batch_size: Batch size for data loaders.
        seed: Random seed for data processing.
        selected_bands: Dictionary mapping SatelliteType to a list of selected bands.
        slice_size: Tuple specifying the size for subtiling.
        train_size: Fraction of data allocated for training.
        transform_list: List of torchvision transforms applied to the data.

    Methods:
        load_and_preprocess(tile_dir: Path) -> Tuple[List[xr.DataArray], xr.DataArray]:
            Loads and preprocesses tile data located in tile_dir.

        prepare_data() -> None:
            Processes raw data by loading, splitting into train-test, and subtiling for training and validation.

        setup(stage: str) -> None:
            Sets up the training and validation datasets (self.train_dataset, self.val_dataset).

        train_dataloader() -> torch.utils.data.DataLoader:
            Creates and returns a DataLoader for the training dataset.

        val_dataloader() -> torch.utils.data.DataLoader:
            Creates and returns a DataLoader for the validation dataset.
    """

    def __init__(
        self,
        processed_dir: Path,
        raw_dir: Path,
        batch_size: int = 32,
        num_workers: int = 15,
        seed: int = 12378921,
        selected_bands: Dict[SatelliteType, List[str]] = None,
        slice_size: Tuple[int, int] = (4, 4),
        train_size: float = 0.8,
        transform_list: list = [
            torchvision_transforms.RandomApply([AddNoise()], p=0.5),
            torchvision_transforms.RandomApply([Blur()], p=0.5),
            torchvision_transforms.RandomApply([RandomHFlip()], p=0.5),
            torchvision_transforms.RandomApply([RandomVFlip()], p=0.5),
            ToTensor(),
        ],
    ):
        super(ESDDataModule, self).__init__()
        self.processed_dir = processed_dir
        self.raw_dir = raw_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.selected_bands = selected_bands
        self.slice_size = slice_size
        self.train_size = train_size
        self.satellite_type_list = list(self.selected_bands.keys())

        # Define data transformations
        self.transform = torchvision_transforms.transforms.Compose(transform_list)

        # Define directories for train and validation data
        self.train_dir = self.processed_dir / "Train"
        self.val_dir = self.processed_dir / "Val"

    def load_and_preprocess(self,tile_dir: Path) -> Tuple[List[xr.DataArray], xr.DataArray]:
        """
        Load and preprocess raw satellite imagery data from `tile_dir`.
        
        Parameters:
            tile_dir (Path): Path to the directory containing raw tile data.

        Returns:
            Tuple[List[xr.DataArray], xr.DataArray]: List of preprocessed data arrays and ground truth data array.
        """
        preprocess_functions = {
            SatelliteType.VIIRS: preprocess_viirs,
            SatelliteType.S1: preprocess_sentinel1,
            SatelliteType.S2: preprocess_sentinel2,
            SatelliteType.LANDSAT: preprocess_landsat,
        }

        # Load and preprocess each satellite type, excluding VIIRS max projection
        preprocessed_data_array_list = [
            preprocess_functions[satellite_type](
                load_satellite(tile_dir, satellite_type).sel(band=self.selected_bands[satellite_type])
            )
            for satellite_type in self.satellite_type_list if satellite_type != SatelliteType.VIIRS_MAX_PROJ
        ]

        # Handle VIIRS max projection separately if it exists in the satellite types
        if SatelliteType.VIIRS_MAX_PROJ in self.satellite_type_list:
            preprocessed_data_array_list.append(
                maxprojection_viirs(load_satellite(tile_dir, SatelliteType.VIIRS))
            )
        # Load ground truth data
        gt_data = load_satellite(tile_dir, SatelliteType.GT)

        return preprocessed_data_array_list, gt_data

    def prepare_data(self) -> None:
        """
        If the data has not been processed before (denoted by whether or not self.processed_dir is an existing directory),
        we will process it.

        The steps for processing are as follows:
            - load all 60 tiles
            - train test split the tiles
            - subtile and save the training split
            - subtile and save the testing split
        """
       
        if not self.processed_dir.exists() or not list(self.processed_dir.glob("*")):
            # Load raw tile data
            tile_dirs = list(self.raw_dir.glob("*"))
            
            # Perform train-test split
            tile_dirs_train, tile_dirs_val = train_test_split(tile_dirs, test_size=1 - self.train_size, train_size=self.train_size, random_state=self.seed)
            
            # Process and save training subtiles
            for tile_dir in tqdm(tile_dirs_train, desc="Processing train tiles"):
                data_array_list, gt_data_array = self.load_and_preprocess(tile_dir)
                subtile = Subtile(data_array_list, gt_data_array, self.slice_size)
                subtile.save(self.train_dir)

            # iterate over the tile directories in tqdm(list(tile_dirs_val), desc="Processing validation tiles"):
            # Process and save validation subtiles
            for tile_dir in tqdm(tile_dirs_val, desc="Processing validation tiles"):
                data_array_list, gt_data_array = self.load_and_preprocess(tile_dir)
                subtile = Subtile(data_array_list, gt_data_array, self.slice_size)
                subtile.save(self.val_dir)
        else:
            print("Processed data already exists. Skipping redundant processing")

    def setup(self, stage: str) -> None:
        """
        Set up the training, validation, or test dataset based on the specified stage.
        
        Parameters:
            stage (str): Either "fit" for training/validation or "test" for test setup.
        """
        if stage == "fit":
            # Setup training and validation datasets
            self.train_dataset = ESDDataset(self.train_dir, self.transform, self.satellite_type_list, self.slice_size)
            self.val_dataset = ESDDataset(self.val_dir, self.transform, self.satellite_type_list, self.slice_size)
        
        if stage == "test":
            # Setup test dataset
            self.test_dataset = ESDDataset(self.train_dir, self.transform, self.satellite_type_list, self.slice_size)


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Creates and returns a DataLoader for the training dataset."""
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers, persistent_workers=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Creates and returns a DataLoader for the validation dataset."""
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers, persistent_workers=True
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Creates and returns a DataLoader for the test dataset."""
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers
        )
