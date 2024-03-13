""" This module contains the PyTorch Lightning ESDDataModule to use with the
PyTorch ESD dataset."""

import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

import numpy as np
import pytorch_lightning as pl  # noqa
import torch  # noqa
from torch import Generator  # noqa
from torch.utils.data import DataLoader, random_split  # noqa
from torchvision import transforms  # noqa
from tqdm import tqdm  # noqa

from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor,
)
from src.preprocessing.file_utils import Metadata

from ..preprocessing.file_utils import load_satellite
from ..preprocessing.preprocess_sat import (
    maxprojection_viirs,
    preprocess_landsat,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_viirs,
)
from ..preprocessing.subtile_esd_hw02 import grid_slice  # noqa
from .dataset import DSE  # noqa


def collate_fn(batch):
    Xs = []
    ys = []
    metadatas = []

    for X, y, metadata in batch:

        X_tensor = torch.tensor(X,dtype=torch.float32) #change this if you want to run float64
        y_tensor = torch.tensor(y,dtype=torch.float32)
        Xs.append(X_tensor)    #float32
        ys.append(y_tensor)    #float64

        metadatas.append(metadata)

    Xs = torch.stack(Xs)
    ys = torch.stack(ys)
    return Xs, ys, metadatas


class ESDDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning ESDDataModule to use with the PyTorch ESD dataset.

    Attributes:
        processed_dir: str | os.PathLike
            Location of the processed data
        raw_dir: str | os.PathLike
            Location of the raw data
        selected_bands: Dict[str, List[str]] | None
            Dictionary mapping satellite type to list of bands to select
        tile_size_gt: int
            Size of the ground truth tiles
        batch_size: int
            Batch size
        seed: int
            Seed for the random number generator
    """

    def __init__(
        self,
        processed_dir: str | os.PathLike,
        raw_dir: str | os.PathLike,
        selected_bands: Dict[str, List[str]] | None = None,
        tile_size_gt=4,
        batch_size=32,
        seed=12378921,
    ):

        super().__init__()
        self.processed_dir = processed_dir
        self.raw_dir = raw_dir
        self.selected_bands = selected_bands
        self.tile_size_gt = tile_size_gt
        self.batch_size = batch_size
        self.seed = seed

        # Seed for reproducibility in transformations
        pl.seed_everything(self.seed)

        # set transform to a composition of the following transforms: AddNoise, Blur, RandomHFlip, RandomVFlip, ToTensor
        # utilize the RandomApply transform to apply each of the transforms with a probability of 0.5

        self.transform = transforms.Compose(
            [
                transforms.RandomApply([AddNoise()], p=0.5),
                transforms.RandomApply([Blur()], p=0.5),
                RandomHFlip(p=0.5),
                RandomVFlip(p=0.5),
                ToTensor(),
            ]
        )

    # raise NotImplementedError("DataModule __init__ function not implemented.")

    def __load_and_preprocess(
        self,
        tile_dir: str | os.PathLike,
        satellite_types: List[str] = [
            "viirs",
            "sentinel1",
            "sentinel2",
            "landsat",
            "gt",
        ]) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Metadata]]]:
        """
        Performs the preprocessing step: for a given tile located in tile_dir,
        loads the tif files and preprocesses them just like in homework 1.

        Input:
            tile_dir: str | os.PathLike
                Location of raw tile data
            satellite_types: List[str]
                List of satellite types to process

        Output:
            satellite_stack: Dict[str, np.ndarray]
                Dictionary mapping satellite_type -> (time, band, width, height) array
            satellite_metadata: Dict[str, List[Metadata]]
                Metadata accompanying the statellite_stack
        """
        preprocess_functions = {
            "viirs": preprocess_viirs,
            "sentinel1": preprocess_sentinel1,
            "sentinel2": preprocess_sentinel2,
            "landsat": preprocess_landsat,
            "gt": lambda x: x,
        }

        satellite_stack = {}
        satellite_metadata = {}
        for satellite_type in satellite_types:
            stack, metadata = load_satellite(tile_dir, satellite_type)

            stack = preprocess_functions[satellite_type](stack)

            satellite_stack[satellite_type] = stack
            satellite_metadata[satellite_type] = metadata

        satellite_stack["viirs_maxproj"] = np.expand_dims(
            maxprojection_viirs(satellite_stack["viirs"], clip_quantile=0.0), axis=0
        )
        satellite_metadata["viirs_maxproj"] = deepcopy(satellite_metadata["viirs"])
        for metadata in satellite_metadata["viirs_maxproj"]:
            metadata.satellite_type = "viirs_maxproj"

        return satellite_stack, satellite_metadata

    def prepare_data(self, seed=1024):
        """
        If the data has not been processed before (denoted by whether or not self.processed_dir is an existing directory)

        For each tile,
            - load and preprocess the data in the tile
            - grid slice the data
            - for each resulting subtile
                - save the subtile data to self.processed_dir
        """
        # if the processed_dir does not exist, process the data and create
        # subtiles of the parent image to save
        if Path(self.processed_dir).exists():
            return
    
        #create "data/processed/nxn/" directory
        self.processed_dir.mkdir(parents=True,exist_ok = True)
        
        train_path = Path(self.processed_dir/'Train')
        train_path.mkdir(parents = True, exist_ok = True)

        #create data/processed/nxn/Val
        val_path = Path(self.processed_dir/'Val')
        val_path.mkdir(parents = True, exist_ok = True)

        # fetch all the parent tiles in the raw_dir
        subdirectories = [d for d in self.raw_dir.iterdir() if d.is_dir()]
    
        # randomly split the directories into train and val
        train_tiles, val_tiles = train_test_split(subdirectories,test_size=0.2,train_size=0.8,random_state=seed)

        #sort the subdirectories
        train_tiles = sorted(train_tiles, key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x.name)])
        val_tiles = sorted(val_tiles, key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x.name)])

        for tile in train_tiles:
            # call __load_and_preprocess to load and preprocess the data for all satellite types
            processed_data = self.__load_and_preprocess(tile_dir=tile)
            # grid slice the data with the given tile_size_gt
            subtiles = grid_slice(
                satellite_stack=processed_data[0],
                metadata_stack=processed_data[1],
                tile_size_gt=self.tile_size_gt,
            )
            # save each subtile
            for subtile in subtiles:
                subtile.save(dir= train_path)
        
        for tile in val_tiles:
            # call __load_and_preprocess to load and preprocess the data for all satellite types
            processed_data = self.__load_and_preprocess(tile_dir=tile)

            # grid slice the data with the given tile_size_gt
            subtiles = grid_slice(
                satellite_stack=processed_data[0],
                metadata_stack=processed_data[1],
                tile_size_gt=self.tile_size_gt,
            )
            # save each subtile
            for subtile in subtiles:
                subtile.save(dir= val_path)
            
    def setup(self, stage: str = None, seed=1024):
        """
        Create self.train_dataset and self.val_dataset.0000ff

        Hint: Use torch.utils.data.random_split to split the Train
        directory loaded into the PyTorch dataset DSE into an 80% training
        and 20% validation set. Set the seed to 1024.
        """
        # Create generator for random number generation.
        gen = Generator()
        gen.manual_seed(seed)

        train = DSE(
            root_dir= self.processed_dir / 'Train',
            selected_bands=self.selected_bands,
        )
        
        val = DSE(
            root_dir= self.processed_dir / 'Val',
            selected_bands=self.selected_bands,
        )
        self.train_dataset = train
        self.val_dataset = val

        '''train_sample, train_label, _ = train[0]
        print(f"Shape of the first training sample: {train_sample.shape}")
        print(f"Shape of the first training label: {train_label.shape}")

        # Access the first sample in the validation dataset
        val_sample, val_label, _ = val[0]
        print(f"Shape of the first validation sample: {val_sample.shape}")
        print(f"Shape of the first validation label: {val_label.shape}")'''

    def train_dataloader(self):
        """
        Create and return a torch.utils.data.DataLoader with
        self.train_dataset
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=7,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        """
        Create and return a torch.utils.data.DataLoader with
        self.val_dataset
        """
        
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=7,
            persistent_workers=True
        )