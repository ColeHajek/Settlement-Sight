""" This module contains the PyTorch Lightning ESDDataModule to use with the
PyTorch ESD dataset."""

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyprojroot
import pytorch_lightning as pl
import torch
from torch import Generator
from torch.utils.data import random_split
from torchvision import transforms

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
from ..preprocessing.subtile_esd_hw02 import grid_slice
from .dataset import DSE

sys.path.append(str(pyprojroot.here()))


def collate_fn(batch):
    Xs = []
    ys = []
    metadatas = []
    for X, y, metadata in batch:
        Xs.append(X)
        ys.append(y)
        metadatas.append(metadata)

    Xs = np.stack(Xs)
    ys = np.stack(ys)
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
        ],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Metadata]]]:
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

    def prepare_data(self):
        """
        If the data has not been processed before (denoted by whether or not self.processed_dir is an existing directory)

        For each tile,
            - load and preprocess the data in the tile
            - grid slice the data
            - for each resulting subtile
                - save the subtile data to self.processed_dir
        """
        # if the processed_dir does not exist, process the data and create
        if Path(self.processed_dir).exists():
            return

        # fetch all the parent images in the raw_dir
        dir = self.raw_dir if type(self.raw_dir) == os.PathLike else Path(self.raw_dir)
        path_raw_dir = Path.cwd() / str(dir)[1:]

        # for each parent image in the raw_dir
        for tile in path_raw_dir.iterdir():
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
                subtile.save(dir=self.processed_dir)

    def setup(self, stage: str):
        """
        Create self.train_dataset and self.val_dataset.0000ff

        Hint: Use torch.utils.data.random_split to split the Train
        directory loaded into the PyTorch dataset DSE into an 80% training
        and 20% validation set. Set the seed to 1024.
        """
        # Create generator for random number generation.
        gen = Generator()
        gen.manual_seed(1024)

        dataset = DSE(
            root_dir=Path(
                "/Users/nathanhuey/Coursework/2024 Winter/CS175/hw2/data/processed/Train1x1/subtiles"  # noqa
            ),
            selected_bands=self.selected_bands,
        )
        train, val = random_split(dataset=dataset, lengths=(0.8, 0.2), generator=gen)

        # self.train_dataset = DataLoader(train)
        # self.val_dataset = DataLoader(val)
        self.train_dataset = train
        self.val_dataset = val

    def train_dataloader(self):
        """
        Create and return a torch.utils.data.DataLoader with
        self.train_dataset
        """
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Create and return a torch.utils.data.DataLoader with
        self.val_dataset
        """
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )
