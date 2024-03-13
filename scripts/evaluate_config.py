from dataclasses import dataclass
import numpy as np
import pyprojroot
import sys
import os
import torch
import re
from copy import deepcopy
from torch.nn import functional as F
from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.resnet_transfer import FCNResnetTransfer
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from torchmetrics import Accuracy,AUROC,F1Score
from src.models.supervised.unet import UNet
import pytorch_lightning as pl 
from typing import Dict, List, Tuple
from torchvision import transforms  
from sklearn.model_selection import train_test_split
from torch import Generator  # noqa
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor,
)
from src.preprocessing.file_utils import Metadata
from src.preprocessing.file_utils import load_satellite
from src.preprocessing.preprocess_sat import (
    maxprojection_viirs,
    preprocess_landsat,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_viirs,
)
from src.esd_data.dataset import DSE  # noqa
from pathlib import Path
from src.preprocessing.subtile_esd_hw02 import grid_slice  
from torch.utils.data import DataLoader, random_split  # noqa
root = pyprojroot.here()
sys.path.append(str(root))
@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    results_dir: str | os.PathLike = root / 'data/predictions' / "SegmentationCNN"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "models" / "SegmentationCNN" / "last.ckpt"


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
            #shuffle=True,
            persistent_workers=True
        )


class ESDSegmentation(pl.LightningModule):
    """
    LightningModule for training a segmentation model on the ESD dataset
    """
    def __init__(self, model_type, in_channels, out_channels, 
                 learning_rate=1e-3, model_params: dict = {}):
        """
        Initializes the model with the given parameters.

        Input:
        model_type (str): type of model to use, one of "SegmentationCNN",
        "UNet", or "FCNResnetTransfer"
        in_channels (int): number of input channels of the image of shape
        (batch, in_channels, width, height)
        out_channels (int): number of output channels of prediction, prediction
        is shape (batch, out_channels, width//scale_factor, height//scale_factor)
        learning_rate (float): learning rate of the optimizer
        model_params (dict): dictionary of parameters to pass to the model
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        if model_type == "SegmentationCNN":
            self.model = SegmentationCNN(in_channels,out_channels, **model_params)
        elif model_type == "UNet":
            self.model = UNet(in_channels, out_channels, **model_params)
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(in_channels, out_channels, **model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        ## Performance Metrics ##
        
        # Accuracy
        self.avg_acc = Accuracy(task='multiclass',num_classes=out_channels)
        self.per_class_acc = Accuracy(task='multiclass',num_classes=out_channels,average=None)

        # Area Under Curve
        self.avg_AUC = MulticlassAUROC(num_classes=out_channels,average='macro',thresholds=None)
        self.per_class_AUC = MulticlassAUROC(num_classes=out_channels,average='none',thresholds=None)

        # F1 Score
        self.avg_F1Score = MulticlassF1Score(num_classes = out_channels)
        self.per_class_F1Score = MulticlassF1Score(num_classes=out_channels,average=None)

    
    def forward(self, X):
        """
        Run the input X through the model

        Input: X, a (batch, input_channels, width, height) image
        Ouputs: y, a (batch, output_channels, width/scale_factor, height/scale_factor) image
        """
        # return self.model.forward(X)
        y_pred = self.model.forward(X)
        return y_pred # list of probabilitiels falls under each class 
    # prob would sum up to one 
    # X is a list of data (batch) 
    # X will give us a list of probabilities (subtiles) and they are in a form of a batch

        
    def training_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then uses CrossEntropyLoss to calculate
        the current loss.

        Note: CrossEntropyLoss requires mask to be of type
        torch.int64 and shape (batches, width, height), 
        it only has one channel as the label is encoded as
        an integer index. As these may not be this shape and
        type from the dataset, you might have to use
        torch.reshape or torch.squeeze in order to remove the
        extraneous dimensions, as well as using Tensor.to to
        cast the tensor to the correct type.

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            train_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        sat_img, target, metadata = batch
        target = target.squeeze(1)
        target = target.to(torch.int64)

        preds = self.forward(sat_img)

        loss = F.cross_entropy(preds, target)
        
        ## Record Performance Metrics ##

        # Accuracy
        acc = self.avg_acc(preds, target)
        per_class_acc = self.per_class_acc(preds,target)
        
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        for c in range(4):
            label = 'train_class_' + str(c+1) + '_acc'
            self.log(label, per_class_acc[c], on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        
        # Area Under Curve
        auc = self.avg_AUC(preds,target)
        per_class_auc = self.per_class_AUC(preds,target)

        self.log('train_auc',auc,on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        for c in range(4):
            label = 'train_class_' + str(c+1) + '_auc'
            self.log(label,per_class_auc[c],on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)

        # F1 Score
        f1 = self.avg_F1Score(preds,target)
        per_class_f1 = self.per_class_F1Score(preds,target)
        
        self.log('train_f1',f1,on_step=True, on_epoch=True, prog_bar=True, logger=True,enable_graph=True)
        for c in range(4):
            label = 'train_class_' + str(c+1) + '_f1'
            self.log(label,per_class_f1[c],on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then evaluates the 

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            val_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Should be the cross_entropy_loss, as it is the main validation
            loss that will be tracked.
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        sat_img, target, batch_metadata = batch
        target = target.squeeze(1)
        target = target.to(torch.int64)

        sat_img = sat_img.float()
        
        preds = self.forward(sat_img)

        loss = F.cross_entropy(preds, target)

        ## Record Performance Metrics ##

        # Accuracy
        acc = self.avg_acc(preds, target)
        per_class_acc = self.per_class_acc(preds,target)
        
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        for c in range(4):
            label = 'val_class_' + str(c+1) + '_acc'
            self.log(label, per_class_acc[c], on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        
        # Area Under Curve
        auc = self.avg_AUC(preds,target)
        per_class_auc = self.per_class_AUC(preds,target)

        self.log('val_auc',auc,on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        for c in range(4):
            label = 'val_class_' + str(c+1) + '_auc'
            self.log(label,per_class_auc[c],on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)

        # F1 Score
        f1 = self.avg_F1Score(preds,target)
        per_class_f1 = self.per_class_F1Score(preds,target)
        
        self.log('val_f1',f1,on_step=True, on_epoch=True, prog_bar=True, logger=True,enable_graph=True)
        for c in range(4):
            label = 'val_class_' + str(c+1) + '_f1'
            self.log(label,per_class_f1[c],on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        return loss
    
    def configure_optimizers(self):
        """
        Loads and configures the optimizer. See torch.optim.Adam
        for a default option.

        Outputs:
            optimizer: torch.optim.Optimizer
                Optimizer used to minimize the loss
        """
        return torch.optim.SGD(self.parameters(),lr=self.learning_rate)