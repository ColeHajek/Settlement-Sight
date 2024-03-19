import pyprojroot
import sys
import torch
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation

import wandb
@dataclass
class ESDConfig:
    """
    IMPORTANT: This class is used to define the configuration for the experiment
    Please make sure to use the correct types and default values for the parameters
    and that the path for processed_dir contain the tiles you would like 
    """
    processed_dir: str | os.PathLike = root / 'data/processed/4x4' #
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    selected_bands: None = None 
    model_type: str = "UNet"
    tile_size_gt: int = 4
    batch_size: int = 16
    max_epochs: int = 10
    seed: int = 12378921
    learning_rate: float = 0.001
    num_workers: int = 11
    accelerator: str = "cpu"
    devices: int = 1
    in_channels: int = 45
    out_channels: int = 4
    depth: int = 2
    n_encoders: int = 3
    embedding_size: int = 128
    pool_sizes: str = "5,5,2" # List[int] = [5,5,2]
    kernel_size: int = 3
    scale_factor: int = 50
    wandb_run_name: str | None = None
    load_from_chkpt = False
    weights = True
    model_path: str | os.PathLike = root / "models" / "UNet" / "last.ckpt"



def calculate_class_weights(dataset):
    # Count the frequencies of each class in the dataset
    class_frequencies = dataset.count_frequencies()
    
    # Adjust the class labels if necessary (e.g., subtract 1 if your class labels start from 1 instead of 0)
    class_frequencies = {int(key) - 1: value for key, value in class_frequencies.items()}
    
    # Sort the class frequencies based on class indices
    sorted_class_frequencies = {k: class_frequencies[k] for k in sorted(class_frequencies)}
    
    # Calculate the total number of samples
    total_samples = sum(sorted_class_frequencies.values())
    
    # Calculate the number of classes
    num_classes = len(sorted_class_frequencies)

    print("Class Frequencies:", sorted_class_frequencies)
    true_class_weights = torch.tensor(
        [(total_samples / (num_classes * value)) if value != 0 else 0 for value in sorted_class_frequencies.values()],
        dtype=torch.float
    )

    return true_class_weights

# potential band combinations that could be good    
#{ "viirs_maxproj": ["0"],"sentinel1": ["VV", "VH"],"sentinel2":["04","08","11"],"landsat":["4","5","6"]}
#{ "viirs_maxproj": ["0"],"sentinel1": ["VV", "VH"],"sentinel2":["04","08","11"],"landsat":["3","4","5","6","7"]}
#{ "viirs_maxproj": ["0"],"sentinel1": ["VV", "VH"],"sentinel2":["02","03","04","08","11","12"],"landsat":["5","6","7","8"]}
#{ "viirs_maxproj": ["0"],"viirs": ["0"],"sentinel1": ["VV", "VH"],"sentinel2":["02","03","04","08","11","12"],"landsat":["5","6","7","8"]}


def train(options: ESDConfig):
    """
    Prepares datamodule and model, then runs the training loop

    Inputs:
        options: ESDConfig
            options for the experiment
    """
    # Initialize the weights and biases logger
    wb_logger = WandbLogger(
        # set the wandb project where this run will be logged
        project="thethundermen",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": options.learning_rate,
        "architecture": options.model_type,
        "dataset": "IEEE sat",
        "epochs": options.max_epochs,
        }
    )

    #temporary hardcoding bands so i dont have to type them in terminal
    options.selected_bands = { "viirs_maxproj": ["0"],"sentinel1": ["VV", "VH"],"sentinel2":["02","03","04","08","11","12"],"landsat":["5","6","7","8"]}
   
    # initiate the ESDDatamodule
    datamodule = ESDDataModule(
        options.processed_dir,
        options.raw_dir,
        options.selected_bands,
        options.tile_size_gt,
        options.batch_size,
        options.seed
        )
    #current val set has tiles: 4, 13, 16, 23, 24, 29, 30, 35, 55, 56, 59, 60
    
    #preprocess data and set up modules if not already done
    datamodule.prepare_data()
    datamodule.setup()

    #if using inverse frequency 
    if options.weights:
        true_class_weights = calculate_class_weights(datamodule.train_dataset)
    else:
        true_class_weights = None

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    
    # create a dictionary with the parameters to pass to the models

    params = {
        "learning_rate": options.learning_rate,
        "architecture": options.model_type,
        "dataset": "IEEE sat",
        "epochs": options.max_epochs,
    }
   
    
    #Load ESD Segmentation model or create new one
    if options.load_from_chkpt:
        model = ESDSegmentation.load_from_checkpoint(
            checkpoint_path = str(options.model_path),
            model_type = options.model_type,
            in_channels = options.in_channels,
            out_channels = options.out_channels,
            learning_rate = options.learning_rate,
            class_weights = true_class_weights,
            params = params  # Add other necessary parameters as needed
        )
    else:
        model = ESDSegmentation(
            model_type = options.model_type,
            in_channels = options.in_channels,
            out_channels = options.out_channels,
            learning_rate = options.learning_rate,
            class_weights = true_class_weights,
            model_params = params
        )
    wb_logger.watch(model)
    
    # ModelCheckpoint: saves intermediate results for the neural network
    # in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    callbacks = [
        ModelCheckpoint(
            dirpath=root / 'models' / options.model_type,
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
            save_top_k=0,
            save_last=True,
            verbose=True,
            monitor='val_loss',
            mode='min',
            every_n_train_steps=1000
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    
    trainer = Trainer(
        logger=wb_logger,
        max_epochs=options.max_epochs,
        accelerator=options.accelerator,
        devices=options.devices,
        callbacks=callbacks,
        log_every_n_steps=50, #adjustable
        check_val_every_n_epoch=1, #check validation set every epoch
        enable_progress_bar=True
    )

    #look into using the datamodule = datamodule option for trainer.fit
    trainer.fit(model, train_dataloader,val_dataloader)

if __name__ == '__main__':
    
    # load dataclass arguments from yml file
    
    config = ESDConfig()
    parser = ArgumentParser()

    
    parser.add_argument("--model_type", type=str, help="The model to initialize.", default=config.model_type)
    parser.add_argument("--learning_rate", type=float, help="The learning rate for training model", default=config.learning_rate)
    parser.add_argument("--max_epochs", type=int, help="Number of epochs to train for.", default=config.max_epochs)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,help=".")
    
    parser.add_argument('--in_channels', type=int, default=config.in_channels, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=config.out_channels, help='Number of output channels')
    parser.add_argument('--depth', type=int, help="Depth of the encoders (CNN only)", default=config.depth)
    parser.add_argument('--n_encoders', type=int, help="Number of encoders (Unet only)", default=config.n_encoders)
    parser.add_argument('--embedding_size', type=int, help="Embedding size of the neural network (CNN/Unet)", default=config.embedding_size)
    parser.add_argument('--pool_sizes', help="A comma separated list of pool_sizes (CNN only)", type=str, default=config.pool_sizes)
    parser.add_argument('--kernel_size', help="Kernel size of the convolutions", type=int, default=config.kernel_size)
    parser.add_argument('--scale_factor', help="Scale factor between the labels and the image (Unet and Transfer Resnet)", type=int, default=config.scale_factor)
    # --pool_sizes=5,5,2 to call it correctly
    
    parse_args = parser.parse_args()
    
    train(ESDConfig(**parse_args.__dict__))