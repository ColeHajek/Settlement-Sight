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
    selected_bands: None = None #dict# {'sentinel1': ['VV', 'VH']} #None = None
    #selected_bands: Dict[str, List[str]]
    model_type: str = "UNet"
    tile_size_gt: int = 4
    batch_size: int = 8
    max_epochs: int = 10
    seed: int = 12378921
    learning_rate: float = 0.001
    num_workers: int = 11
    accelerator: str = "cpu"
    devices: int = 1
    in_channels: int = 99
    out_channels: int = 4
    depth: int = 2
    n_encoders: int = 2
    embedding_size: int = 64
    pool_sizes: str = "5,5,2" # List[int] = [5,5,2]
    kernel_size: int = 3
    scale_factor: int = 50
    wandb_run_name: str | None = None
    load_from_chkpt = True
    model_path: str | os.PathLike = root / "models" / "UNet" / "last.ckpt"

def calculate_class_weights(dataset):
    # Count the frequencies of each class in the dataset
    class_frequencies = dataset.count_frequencies()
    
    # Adjust the class labels if necessary (e.g., subtract 1 if your class labels start from 1 instead of 0)
    class_frequencies = {int(key) - 1: value for key, value in class_frequencies.items()}
    
    # Sort the class frequencies based on class indices
    sorted_class_frequencies = {k: class_frequencies[k] for k in sorted(class_frequencies)}
    
    # Calculate the inverse of each class frequency
    # If a frequency is zero (indicating no samples for a class), handle it to avoid division by zero
    print("Class Frequencies:",sorted_class_frequencies)
    true_class_weights = torch.tensor([1000.0 / value if value != 0 else 0 for value in sorted_class_frequencies.values()], dtype=torch.float)
    #sum_weights = torch.sum(true_class_weights)
    #normalized_weights = true_class_weights / sum_weights
    return true_class_weights
    
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
    #models/SegmentationCNN/last.ckpt
    
    # initiate the ESDDatamodule
    datamodule = ESDDataModule(options.processed_dir, options.raw_dir,
                              options.selected_bands, options.tile_size_gt,
                               options.batch_size, options.seed)

    # make sure to prepare_data in case the data has not been preprocessed
    datamodule.prepare_data()
    datamodule.setup()

    #print validation
    val_class_dist = calculate_class_weights(datamodule.val_dataset)
    true_class_weights = calculate_class_weights(datamodule.train_dataset)

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    
    # create a dictionary with the parameters to pass to the models

    params = {
        "learning_rate": options.learning_rate,
        "architecture": options.model_type,
        "dataset": "IEEE sat",
        "epochs": options.max_epochs,
        #"device": options.accelerator
    }
    # initialize the ESDSegmentation module
    
    #Load model or create new one
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
    # Use the following callbacks, they're provided for you,
    # but you may change some of the settings
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

    # create a pytorch Trainer
    # see pytorch_lightning.Trainer
    # make sure to use the options object to load it with the correct options
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

    # run trainer.fit
    # make sure to use the datamodule option
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