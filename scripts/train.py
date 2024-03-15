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
    in_channels: int = 30
    out_channels: int = 4
    depth: int = 2
    n_encoders: int = 2
    embedding_size: int = 64
    pool_sizes: str = "5,5,2" # List[int] = [5,5,2]
    kernel_size: int = 3
    scale_factor: int = 50
    wandb_run_name: str | None = None
    load_from_chkpt = False
    model_path: str | os.PathLike = root / "models" / "UNet" / "last.ckpt"
    
#{ "viirs_maxproj": ["0"],"sentinel1": ["VV", "VH"],"sentinel2":["04","08","11"],"landsat":["4","5","6"]}
#{ "viirs_maxproj": ["0"],"sentinel1": ["VV", "VH"],"sentinel2":["04","08","11"],"landsat":["3","4","5","6","7"]}
##{ "viirs_maxproj": ["0"],"sentinel1": ["VV", "VH"],"sentinel2":["04","08","11"],"landsat":["3","4","5","6","7"]}

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
#potentially useful bands:
# need: viirs_maxproj
# landsat 9 for detecting if theres a cloud, this would help tell other tiles to ignore it
# landsat maybe 10 or 11? they're used for measuring surface temperatures
# landsat 4 red wavelength good for urban areas and different types of vegitation/soils
# landsat 3 (green) for detecting vegetation and water bodies
# probably exclude landsat 1 and 2
# 

'''
Explination of band functions:
with importance ranking
Landsat Bands:
1/5 Band 1 (Coastal/Aerosol): Captures light in the coastal and aerosol wavelengths, useful for studying coastal water and atmospheric aerosols.            Probably exclude
1/5 Band 2 (Blue): Primarily captures blue light, used in marine and atmospheric studies.                                                                   Probably exclude
4/5 Band 3 (Green): Captures green light, useful for analyzing vegetation and water bodies.                                                                 Use?
4/5 Band 4 (Red): Captures red light, helpful for distinguishing different types of vegetation, soils, and urban areas.                                     
4/5 Band 5 (Near-Infrared): Reflects plant health and biomass content, useful for assessing vegetation and water body delineation.
3/5 Band 6 (SWIR 1): Sensitive to moisture in soil and vegetation, helps in plant stress analysis and fire detection.
3/5? Band 7 (SWIR 2): Penetrates atmospheric haze well, useful for geological and soil mapping.
3/5? Band 8 (Panchromatic): Provides high-resolution black-and-white imagery, useful for detailed mapping.
4/5? Band 9 (Cirrus): Detects high atmospheric clouds, aiding in cloud correction for other bands.
Band 10 (Thermal Infrared 1): Measures soil and surface temperatures, useful in geothermal and vegetation studies.
Band 11 (Thermal Infrared 2): Similar to Band 10 but at a different wavelength, providing additional temperature information.

Sentinel-1 Bands:
6/10 VH (Vertical Transmit, Horizontal Receive): Useful for differentiating between types of surfaces and moisture levels, particularly in vegetation.
9/10 VV (Vertical Transmit, Vertical Receive): Provides details on surface texture and moisture, better for urban and water body mapping.

Sentinel-2 Bands:
Band 01 (Coastal/Aerosol): Helps with coastal and atmospheric studies, similar to Landsat.
Band 02 (Blue): Used for water body detection and atmospheric correction.
Band 03 (Green): Important for analyzing plant health and land.
Band 04 (Red): Essential for vegetation differentiation and health assessment.
Band 05 (Vegetation Red Edge): Provides information on plant chlorophyll content.
Band 06 (Vegetation Red Edge): Further details on plant health, used in vegetation indices.
Band 07 (Vegetation Red Edge): Adds depth to vegetation mapping and health assessment.
Band 08 (NIR): Key for assessing vegetation biomass and water bodies.
Band 09 (Water Vapour): Used for atmospheric correction and studying moisture.
Band 11 (SWIR): Good for soil and vegetation moisture content analysis.
Band 12 (SWIR): Used for atmospheric contamination detection and correction.
Band 8A (Narrow NIR): Provides detailed vegetation information, particularly for crop monitoring.

VIIRS Bands:
Band 0: Utilized for night-time light detection, essential for observing human activity, urbanization, and electricity usage.

VIIRS MaxProj Bands:
Band 0: Essentially the same as the standard VIIRS Band 0, designed for capturing night-time light, but typically used in different data products or processing techniques.


'''
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
                              { "viirs_maxproj": ["0"],"sentinel1": ["VV", "VH"],"sentinel2":["04","08","11"],"landsat":["4","5","6"]}, options.tile_size_gt,
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