from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import xarray as xr



class SatelliteType(Enum):
    VIIRS_MAX_PROJ = "viirs_max_projection"
    VIIRS = "viirs"
    SENTINEL_1 = "sentinel1"
    SENTINEL_2 = "sentinel2"
    LANDSAT = "landsat"
    GROUND_TRUTH = "gt"


ROOT = Path.cwd()
PROJ_NAME = "Settlement-Sight Revision"
MODEL = "unet"  


@dataclass
class ESDConfig:
    processed_dir: Path = ROOT / "data" / "processed" 
    raw_dir: Path = ROOT / "data" / "raw" / "Train"
    checkpoint_file: str = ""
    results_dir: Path = ROOT / "results" / "predictions" / MODEL

    model_path: Path = ROOT / "model_checkpoints" / MODEL / "last-v2.ckpt"
    load_from_check_point = False
    model_type: str = MODEL
    wandb_run_name: str | None = None
    '''
    selected_bands = {
        SatelliteType.VIIRS: ["0"],
        SatelliteType.SENTINEL_1: ["VV", "VH"],
        SatelliteType.SENTINEL_2: [
            "12",
            "11",
            # "09",
            # "8A",
            "08",
            # "07",
            # "06",
            # "05",
            "04",
            "03",
            "02",
            # "01",
        ],
        SatelliteType.LANDSAT: [
            #"11",
            #"10",
            # "9",
            "8",
            "7",
            "6",
            "5",
            #"4",
            #"3",
            #"2",
            # "1",
        ],
        SatelliteType.VIIRS_MAX_PROJ: ["0"],
    }
    '''
    selected_bands = {
        SatelliteType.VIIRS: ["0"],
        SatelliteType.SENTINEL_1: ["VV", "VH"],
        SatelliteType.SENTINEL_2: [
            "12",
            "11",
             "09",
            "8A",
            "08",
            "07",
            "06",
            "05",
            "04",
            "03",
            "02",
            "01",
        ],
        SatelliteType.LANDSAT: [
            "11",
            "10",
            "9",
            "8",
            "7",
            "6",
            "5",
            "4",
            "3",
            "2",
            "1",
        ],
        SatelliteType.VIIRS_MAX_PROJ: ["0"],
    }

    accelerator: str = "gpu"
    devices: int = 1
    num_workers: int = 4


    batch_size: int = 16
    depth: int = 3
    embedding_size: int = 128
    kernel_size: int = 5
    n_encoders: int = 2
    learning_rate: float = 0.0001
    lambda_l1: float =0.0001
    pool_sizes: str = "5,5,2"
    slice_size: tuple = (4, 4)
    max_epochs: int = 100
    dropout_prob: float = 0.2
    in_channels: int = 54
    out_channels: int = 4
    
    seed: int = 12345678
    '''
    Batch size: 16
    Max epochs: 10
    Learning rate: 0.001
    Embedding Size: 128
    Number of Encoders: 2
    Using lr_Adam
    '''
def get_satellite_dataset_size(
    data_set: xr.Dataset, dims: List[str] = ["date", "band", "height", "width"]
):
    """
    Gets the shape of a dataset

    Parameters
    ----------
    data_set : xr.Dataset
        A satellite dataset
    dims: List[str]
        A list of dimensions of the data_set data_arrays
    Returns
    -------
    Tuple:
        Shape of the data_set, default is (date, band, height, width)
    """
    return tuple(data_set.sizes[d] for d in dims)
