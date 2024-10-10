from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import xarray as xr



class SatelliteType(Enum):
    VIIRS_MAX_PROJ = "viirs_max_projection"
    VIIRS = "viirs"
    S1 = "sentinel1"
    S2 = "sentinel2"
    LANDSAT = "landsat"
    GT = "gt"


ROOT = Path.cwd()
PROJ_NAME = "Settlement-Sight Revision"
MODEL = "unet"  


@dataclass
class ESDConfig:
    processed_dir: Path = ROOT / "data" / "processed" 
    raw_dir: Path = ROOT / "data" / "raw" / "Train"
    checkpoint_file: str = ""
    results_dir: Path = ROOT / "results" / "predictions" / MODEL

    model_path: Path = ROOT / "model_checkpoints" / MODEL / "last.ckpt"
    load_from_check_point = False
    model_type: str = MODEL
    wandb_run_name: str | None = None
    
    selected_bands = {
        SatelliteType.VIIRS: ["0"],
        SatelliteType.S1: ["VV", "VH"],
        SatelliteType.S2: [
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
            "4",
            #"3",
            #"2",
            # "1",
        ],
        SatelliteType.VIIRS_MAX_PROJ: ["0"],
    }

    accelerator: str = "gpu"
    devices: int = 1
    num_workers: int = 4
    optomizer: str = "adam"
    batch_size: int = 32
    depth: int = 4
    embedding_size: int = 64
    kernel_size: int = 3
    n_encoders: int = 3
    learning_rate: float = 0.0001
    lambda_l1: float =0.00
    pool_sizes: str = "5,5,2"
    slice_size: tuple = (4, 4)
    max_epochs: int = 100

    in_channels: int = 57  
    out_channels: int = 4  
    
    seed: int = 12378921
    
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
