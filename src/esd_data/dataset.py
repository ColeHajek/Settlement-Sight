import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as torchvision_transforms

# Local modules
sys.path.append(".")
from src.preprocessing.subtile import Subtile
from src.utilities import SatelliteType


class ESDDataset(Dataset):
    """
    A PyTorch Dataset class for handling satellite image subtiles in DataLoader batches.
    
    Attributes:
        processed_dir (Path): Path to the directory containing processed subtiles.
        transform (Compose): Composition of image transformations to be applied.
        satellite_type_list (List[SatelliteType]): List of satellite types to include.
        slice_size (Tuple[int, int]): Size of subtiles.
        subtile_dirs (List[Path]): List of paths to individual subtiles.
        
    Methods:
        __len__() -> int:
            Returns the number of subtiles in the dataset.
            
        __getitem__(idx: int) -> Tuple[np.ndarray, np.ndarray]:
            Retrieves and processes the subtile at the specified index, 
            applying transformations and returning the input and target tensors.
    """

    def __init__(
        self,
        processed_dir: Path,
        transform: torchvision_transforms.transforms.Compose,
        satellite_type_list: List[SatelliteType],
        slice_size: Tuple[int, int] = (4, 4),
    ) -> None:
        """
        Initialize the ESDDataset with the processed directory and configurations.
        
        Parameters:
            processed_dir (Path): Path to the directory containing processed subtiles.
            transform (Compose): Composition of image transformations to be applied.
            satellite_type_list (List[SatelliteType]): List of satellite types to include.
            slice_size (Tuple[int, int]): Size of subtiles for each image (default: (4, 4)).
        """
        self.transform = transform
        self.satellite_type_list = satellite_type_list
        self.slice_size = slice_size

        # Gather the paths of all subtiles within the processed directory
        self.subtile_dirs = []
        for tile_dir in processed_dir.iterdir():
            # Iterate through each tile directory to access individual subtiles
            for subtile_dir in tile_dir.iterdir():
                for subtile in subtile_dir.iterdir():
                    # Append each subtile path to the subtile_dirs list
                    self.subtile_dirs.append(subtile)

    def __len__(self) -> int:
        """
        Returns the number of subtiles in the dataset.
        
        Returns:
            int: Total number of subtiles.
        """
        return len(self.subtile_dirs)

    def __aggregate_time(self, array: np.ndarray) -> np.ndarray:
        """
        Combine the time and band dimensions of the array for model input.
        
        This method reshapes an array of shape (time, bands, width, height) to 
        (time*bands, width, height) to flatten the time and band dimensions.
        
        Parameters:
            array (np.ndarray): Array of shape (time, bands, width, height).
            
        Returns:
            np.ndarray: Flattened array of shape (time*bands, width, height).
        """
        return array.reshape(array.shape[0] * array.shape[1], array.shape[2], array.shape[3])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves a subtile at a specified index, aggregates the time bands, 
        applies transformations, and returns the input data and ground truth labels.
        
        Parameters:
            idx (int): Index of the subtile to retrieve.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed input data and target labels.
        """
        # Load the subtile data from the directory at the specified index
        subtile = Subtile.load_subtile_by_dir(self.subtile_dirs[idx], self.satellite_type_list, self.slice_size)

        # Extract satellite data arrays and ground truth from the subtile
        data_array_list = subtile.satellite_list
        ground_truth = subtile.ground_truth

        # Convert xarray data to numpy arrays for processing
        data_values_list = [data.values for data in data_array_list]
        ground_truth_values = ground_truth.values

        # Aggregate the time bands for each satellite data array
        X = [self.__aggregate_time(data) for data in data_values_list]
        X = np.concatenate(X, axis=0)  # Concatenate along the band dimension

        # Squeeze ground truth dimensions to get shape (width, height)
        y = np.squeeze(ground_truth_values, axis=(0, 1))

        # If transformations are provided, apply them to the input and labels
        if self.transform:
            transformed_sample = self.transform({"X": X, "y": y})
            X, y = transformed_sample["X"], transformed_sample["y"]

        # Return input data and ground truth labels (zero-indexed)
        return X, y - 1
