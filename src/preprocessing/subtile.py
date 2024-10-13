import sys
from pathlib import Path
from typing import List, Tuple, Union
import xarray as xr

# local modules
sys.path.append(".")
from src.utilities import SatelliteType


class Subtile:
    def __init__(
        self,
        satellite_list: List[xr.DataArray],
        ground_truth: Union[xr.DataArray, None] = None,
        slice_size: Tuple[int, int] = (4, 4),
        parent_tile_id: Union[int, None] = None,
    ):
        """
        Initializes the Subtile class, which handles saving and loading subtiles for a parent image.

        Parameters:
            satellite_list (List[xr.DataArray]): List of satellite data arrays.
            ground_truth (xr.DataArray | None): Ground truth data array, if available.
            slice_size (Tuple[int, int]): The slice size of the image, e.g., (4, 4) for 4x4 subtiles.
            parent_tile_id (int | None): ID of the parent tile. If not provided, it is extracted from the data.
        """
        self.satellite_list = satellite_list
        self.ground_truth = ground_truth
        self.slice_size = slice_size

        # Set parent_tile_id from attributes if not provided
        if parent_tile_id is None:
            raw_parent_tile_id = satellite_list[0].attrs["parent_tile_id"]
            self.parent_tile_id = Path(raw_parent_tile_id).name
        else:
            self.parent_tile_id = parent_tile_id

    def __calculate_slice_indices(
        self, x: int, y: int, slice_size: Tuple[int, int], shape: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Calculates the start and end indices for slicing based on the slice size and coordinates.

        Parameters:
            x (int): X index of the slice.
            y (int): Y index of the slice.
            slice_size (Tuple[int, int]): Size of the slice (rows, columns).
            shape (Tuple[int, int]): Shape of the array (height, width).

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: Start and end indices for slicing.
        """
        row_factor, col_factor = shape[0] / slice_size[0], shape[1] / slice_size[1]
        start_idx = (int(row_factor * x), int(col_factor * y))
        end_idx = (int(row_factor * (x + 1)), int(col_factor * (y + 1)))

        self.__validate_indices(start_idx, end_idx, shape)
        return start_idx, end_idx

    def __validate_indices(self, start_idx: Tuple[int, int], end_idx: Tuple[int, int], shape: Tuple[int, int]):
        """Ensure indices created are within bounds"""
        if any(s > l for s, l in zip(start_idx, shape)) or any(e > l for e, l in zip(end_idx, shape)):
            raise IndexError(f"Slice indices {start_idx}, {end_idx} are out of range for shape {shape}.")

    def get_subtile_from_parent_image(self, x: int, y: int) -> Tuple[List[xr.DataArray], xr.DataArray]:
        """
        Subtiles the satellite and ground truth data at given x, y coordinates.

        Parameters:
            x (int): X index of the subtile.
            y (int): Y index of the subtile.

        Returns:
            Tuple[List[xr.DataArray], xr.DataArray]: Subtiled satellite data and ground truth.
        """
        img_shape = self.satellite_list[0].shape[-2:]
        label_shape = self.ground_truth.shape[-2:]

        start_img, end_img = self.__calculate_slice_indices(x, y, self.slice_size, img_shape)
        start_label, end_label = self.__calculate_slice_indices(x, y, self.slice_size, label_shape)

        subtiled_satellites = [data[:, :, start_img[0]:end_img[0], start_img[1]:end_img[1]] for data in self.satellite_list]
        subtiled_ground_truth = self.ground_truth[:, :, start_label[0]:end_label[0], start_label[1]:end_label[1]]

        # Add x, y attributes to the subtiles for reference
        for sub_tile in subtiled_satellites:
            sub_tile.attrs.update({"x": x, "y": y})
        subtiled_ground_truth.attrs.update({"x": x, "y": y})

        return subtiled_satellites, subtiled_ground_truth

    def _save_array(self, data_array: xr.DataArray, directory: Path, x: int, y: int, filename: str):
        """
        Saves a given data array to a NetCDF file.

        Parameters:
            data_array (xr.DataArray): Data array to be saved.
            directory (Path): Directory to save the data.
            x (int): X index of the subtile.
            y (int): Y index of the subtile.
            filename (str): Name of the file to save the data.
        """
        path = directory / self.parent_tile_id / f"{x}_{y}" / filename
        path.parent.mkdir(parents=True, exist_ok=True)  # Create directory structure if not exists
        data_array.to_netcdf(path)

    def save(self, directory_to_save: Path) -> None:
        """
        Saves subtiles of the parent tile to the specified directory.

        Parameters:
            directory_to_save (Path): Base directory to save the subtiles.
        """
        for x in range(self.slice_size[0]):
            for y in range(self.slice_size[1]):
                satellite_subtiles, ground_truth_subtile = self.get_subtile_from_parent_image(x, y)

                # Save each satellite subtile
                for subtile in satellite_subtiles:
                    filename = f"{subtile.attrs['satellite_type']}.nc"
                    self._save_array(subtile, directory_to_save / "subtiles", x, y, filename)

                # Save ground truth subtile
                self._save_array(ground_truth_subtile, directory_to_save / "subtiles", x, y, f"{SatelliteType.GROUND_TRUTH.value}.nc")

        # Clear data to free memory after saving
        self.satellite_list = None
        self.ground_truth = None

    def load_subtile(
        self, directory_to_load: Path, satellite_type_list: List[SatelliteType], x: int, y: int
    ) -> List[xr.DataArray]:
        """
        Loads a single subtile given the directory, satellite type, and slice coordinates.

        Parameters:
            directory_to_load (Path): Base directory to load subtiles from.
            satellite_type_list (List[SatelliteType]): List of satellite types to load.
            x (int): X index of the subtile.
            y (int): Y index of the subtile.

        Returns:
            List[xr.DataArray]: List of loaded subtiles as data arrays.
        """
        subtile_path = directory_to_load / "subtiles" / self.parent_tile_id / f"{x}_{y}"
        return [self._load_data_array(subtile_path / f"{satellite_type.value}.nc", x, y) for satellite_type in satellite_type_list]

    @staticmethod
    def _load_data_array(file_path: Path, x: int, y: int) -> xr.DataArray:
        """
        Loads a data array from a NetCDF file and validates its coordinates.

        Parameters:
            file_path (Path): Path to the NetCDF file.
            x (int): Expected X index in the file attributes.
            y (int): Expected Y index in the file attributes.

        Returns:
            xr.DataArray: Loaded data array.
        """
        assert file_path.exists(), f"{file_path} does not exist"
        data_array = xr.load_dataarray(file_path)
        assert data_array.attrs["x"] == x and data_array.attrs["y"] == y, f"Invalid coordinates in {file_path}"
        return data_array

    @staticmethod
    def load_subtile_by_dir(
        directory_to_load: Path, satellite_type_list: List[SatelliteType], slice_size: Tuple[int, int] = (4, 4), has_gt: bool = True
    ) -> "Subtile":
        """
        Loads all subtiles from a given directory.

        Parameters:
            directory_to_load (Path): Directory containing the subtile files.
            satellite_type_list (List[SatelliteType]): List of satellite types to load.
            slice_size (Tuple[int, int]): Size of the subtiles.
            has_gt (bool): Whether ground truth data is available.

        Returns:
            Subtile: An instance of the Subtile class with loaded data arrays.
        """
        subtiled_data = [xr.load_dataarray(directory_to_load / f"{satellite_type.value}.nc") for satellite_type in satellite_type_list]

        ground_truth = None
        if has_gt:
            ground_truth = xr.load_dataarray(directory_to_load / f"{SatelliteType.GROUND_TRUTH.value}.nc")

        return Subtile(satellite_list=subtiled_data, ground_truth=ground_truth, slice_size=slice_size)

    def restitch(self, directory_to_load: Path, satellite_type_list: List[SatelliteType]) -> None:
        """
        Restitches subtiles into their original image.

        Parameters:
            directory_to_load (Path): Directory containing the subtiles.
            satellite_type_list (List[SatelliteType]): List of satellite types to restitch.
        """
        satellite_type_list_with_gt = satellite_type_list + [SatelliteType.GROUND_TRUTH]
        stitched_data = [self._stitch_images(directory_to_load, satellite_type) for satellite_type in satellite_type_list_with_gt]

        self.satellite_list = stitched_data[:-1]
        self.ground_truth = stitched_data[-1]

    def _stitch_images(self, directory_to_load: Path, satellite_type: SatelliteType) -> xr.DataArray:
        """
        Helper function to stitch images together from subtiles.

        Parameters:
            directory_to_load (Path): Directory containing the subtiles.
            satellite_type (SatelliteType): The type of satellite data to stitch.

        Returns:
            xr.DataArray: The stitched data array.
        """
        rows = [
            xr.concat([self.load_subtile(directory_to_load, [satellite_type], x, y)[0] for y in range(self.slice_size[1])], dim="width")
            for x in range(self.slice_size[0])
            ]
        return xr.concat(rows, dim="height")