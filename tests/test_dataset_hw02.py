import sys

import pyprojroot

sys.path.append(str(pyprojroot.here()))
import unittest
from pathlib import Path

import numpy as np
import torch

from src.esd_data.dataset import DSE  # Import your actual module where DSE is defined
from src.preprocessing.subtile_esd_hw02 import (  # Import your actual module where Subtile and TileMetadata are defined
    SatelliteMetadata,
    Subtile,
    TileMetadata,
)

CURRENT_POINTS_TEST = 0
MAX_POINTS_TEST = 25


class TestDSE(unittest.TestCase):
    def setUp(self):
        root = pyprojroot.here()
        # Set up any necessary objects, paths, or configurations for testing
        self.root_dir = (
            root / "data/processed/Train1x1/subtiles/"
        )  # "/path/to/your/data"
        self.selected_bands = {"sentinel1": ["VV"], "gt": ["0"]}
        self.transform = None  # You can set a transformation function if needed
        self.dse_instance = DSE(
            root_dir=self.root_dir,
            selected_bands=self.selected_bands,
            transform=self.transform,
        )

    def test_len_method(self):
        # Test the __len__ method
        self.assertEqual(
            len(self.dse_instance),
            len(list(Path(self.root_dir).glob("*.npz"))),
            "The length of the dataset is not correct.",
        )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
            print("Test len method SUCCESS")
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_aggregate_time_method(self):
        # Test the __aggregate_time method
        input_data = np.random.rand(5, 2, 10, 10)  # Example input data
        output_data = self.dse_instance._DSE__aggregate_time(input_data)
        expected_shape = (5 * 2, 10, 10)
        self.assertEqual(
            output_data.shape,
            expected_shape,
            "The __aggregate_time method is not working as expected.",
        )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
            print("Test aggregate time method SUCCESS")
        else:
            print("Test aggregate time method FAIL")
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_select_indices_method(self):
        # Test the __select_indices method
        bands = ["band1", "band2", "band3"]
        selected_bands = ["band2", "band3"]
        indices = self.dse_instance._DSE__select_indices(bands, selected_bands)
        expected_indices = [1, 2]
        self.assertEqual(
            indices,
            expected_indices,
            "The __select_indices method is not working as expected.",
        )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
            print("Test select indices method SUCCESS")
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_select_bands_method(self):
        subtile_satellite_stack = {
            "sentinel1": np.random.rand(4, 2, 100, 100),
            "viirs": np.random.rand(9, 1, 100, 100),
            "gt": np.random.rand(1, 1, 2, 2),
        }
        satellitemetadata_to_fill_tilemetadata_satellites_dict = {
            "sentinel1": SatelliteMetadata(
                satellite_type="sentinel1",
                bands=["VV", "VH"],
                times=["20200723", "20200804", "20200816", "20200828"],
                original_filenames=[
                    ["S1A_IW_GRDH_20200723_VH.tif", "S1A_IW_GRDH_20200723_VV.tif"],
                    ["S1A_IW_GRDH_20200804_VH.tif", "S1A_IW_GRDH_20200804_VV.tif"],
                    ["S1A_IW_GRDH_20200816_VH.tif", "S1A_IW_GRDH_20200816_VV.tif"],
                    ["S1A_IW_GRDH_20200828_VH.tif", "S1A_IW_GRDH_20200828_VV.tif"],
                ],
            ),
            "viirs": SatelliteMetadata(
                satellite_type="viirs",
                bands=["0"],
                times=[
                    "20200221",
                    "20200224",
                    "20200225",
                    "20200226",
                    "20200227",
                    "20200231",
                    "20200235",
                    "20200236",
                    "20200237",
                ],
                original_filenames=[
                    ["DNB_VNP46A1_A2020221.tif"],
                    ["DNB_VNP46A1_A2020224.tif"],
                    ["DNB_VNP46A1_A2020225.tif"],
                    ["DNB_VNP46A1_A2020226.tif"],
                    ["DNB_VNP46A1_A2020227.tif"],
                    ["DNB_VNP46A1_A2020231.tif"],
                    ["DNB_VNP46A1_A2020235.tif"],
                    ["DNB_VNP46A1_A2020236.tif"],
                    ["DNB_VNP46A1_A2020237.tif"],
                ],
            ),
            "gt": SatelliteMetadata(
                satellite_type="gt",
                bands=["0"],
                times=[""],
                original_filenames=[["groundTruth.tif"]],
            ),
        }

        subtile_tilemetadata = TileMetadata(
            satellites=satellitemetadata_to_fill_tilemetadata_satellites_dict,
            x_gt=2,
            y_gt=2,
            subtile_size=100,
            parent_tile_id="Tile1",
        )
        subtile = Subtile(
            satellite_stack=subtile_satellite_stack, tile_metadata=subtile_tilemetadata
        )
        selected_sat_stack, selected_metadata = self.dse_instance._DSE__select_bands(
            subtile
        )

        print("selected_sat_stack", selected_sat_stack["sentinel1"].shape)
        print("selected_metadata", selected_metadata.satellites["sentinel1"].bands)
        self.assertEqual(
            selected_sat_stack["sentinel1"].shape,
            (4, 1, 100, 100),
            "__select_bands method is not filtering by bands.",
        )
        self.assertEqual(
            selected_metadata.satellites["sentinel1"].bands[0],
            "VV",
            "__select_bands method is not filtering by bands.",
        )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
            print("Test select bands method SUCCESS")
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_getitem_method(self):
        # Test the __getitem__ method
        idx = 0  # Choose an index for testing
        X, y, tile_metadata = self.dse_instance.__getitem__(idx)
        self.assertIsInstance(
            X,
            (np.ndarray, torch.Tensor),
            "The __getitem__ method is not returning the expected type for X",
        )
        self.assertIsInstance(
            y,
            (np.ndarray, torch.Tensor),
            "The __getitem__ method is not returning the expected type for y",
        )
        self.assertIsInstance(
            tile_metadata,
            TileMetadata,
            "The __getitem__ method is not returning the expected type for tile_metadata",
        )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
            print("Test getitem method SUCCESS")
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")


if __name__ == "__main__":
    unittest.main()
