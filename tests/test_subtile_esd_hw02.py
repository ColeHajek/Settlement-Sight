""" Unit tests for the subtile_esd_hw02 module."""

import unittest
import os
import sys
import pyprojroot
sys.path.append(str(pyprojroot.here()))
import numpy as np
from pathlib import Path
import json
from src.preprocessing.subtile_esd_hw02 import(
    SatelliteMetadata,
    TileMetadata,
    Subtile,
    metadata_to_tile_metadata,
    tile_metadata_to_metadata,
    get_tile_ground_truth,
    get_tile_satellite,
    grid_slice,
    restitch
)
from src.preprocessing.file_utils import Metadata
CURRENT_POINTS_TEST = 0
MAX_POINTS_TEST = 20

class TestTileMetadata(unittest.TestCase):
    def setUp(self):
        # Set up any necessary objects, paths, or configurations for testing
        self.satellite_metadata = SatelliteMetadata(satellite_type="sentinel1", bands=["VV", "VH"], times=["20200723"], original_filenames=[["S1A_IW_GRDH_20200723_VV.tif", "S1A_IW_GRDH_20200723_VH.tif"]])
        self.satellite_object_dict = {"sentinel1": self.satellite_metadata}
        self.tile_metadata = TileMetadata(satellites=self.satellite_object_dict, x_gt=1, y_gt=1, subtile_size=50, parent_tile_id="Tile1")
        root = pyprojroot.here()
        self.file_name = root / "data/processed/unit_test_dummies/Tile1_dummy_0_0.json"

    def test_toJSON(self):
        expected_json = {
            "parent_tile_id": "Tile1",
            "satellites": {
                "sentinel1": {
                    "bands": [
                        "VV",
                        "VH"
                    ],
                    "original_filenames": [
                        [
                            "S1A_IW_GRDH_20200723_VV.tif",
                            "S1A_IW_GRDH_20200723_VH.tif"
                        ]
                    ],
                    "satellite_type": "sentinel1",
                    "times": [
                        "20200723"
                    ]
                }
            },
            "subtile_size": 50,
            "x_gt": 1,
            "y_gt": 1
        }
        self.assertEqual(json.loads(self.tile_metadata.toJSON()), expected_json, "JSON serialization failed")
    
    def test_saveJSON(self):
        self.tile_metadata.saveJSON(file_name=self.file_name)
        self.assertTrue(os.path.exists(self.file_name), "JSON file not saved")
    
    def test_loadJSON(self):
        loaded_tile_metadata_json = self.tile_metadata.loadJSON(self.file_name)
        self.assertIsInstance(loaded_tile_metadata_json, TileMetadata, "loadJSON member function did not return a TileMetadata object.")
        # Check that the attributes of the loaded TileMetadata object match the expected values
        self.assertEqual(loaded_tile_metadata_json.parent_tile_id, "Tile1", "parent_tile_id attribute does not match expected value.")
        self.assertEqual(loaded_tile_metadata_json.x_gt, 1, "x_gt attribute does not match expected value.")
        self.assertEqual(loaded_tile_metadata_json.y_gt, 1, "y_gt attribute does not match expected value.")
        self.assertEqual(loaded_tile_metadata_json.subtile_size, 50, "subtile_size attribute does not match expected value.")

        # Check that the nested SatelliteMetadata objects have the correct attributes
        sentinel1_metadata = loaded_tile_metadata_json.satellites['sentinel1']
        self.assertIsInstance(sentinel1_metadata, SatelliteMetadata, "satellites['sentinel1'] attribute does not contain a SatelliteMetadata object.")
        self.assertEqual(sentinel1_metadata.satellite_type, "sentinel1", "satellite_type attribute does not match expected value.")
        self.assertEqual(sentinel1_metadata.bands, ["VV", "VH"], "bands attribute does not match expected value.")
        self.assertEqual(sentinel1_metadata.times, ["20200723"], "time string attribute does not match expected value.")
        self.assertEqual(sentinel1_metadata.original_filenames, [["S1A_IW_GRDH_20200723_VV.tif", "S1A_IW_GRDH_20200723_VH.tif"]], "original_filenames attribute does not match expected value.")

    def tearDown(self):
        pass


class TestGridSlice(unittest.TestCase):
    def setUp(self):
        self.satellite_stack = {
            "sentinel1": np.random.rand(4, 2, 800, 800),  # 16 tiles
            "viirs": np.random.rand(9, 1, 800, 800),
            "gt": np.random.rand(1, 1, 16, 16)
        }

        list_metadata_s1 = [
            Metadata(
                satellite_type="sentinel1",
                file_name=["S1A_IW_GRDH_20200723_VV.tif", "S1A_IW_GRDH_20200723_VH.tif"],
                tile_id="Tile1",
                bands=["VV", "VH"],
                time="20200723"
            ),
            Metadata(
                satellite_type="sentinel1",
                file_name=["S1A_IW_GRDH_20200804_VV.tif", "S1A_IW_GRDH_20200804_VH.tif"],
                tile_id="Tile1",
                bands=["VV", "VH"],
                time="20200804"
            ),
            Metadata(
                satellite_type="sentinel1",
                file_name=["S1A_IW_GRDH_20200816_VV.tif", "S1A_IW_GRDH_20200816_VH.tif"],
                tile_id="Tile1",
                bands=["VV", "VH"],
                time="20200816"
            ),
            Metadata(
                satellite_type="sentinel1",
                file_name=["S1A_IW_GRDH_20200828_VV.tif", "S1A_IW_GRDH_20200828_VH.tif"],
                tile_id="Tile1",
                bands=["VV", "VH"],
                time="20200828"
            )   
        ]
        
        list_metadata_viirs = [
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020221.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020221"
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020224.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020224"
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020225.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020225"
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020226.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020226"
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020227.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020227"
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020231.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020231"
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020235.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020235"
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020236.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020236"
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020237.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020237"
            )
        ]
        
        list_metadata_gt = [
            Metadata(
                satellite_type="gt",
                file_name=["groundTruth.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="0"
            )
        ]


        self.metadata_stack = {
            "sentinel1": list_metadata_s1,
            "viirs": list_metadata_viirs,
            "gt": list_metadata_gt
        }
        self.tile_size_gt = 2

    def test_grid_slice(self):
        subtiles = grid_slice(
            self.satellite_stack,
            self.metadata_stack,
            self.tile_size_gt
            )
        self.assertEqual(len(subtiles), 64, "grid_slice did not return the expected number of subtiles")
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")
    
class TestRestitch(unittest.TestCase):
    def setUp(self):
        root = pyprojroot.here()
        self.dir = root / "data/processed/Train1x1/subtiles"
        self.satellite_type = "sentinel1"
        self.tile_id = "Tile1"
        self.range_x = (0, 5)
        self.range_y = (0, 6)

    def test_restitch(self):
        restitched_np, restitched_tilemetadata_list = restitch(
            dir=self.dir,
            satellite_type=self.satellite_type,
            tile_id=self.tile_id,
            range_x=self.range_x,
            range_y=self.range_y
        )
        self.assertEqual(len(restitched_tilemetadata_list), 5, "restitched list of TileMetadata is of incorrect length")
        self.assertEqual(restitched_np.shape, (4, 2, 250, 300), "restitched image is not of correct dimensions.")
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

class TestGetFunctions(unittest.TestCase):
    def setUp(self):
        self.gt_parent = np.random.rand(1,1,16,16)
        self.satellite_parent = np.random.rand(4, 2, 800, 800)
        self.x_gt = 2
        self.y_gt = 2
        self.size_gt = 2
        self.x_sat = 0
        self.y_sat = 0
    

    def test_get_tile_ground_truth(self):
        subtile_gt = get_tile_ground_truth(
            gt_parent=self.gt_parent,
            x_gt=self.x_gt,
            y_gt=self.y_gt,
            size_gt=self.size_gt

        )
        self.assertEqual(subtile_gt.shape, (1,1,2,2), "ground truth subtile incorrect dimensions")
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_get_tile_satellite(self):
        subtile_satellite = get_tile_satellite(
            sat_parent=self.satellite_parent,
            x_sat=self.x_sat,
            y_sat=self.y_sat,
            size_gt=self.size_gt,
            scale_factor=50
        )
        self.assertEqual(subtile_satellite.shape, (4, 2, 100, 100), "subtile satellite incorrect dimensions.")
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")





if __name__ == "__main__":
    unittest.main()


