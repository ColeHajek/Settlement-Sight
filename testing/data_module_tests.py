""" Unit tests for dataset_hw02.py """

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyprojroot

sys.path.append(str(pyprojroot.here()))
from src.esd_data.datamodule import ESDDataModule, collate_fn
from src.esd_data.dataset import DSE
from src.preprocessing.file_utils import Metadata

CURRENT_POINTS_TEST = 0
MAX_POINTS_TEST = 20


class TestESDDataModule(unittest.TestCase):
    def setUp(self):
        root = pyprojroot.here()
        self.processed_dir = Path(root / "data/processed/Train/subtiles")
        self.raw_dir = Path(root / "/data/raw/Train/")
        self.selected_bands = {
            "sentinel1": ["VV", "VH"],
            "viirs_maxproj": ["0"],
            "gt": ["0"],
        }
        self.dm = ESDDataModule(
            self.processed_dir, self.raw_dir, self.selected_bands, tile_size_gt=2
        )

    @patch("src.esd_data.dataset.DSE")
    @patch("os.path.exists", return_value=False)
    @patch("pathlib.Path.is_dir", return_value=False)
    @patch("pathlib.Path.glob", side_effect=["Tile1"])
    @patch("src.preprocessing.subtile_esd_hw02.Subtile.save")
    def test_prepare_data(
        self, mock_dse, mock_exists, mock_isdir, mock_glob, mock_save
    ):
        # Mock os.path.exists to simulate that processed_dir does not exist
        mock_exists.return_value = False

        # Mock the DSE class to avoid actual file operations
        mock_dse_instance = MagicMock(spec=DSE)
        mock_dse.return_value = mock_dse_instance

        satellite_stack = {
            "sentinel1": np.random.rand(4, 2, 800, 800),  # 16 tiles
            "viirs": np.random.rand(9, 1, 800, 800),
            "gt": np.random.rand(1, 1, 16, 16),
        }

        list_metadata_s1 = [
            Metadata(
                satellite_type="sentinel1",
                file_name=[
                    "S1A_IW_GRDH_20200723_VV.tif",
                    "S1A_IW_GRDH_20200723_VH.tif",
                ],
                tile_id="Tile1",
                bands=["VV", "VH"],
                time="20200723",
            ),
            Metadata(
                satellite_type="sentinel1",
                file_name=[
                    "S1A_IW_GRDH_20200804_VV.tif",
                    "S1A_IW_GRDH_20200804_VH.tif",
                ],
                tile_id="Tile1",
                bands=["VV", "VH"],
                time="20200804",
            ),
            Metadata(
                satellite_type="sentinel1",
                file_name=[
                    "S1A_IW_GRDH_20200816_VV.tif",
                    "S1A_IW_GRDH_20200816_VH.tif",
                ],
                tile_id="Tile1",
                bands=["VV", "VH"],
                time="20200816",
            ),
            Metadata(
                satellite_type="sentinel1",
                file_name=[
                    "S1A_IW_GRDH_20200828_VV.tif",
                    "S1A_IW_GRDH_20200828_VH.tif",
                ],
                tile_id="Tile1",
                bands=["VV", "VH"],
                time="20200828",
            ),
        ]

        list_metadata_viirs = [
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020221.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020221",
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020224.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020224",
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020225.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020225",
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020226.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020226",
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020227.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020227",
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020231.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020231",
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020235.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020235",
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020236.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020236",
            ),
            Metadata(
                satellite_type="viirs",
                file_name=["DNB_VNP46A1_A2020237.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="2020237",
            ),
        ]

        list_metadata_gt = [
            Metadata(
                satellite_type="gt",
                file_name=["groundTruth.tif"],
                tile_id="Tile1",
                bands=["0"],
                time="0",
            )
        ]

        metadata_stack = {
            "sentinel1": list_metadata_s1,
            "viirs": list_metadata_viirs,
            "gt": list_metadata_gt,
        }

        # Mock the __load_and_preprocess method to avoid actual data processing
        with patch.object(
            ESDDataModule,
            "_ESDDataModule__load_and_preprocess",
            return_value=(satellite_stack, metadata_stack),
        ) as mock_load:
            self.dm.prepare_data()

            # Verify that __load_and_preprocess was called
            mock_load.assert_called()
            if self._outcome.success:
                global CURRENT_POINTS_TEST
                CURRENT_POINTS_TEST += 5
            print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    @patch("src.esd_data.datamodule.random_split")
    def test_setup(self, mock_random_split):
        # Mock random_split to control the output of training and validation dataset split
        mock_train_dataset = MagicMock(spec=DSE)
        mock_val_dataset = MagicMock(spec=DSE)
        mock_random_split.return_value = [mock_train_dataset, mock_val_dataset]

        # Call setup method
        self.dm.setup("fit")

        # Verify that random_split was called with expected arguments
        mock_random_split.assert_called_once()

        # Check if train_dataset and val_dataset are correctly set
        self.assertEqual(
            self.dm.train_dataset,
            mock_train_dataset,
            "train_dataset not set up correctly.",
        )
        self.assertEqual(
            self.dm.val_dataset, mock_val_dataset, "val_dataset not set up correctly."
        )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    @patch("torch.utils.data.DataLoader", autospec=True)
    def test_train_dataloader(self, mock_dataloader):
        self.dm.setup("fit")
        dataloader = self.dm.train_dataloader()

        mock_dataloader.assert_called_once_with(
            self.dm.train_dataset, batch_size=self.dm.batch_size, collate_fn=collate_fn
        )
        self.assertEqual(
            dataloader,
            mock_dataloader.return_value,
            "train_dataloader not returning DataLoader.",
        )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    @patch("torch.utils.data.DataLoader", autospec=True)
    def test_val_dataloader(self, mock_dataloader):
        self.dm.setup("fit")
        dataloader = self.dm.val_dataloader()

        mock_dataloader.assert_called_once_with(
            self.dm.val_dataset, batch_size=self.dm.batch_size, collate_fn=collate_fn
        )
        self.assertEqual(
            dataloader,
            mock_dataloader.return_value,
            "val_dataloader not returning DataLoader.",
        )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 5
        print(f"CURRENT_POINTS_TEST: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")


if __name__ == "__main__":
    root = pyprojroot.here()
    processed_dir = Path(root / 'data'/'processed'/'4x4')
    raw_dir = Path(root / 'data'/ 'raw'/'Train')

    selected_bands = {
        "sentinel1": ["VV", "VH"],
        "viirs_maxproj": ["0"],
        "gt": ["0"],
    }
    dm = ESDDataModule(processed_dir,raw_dir,selected_bands,4)
    dm.prepare_data()