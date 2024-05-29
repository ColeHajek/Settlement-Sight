import sys
import unittest
import numpy as np
import pyprojroot
root = pyprojroot.here()
sys.path.append(str(root))

CURRENT_POINTS_TEST = 0
MAX_POINTS_TEST = 0

from src.preprocessing.preprocess_sat import (
    per_band_gaussian_filter,
    quantile_clip,
    minmax_scale,
    brighten,
    gammacorr,
    maxprojection_viirs,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
    preprocess_viirs
)

CURRENT_POINTS_TEST = 0
MAX_POINTS_TEST = 15


class TestPreprocessingUtilities(unittest.TestCase):
    """
    Class to test the preprocessing utilities.
    """
    def setUp(self):
        self.test_sat_img = np.random.rand(2, 3, 800, 800)

    def tearDown(self):
        pass

    def test_per_band_gaussian_filter(self):
        """
        Test the per_band_gaussian_filter function.
        """
        result = per_band_gaussian_filter(self.test_sat_img)
        self.assertEqual(result.shape, self.test_sat_img.shape)
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_quantile_clip(self):
        """
        Test the quantile_clip function.
        """
        sample_img = self.test_sat_img.copy()
        sample_img[:, :, 0, 0] = -1e10
        sample_img[:, :, -1, -1] = 1e10
        result = quantile_clip(sample_img, clip_quantile=0.01)
        self.assertNotAlmostEqual(result.min(), -1e10)
        self.assertNotAlmostEqual(result.max(), 1e10)

        sample_img = self.test_sat_img.copy()
        sample_img[0, :, 0, 0] = -1e10
        sample_img[-1, :, -1, -1] = 1e10
        result = quantile_clip(
            sample_img,
            clip_quantile=0.01,
            group_by_time=False
            )
        self.assertNotAlmostEqual(result.min(), -1e10)
        self.assertNotAlmostEqual(result.max(), 1e10)
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 2
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_minmax_scale(self):
        """
        Test the minmax_scale function.
        """
        sample_img = self.test_sat_img.copy()
        sample_img[:, :, 0, 0] = -1e10
        sample_img[:, :, -1, -1] = 1e10
        result = minmax_scale(sample_img)
        self.assertAlmostEqual(result.min(), 0)
        self.assertAlmostEqual(result.max(), 1)

        sample_img = self.test_sat_img.copy()
        sample_img[0, :, 0, 0] = -1e10
        sample_img[-1, :, -1, -1] = 1e10
        result = minmax_scale(sample_img, group_by_time=False)
        self.assertAlmostEqual(result.min(), 0)
        self.assertAlmostEqual(result.max(), 1)
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 2
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_brighten(self):
        """
        Test the brighten function.
        """
        sample_img = np.ones((1, 1, 2, 2))
        sample_img[:, :, 0, 0] = 0
        result = brighten(sample_img, alpha=2, beta=0.5)
        self.assertAlmostEqual(result.max(), 1)
        self.assertAlmostEqual(result.min(), 0.5)
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_gamma_corr(self):
        """
        Test the gammacorr function.
        """
        sample_img = 2*np.ones((1, 1, 1, 1))
        result = gammacorr(sample_img, gamma=2)
        self.assertAlmostEqual(result[0, 0, 0, 0], np.sqrt(2))
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_maxprojection_viirs(self):
        """
        Test the maxprojection_VIIRS function.
        """
        sample_img = np.ones((4, 1, 2, 2))
        sample_img[3, 0, 1, 1] = 100
        sample_img[0, 0, 0, 0] = 0
        result = maxprojection_viirs(sample_img)
        self.assertAlmostEqual(result[0, 1, 1], 1)
        self.assertAlmostEqual(result[0, 0, 0], 0)
        self.assertEqual(result.shape, sample_img.shape[1:])
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 4
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_preprocess_sentinel1(self):
        """
        Test the preprocess_sentinel1 function.
        """
        result = preprocess_sentinel1(self.test_sat_img)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 1)
        self.assertEqual(result.shape, self.test_sat_img.shape)
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_preprocess_sentinel2(self):
        """
        Test the preprocess_sentinel2 function.
        """
        result = preprocess_sentinel2(self.test_sat_img)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 1)
        self.assertEqual(result.shape, self.test_sat_img.shape)
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_preprocess_landsat(self):
        """
        Test the preprocess_landsat function.
        """
        result = preprocess_landsat(self.test_sat_img)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 1)
        self.assertEqual(result.shape, self.test_sat_img.shape)
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_preprocess_viirs(self):
        """
        Test the preprocess_viirs function.
        """
        result = preprocess_viirs(self.test_sat_img)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 1)
        self.assertEqual(result.shape, self.test_sat_img.shape)
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")


if __name__ == '__main__':
    unittest.main()