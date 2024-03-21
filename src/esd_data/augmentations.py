""" Augmentations Implemented as Callable Classes."""

import random
from typing import Dict

import cv2
import numpy as np
import torch


def apply_per_band(img, transform):
    """
    Helpful function to allow you to more easily implement
    transformations that are applied to each band separately.
    Not necessary to use, but can be helpful.
    """
    result = np.zeros_like(img)
    for band in range(img.shape[0]):
        transformed_band = transform(img[band].copy())
        result[band] = transformed_band

    return result


class Blur(object):
    """
    Blurs each band separately using cv2.blur

    Parameters:
        kernel: Size of the blurring kernel # kernal is the little matrix that slides over image (slidng window type stuff)
        in both x and y dimensions, used
        as the input of cv.blur

    This operation is only done to the X input array.
    """

    def __init__(self, kernel=3):
        self.kernel = kernel

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Performs the blur transformation.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        # Sample must have X and y in a dictionary format
        # dimensions of img: (t, bands, tile_height, tile_width)
        if "X" not in sample:
            raise ValueError("The sample must have a key 'X'")

        transformed_X = apply_per_band(
            sample["X"], lambda x: cv2.blur(x, (self.kernel, self.kernel))
        )
        transformed_sample = {"X": transformed_X}

        if "y" in sample:
            transformed_sample["y"] = sample["y"]
        return transformed_sample


class AddNoise(object):
    """
    Adds random gaussian noise using np.random.normal.

    Parameters:
        mean: float
            Mean of the gaussian noise
        std_lim: float
            Maximum value of the standard deviation
    """

    def __init__(self, mean=0, std_lim=0.0):
        self.mean = mean
        self.std_lim = std_lim

    def __call__(self, sample):
        """
        Performs the add noise transformation.
        A random standard deviation is first calculated using
        random.uniform to be between 0 and self.std_lim

        Random noise is then added to each pixel with
        mean self.mean and the standard deviation
        that was just calculated

        The resulting value is then clipped using
        numpy's clip function to be values between
        0 and 1.

        This operation is only done to the X array.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        if "X" not in sample:
            raise ValueError("The sample must have a key 'X'")

        random_std = random.uniform(0, self.std_lim)
        transformed_X = apply_per_band(
            sample["X"],
            lambda x: np.clip(
                x + np.random.normal(self.mean, random_std, x.shape), 0, 1
            ),
        )
        transformed_sample = {"X": transformed_X}

        if "y" in sample:
            transformed_sample["y"] = sample["y"]

        return transformed_sample


class RandomVFlip(object):
    """
    Randomly flips all bands vertically in an image with probability p.

    Parameters:
        p: probability of flipping image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Performs the random flip transformation using cv.flip.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        if "X" not in sample:
            raise ValueError("The sample must have a key 'X'")

        if random.random() < self.p:
            transformed_X = apply_per_band(sample["X"], lambda x: cv2.flip(x, 0))
        else:
            transformed_X = sample["X"].copy()

        transformed_sample = {"X": transformed_X}

        if "y" in sample:
            transformed_sample["y"] = apply_per_band(
                sample["y"], lambda x: cv2.flip(x, 0)
            )
        return transformed_sample


class RandomHFlip(object):
    """
    Randomly flips all bands horizontally in an image with probability p.

    Parameters:
        p: probability of flipping image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Performs the random flip transformation using cv.flip.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        if "X" not in sample:
            raise ValueError("The sample must have a key 'X'")

        if random.random() < self.p:
            transformed_X = apply_per_band(sample["X"], lambda x: cv2.flip(x, 1))
        else:
            transformed_X = sample["X"].copy()

        transformed_sample = {"X": transformed_X}

        if "y" in sample:
            transformed_sample["y"] = apply_per_band(
                sample["y"], lambda x: cv2.flip(x, 1)
            )

        return transformed_sample


class ToTensor(object):
    """
    Converts numpy.array to torch.tensor
    """

    def __call__(self, sample):
        """
        Transforms all numpy arrays to tensors

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, torch.Tensor]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        transformed = {}
        for k, v in sample.items():
            transformed[k] = torch.from_numpy(v)

        return transformed
