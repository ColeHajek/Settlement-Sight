import cv2
import numpy as np
import torch
import random
from typing import Dict


def apply_per_band(img: np.ndarray, transform) -> np.ndarray:
    """
    Apply a given transformation to each band of the input image separately.
    
    Parameters:
        img (np.ndarray): The input image of shape (bands, width, height).
        transform (function): The transformation function to apply to each band.
    
    Returns:
        np.ndarray: The transformed image.
    """
    result = np.zeros_like(img)
    for band in range(img.shape[0]):
        transformed_band = transform(img[band].copy())
        result[band] = transformed_band
    return result


class Blur:
    """
    Applies a blur filter to each band separately using OpenCV's cv2.blur.

    Parameters:
        kernel (int): Maximum size of the blurring kernel in both x and y dimensions.
                      A random size between 1 and kernel will be used for each call.
    """

    def __init__(self, kernel: int = 3):
        self.kernel = kernel

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply the blur transformation to the 'X' component of the sample.
        
        Parameters:
            sample (Dict[str, np.ndarray]): Dictionary containing the 'X' (input image)
                                            and 'y' (label/mask).
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with the blurred 'X' and unchanged 'y'.
        """
        img, mask = sample["X"], sample["y"]
        kernel_size = random.randint(1, self.kernel)

        # Apply blur to each band of the image
        img = apply_per_band(img, lambda x: cv2.blur(x, (kernel_size, kernel_size)))

        return {"X": img, "y": mask}


class AddNoise:
    """
    Adds random Gaussian noise to the input image.

    Parameters:
        mean (float): Mean of the Gaussian noise distribution.
        std_lim (float): Maximum value for the standard deviation of the noise.
    """

    def __init__(self, mean: float = 0, std_lim: float = 0.0):
        self.mean = mean
        self.std_lim = std_lim

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Add Gaussian noise to the 'X' component of the sample.

        Parameters:
            sample (Dict[str, np.ndarray]): Dictionary containing 'X' (input image) and 'y' (label/mask).

        Returns:
            Dict[str, np.ndarray]: Dictionary with noisy 'X' and unchanged 'y'.
        """
        img, mask = sample["X"], sample["y"]

        # Calculate random standard deviation for noise
        std_dev = random.uniform(0, self.std_lim)
        
        # Add Gaussian noise to the image
        noise = np.random.normal(self.mean, std_dev, size=img.shape)
        img = np.clip(img + noise, 0, 1)  # Ensure pixel values stay between 0 and 1

        return {"X": img.astype(np.float32), "y": mask}


class RandomVFlip:
    """
    Randomly flip all bands vertically with a given probability.

    Parameters:
        p (float): Probability of flipping the image vertically (default is 0.5).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Vertically flip the 'X' and 'y' components of the sample with probability p.

        Parameters:
            sample (Dict[str, np.ndarray]): Dictionary containing 'X' (input image) and 'y' (label/mask).

        Returns:
            Dict[str, np.ndarray]: Dictionary with flipped 'X' and 'y' (if flip occurs).
        """
        img, mask = sample["X"], sample["y"]

        if random.random() <= self.p:
            # Apply vertical flip to each band of the image
            img = apply_per_band(img, lambda x: cv2.flip(x, 0))
            # Apply vertical flip to the mask
            mask = cv2.flip(mask, 0)

        return {"X": img, "y": mask}


class RandomHFlip:
    """
    Randomly flip all bands horizontally with a given probability.

    Parameters:
        p (float): Probability of flipping the image horizontally (default is 0.5).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Horizontally flip the 'X' and 'y' components of the sample with probability p.

        Parameters:
            sample (Dict[str, np.ndarray]): Dictionary containing 'X' (input image) and 'y' (label/mask).

        Returns:
            Dict[str, np.ndarray]: Dictionary with flipped 'X' and 'y' (if flip occurs).
        """
        img, mask = sample["X"], sample["y"]

        if random.random() <= self.p:
            # Apply horizontal flip to each band of the image
            img = apply_per_band(img, lambda x: cv2.flip(x, 1))
            # Apply horizontal flip to the mask
            mask = cv2.flip(mask, 1)

        return {"X": img, "y": mask}


class ToTensor:
    """
    Converts numpy arrays to torch tensors.
    """

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert 'X' and 'y' from numpy arrays to torch tensors.

        Parameters:
            sample (Dict[str, np.ndarray]): Dictionary containing 'X' (input image) and 'y' (label/mask).

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'X' and 'y' as torch tensors.
        """
        img, mask = sample["X"], sample["y"]
        return {"X": torch.from_numpy(img), "y": torch.from_numpy(mask)}
