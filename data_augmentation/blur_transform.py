from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np

from myrtle_core import normalise, unnormalise
from myrtle_torch_backend import cifar10_mean, cifar10_std
from quantum_blur import blur_image


class QuantumBlur(namedtuple("QuantumBlur", ("h", "w", "alpha"))):
    """Quantum blur transformation for data augmentation.

    This transforms (3 x n x m) data i.e. image data using Wooton et
    al.'s Quantum Blur: arXiv:2112.01646.

    Note: due to the normalising and unnormalising, this method is
    currently restricted to the cifar10 dataset.

    Attributes
    ----------
    h: int
        Height of the patch to blur
    w: int
        Width of the patch to blur
    alpha: float
        RZ rotational parameter for quantum blur.
    """

    def __call__(self, x: np.ndarray, x0: int, y0: int) -> np.ndarray:
        """Apply quantum blur to the patch.

        Parameters
        ----------
        x : np.ndarray
            Image tensor in shape (3 x n x m).
        x0 , y0 : int
            Corner such that (x0, y0), (x0 + W, y0), (x0 + W, y0 + H),
            (x0, y0 + H) is a valid rectangle

        Returns
        -------
        np.ndarray
            Image with a patch blurred
        """
        x_unnormalised = unnormalise(
            x[..., y0 : y0 + self.h, x0 : x0 + self.w],
            mean=np.array(cifar10_mean, dtype=np.float32),
            std=np.array(cifar10_std, dtype=np.float32),
        )
        x[..., y0 : y0 + self.h, x0 : x0 + self.w] = normalise(
            blur_image(x_unnormalised, self.alpha).transpose((1, 2, 0)),
            mean=np.array(cifar10_mean, dtype=np.float32),
            std=np.array(cifar10_std, dtype=np.float32),
        ).transpose((2, 0, 1))
        return x

    def options(self, shape: Tuple[int]) -> List[Dict[str, int]]:
        """Get the set of choices for bounding corner positions.

        i.e. return pairs such that (x0, y0), (x0 + W, y0),
        (x0 + W, y0 + H), (x0, y0 + H) make a valid rectangle in the
        image coordinates.

        Parameters
        ----------
        shape : Tuple[int]
            Shape of the image tensor expected to be either (1 x n x m)
            or (3 x m x n)

        Returns
        -------
        List[Dict[str, int]]
            Pairs of possible values for x0, y0.
        """
        *_, H, W = shape
        return [
            {"x0": x0, "y0": y0}
            for x0 in range(W + 1 - self.w)
            for y0 in range(H + 1 - self.h)
        ]


class GaussianBlur(namedtuple("GaussianBlur", ("h", "w"))):
    """Gaussian blur transformation for data augmentation.

    This transforms (3 x n x m) data i.e. image data using a normal
    distribution with location = 0 and scale = 0.01.


    Attributes
    ----------
    h: int
        Height of the patch to blur
    w: int
        Width of the patch to blur
    """

    def __call__(self, x, x0, y0):
        """Apply Gaussian blur to the patch.

        Parameters
        ----------
        x : np.ndarray
            Image tensor in shape (3 x n x m).
        x0 , y0 : int
            Corner such that (x0, y0), (x0 + W, y0), (x0 + W, y0 + H),
            (x0, y0 + H) is a valid rectangle

        Returns
        -------
        np.ndarray
            Image with a patch blurred
        """
        x[..., y0 : y0 + self.h, x0 : x0 + self.w] = x[
            ..., y0 : y0 + self.h, x0 : x0 + self.w
        ] + np.random.normal(loc=0, scale=0.01, size=(3, self.h, self.w))
        return x

    def options(self, shape: Tuple[int]) -> List[Dict[str, int]]:
        """Get the set of choices for bounding corner positions.

        i.e. return pairs such that (x0, y0), (x0 + W, y0),
        (x0 + W, y0 + H), (x0, y0 + H) make a valid rectangle in the
        image coordinates.

        Parameters
        ----------
        shape : Tuple[int]
            Shape of the image tensor expected to be either (1 x n x m)
            or (3 x m x n)

        Returns
        -------
        List[Dict[str, int]]
            Pairs of possible values for x0, y0.
        """
        *_, H, W = shape
        return [
            {"x0": x0, "y0": y0}
            for x0 in range(W + 1 - self.w)
            for y0 in range(H + 1 - self.h)
        ]
