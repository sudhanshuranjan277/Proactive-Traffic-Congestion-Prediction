"""
transforms.py

Custom dataset transforms.
"""

from typing import Callable


class IdentityTransform:
    """
    Returns input unchanged.
    """

    def __call__(self, sample):
        return sample


class Compose:
    """
    Compose multiple transforms.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample