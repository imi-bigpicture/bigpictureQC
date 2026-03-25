"""
This file contains class normalizing the image patches that were extracted from whole-slide image to a target range.
"""

from . import rangenormalizerbase as dptrangenormalizerbase

import numpy as np

#----------------------------------------------------------------------------------------------------

class RgbToZeroOneRangeNormalizer(dptrangenormalizerbase.RangeNormalizerBase):
    """
    This class can normalize a batch of RGB image patches with [0, 255] np.uint8 value range that were extracted from whole-slide images to [0.0, 1.0] np.float32 target value range.
    """

    def __init__(self):
        """Initialize the object."""

        # Initialize the base class.
        #
        super().__init__()

    def process(self, patches):
        """
        Normalize the batch to the target range.

        Args:
            patches (dict): A {level: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with extracted patches, corresponding labels or label patches and weight maps.

        Returns:
            dict: Batch dictionary with range normalized image patches.
        """

        for spacing in patches:
            # Convert the batch to float representation to the target interval.
            #
            patches[spacing]['patches'] = patches[spacing]['patches'].astype(np.float32) / np.float32(255.0)

        return patches
