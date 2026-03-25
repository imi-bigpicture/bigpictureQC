"""
This file contains class normalizing the image patches that were extracted from whole-slide image to a target range.
"""

from . import rangenormalizerbase as dptrangenormalizerbase

from ...errors import rangeerrors as dptrangeerrors

import numpy as np

#----------------------------------------------------------------------------------------------------

class RgbRangeNormalizer(dptrangenormalizerbase.RangeNormalizerBase):
    """
    This class can normalize a batch of RGB image patches with [0, 255] np.uint8 value range that were extracted from whole-slide images to a target np.float32 value range.
    """

    def __init__(self, target_range):
        """
        Initialize the object.

        Args:
            target_range (tuple): Target range. Tuple of two.

        Raises:
            InvalidNormalizationRangeError: Target range is invalid.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__target_range = None      # Target range of normalization.
        self.__target_range_size = 0.0  # Calculated size of the target range.

        # Process the configured parameters.
        #
        self.__setrange(target_range)

    def __setrange(self, target_range):
        """
        Configure the target range.

        Args:
            target_range (tuple): Target range. Tuple of two.

        Raises:
            InvalidNormalizationRangeError: The target range is invalid.
        """

        # Check target range.
        #
        if len(target_range) < 2 or target_range[1] <= target_range[0]:
            dptrangeerrors.InvalidNormalizationRangeError('target', target_range)

        # Save the range.
        #
        self.__target_range = (float(target_range[0]), float(target_range[1]))
        self.__target_range_size = float(target_range[1] - target_range[0])

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
            patches[spacing]['patches'] = patches[spacing]['patches'].astype(np.float32) * np.float32(self.__target_range_size / 255.0) + self.__target_range[0]

        return patches
