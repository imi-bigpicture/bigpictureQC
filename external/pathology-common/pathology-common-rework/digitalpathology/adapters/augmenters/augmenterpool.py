"""
This file contains a class for handling and applying multiple augmentations on patches from whole slide images.
"""

from . import augmenterbase as dptaugmenterbase
from .spatial import spatialaugmenterbase as dptspatialaugmenterbase

from ...errors import augmentationerrors as dptaugmentationerrors

import numpy as np
import random

#----------------------------------------------------------------------------------------------------

class AugmenterPool(object):
    """Class for augmenting patches from whole slide images."""

    def __init__(self):
        """Initialize object."""

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__groups = []          # List of augmenter groups in order of application.
        self.__randomized = {}      # Flag for each group if its order is randomized.
        self.__augmenters = {}      # List of augmenter object by groups.
        self.__probabilities = {}   # Probability of execution for each augmenter.
        self.__initialized = False  # Flag indicating that the augmenters were initialized.

    def __cropbatch(self, patches, labels, shape):
        """
        Crop the batch of patches to the target shape. For efficiency considerations this is a separate function from the crop() public function.

        Args:
            patches (np.ndarray): Image patches to crop.
            labels (np.ndarray): Label images to crop.
            shape (tuple): Target shape.

        Returns:
            (np.ndarray): Cropped batch.

        Raises:
            BatchCroppingError: The target shape for cropping is smaller than the batch to crop itself.
        """

        # Check quickly if cropping is needed at all.
        #
        if patches.shape[1:3] == shape:
            # Return the original batch.
            #
            return patches, labels

        else:
            # Check if the shape is larger or equal to the target.
            #
            if patches.shape[1] < shape[0] or patches.shape[2] < shape[1]:
                dptaugmentationerrors.BatchCroppingError(patches.shape, shape)

            # Calculate the shift.
            #
            shift = ((patches.shape[1] - shape[0]) // 2, (patches.shape[2] - shape[1]) // 2)

            # Return the cropped patch.
            #
            return patches[:, shift[0]: shift[0] + shape[0], shift[1]: shift[1] + shape[1]], labels[:, shift[0]: shift[0] + shape[0], shift[1]: shift[1] + shape[1]] if 2 < labels.ndim else labels

    def appendgroup(self, group, randomized):
        """
        Append augmentation group. If the created group is randomized the augmenter pool will execute the augmenters in pool for in random order, otherwise the augmenters in the group are
        executed in order.

        Args:
            group (str): Group identifier.
            randomized (bool): Flag indicating group type.

        Raises:
            AugmentationGroupAlreadyExistsError: Augmentation group already exists.
        """

        # Check if group exists already.
        #
        if group in self.__groups:
            raise dptaugmentationerrors.AugmentationGroupAlreadyExistsError(group)

        # Set up the per-group data.
        #
        self.__groups.append(group)
        self.__randomized[group] = randomized
        self.__augmenters[group] = []
        self.__probabilities[group] = []

    def appendaugmenter(self, augmenter, group, probability=1.0):
        """
        Append an augmenter object to the group.

        Args:
            augmenter (dptaugmenterbase.AugmenterBase): Augmenter object.
            group (str): Group identifier.
            probability (float): Probability of selection.

        Raises:
            UnknownAugmentationGroupError: Unknown augmentation group.
            InvalidAugmentationProbabilityError: The probability of the augmentation execution is invalid.
        """

        # Check if the group is known.
        #
        if group not in self.__groups:
            raise dptaugmentationerrors.UnknownAugmentationGroupError(group)

        # Check the probability of execution.
        #
        if probability < 0.0 or 1.0 < probability:
            raise dptaugmentationerrors.InvalidAugmentationProbabilityError(probability)

        # Append the augmenter object and its execution probability to the per-group list.
        #
        self.__augmenters[group].append(augmenter)
        self.__probabilities[group].append(probability)
        self.__initialized = False

    def shapes(self, target_shapes):
        """
        Calculate the required shape of the input to achieve the target output shape.

        Args:
            target_shapes (dict): Target output shape per level.

        Returns:
            (dict): Required input shape per level.
        """

        # Copy the target sizes.
        #
        required_shapes = target_shapes.copy()

        # Get the maximal required sizes.
        #
        for group_id in self.__augmenters:
            for augmenter_item in self.__augmenters[group_id]:
                required_shapes_for_item = augmenter_item.shapes(target_shapes)

                for level in required_shapes:
                    required_shapes[level] = tuple(max(required_shapes[level][index], required_shapes_for_item[level][index]) for index in range(len(required_shapes[level])))

        return required_shapes

    def transform(self, patch, label=None):
        """
        Randomly select one from each group and apply transformations on the patch and the label map in order.

        Args:
            patch (np.ndarray): Patch to transform.
            label (np.ndarray, None): Patch labels to transform.

        Returns:
            np.ndarray, (np.ndarray, None): Transformed patch, transformed labels.

        Raises:
            MissingAugmentationRandomizationError: Augmentations are configured but not randomized.
            InvalidElasticImageShapeError: Not a 2D grayscale or 3 channel image is transformed.
        """

        # Check if randomization is done.
        #
        if not self.__initialized:
            raise dptaugmentationerrors.MissingAugmentationRandomizationError()

        # Prepare the result.
        #
        transformed_patch = patch
        transformed_label = label

        # Go through the groups in order.
        #
        for group_id in self.__groups:
            # Randomize the order of the augmenters if necessary.
            #
            augmenter_count = len(self.__augmenters[group_id])
            augmenter_indices = random.sample(population=range(augmenter_count), k=augmenter_count) if self.__randomized[group_id] else range(augmenter_count)

            # Apply all augmenters.
            #
            for augmenter_index in augmenter_indices:
                if 1.0 <= self.__probabilities[group_id][augmenter_index] or random.uniform(a=0.0, b=1.0) < self.__probabilities[group_id][augmenter_index]:
                    augmenter_item = self.__augmenters[group_id][augmenter_index]

                    if group_id == 'blackout':
                        transformed_patch, transformed_label = augmenter_item.transform(patch=transformed_patch, label=transformed_label) # temporary add

                    else:
                        transformed_patch = augmenter_item.transform(patch=transformed_patch)

                        if transformed_label is not None and 2 <= transformed_label.ndim <= 3 and isinstance(augmenter_item, dptspatialaugmenterbase.SpatialAugmenterBase):
                            transformed_label = augmenter_item.transform(patch=transformed_label)

        # Return the result patch, label pair.
        #
        return transformed_patch, transformed_label

    def randomize(self):
        """Randomize the parameters of the augmenters."""

        # Randomize the parameters of all augmenters in all groups.
        #
        for group_id in self.__groups:
            for augmenter_item in self.__augmenters[group_id]:
                augmenter_item.randomize()

        self.__initialized = True

    def crop(self, patch, shape):
        """
        Crop the patch to the target shape.

        Args:
            patch (np.ndarray): Patch to crop.
            shape (tuple): Target shape.

        Returns:
            (np.ndarray): Cropped patch.

        Raises:
            PatchCroppingError: The target shape for cropping is smaller than the patch to crop itself.
        """

        # Check quickly if cropping is needed at all.
        #
        if patch.shape[1:] == shape:
            # Return the original patch.
            #
            return patch

        else:
            # Check if the patch shape is larger or equal to the target.
            #
            if patch.shape[1] < shape[0] or patch.shape[2] < shape[1]:
                dptaugmentationerrors.PatchCroppingError(patch.shape, shape)

            # Calculate the shift.
            #
            shift = ((patch.shape[1] - shape[0]) // 2, (patch.shape[2] - shape[1]) // 2)

            # Return the cropped patch.
            #
            return patch[:, shift[0]: shift[0] + shape[0], shift[1]: shift[1] + shape[1]]

    def process(self, patches, shapes=None, randomize=True):
        """
        Process a batch of multi-level patches.

        Args:
            patches (dict): RGB patches and labels per level as given by the PatchSampler.
            randomize (flag to control if parameters should be randomized before each patch augmentation.
            shapes (dict, None): Target patch shapes (rows, columns) per level.

        Returns:
            dict: Augmented patch collection.

        Raises:
            MissingTargetShapeForLevelError: Target shape for cropping is missing for a level.
            MissingAugmentationRandomizationError: Augmentations are configured but not randomized.
            BatchCroppingError: The target shape for cropping is smaller than the batch to crop itself.
        """

        # Check if the target shapes are valid.
        #
        if shapes:
            if any(level not in shapes for level in patches):
                dptaugmentationerrors.MissingTargetShapeForLevelError(list(patches.keys()), list(shapes.keys()))

        # Get patch collection length.
        #
        patch_count = next(iter(patches.values()))['patches'].shape[0]

        # Go through all the patches: randomize the augmenters and apply the same augmentation across all the levels for the same patch.
        #
        for index in range(patch_count):
            if randomize:
                self.randomize()

            for level in patches:
                patches[level]['patches'][index], patches[level]['labels'][index] = self.transform(patch=patches[level]['patches'][index], label=patches[level]['labels'][index])

        # Crop the central part of the patches to remove augmentation artifacts.
        #
        if shapes:
            for level in patches:
                patches[level]['patches'], patches[level]['labels'] = self.__cropbatch(patches=patches[level]['patches'], labels=patches[level]['labels'], shape=shapes[level])

        # Return the transformed patch set.
        #
        return patches
