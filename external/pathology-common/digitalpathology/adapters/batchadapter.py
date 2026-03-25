"""
This file contains a class that can adapt the extracted raw patches for network training.
"""

from ..errors import configerrors as dptconfigerrors
from ..errors import weighterrors as dptweighterrors

import numpy as np

#----------------------------------------------------------------------------------------------------

class BatchAdapter(object):
    """
    This class can adapt a set of raw patches and labels extracted from whole-slide images and corresponding mask files to the right format for training a network.

    The range normalization and the weight mapping can be done either right after extraction, or just before releasing the mini-batch. Doing it early (right after
    the extraction) can be computationally more efficient, but in that case the float representation of the data is cached that makes it 4x larger in memory.

    Doing the normalization early has the additional benefit, that most of the image processing libraries (e.g. the ones used in the augmentation pool) can handle
    input from the [0.0, 1.0] float32 range, therefore a lot of back and fort conversion between float32 and uint8 can be saved. In this case it is recommended to
    enable range squashing, that converts the [0, 255] uint8 pixel value range to [0.0, 1.0] float32 range immediately, even before passing the patches to the
    augmenters, and adding a range normalizer that can convert the resulting (also augmented) [0.0, 1.0] patch to the target range.
    """

    def __init__(self,
                 squash_range,
                 augmenter_pool,
                 range_normalizer,
                 label_mapper,
                 labels_one_hot,
                 weight_mapper,
                 late_range_normalizer,
                 late_weight_mapper,
                 batch_weight_mapper,
                 late_labels_one_hot,
                 label_count=None):
        """
        Initialize the object.

        Args:
            squash_range (bool): Squash the [0, 255] np.uint8 input range to [0.0, 1.0] np.float32 as a very first step.
            augmenter_pool (augmenters.augmenterpool.AugmenterPool, None): Image augmenter pool.
            range_normalizer (range.rangenormalizerbase.RangeNormalizerBase, None): Pixel value range normalizer.
            label_mapper (label.labelmapper.LabelMapper, None): Label mapper.
            labels_one_hot (bool): Convert labels to one-hot representation in adapt function.
            weight_mapper (weight.weightmapperbase.WeightMapperBase, None): Label weight mapper.
            late_range_normalizer (range.rangenormalizerbase.RangeNormalizerBase, None): Pixel value range normalizer for late execution.
            late_weight_mapper (weight.weightmapperbase.WeightMapperBase, None): Label weight mapper for late execution.
            batch_weight_mapper (weight.weightmapperbase.WeightMapperBase, None): Label weight mapper for a mini-batch.
            late_labels_one_hot (bool): Convert labels to one-hot representation in adjust function.
            label_count (int, None): Number of labels. Only necessary if one-hot conversion is configured without label mapper.

        Raises:
            InvalidLabelCountError: Invalid label count for one-hot encoding.
            MissingLabelMapperForWeightMapperError: Weight mapping is configured without label mapping.
            WeightMapperLabelMapperClassesMismatchError: The network labels known by the label mapper and weight mapper does not match.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__squash_range = False            # Squash the [0, 255] np.uint8 input range to [0.0, 1.0] np.float32.
        self.__augmenter_pool = None           # Image augmenter pool.
        self.__range_normalizer = None         # Pixel value range normalizer.
        self.__label_mapper = None             # Label value mapper.
        self.__labels_one_hot = False          # Convert labels to one hot representation in adapt function.
        self.__weight_mapping_enabled = False  # There is a weight mapping step enabled.
        self.__weight_mapper = None            # Label weight mapper.
        self.__late_range_normalizer = None    # Pixel value range normalizer for late execution.
        self.__late_weight_mapper = None       # Label weight mapper for late execution.
        self.__batch_weight_mapper = None      # Label weight mapper for a single mini-batch.
        self.__late_labels_one_hot = False     # Convert labels to one-hot representation in adjust function.
        self.__label_count = 0                 # Number of labels for one-hot conversion.

        # Save the configured adapters.
        #
        self.__setaugmenterpool(augmenter_pool=augmenter_pool)
        self.__setrangenormalizers(squash_range=squash_range, range_normalizer=range_normalizer, late_range_normalizer=late_range_normalizer)
        self.__setlabelmapper(label_mapper=label_mapper)
        self.__setonehot(labels_one_hot=labels_one_hot, late_labels_one_hot=late_labels_one_hot, label_count=label_count)
        self.__setweightmappers(weight_mapper=weight_mapper, late_weight_mapper=late_weight_mapper, batch_weight_mapper=batch_weight_mapper)

    def __setaugmenterpool(self, augmenter_pool):
        """
        Set the image augmenter pool.

        Args:
            augmenter_pool (augmenters.augmenterpool.AugmenterPool, None): Image augmenter pool.
        """

        # Save the augmenter pool object.
        #
        self.__augmenter_pool = augmenter_pool

    def __setrangenormalizers(self, squash_range, range_normalizer, late_range_normalizer):
        """
        Set the pixel value range normalizer objects.

        Args:
            squash_range (bool): Squash the [0, 255] np.uint8 input range to [0.0, 1.0] np.float32 as a very first step.
            range_normalizer (range.rangenormalizerbase.RangeNormalizerBase, None): Pixel value range normalizer.
            late_range_normalizer (range.rangenormalizerbase.RangeNormalizerBase, None): Pixel value range normalizer for late execution.
        """

        # Save the range normalizers.
        #
        self.__squash_range = squash_range
        self.__range_normalizer = range_normalizer
        self.__late_range_normalizer = late_range_normalizer

    def __setlabelmapper(self, label_mapper):
        """
        Set the label mapper object.

        Args:
            label_mapper (label.labelmapper.LabelMapper, None): Label mapper.
        """

        # Save the label mapper object.
        #
        self.__label_mapper = label_mapper

    def __setonehot(self, labels_one_hot, late_labels_one_hot, label_count):
        """
        Configure the one-hot label conversion.

        Args:
            labels_one_hot (bool): Convert labels to one hot representation in adapt function.
            late_labels_one_hot (bool): Convert labels to one hot representation in adjust function.
            label_count (int, None): Number of labels. Only necessary if one-hot conversion is configured without label mapper.

        Raises:
            InvalidLabelCountError: Invalid label count for one-hot encoding.
        """

        # Check if the label count is configured in case the label mapper is not.
        #
        if (labels_one_hot or late_labels_one_hot) and self.__label_mapper is None and (label_count is None or label_count < 1):
            raise dptconfigerrors.InvalidLabelCountError(label_count)

        # Save the one-hot configuration. Make sure that the labels are only converted to one hot once.
        #
        self.__labels_one_hot = labels_one_hot
        self.__late_labels_one_hot = not labels_one_hot and late_labels_one_hot
        self.__label_count = self.__label_mapper.classes if self.__label_mapper is not None else label_count

    def __setweightmappers(self, weight_mapper, late_weight_mapper, batch_weight_mapper):
        """
        Set the weight mappers.

        Args:
            weight_mapper (weight.weightmapperbase.WeightMapperBase, None): Label weight mapper.
            late_weight_mapper (weight.weightmapperbase.WeightMapperBase, None): Label weight mapper for late execution.
            batch_weight_mapper (weight.weightmapperbase.WeightMapperBase, None): Label weight mapper for a mini-batch.

        Raises:
            MissingLabelMapperForWeightMapperError: Weight mapping is configured without label mapping.
            WeightMapperLabelMapperClassesMismatchError: The network labels known by the label mapper and weight mapper does not match.
        """

        if weight_mapper is not None or batch_weight_mapper is not None:
            # Check if label mapper is configured. It is necessary for the valid pixel map generation.
            #
            if self.__label_mapper is None:
                raise dptweighterrors.MissingLabelMapperForWeightMapperError()

        if weight_mapper is not None:
            # Check if the label mapper and the patch weight mapper has the same network labels.
            #
            if weight_mapper.classes != self.__label_mapper.classes:
                raise dptweighterrors.WeightMapperLabelMapperClassesMismatchError(weight_mapper.classes, self.__label_mapper.classes)

        if late_weight_mapper is not None:
            # Check if the label mapper and the patch weight mapper has the same network labels.
            #
            if late_weight_mapper.classes != self.__label_mapper.classes:
                raise dptweighterrors.WeightMapperLabelMapperClassesMismatchError(late_weight_mapper.classes, self.__label_mapper.classes)

        if batch_weight_mapper is not None:
            # Check if the label mapper and the batch weight mapper has the same network labels.
            #
            if batch_weight_mapper.classes != self.__label_mapper.classes:
                raise dptweighterrors.WeightMapperLabelMapperClassesMismatchError(batch_weight_mapper.classes, self.__label_mapper.classes)

            # Check if the batch weight mapper is accompanied by a patch weight mapper.
            #
            if weight_mapper is None and late_weight_mapper is None:
                raise dptweighterrors.MissingPatchMapperWithBatchMapper()

        # Store the weight mappers.
        #
        self.__weight_mapping_enabled = weight_mapper is not None or late_weight_mapper is not None or batch_weight_mapper is not None
        self.__weight_mapper = weight_mapper
        self.__late_weight_mapper = late_weight_mapper
        self.__batch_weight_mapper = batch_weight_mapper

    @staticmethod
    def __squashrange(patches):
        """

        Args:
            patches (dict): RGB patches and labels per level as given by the PatchSampler.

        Returns:
            dict: Patch collection with pixel values squashed to the [0.0, 1.0] float range.
        """

        # Process all levels of pixel spacing.
        #
        for spacing in patches:
            patches[spacing]['patches'] = patches[spacing]['patches'].astype(dtype=np.float32) / np.float32(255.0)

        return patches

    @staticmethod
    def __formatweightmaps(patches):
        """
        Format the weight maps to have the right amount of dimensions.

        Args:
            patches (dict): A {spacing: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with extracted patches, corresponding labels or label patches and weight maps.

        Returns:
             dict: Batch dictionary with formatted weight maps.
        """

        # Process all levels of pixel spacing.
        #
        for spacing in patches:
            patches[spacing]['weights'] = patches[spacing]['weights'][..., None]

        return patches

    def __labelstoonehot(self, patches):
        """
        Converts a class vector (integers) to binary class matrix (one hot representation). E.g. for use with categorical_crossentropy.

        Args:
            patches (dict): A {spacing: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with extracted patches, corresponding labels or label patches and weight maps.

        Returns:
             dict: Batch dictionary with one-hot labels.
        """

        # Process all levels of pixel spacing.
        #
        for spacing in patches:
            # Get the labels from this level of pixel spacing.
            #
            labels = patches[spacing]['labels']
            labels_shape = labels.shape

            # Ravel the label array to 1D.
            #
            labels = labels.ravel()

            # Convert to one-hot.
            #
            categorical = np.zeros(shape=(labels.shape[0], self.__label_count), dtype=np.float32)
            categorical[np.arange(labels.shape[0]), labels.astype(np.uint8)] = 1

            # Restore shape.
            #
            target_shape = (labels_shape[0], self.__label_count) if len(labels_shape) < 2 else labels_shape + (self.__label_count,)
            patches[spacing]['labels'] = np.reshape(a=categorical, newshape=target_shape)

        return patches

    @property
    def normalized(self):
        """
        Check if the pixel value range is normalized in the adapt function.

        Returns:
            bool: True if the pixel value range is normalized in the adapt function.
        """

        return self.__squash_range or self.__range_normalizer is not None

    @property
    def mapping(self):
        """
        Get the label mapping configuration.

        Returns:
            dict: Label mapping dictionary if label mapping is configured, else empty dictionary.
        """

        return self.__label_mapper.mapping if self.__label_mapper is not None else dict()

    @property
    def onehot(self):
        """
        Get the number of labels used for one-hot representation if enabled in adapt function.

        Returns:
            int: Number of labels used for obe-hot representation conversion in the adapt function, 0 if conversion not enabled.
        """

        return self.__label_count if self.__labels_one_hot else 0

    @property
    def weights(self):
        """
        Check if weight maps are generated.

        Returns:
            bool: True if weight mapping is enabled, false otherwise.
        """

        return self.__weight_mapper is not None

    def shapes(self, target_shapes):
        """
        Calculate the required shape of the input to achieve the target output shape.

        Args:
            target_shapes (dict): Target output shape per level.

        Returns:
            (dict): Required input shape per level.
        """

        return self.__augmenter_pool.shapes(target_shapes=target_shapes) if self.__augmenter_pool is not None else target_shapes

    def adapt(self, patches, shapes, randomize):
        """
        Adapt a collection multi-level patches.

        Args:
            patches (dict): RGB patches and labels per level as given by the PatchSampler.
            shapes (dict, None): Target patch shapes (rows, columns) per level.
            randomize (bool): Flag to control if parameters should be randomized before each patch augmentation.

        Returns:
            dict: Adapted patch collection.

        Raises:
            MissingTargetShapeForLevelError: Target shape for cropping is missing for a level.
            MissingAugmentationRandomizationError: Augmentations are configured but not randomized.
            InvalidElasticImageShapeError: Not a 2D grayscale or 3 channel image is transformed.
            BatchCroppingError: The target shape for cropping is smaller than the batch to crop itself.
        """

        # Squash the input range to [0.0, 1.0].
        #
        if self.__squash_range:
            patches = self.__squashrange(patches=patches)

        # Augment the images.
        #
        if self.__augmenter_pool is not None:
            patches = self.__augmenter_pool.process(patches=patches, shapes=shapes, randomize=randomize)

        # Normalize the value range of the pixels.
        #
        if self.__range_normalizer is not None:
            patches = self.__range_normalizer.process(patches=patches)

        # Map the labels.
        #
        if self.__label_mapper is not None:
            patches = self.__label_mapper.process(patches=patches, valid_map=self.__weight_mapper is not None)

        # Calculate label weights.
        #
        if self.__weight_mapper is not None:
            patches = self.__weight_mapper.process(patches=patches)

        # Convert the labels to one-hot representation.
        #
        if self.__labels_one_hot:
            patches = self.__labelstoonehot(patches=patches)

        return patches

    def adjust(self, batch):
        """
        Adjust the label weights of a mini-batch of multi-level patches.

        Args:
            batch (dict): A mini-batch of {level: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with patches, corresponding labels or label patches and weight maps.

        Returns:
            dict: Adapted patch collection.
        """

        # Normalize the value range of the pixels in the mini-batch.
        #
        if self.__late_range_normalizer is not None:
            batch = self.__late_range_normalizer.process(patches=batch)

        # Calculate label weights.
        #
        if self.__late_weight_mapper is not None:
            batch = self.__late_weight_mapper.process(patches=batch)

        # Adjust the label weights in the mini-batch.
        #
        if self.__batch_weight_mapper is not None:
            batch = self.__batch_weight_mapper.process(patches=batch)

        # Format the label weights.
        #
        if self.__weight_mapping_enabled:
            batch = self.__formatweightmaps(patches=batch)

        # Convert the labels to one-hot representation.
        #
        if self.__late_labels_one_hot:
            batch = self.__labelstoonehot(patches=batch)

        return batch
