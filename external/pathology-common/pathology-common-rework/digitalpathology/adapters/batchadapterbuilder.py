"""
This file contains a builder for BatchAdapter class.
"""

from . import batchadapter as dptbatchadapter

from .augmenters import augmenterpool as dptaugmenterpool
from .augmenters.color import contrastaugmenter as dptcontrastaugmenter
from .augmenters.color import hedcoloraugmenter as dpthedcoloraugmenter
from .augmenters.color import hsbcoloraugmenter as dpthsbcoloraugmenter
from .augmenters.noise import additiveguassiannoiseaugmenter as dtpadditiveguassiannoiseaugmenter
from .augmenters.noise import gaussianbluraugmenter as dptgaussianbluraugmenter
from .augmenters.noise import blackoutaugmenter as dptblackoutaugmenter
from .augmenters.spatial import elasticagumenter as dptelasticagumenter
from .augmenters.spatial import flipaugmenter as dptflipaugmenter
from .augmenters.spatial import rotate90augmenter as dptrotate90augmenter
from .augmenters.spatial import scalingaugmenter as dptscalingaugmenter
from .range import additionalrangenormalizer as dtpgadditionalrangenormalizer
from .range import generalrangenormalizer as dtpgeneralrangenormalizer
from .range import rgbrangenormalizer as dtprgbrangenormalizer
from .range import rgbtozeroonerangenormalizer as dtprgbtozeroonerangenormalizer
from .label import labelmapper as dptlabelmapper
from .weight import cleanweightmapper as dptcleanweightmapper
from .weight.normalizing import batchweightmapper as dptbatchweightmapper
from .weight.normalizing import patchweightmapper as dptpatchweightmapper

from ..errors import configerrors as dptconfigerrors

import logging

#----------------------------------------------------------------------------------------------------

class BatchAdapterBuilder(object):
    """Builder for BatchAdapter class."""

    def __init__(self):
        """Initialize the object."""

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__logger = None                   # Configured logger object.
        self.__augmentation_config = None      # Augmentation configuration.
        self.__range_normalizer_config = None  # Range normalizer configuration.
        self.__label_mapper_config = None      # Label mapper configuration.
        self.__weight_mapper_config = None     # Label weight mapper configuration.
        self.__class_count = None              # Number of classes to assume if label mapper is not configured.

        # Initialize logging.
        #
        self.__initlogging()

    def __initlogging(self):
        """Initialize logging."""

        # Configure logging. This class relies on configured logging somewhere down on the hierarchy.
        #
        qualified_class_name = '{module_name}.{class_name}'.format(module_name=self.__class__.__module__, class_name=self.__class__.__name__)
        self.__logger = logging.getLogger(name=qualified_class_name)

    def __buildaugmenter(self):
        """
        Construct patch augmenter.

        Returns:
            dptaugmenterpool.AugmenterPool: Patch augmenter.

        Raises:
            UnknownAugmentationTypeError: Unknown augmentation type.

            DigitalPathologyAugmentationError: Augmentation errors.
        """

        self.__logger.info('Configuring augmenters')

        # Log configuration.
        #
        self.__logger.debug('Patch augmenters configuration: {config}'.format(config=self.__augmentation_config))

        # Check if augmentation is configured.
        #
        if self.__augmentation_config is not None:
            # Initialize augmenter pool.
            #
            patch_augmenter = dptaugmenterpool.AugmenterPool()

            # Instantiate and add each augmenter from the configuration to the pool.
            #
            for group_item in self.__augmentation_config:
                # Add augmentation group.
                #
                patch_augmenter.appendgroup(group=group_item['group'], randomized=group_item['random'])

                # Add all augmenter items in the group.
                #
                for augmenter_item in group_item['items']:
                    # Create augmenter object.
                    #
                    if augmenter_item['type'] == 'contrast':
                        # Contrast enhancement patch augmentation.
                        #
                        current_augmenter = dptcontrastaugmenter.ContrastAugmenter(sigma_range=augmenter_item['sigma'])

                    elif augmenter_item['type'] == 'hed_color':
                        # Saturation enhancement patch augmentation.
                        #
                        current_augmenter = dpthedcoloraugmenter.HedColorAugmenter(haematoxylin_sigma_range=augmenter_item['haematoxylin']['sigma'],
                                                                                   haematoxylin_bias_range=augmenter_item['haematoxylin']['bias'],
                                                                                   eosin_sigma_range=augmenter_item['eosin']['sigma'],
                                                                                   eosin_bias_range=augmenter_item['eosin']['bias'],
                                                                                   dab_sigma_range=augmenter_item['dab']['sigma'],
                                                                                   dab_bias_range=augmenter_item['dab']['bias'],
                                                                                   cutoff_range=augmenter_item['cutoff'])

                    elif augmenter_item['type'] == 'hsb_color':
                        # Saturation enhancement patch augmentation.
                        #
                        current_augmenter = dpthsbcoloraugmenter.HsbColorAugmenter(hue_sigma_range=augmenter_item['hue'],
                                                                                   saturation_sigma_range=augmenter_item['saturation'],
                                                                                   brightness_sigma_range=augmenter_item['brightness'])

                    elif augmenter_item['type'] == 'additive':
                        # Additive Gaussian noise patch augmentation.
                        #
                        current_augmenter = dtpadditiveguassiannoiseaugmenter.AdditiveGaussianNoiseAugmenter(sigma_range=augmenter_item['sigma'])

                    elif augmenter_item['type'] == 'blur':
                        # Gaussian blur patch augmentation.
                        #
                        current_augmenter = dptgaussianbluraugmenter.GaussianBlurAugmenter(sigma_range=augmenter_item['sigma'])

                    elif augmenter_item['type'] == 'elastic':
                        # Elastic deformation patch augmentation.
                        #
                        current_augmenter = dptelasticagumenter.ElasticAugmenter(sigma_interval=augmenter_item['sigma'],
                                                                                 alpha_interval=augmenter_item['alpha'],
                                                                                 map_count=augmenter_item['maps'],
                                                                                 interpolation_order=augmenter_item['order'])

                    elif augmenter_item['type'] == 'flip':
                        # Flipping patch augmentation.
                        #
                        current_augmenter = dptflipaugmenter.FlipAugmenter(flip_list=augmenter_item['flips'])

                    elif augmenter_item['type'] == 'rotate_90':
                        # Rotation by multiples of 90 degrees patch augmentation.
                        #
                        current_augmenter = dptrotate90augmenter.Rotate90Augmenter(k_list=augmenter_item['rotations'])

                    elif augmenter_item['type'] == 'scale':
                        # Scaling patch augmentation.
                        #
                        current_augmenter = dptscalingaugmenter.ScalingAugmenter(scaling_range=augmenter_item['scaling'], interpolation_order=augmenter_item['order'])

                    elif augmenter_item['type'] == 'blackout':
                        current_augmenter = dptblackoutaugmenter.BlackoutAugmenter(blackout_class=augmenter_item['blackout_class'])

                    else:
                        # The given augmentation type is unknown.
                        #
                        self.__logger.error('Unknown augmentation type: \'{augmentation_type}\''.format(augmentation_type=augmenter_item['type']))
                        raise dptconfigerrors.UnknownAugmentationTypeError(augmenter_item['type'])

                    # Append the augmenter object to the pool.
                    #
                    patch_augmenter.appendaugmenter(augmenter=current_augmenter, group=group_item['group'], probability=augmenter_item['probability'])

        else:
            # Patch augmentation is not configured.
            #
            patch_augmenter = None

        return patch_augmenter

    def __buildrangenormalizers(self):
        """
        Construct value range normalizers.

        Returns:
            bool, RangeNormalizerBase, RangeNormalizerBase: Squash range flag, range normalizer, and mini-bath range normalizer.

        Raises:
            UnknownRangeNormalizerTypeError: Unknown range normalization type.

            DigitalPathologyRangeError: Range normalization errors.
        """

        self.__logger.info('Configuring range normalizer')

        # Log configuration.
        #
        self.__logger.debug('Range normalization configuration: {config}'.format(config=self.__range_normalizer_config))

        # Check if value range normalization is configured.
        #
        if self.__range_normalizer_config is not None:
            # Check if early range normalization is enabled.
            #
            if self.__range_normalizer_config['early']:
                # Early range normalization is enabled. The normalization will be done right after patch extraction. Start with squashing to [0.0, 1.0] to prevent
                # back and forth conversion during augmentation. That assumes the [0, 255] input range.
                #
                late_range_normalizer = None

                if self.__range_normalizer_config['type'] == 'general':

                    # The input range is not the standard [0, 255], the range squashing cannot be enabled.
                    #
                    squash_range = False
                    range_normalizer = dtpgeneralrangenormalizer.GeneralRangeNormalizer(target_range=self.__range_normalizer_config['target range'],
                                                                                        source_range=self.__range_normalizer_config['source range'])

                elif self.__range_normalizer_config['type'] == 'rgb':
                    # The input range is the standard [0, 255] but the output is something different. After squashing further normalization is required.
                    #
                    squash_range = True
                    range_normalizer = dtpgadditionalrangenormalizer.AdditionalRangeNormalizer(target_range=self.__range_normalizer_config['target range'])

                elif self.__range_normalizer_config['type'] == 'rgb to 0-1':
                    squash_range = True
                    range_normalizer = None

                else:
                    # The given range normalization type is unknown.
                    #
                    self.__logger.error('Unknown range normalization type: \'{normalization_type}\''.format(normalization_type=self.__range_normalizer_config['type']))
                    raise dptconfigerrors.UnknownRangeNormalizerTypeError(normalization_type=self.__range_normalizer_config['type'])

            else:
                # Early range normalization is disabled. The normalization will be done after mini-batch assembly.
                #
                squash_range = False
                range_normalizer = None

                if self.__range_normalizer_config['type'] == 'general':
                    late_range_normalizer = dtpgeneralrangenormalizer.GeneralRangeNormalizer(target_range=self.__range_normalizer_config['target range'],
                                                                                             source_range=self.__range_normalizer_config['source range'])

                elif self.__range_normalizer_config['type'] == 'rgb':
                    late_range_normalizer = dtprgbrangenormalizer.RgbRangeNormalizer(target_range=self.__range_normalizer_config['target range'])

                elif self.__range_normalizer_config['type'] == 'rgb to 0-1':
                    late_range_normalizer = dtprgbtozeroonerangenormalizer.RgbToZeroOneRangeNormalizer()

                else:
                    # The given range normalization type is unknown.
                    #
                    self.__logger.error('Unknown range normalization type: \'{normalization_type}\''.format(normalization_type=self.__range_normalizer_config['type']))
                    raise dptconfigerrors.UnknownRangeNormalizerTypeError(self.__range_normalizer_config['type'])

        else:
            # Value range normalization is not configured.
            #
            squash_range = False
            range_normalizer = None
            late_range_normalizer = None

        # Return the configured batch normalizer objects.
        #
        return squash_range, range_normalizer, late_range_normalizer

    def __buildlabelmapper(self):
        """
        Construct the label value to label index mappers.

        Returns:
            LabelMapper, bool, bool: Label mapper, early one-hot conversion flag, and late one-hot conversion flag.

        Raises:
            DigitalPathologyLabelError: Label errors.
        """

        self.__logger.info('Configuring label mapper')

        # Log parameters.
        #
        self.__logger.debug('Label mapping configuration: {config}'.format(config=self.__label_mapper_config))

        # The label mapping is always done immediately after extraction to keep the library consistent. There is always a label mapper configured so the
        # label counts are not filled in.
        #
        label_mapper = dptlabelmapper.LabelMapper(label_map=self.__label_mapper_config['label map'])
        # if 'label map' in self.__label_mapper_config.keys():
        #     label_mapper = dptlabelmapper.LabelMapper(label_map=self.__label_mapper_config['label map'])
        # else:
        #     return None, None, None

        # Check if early one-hot conversion is enabled.
        #
        labels_one_hot = False
        labels_late_one_hot = False

        if self.__label_mapper_config['one hot']:
            self.__logger.debug('Labels early one-hot: {flag}'.format(flag=self.__label_mapper_config['early']))

            if self.__label_mapper_config['early']:
                labels_one_hot = True
            else:
                labels_late_one_hot = True

        # Return training and validation label mappers.
        #
        return label_mapper, labels_one_hot, labels_late_one_hot

    def __buildweightmappers(self, classes):
        """
        Construct the weight mapper objects.

        Args:
            classes (int): Number of output classes.

        Returns:
            WeightMapperBase, WeightMapperBase, WeightMapperBase: Patch weight mappers, late patch weight mappers and batch weight mappers.

        Raises:
            InvalidClassCountError: Neither a label mapper nor a valid number of classes are configured.
            UnknownWeightMapperTypeError: Unknown weight mapping type.

            DigitalPathologyWeightError: Weight mapping errors.
        """

        self.__logger.info('Configuring weight mapper')

        # Log parameters.
        #
        self.__logger.debug('Weight mapping configuration: {config}'.format(config=self.__weight_mapper_config))

        # Check class if count is configured.
        #
        if classes is None or classes < 2:
            dptconfigerrors.InvalidClassCountError(classes)

        # Check if label weight mapping is configured.
        #
        if self.__weight_mapper_config is not None:
            # Configure the weight mapper objects.
            #
            if self.__weight_mapper_config['type'] == 'clean':
                patch_weight_mapper = dptcleanweightmapper.CleanWeightMapper(classes=classes)
                batch_weight_mapper = None

            elif self.__weight_mapper_config['type'] == 'balancing':
                patch_weight_mapper = dptpatchweightmapper.PatchWeightMapper(classes=classes,
                                                                             normalize=self.__weight_mapper_config['normalize'],
                                                                             clip_min=self.__weight_mapper_config['clipping']['min'],
                                                                             clip_max=self.__weight_mapper_config['clipping']['max'])
                batch_weight_mapper = None

            elif self.__weight_mapper_config['type'] == 'batch balancing':
                patch_weight_mapper = dptcleanweightmapper.CleanWeightMapper(classes=classes)
                batch_weight_mapper = dptbatchweightmapper.BatchWeightMapper(classes=classes,
                                                                             normalize=self.__weight_mapper_config['normalize'],
                                                                             clip_min=self.__weight_mapper_config['clipping']['min'],
                                                                             clip_max=self.__weight_mapper_config['clipping']['max'])
            else:
                # The given label weight mapping type is unknown.
                #
                self.__logger.error('Unknown weight mapping type: \'{mapping_type}\''.format(mapping_type=self.__weight_mapper_config['type']))
                raise dptconfigerrors.UnknownWeightMapperTypeError(self.__weight_mapper_config['type'])

            # Check if early mapping is enabled.
            #
            if self.__weight_mapper_config['early']:
                weight_mapper = patch_weight_mapper
                late_weight_mapper = None
            else:
                weight_mapper = None
                late_weight_mapper = patch_weight_mapper

        else:
            # Label weight mapping is not configured.
            #
            weight_mapper = None
            late_weight_mapper = None
            batch_weight_mapper = None

        # Return the configured weight mapper objects.
        #
        return weight_mapper, late_weight_mapper, batch_weight_mapper

    def setaugmenters(self, config):
        """
        Configure the augmenters.

        Args:
            config (list, None): List of augmentation group configurations. See 'augmentation' section of parameter configuration example.
        """

        self.__augmentation_config = config

    def setrangenormalizer(self, config):
        """
        Configure the value range normalizers.

        Args:
            config (dict, None): Range normalizer configuration. See 'data/range normalization' section of parameter configuration example.
        """

        self.__range_normalizer_config = config

    def setlabelmapper(self, config):
        """
        Configure the label mappers.

        Args:
            config (dict): Label mapper configuration. See 'data/labels' section of parameter configuration example.
        """

        self.__label_mapper_config = config

    def setweightmapper(self, config, classes=None):
        """
        Configure the label weight mappers.

        Args:
            config (dict, None): Label weight mapper configuration. See 'data/weight mapping' section of parameter configuration example.
            classes (int, None): Class count. Only used if label mapper is not configured.
        """

        self.__weight_mapper_config = config
        self.__class_count = classes

    def build(self):
        """
        Build the configured BatchAdapter object.

        Returns:
            BatchAdapter: The configured batch adapter object.

        Raises:
            UnknownAugmentationTypeError: Unknown augmentation type.
            UnknownRangeNormalizerTypeError: Unknown range normalization type.
            InvalidClassCountError: Neither a label mapper nor a valid number of classes are configured.
            UnknownWeightMapperTypeError: Unknown weight mapping type.
            InvalidLabelCountError: Invalid label count for one-hot encoding.
            MissingLabelMapperForWeightMapperError: Weight mapping is configured without label mapping.
            WeightMapperLabelMapperClassesMismatchError: The network labels known by the label mapper and weight mapper does not match.

            DigitalPathologyAugmentationError: Augmentation errors.
            DigitalPathologyLabelError: Label errors.
            DigitalPathologyRangeError: Range normalization errors.
            DigitalPathologyWeightError: Weight mapping errors.
        """

        self.__logger.info('Building batch adapter')

        # Build the augmenter pool.
        #
        augmenter_pool = self.__buildaugmenter()

        # Build the value range normalizers.
        #
        squash_range, range_normalizer, late_range_normalizer = self.__buildrangenormalizers()

        # Build the label mapper.
        #
        label_mapper, labels_one_hot, late_labels_one_hot = self.__buildlabelmapper()

        # Calculate the label count.
        #
        class_count = label_mapper.classes if label_mapper is not None else self.__class_count

        # Build the weight mappers.
        #
        weight_mapper, late_weight_mapper, batch_weight_mapper = self.__buildweightmappers(classes=class_count)

        # Build the batch adapter and return it.
        #
        return dptbatchadapter.BatchAdapter(squash_range=squash_range,
                                            augmenter_pool=augmenter_pool,
                                            range_normalizer=range_normalizer,
                                            label_mapper=label_mapper,
                                            labels_one_hot=labels_one_hot,
                                            weight_mapper=weight_mapper,
                                            late_range_normalizer=late_range_normalizer,
                                            late_weight_mapper=late_weight_mapper,
                                            batch_weight_mapper=batch_weight_mapper,
                                            late_labels_one_hot=late_labels_one_hot,
                                            label_count=class_count)
