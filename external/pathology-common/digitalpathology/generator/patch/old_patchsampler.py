"""
This file contains class for sampling patches from whole slide images.
"""

from . import patchsource as dptpatchsource
from . import basepatchsampler as dptbasepatchsampler

from ..mask import randomizer as dptrandom
from ..mask import maskstats as dptmaskstats

from ...errors import imageerrors as dptimageerrors
from ...errors import dataerrors as dptdataerrors
from ...errors import configerrors as dptconfigerrors
from ...image.io import imagereader as dptimagereader

import numpy as np
import os
import cv2

#----------------------------------------------------------------------------------------------------

class PatchSampler(dptbasepatchsampler.BasePatchSampler):
    """This class can sample patches from an image considering various conditions like mask and probability."""

    def __init__(self, patch_source, create_stat, mask_spacing, spacing_tolerance, input_channels, label_mode):
        """
        Initialize the object: load the image, load a mask for the configured image and check compatibility, extract and store necessary mask data
        in memory for efficient patch extraction and initialize index randomizer.

        Args:
            patch_source (dptpatchsource.PatchSource): Image patch source descriptor with image and mask paths, mask level and labels to use.
            create_stat (bool): Allow missing stat file, and create it if necessary.
            mask_spacing (float, None): Pixel spacing of the mask to process (micrometer).
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
            input_channels (list): Desired channels that are extracted for each patch.
            label_mode (str): Label generation mode. Accepted values:
                'central': Just return the label of the central pixel.
                'synthesize': Synthesize the label map from the mask statistics and zoom it to the appropriate level of pixel spacing.
                'load': Load the label map from the label image and zoom it to the appropriate level of pixel spacing if necessary.

        Raises:
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
            UnknownLabelModeError: The label generation mode is unknown.
            MissingMaskImageError: The label generation mode is set to 'load' but the mask image is not present.
            StatShapeMismatchError: The shape of the loaded stat and the given mask file cannot be matched.
            StatSpacingMismatchError: The pixel spacing of the loaded stat and the given mask file cannot be matched.
            MissingMaskImageError: Stat is not configured but mask image is also not configured for stat calculation.
            UnfixableImageSpacingError: The missing spacing information of he mask image cannot be fixed.
            MaskLabelListMismatchError: The configured labels does not match the list of the labels collected by the stat object.
            ImageShapeMismatchError: The shape of the image cannot be matched the shape of the mask.

            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
            DigitalPathologyStatError: Stat errors.
        """

        super().__init__()
        np.random.seed(10)
        self._loader = None             # Image patch loader.
        self._mask = None               # Mask patch loader.
        self._stats = None              # Mask statistics.
        self._rand = None               # Index randomizer object.
        self._spacing_tolerance = 0.0   # Tolerance for finding a level for the given pixel spacing.
        self._input_channel_count = 0   # Desired channels that are extracted for each patch.
        self._label_mode = ''
        self._stats_level = 0           # Level of the image where its dimensions match the mask.
        self._stats_downsamplings = []  # Downsampling factors for each used level relative to the matching level.
        self._stats_shifts = []         # Central pixel shift values for each used level relative to the matching level.
        self._mask_level = []           # Level of mask to use of each image level for label map extraction.
        self._mask_downsamplings = []   # Downsampling factors for each used level relative to the mask level to use.
        self._mask_shifts = []          # Central pixel shift values for each used level relative to the mask level to use.
        self._sampler_items = {}
        # Configure parameters.
        #
        self._setspacing(spacing_tolerance=spacing_tolerance)
        self._openimage(image_path=patch_source.image, mask_path=patch_source.mask, input_channels=input_channels)
        self._configuremode(label_mode=label_mode)
        self._collectdata(stat_path=patch_source.stat, create_stat=create_stat, mask_spacing=mask_spacing, mask_labels=patch_source.labels)
        self._initrandomizer()

    @property
    def labels(self):
        return self._stats.labels

    def _setspacing(self, spacing_tolerance):
        """
        Set the spacing tolerance.

        Args:
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
        """
        if spacing_tolerance < 0.0:
            Warning("Spacing tolerance <0 (current value: {})".format(spacing_tolerance))
        self._spacing_tolerance = spacing_tolerance

    def _openimage(self, image_path, mask_path, input_channels):
        """
        Load an image.

        Args:
            image_path (str): Path of the image to load.
            mask_path (str, None): Path of the mask to load.
            input_channels (list): Desired channels that are extracted for each patch.

        Raises:
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
        """

        # Open and configure the multi-resolution images.
        #
        self._loader = dptimagereader.ImageReader(image_path=image_path, spacing_tolerance=self._spacing_tolerance, input_channels=input_channels, cache_path=None)

        if mask_path:
            self._mask = dptimagereader.ImageReader(image_path=mask_path, spacing_tolerance=self._spacing_tolerance, input_channels=[0], cache_path=None)

            # If the spacing information is missing from the mask file, assume that the spacing of the mask is exactly the same as of the
            # image file where the shape of the two files first match.
            #
            if None in self._mask.spacings:
                for image_level in range(len(self._loader.shapes)):
                    if self._loader.shapes[image_level] == self._mask.shapes[0]:
                        self._mask.correct(spacing=self._loader.spacings[image_level], level=0)
                        break

            # Check if the mask spacings are correct now: the spacings for all levels are valid.
            #
            if None in self._mask.spacings:
                raise dptimageerrors.UnfixableImageSpacingError(self._loader.path, self._loader.shapes, self._mask.path, self._mask.shapes)

        self._input_channel_count = self._loader.channels

    def _configuremode(self, label_mode):
        """
        Configure the label generation mode.

        Args:
            label_mode (str): Label generation mode.

        Raises:
            UnknownLabelModeError: The label generation mode is unknown.
            MissingMaskImageError: The label generation mode is set to 'load' but the mask image is not present.
        """

        # Check if the label patch generation mode is valid.
        #
        if label_mode not in ['central', 'synthesize', 'load']:
            raise dptconfigerrors.UnknownLabelModeError(label_mode)

        # Check if mask patch loader is ready if the label patches are loaded from file.
        #
        if label_mode == 'load' and not self._mask:
            raise dptconfigerrors.MissingMaskImageError()

        # Save the label mode.
        #
        self._label_mode = label_mode

    def _collectdata(self, stat_path, create_stat, mask_spacing, mask_labels):
        """
        Extract and store necessary mask data in memory for efficient patch extraction.

        Args:
            stat_path (str, None): Path of the mask stat to load.
            create_stat (bool): Allow missing stat file, and create it if necessary.
            mask_spacing (float): Pixel spacing of the mask to process (micrometer).
            mask_labels (tuple): List of mask labels to use. All other labels will be considered as non-labeled area.
        """

        # Load the mask data.
        #
        if stat_path and (os.path.isfile(stat_path) or not create_stat):
            # Load the stats file from disk.
            #
            self._stats = dptmaskstats.MaskStats(file=stat_path, mask_spacing=mask_spacing, spacing_tolerance=self._spacing_tolerance, mask_labels=mask_labels)

            source_path = stat_path

            # Check if the loaded stats matches the label image (mask) shape.
            #
            if self._mask:
                mask_level = self._mask.level(spacing=mask_spacing)

                if self._mask.shapes[mask_level] != self._stats.shape:
                    raise dptimageerrors.StatShapeMismatchError(self._stats.path, self._stats.shape, self._mask.path, self._mask.shapes[mask_level], mask_spacing)

                if self._spacing_tolerance < abs(self._mask.spacings[mask_level] - self._stats.spacing):
                    raise dptimageerrors.StatSpacingMismatchError(self._stats.path, self._stats.spacing, self._mask.path, self._mask.spacings, self._spacing_tolerance)
        else:
            # Stat is not configured or it should be created dynamically. Check if the mask image path is set.
            #
            if not self._mask:
                raise dptconfigerrors.MissingMaskImageError()

            self._stats = dptmaskstats.MaskStats(file=self._mask, mask_spacing=mask_spacing, spacing_tolerance=self._spacing_tolerance, mask_labels=mask_labels)
            source_path = self._mask.path

            # Save the stat file if it should be dynamically created.
            #
            if stat_path and create_stat:
                self._stats.save(file_path=stat_path)

        # Check if the labels match.
        #
        if not set(mask_labels) <= set(self._stats.labels):
            raise dptdataerrors.MaskLabelListMismatchError(source_path, self._stats.labels, mask_labels)

        # Find the level in the image where it matches the dimensions of the mask image on the used mask level.
        #
        matching_level = None
        for image_level in range(len(self._loader.shapes)):
            if self._loader.shapes[image_level] == self._stats.shape and abs(self._loader.spacings[image_level] - self._stats.spacing) < self._spacing_tolerance:
                matching_level = image_level
                break

        # Check if the stat shape matches the image at any level.
        #
        if matching_level is not None:
            # Calculate the downsampling factor between the mask level and each image level.
            #
            self._stats_level = matching_level
            self._stats_downsamplings = [self._loader.downsamplings[matching_level] / level_downsampling for level_downsampling in self._loader.downsamplings]
            self._stats_shifts = [max(0, int(stats_downsampling) // 2 - 1) for stats_downsampling in self._stats_downsamplings]
        else:
            raise dptimageerrors.ImageShapeMismatchError(self._loader.path, self._loader.shapes, self._loader.spacings, self._stats.path, self._stats.shape, self._stats.spacing)

        # Check if the image shape matches the mask shape on the given level.
        #
        if self._mask:
            # Get the mask level to use by the pixel spacing.
            #
            mask_level = self._mask.level(spacing=mask_spacing)

            if matching_level is not None and \
               self._loader.shapes[matching_level] == self._mask.shapes[mask_level] and \
               abs(self._loader.spacings[matching_level] - self._mask.spacings[mask_level]) < self._spacing_tolerance:
                # Calculate the closest matching level from the mask for each level in the image and the downsampling level to zoom to the same level as the image.
                #
                level_diff = matching_level - mask_level
                self._mask_level = [min(max(0, level - level_diff), len(self._mask.shapes) - 1) for level in range(len(self._loader.shapes))]
                self._mask_downsamplings = [self._loader.downsamplings[level] / self._loader.downsamplings[self._mask_level[level] + level_diff] for level in range(len(self._loader.shapes))]
                self._mask_shifts = [max(0, int(mask_downsampling) // 2 - 1) for mask_downsampling in self._mask_downsamplings]
            else:
                raise dptimageerrors.ImageShapeMismatchError(self._loader.path,
                                                             self._loader.shapes,
                                                             self._loader.spacings,
                                                             self._mask.path,
                                                             self._mask.shapes[mask_level],
                                                             self._mask.spacings[mask_level])

    def _initrandomizer(self):
        """
        Initialize index randomizer.
        """
        self._rand = dptrandom.IndexRandomizer(pixel_counts=self._stats.counts)

    def _randomizecoordinates(self, counts):
        """
        Randomize coordinates with the previously configured distribution. The results are in numpy notation (row, col).

        Args:
            counts (dict): Label value to label count to extract mapping.

        Returns:
            np.ndarray: List of randomized coordinates organized in a list per label.

        Raises:
            LabelListMismatchError: The available labels does not match the requested labels
        """

        # Collect random pixel indexes in ascending order.
        #
        per_label_random_indexes = self._rand.randomindices(counts=counts)

        # Convert indexes to (row, col, label) coordinates.
        #
        coordinate_array = np.empty((sum(pixel_count for pixel_count in counts.values()), 3), dtype=np.int32)
        filled_up_index = 0
        for label in per_label_random_indexes:
            index_array = per_label_random_indexes[label]
            index_count = len(index_array)
            coordinate_array[filled_up_index:filled_up_index + index_count] = self._stats.indextocoorindate(index_array=index_array, label=label)
            filled_up_index += index_count

        return coordinate_array

    def reset_sampler(self):
        self._sampler_items = {}

    def single_sample(self, item, label, shape, spacing):
        if item not in self._sampler_items:
            sample_index = self._rand.randomindex(label)
            central_coordinate = self._stats.indextocoorindate([sample_index], label=label)[0][0:2]
            self._sampler_items[item] = central_coordinate
        else:
            central_coordinate = self._sampler_items[item]

        level = self._loader.level(spacing=spacing)
        patch = self._get_patch(central_coordinate, level, shape, spacing)

        # label mode is 'load' we need to sample the mask patch. In case of 'central' mode, we can just pass the label.
        if self._label_mode == 'load':
            # Load label patch from the mask image. First convert the coordinate from the level of the statistics (matching level) back to the target level
            # of the image then convert both the coordinate and the patch shape from the target level of the image to the actual mask image level to use.
            #
            load_level = self._mask_level[level]
            load_level_center_image = central_coordinate * self._stats_downsamplings[level] + self._stats_shifts[level]
            load_level_shape_mask = np.multiply(shape, self._mask_downsamplings[level]).astype(int)
            load_level_center_mask = load_level_center_image * self._mask_downsamplings[level] + self._mask_shifts[level]
            load_level_start_mask = np.subtract(load_level_center_mask, [(load_level_shape_mask[0] - 1) // 2,
                                                                         (load_level_shape_mask[1] - 1) // 2])

            loaded_level_patch = self._mask.read(spacing=self._mask.spacings[load_level],
                                                 row=load_level_start_mask[0],
                                                 col=load_level_start_mask[1],
                                                 height=load_level_shape_mask[0],
                                                 width=load_level_shape_mask[1])

            if np.array_equal(load_level_shape_mask, shape):
                label = loaded_level_patch.squeeze()
            else:
                label = cv2.resize(loaded_level_patch.squeeze(), tuple(shape), cv2.INTER_NEAREST)

        return patch, label

    def _get_patch(self, central_coordinate, level, shape, spacing):
        patch_level_center = np.multiply(central_coordinate, self._stats_downsamplings[level]).astype(int) + \
                             self._stats_shifts[level]
        patch_level_start = patch_level_center - [(shape[0] - 1) // 2, (shape[1] - 1) // 2]
        patch = self._loader.read(spacing=spacing,
                                  row=patch_level_start[0],
                                  col=patch_level_start[1],
                                  height=shape[0],
                                  width=shape[1])
        return patch

    def sample(self, counts, shapes):
        """
        Collect a batch of patches with the configured distribution from the opened image.

        Args:
            counts (dict): Label value to label count to extract mapping.
            shapes (dict): Dictionary mapping pixel spacings to (rows, columns) patch shape.

        Returns:
            dict: Collected batch of RGB patches and corresponding labels or crops of mask data with label indices (not label values) per level.

        Raises:
            PixelSpacingLevelError: There is no level found for the given pixel spacing and tolerance.
            LabelListMismatchError: The available labels does not match the requested labels
        """

        # Extract random coordinates and sort them for more efficient patch extraction.
        #
        valid_counts = {label: label_count for label, label_count in counts.items() if 0 < label_count}
        non_organized_coordinates = self._randomizecoordinates(valid_counts)
        lex_sort_order = np.lexsort((non_organized_coordinates[:, 1], non_organized_coordinates[:, 0]))
        patch_coordinates = non_organized_coordinates[lex_sort_order]

        # Initialize output.
        #
        sum_count = sum(valid_counts.values())
        patch_dict = {spacing: {'patches': np.empty(shape=(sum_count,) + tuple(shapes[spacing]) + (self._input_channel_count,), dtype=np.uint8),
                                'labels': np.empty(shape=(sum_count,) if self._label_mode == 'central' else (sum_count,) + tuple(shapes[spacing]), dtype=np.uint8)}
                      for spacing in shapes}
        # Extract patch from each coordinate.
        #
        for sample_index in range(patch_coordinates.shape[0]):
            central_coordinate = patch_coordinates[sample_index, 0:2]

            # Extract patches of the image.
            #
            for spacing in shapes:
                level = self._loader.level(spacing=spacing)
                patch_level_shape = shapes[spacing]
                patch_level_center = np.multiply(central_coordinate, self._stats_downsamplings[level]).astype(int) + self._stats_shifts[level]
                patch_level_start = patch_level_center - [(patch_level_shape[0] - 1) // 2, (patch_level_shape[1] - 1) // 2]
                patch_dict[spacing]['patches'][sample_index] = self._loader.read(spacing=spacing,
                                                                                 row=patch_level_start[0],
                                                                                 col=patch_level_start[1],
                                                                                 height=patch_level_shape[0],
                                                                                 width=patch_level_shape[1])

            # Extract or synthesize patches of the mask.
            #
            for spacing in shapes:
                level = self._loader.level(spacing=spacing)
                level_shape = shapes[spacing]

                if self._label_mode == 'load':
                    # Load label patch from the mask image. First convert the coordinate from the level of the statistics (matching level) back to the target level
                    # of the image then convert both the coordinate and the patch shape from the target level of the image to the actual mask image level to use.
                    #
                    load_level = self._mask_level[level]
                    load_level_center_image = central_coordinate * self._stats_downsamplings[level] + self._stats_shifts[level]
                    load_level_shape_mask = np.multiply(level_shape, self._mask_downsamplings[level]).astype(int)
                    load_level_center_mask = load_level_center_image * self._mask_downsamplings[level] + self._mask_shifts[level]
                    load_level_start_mask = np.subtract(load_level_center_mask, [(load_level_shape_mask[0] - 1) // 2, (load_level_shape_mask[1] - 1) // 2])

                    loaded_level_patch = self._mask.read(spacing=self._mask.spacings[load_level],
                                                         row=load_level_start_mask[0],
                                                         col=load_level_start_mask[1],
                                                         height=load_level_shape_mask[0],
                                                         width=load_level_shape_mask[1])

                    if np.array_equal(load_level_shape_mask, level_shape):
                        patch_dict[spacing]['labels'][sample_index] = loaded_level_patch.squeeze()
                    else:
                        patch_dict[spacing]['labels'][sample_index] = np.expand_dims(cv2.resize(loaded_level_patch.squeeze(), tuple(shapes[spacing]), cv2.INTER_NEAREST), axis=0)

                elif self._label_mode == 'central':
                    # Just store the label of the central pixel.
                    #
                    patch_dict[spacing]['labels'][sample_index] = patch_coordinates[sample_index, 2]
        # Return the constructed patch, label collection.
        #
        return patch_dict

    def close(self):
        """Close the opened image objects."""

        # Close the image.
        #
        if self._loader is not None:
            self._loader.close()
            self._loader = None

        # Close the mask.
        #
        if self._mask is not None:
            self._mask.close()
            self._mask = None
