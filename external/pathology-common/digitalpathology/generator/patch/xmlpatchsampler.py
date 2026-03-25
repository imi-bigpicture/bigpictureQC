"""
This file contains class for sampling patches from whole slide images.
"""

from . import patchsource as dptpatchsource
from . import basepatchsampler as dptbasepatchsampler

from ..mask import randomizer as dptrandom
from ..mask import xml_maskstats as dptxmlmaskstats
from ...errors import configerrors as dptconfigerrors
from ...image.io import imagereader as dptimagereader

import numpy as np

#----------------------------------------------------------------------------------------------------

class XmlPatchSampler(dptbasepatchsampler.BasePatchSampler):
    """This class can sample patches from an image considering various conditions like mask and probability."""

    def __init__(self, patch_source, mask_spacing, spacing_tolerance, input_channels, label_mode):
        """
        Initialize the object: load the image, load a mask for the configured image and check compatibility, extract and store necessary mask data
        in memory for efficient patch extraction and initialize index randomizer.

        Args:
            patch_source (dptpatchsource.PatchSource): Image patch source descriptor with image and mask paths, mask level and labels to use.
            create_stat (bool): Allow missing stat file, and create it if necessary.
            mask_spacing (float, None): Pixel spacing of the mask to process (micrometer).
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
            input_channels (list): Desired channels that are extracted for each patch.
            label_mode (str): Currently not used as only ASAP xml files are supported.
        """

        super().__init__()

        self._loader = None
        self._xml_path = None
        self._stats = None
        self._spacing_tolerance = 0.0
        self._input_channel_count = 0
        self._label_mode = label_mode
        self._stats_level = 0

        # Configure parameters.
        #
        self._setspacing(spacing_tolerance=spacing_tolerance)
        self._openimage(image_path=patch_source.image, xml_path=patch_source.mask, input_channels=input_channels)
        self._initialize_stats(target_spacing=mask_spacing, xml_labels=patch_source.labels)
        self._rand = dptrandom.IndexRandomizer(pixel_counts=self._stats.counts)

    def _setspacing(self, spacing_tolerance):
        if spacing_tolerance < 0.0:
            raise dptconfigerrors.InvalidPixelSpacingToleranceError(spacing_tolerance)
        self._spacing_tolerance = spacing_tolerance

    def _openimage(self, image_path, xml_path, input_channels):
        self._loader = dptimagereader.ImageReader(image_path=image_path, spacing_tolerance=self._spacing_tolerance, input_channels=input_channels, cache_path=None)
        self._xml_path = xml_path
        self._input_channel_count = self._loader.channels

    def _initialize_stats(self, target_spacing, xml_labels):
        """
        Extract and store necessary mask data in memory for efficient patch extraction.

        Args:
            target_spacing (float): Target spacing to which the bounding boxes need to be rescaled.
            xml_labels (tuple): List of mask labels to use. All other labels will be considered as non-labeled area.
        """

        if not self._xml_path:
            raise dptconfigerrors.MissingMaskImageError()
        refined_spacing = self._loader.refine(target_spacing)
        self._stats = dptxmlmaskstats.XML_MaskStats(file=self._xml_path, spacing_factor=refined_spacing / self._loader.spacings[0], mask_labels=xml_labels)

    def _randomizecoordinates(self, counts):
        """
        Randomize coordinates with the previously configured distribution. The results are in numpy notation (row, col).

        Args:
            counts (dict): Label value to label count to extract mapping.

        Returns:
            np.ndarray: List of randomized coordinates organized in a list per label.
        """

        per_label_random_indexes = self._rand.randomindices(counts=counts)
        coordinate_array = np.empty((sum(pixel_count for pixel_count in counts.values()), 3), dtype=np.int32)
        filled_up_index = 0
        for label in per_label_random_indexes:
            index_array = per_label_random_indexes[label]
            index_count = len(index_array)
            coordinate_array[filled_up_index:filled_up_index + index_count] = self._stats.indextocoorindate(index_array=index_array, label=label)
            filled_up_index += index_count

        return coordinate_array

    def _sample_random_coordinates(self, counts):
        valid_counts = {label: label_count for label, label_count in counts.items() if 0 < label_count}
        non_organized_coordinates = self._randomizecoordinates(valid_counts)
        lex_sort_order = np.lexsort((non_organized_coordinates[:, 1], non_organized_coordinates[:, 0]))
        patch_coordinates = non_organized_coordinates[lex_sort_order]
        return patch_coordinates, valid_counts

    def _sample_patch(self, central_coordinate, shapes, spacing):
        patch_level_shape = shapes[spacing]
        patch_level_center = central_coordinate
        patch_level_start = patch_level_center - [(patch_level_shape[0] - 1) // 2, (patch_level_shape[1] - 1) // 2]
        return self._loader.read(spacing=spacing, row=patch_level_start[0], col=patch_level_start[1],
                                 height=patch_level_shape[0],
                                 width=patch_level_shape[1])


    def sample(self, counts, shapes):
        """
        Collect a batch of patches with the configured distribution from the opened image.
        """
        # TODO: currently only works for single spacing sampling
        if len(shapes) > 1:
            raise NotImplementedError("XML sampling currently only works for single spacing.")

        patch_coordinates, label_count_dict = self._sample_random_coordinates(counts)
        sum_count = sum(label_count_dict.values())
        patch_dict = {spacing: {'patches': np.empty(shape=(sum_count,) + shapes[spacing] + (self._input_channel_count,),
                                                    dtype=np.uint8),
                                'labels': np.empty(shape=(sum_count,), dtype=np.object)}
                      for spacing in shapes}

        for sample_index in range(patch_coordinates.shape[0]):
            center_coordinate = patch_coordinates[sample_index, 0:2]
            for spacing in shapes:
                level_shape = shapes[spacing]
                patch_dict[spacing]['patches'][sample_index] = self._sample_patch(center_coordinate, shapes, spacing)
                patch_dict[spacing]['labels'][sample_index] = self._stats.construct(center_coordinate[0],
                                                                                    center_coordinate[1],
                                                                                    level_shape[0],
                                                                                    level_shape[1])

        return patch_dict

    def close(self):
        """Close the opened image objects."""

        if self._loader is not None:
            self._loader.close()
            self._loader = None