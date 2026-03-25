"""
This file contains class for extracting patches from a collection of whole slide images.
"""

from ..patch import patchsampler as dptpatchsampler
from ..patch import patchsamplerdaemon as dptpatchsamplerdaemon
from ..batch import batchsource as dptbatchsource

from ...utils import population as dptpopulation

import logging
import numpy as np
import queue
import os
import sys
import random
import multiprocessing as mp
import threading


# ----------------------------------------------------------------------------------------------------

class SimpleSampler(object):
    """This class is a batch sampler class that extracts patches from a collection of whole slide images."""

    def __init__(self,
                 patch_source_filepath,
                 partition='training',
                 iterations=None,
                 label_dist=None,
                 label_map=None,
                 patch_shapes=None,
                 mask_spacing=None,
                 category_dist=None,
                 input_channels=None,
                 label_mode='central',
                 data_adapter=None,
                 sampler_count=None,
                 spacing_tolerance=0.25,
                 strict_selection=False,
                 create_stats=False,
                 free_label_range=False,
                 seed=None):
        """
        Initialize the object. Set the list of labels, store the patch shape, the ratio of data sets per purpose (e.g. 'training', 'validation' and 'testing'),
        add data source, configure the augmenter object and  set collection size that is the number of opened images of patch extraction at once per image category.

        Args:
            label_dist (dict): Label sampling distribution mapping from label value to ratio in a single batch.
            patch_shapes (dict): Dictionary mapping pixel spacings to (rows, columns) patch shape.
            mask_spacing (float): Pixel spacing of the masks to process (micrometer).
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
            input_channels (list): Desired channels that are extracted for each patch.
            label_mode (str): Label generation mode. Accepted values:
                'central': Just return the label of the central pixel.
                'load': Load the label map from the label image and zoom it to the appropriate level of pixel spacing if necessary.
            patch_source_filepath (str): image sources file
            data_adapter (adapters.batchadapter.BatchAdapter): Data adapter.
            category_dist (dict): Image category sampling distribution mapping from image category to ratio in a single batch.
            strict_selection (bool): If true, every label that has higher than 0.0 ratio must be available in every source image selection.
            create_stats (bool): Allow missing stat files, and create them if necessary.
            free_label_range (bool): If this flag is True, non-continuous label ranges, and ranges that do not start at 0 are also allowed.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #

        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self._name_tag = 'simplesampler'  # Name tag of the batch sampler.
        self._log_name_tag = 'simplesampler'  # Logging name tag of the batch sampler for easy identification in logs.

        self._category_dist = {'default': 1.0} if category_dist is None else category_dist
        self._strict_selection = strict_selection  # Strict source item selection.
        self._create_stats = create_stats  # Create missing .stat files.
        self._patch_shapes = patch_shapes
        self._label_map = label_map
        self._spacing = list(patch_shapes.keys())[0]
        self._shape = list(patch_shapes.values())[0]

        self._mask_spacing = list(patch_shapes.keys())[0] if mask_spacing is None else mask_spacing
        self._spacing_tolerance = spacing_tolerance
        self._input_channels = [0,1,2] if input_channels is None else input_channels  # List of channels used in a patch
        self._label_mode = label_mode  # Label generation mode.

        self._patch_samplers = {}  # Patch sampler collection (either PatchSampler objects or host process identifiers).
        self._data_adapter = data_adapter
        self._free_label_range = False  # Free label mapping flag: no label range checking if True.

        self._data_source = {}  # List of PatchSource objects per source category.
        self._label_dist = {}  # Label distribution ratio in a batch.

        self._initlogging()
        self._setlabels(label_dist=label_dist, label_mode=label_mode, strict_selection=strict_selection,
                        free_label_range=free_label_range)
        self._setsource(source_path=patch_source_filepath, partition=partition)

        self._sampler_count = len(self._data_source['default']['items']) if sampler_count is None else sampler_count
        self._iterations = self._sampler_count * 20 if iterations is None else iterations
        self._generator_samples = None

    @property
    def labels(self):
        return list(self._label_dist.keys())

    def _setsource(self, source_path, partition):
        batch_source = dptbatchsource.BatchSource(source_items=None)
        batch_source.load(file_path=source_path)
        patch_sources = batch_source.collection(purpose_id=partition, category_id=None, replace=True)
        for category_id, item_list in patch_sources.items():
            # Store the source list.
            #
            self._data_source[category_id] = {'items': item_list, 'search': {label: list() for label in self._label_dist}}

            # Build the per label search index.
            #
            for item_index in range(len(item_list)):
                for label in self._label_dist:
                    if label in item_list[item_index].labels:
                        self._data_source[category_id]['search'][label].append(item_index)
        category_dist_sum = sum(self._category_dist.values())
        self._category_dist = {category_id: category_prob / category_dist_sum for category_id, category_prob in self._category_dist.items()}

    def _map_classes(self, mask):
        new_mask = np.zeros_like(mask)
        for key, value in self._label_map.items():
            new_mask[mask == key] = value
        return new_mask

    def _initlogging(self):
        """Initialize logging."""

        # Configure logging. This class relies on configured logging somewhere down on the hierarchy.
        #
        qualified_class_name = '{module_name}.{class_name}'.format(module_name=self.__class__.__module__,
                                                                   class_name=self.__class__.__name__)
        self._logger = logging.getLogger(name=qualified_class_name)

        # Report own process and thread identifiers.
        #
        self._logger.debug(
            '{tag}Batch sampler initializing. process: {pid}; thread: {tid}'.format(tag=self._log_name_tag,
                                                                                    pid=os.getpid(),
                                                                                    tid=threading.current_thread().ident))

    def __flushlogging(self):
        """Flush all log handlers."""

        for log_handler in self._logger.handlers:
            log_handler.flush()

    def _setlabels(self, label_dist, label_mode, strict_selection, free_label_range):
        """
        The list of labels.

        Args:
            label_dist (dict): Label sampling distribution mapping from label value to ratio in a single batch.
            label_mode (str): Label generation mode.
            strict_selection (bool): If true, every label that has higher than 0.0 ratio must be available in every source image selection.
        """

        # Check label list: it cannot be empty.
        #

        label_dist_sum = sum(label_dist.values())

        self._label_dist = {label: weight / label_dist_sum for label, weight in label_dist.items()}
        self._strict_selection = strict_selection

    def _samplesources(self):
        """
        Select a set of source items based on the categories and their available labels.

        Returns:
            set: Set of patch source items.
        """

        # Calculate total count of samples and check if any data source is present.
        #
        source_count = sum(len(self._data_source[category_id]['items']) for category_id in self._category_dist)

        # Distribute the image collection count (sampler count) among the image categories.
        #
        category_population = min(source_count, self._sampler_count)
        category_minimum_counts = {
        category_id: 1 if len(self._data_source) <= category_population and 0.0 < self._category_dist[
            category_id] else 0 for category_id in self._category_dist}
        category_distribution = {
        category_id: (self._category_dist[category_id], category_minimum_counts[category_id], category_population) for
        category_id in self._category_dist}
        category_source_count = dptpopulation.distribute_population(population=category_population,
                                                                    ratios=category_distribution)
        # Check if the target distribution is possible.
        #
        if any(len(self._data_source[category_id]['items']) < category_source_count[category_id] for category_id in
               self._category_dist):
            available_image_counts = {category_id: len(self._data_source[category_id]['items']) for category_id in
                                      self._category_dist}
            self._logger.warning(
                '{tag}The calculated {target} image distribution is not possible with {available} available images'.format(
                    tag=self._log_name_tag,
                    target=category_source_count,
                    available=available_image_counts))

        # Distribute the per category image collection count among the labels.
        #
        new_source_item_set = set()
        shuffled_label_list = list(self._label_dist.keys())
        for category_id in self._category_dist:
            # Check if there are images distributed into this category.
            #
            if category_source_count[category_id]:
                # Calculate per label image count.
                #
                label_population = category_source_count[category_id]
                label_minimum_counts = {
                label: 1 if len(self._label_dist) <= label_population and 0.0 < self._label_dist[label] else 0 for
                label in self._label_dist}
                label_distribution = {label: (self._label_dist[label], label_minimum_counts[label], label_population)
                                      for label in self._label_dist}
                label_source_count = dptpopulation.distribute_population(population=label_population,
                                                                         ratios=label_distribution)
                # Calculate set of all available image indices.
                #
                data_source_category = self._data_source[category_id]
                category_image_indices = set()
                available_image_indices = set(range(len(data_source_category['items'])))
                category_data_source_search = data_source_category['search']

                # Select source images.
                #
                random.shuffle(shuffled_label_list)
                for label in shuffled_label_list:
                    # Calculate the image list and select from it: the intersection of the images that contains this label and the images that has not been selected yet.
                    #
                    label_available_image_indices = available_image_indices.intersection(
                        category_data_source_search[label])
                    label_selection = random.sample(population=label_available_image_indices,
                                                    k=min(label_source_count[label],
                                                          len(label_available_image_indices)))
                    # Add the selected images to the per category image index set and remove them from the set of not selected images.
                    #
                    category_image_indices.update(label_selection)
                    available_image_indices.difference_update(label_selection)

                # Fill up the rest randomly if there are available images left.
                #
                if available_image_indices and len(category_image_indices) < label_population:
                    category_image_indices.update(random.sample(population=available_image_indices,
                                                                k=min(len(available_image_indices),
                                                                      label_population - len(category_image_indices))))

                # Add the selected image index set to the result list.
                #
                for item_index in category_image_indices:
                    new_source_item_set.add(data_source_category['items'][item_index])

        # Add random images to the list if necessary.
        #
        if len(new_source_item_set) < category_population:
            # Collect all the images that has not been selected yet, but contains any of the necessary labels.
            #
            available_images = set()
            for category_id in self._category_dist:
                data_source_category = self._data_source[category_id]
                category_available_image_indices = set()
                for label in self._label_dist:
                    category_available_image_indices.update(data_source_category['search'][label])

                for item_index in category_available_image_indices:
                    available_images.add(data_source_category['items'][item_index])

            available_images.difference_update(new_source_item_set)

            # Select images randomly from the available set.
            #
            item_count_to_add = min(len(available_images), category_population - len(new_source_item_set))
            new_source_item_set.update(random.sample(population=available_images, k=item_count_to_add))

        # Check if all the labels are represented if necessary.
        #
        if self._strict_selection:
            available_labels_in_selection = set()
            for new_source_item in new_source_item_set:
                available_labels_in_selection.update(new_source_item.labels)

            if any(label not in available_labels_in_selection and 0.0 < self._label_dist[label] for label in
                   self._label_dist):
                Warning("Strict selection not possible with current image set")
        return new_source_item_set

    def _patchdistribution(self, count):
        """
        Calculate distribution of patches over different images to sample.

        Args:
            count (int): Number of patches to sample.

        Returns:
            dict: per sampler patch count.

        Raises:
            LabelSourceConfigurationError: Label selected without source mask.
        """

        # Calculate the label extract ratios. If strict selection is enabled it is just the configured label distribution as it is guaranteed that each label can be
        # sampled from the current source set.
        #

        available_labels = set()
        for sampler in self._patch_samplers.values():
            available_labels.update(sampler.labels)

        label_extract_weights = {label: self._label_dist[label] for label in self._label_dist if
                                 label in available_labels}

        label_extract_weights_sum = sum(extract_item for extract_item in label_extract_weights.values())
        label_extract_ratios = {label: (label_extract_weights[label] / label_extract_weights_sum, 0, count) for
                                label in label_extract_weights}

        # Distribute the the label of the patches to extract between the available labels.
        #
        label_extract_counts = dptpopulation.distribute_population(population=count, ratios=label_extract_ratios)
        # Calculate the list of source images per label.
        #
        label_source_counts = {label: 0 for label in self._label_dist}
        for sampler in self._patch_samplers.values():
            for label in sampler.labels:
                if label in self._label_dist:
                    label_source_counts[label] += 1

        # Distribute each patches to extract from each label between the available samplers.
        #
        counts_per_sampler = dict()
        for label in label_extract_counts:
            # Only consider the labels that are sampled in this round. Sampling 0 number of a label causes exception in the population distribution.
            #
            label_count = label_extract_counts[label]
            if 0 < label_count:
                source_count = label_source_counts[label]
                # Calculate distribution.
                #
                label_ratios = {
                sampler: (1.0 / source_count, 0, label_count) if label in sampler.labels else (0.0, 0, 0) for
                sampler in self._patch_samplers.values()}
                label_extract_per_source = dptpopulation.distribute_population(population=label_count,
                                                                               ratios=label_ratios)

                # Rearrange data into per sampler layout. Only store the positive values to prevent unnecessary messages with 0 patches to extract.
                #
                for i, sampler in enumerate(self._patch_samplers.values()):
                    if 0 < label_extract_per_source[sampler]:
                        if i in counts_per_sampler:
                            counts_per_sampler[i][label] = label_extract_per_source[sampler]
                        else:
                            counts_per_sampler[i] = {label: label_extract_per_source[sampler]}
        # Return the calculated patch distribution.
        #
        return counts_per_sampler

    def error(self, message):
        """
        Log an error that occurred outside the BatchSampler to save the reason of imminent shutdown.

        Args:
            message (str): Message to log.
        """

        self._logger.error('{tag}{message}'.format(tag=self._log_name_tag, message=message))

    def step(self):
        """
        Clear the patch samplers and create new ones with randomized sources.
        """

        # remove old sources
        for i in list(self._patch_samplers):
            sampler = self._patch_samplers.pop(i)
            del sampler

        new_source_items = self._samplesources()

        for i, source_item in enumerate(sorted(new_source_items)):
            self._patch_samplers[i] = dptpatchsampler.PatchSampler(patch_source=source_item,
                                                                   create_stat=self._create_stats,
                                                                   mask_spacing=self._mask_spacing,
                                                                   spacing_tolerance=self._spacing_tolerance,
                                                                   input_channels=self._input_channels,
                                                                   label_mode=self._label_mode)

        self._generator_samples = self._flatten_dist(self._patchdistribution(self._iterations))
        self.reset_sampler_indices()

    def batch(self, batch_size):
        """
        Collect a batch of patches.

        Args:
            batch_size (int): Batch size.

        Returns:
            (dict): Dictionary {spacing: {'patches': patch array, 'labels': label array}} with as keys the pixel spacings where the patches were taken.
        """

        self._logger.debug('{tag}Sampling {count} patches'.format(tag=self._log_name_tag, count=batch_size))

        # Prepare result arrays.
        #
        sample_counts = self._patchdistribution(count=batch_size)

        # patches_dtype = np.float32 if self._data_adapter.normalized else np.uint8
        # labels_dtype = np.float32 if self._data_adapter.onehot else np.uint8

        batch_dict = {spacing: {} for spacing in self._patch_shapes}
        for spacing in self._patch_shapes:
            patches_shape = (batch_size,) + tuple(self._patch_shapes[spacing]) + (len(self._input_channels),)
            batch_dict[spacing]['patches'] = np.zeros(shape=patches_shape)

            labels_shape = (batch_size,)

            if self._label_mode != 'central':
                labels_shape = labels_shape + tuple(self._patch_shapes[spacing])

            if self._data_adapter and self._data_adapter.onehot:
                labels_shape = labels_shape + (self._data_adapter.onehot,)

            batch_dict[spacing]['labels'] = np.zeros(shape=labels_shape)

            # Add weight arrays if necessary.
            #
            if self._data_adapter and self._data_adapter.weights:
                weights_shape = (batch_size,) + self._patch_shapes[spacing]
                batch_dict[spacing]['weights'] = np.empty(shape=weights_shape, dtype=np.float32)

        # Construct batch.
        #
        first_patch_index = 0
        for i in sample_counts:
            # Extract patches.
            #
            source_item = self._patch_samplers[i]
            patch_counts = sample_counts[source_item]
            total_patch_count = sum(count for count in patch_counts.values())

            if self._data_adapter:
                extract_shapes = self._data_adapter.shapes(target_shapes=self._patch_shapes)
            else:
                extract_shapes = self._patch_shapes

            self._logger.debug(
                '{tag}Extracting patches of shapes {shape} with {dist} distribution from \'{source}\''.format(
                    tag=self._log_name_tag,
                    shape=self._patch_shapes,
                    dist=patch_counts,
                    source=source_item.image))

            patch_dict = self._patch_samplers[source_item].sample(counts=patch_counts, shapes=extract_shapes)

            # Adapt the extracted data.
            #
            if self._data_adapter:
                self._data_adapter.adapt(patches=patch_dict, shapes=self._patch_shapes, randomize=True)

            # Push extracted patches, labels, and weights to the batch.
            #
            for spacing in patch_dict:
                batch_dict[spacing]['patches'][first_patch_index: first_patch_index + total_patch_count] = \
                patch_dict[spacing]['patches']
                batch_dict[spacing]['labels'][first_patch_index: first_patch_index + total_patch_count] = \
                patch_dict[spacing]['labels']

                if self._data_adapter and self._data_adapter.weights:
                    batch_dict[spacing]['weights'][first_patch_index: first_patch_index + total_patch_count] = \
                    patch_dict[spacing]['weights']

            first_patch_index += total_patch_count
        # Return the assembled batch of patches and labels.
        #
        return batch_dict

    def _flatten_dist(self, samples):
        sampler_list = []
        for sampler, label_dict in samples.items():
            for label, count in label_dict.items():
                sampler_list.extend([(sampler, label) for _ in range(count)])
        np.random.shuffle(sampler_list)
        return sampler_list

    def reset_sampler_indices(self):
        for sampler in self._patch_samplers.values():
            sampler.reset_sampler()

    def __len__(self):
        return self._iterations

    def __getitem__(self, item):
        nr, label = self._generator_samples[item]
        patch, mask = self._patch_samplers[nr].single_sample(item, label=label, shape=self._shape, spacing=self._spacing)
        weights = np.asarray(mask > 0, dtype=np.float32)
        if self._label_map:
            mask = self._map_classes(mask)
        return patch, mask, weights



