from functools import partial

import tensorflow as tf
import numpy as np


class TFClassificationGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_gen, augmentations_pipeline=None, batch_size=8, reset_samplers=True, n_classes=2):
        self._data_gen = data_gen
        if augmentations_pipeline:
            self._aug_fn = partial(self._augment_fn, transform=augmentations_pipeline)
        else:
            self._aug_fn = None
        self._batch_size = batch_size
        self._reset = reset_samplers
        self._num_classes = n_classes
        self._data_gen.step()

    def __getitem__(self, index):
        batch, masks = self._preprocess_batch(index)
        return batch, masks

    def __len__(self):
        return self._data_gen._iterations // self._batch_size

    def _augment_fn(self, patch, transform):
        transformed = transform(image=patch)
        return transformed["image"]

    def on_epoch_end(self):
        if self._reset:
            self._data_gen.reset_sampler_indices()

    def _preprocess_batch(self, index):
        patches = np.zeros((self._batch_size, self._data_gen.shape[0], self._data_gen.shape[1], 3), dtype=np.float32)
        masks = np.zeros(self._batch_size, dtype=np.float32)
        patch_ind = index * self._batch_size
        for ind, i in enumerate(range(patch_ind, patch_ind + self._batch_size)):
            patch, mask, weight = self._data_gen[i]
            if self._aug_fn:
                patch = self._aug_fn(patch)
            patches[ind] = patch / 255.0
            masks[ind] = mask
        masks = tf.keras.utils.to_categorical(masks, num_classes=self._num_classes, dtype='float32')
        return patches, masks
