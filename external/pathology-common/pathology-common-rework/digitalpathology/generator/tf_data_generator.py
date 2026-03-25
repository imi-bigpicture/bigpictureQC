from functools import partial

import tensorflow as tf
import numpy as np


class TFDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_gen, augmentations_pipeline=None, batch_size=8, reset_samplers=True):
        self._data_gen = data_gen
        if augmentations_pipeline:
            self._aug_fn = partial(self.augment_fn, transform=augmentations_pipeline)
        else:
            self._aug_fn = None
        self._batch_size = batch_size
        self._reset = reset_samplers
        self._counter = 0
        self._num_classes = len(self._data_gen.labels)
        self._data_gen.step()

    def __getitem__(self, index):
        batch, masks, weights = self._preprocess_batch(index)
        if weights is not None:
            return batch, masks, weights
        else:
            return batch, masks

    def __len__(self):
        return self._data_gen._iterations // self._batch_size

    def augment_fn(self, patch, mask, weight, transform):
        transformed = transform(image=patch, masks=[mask, weight])
        return transformed["image"], transformed["masks"][0], transformed["masks"][1]

    def on_epoch_end(self):
        if self._reset:
            self._data_gen.reset_sampler_indices()

    def _preprocess_batch(self, index):
        patches, masks, weights = [], [], []
        patch_ind = index * self._batch_size
        for i in range(patch_ind, patch_ind + self._batch_size):
            patch, mask, weight = self._data_gen[i]
            if self._aug_fn:
                patch, mask, weight = self._aug_fn(patch, mask, weight)
            patches.append(patch / 255.0)
            if self._num_classes > 2:
                weight = np.reshape(weight, (weight.shape[0] * weight.shape[1]))
                mask = tf.keras.utils.to_categorical(mask, num_classes=self._num_classes, dtype='float32')
                mask = np.reshape(mask, (mask.shape[0] * mask.shape[1], self._num_classes))
                weights.append(weight)
            masks.append(mask)


        patches = np.stack(patches, axis=0)
        weights = np.stack(weights, axis=0)
        if self._num_classes > 2:
            return patches, np.stack(masks, axis=0)[:, :, :].astype(np.float32), weights
        else:
            return patches, np.stack(masks, axis=0)[:, :, :, np.newaxis].astype(np.float32), None

