from typing import Dict, Tuple

import numpy as np
import skimage.color
import cv2
from wholeslidedata.samplers.utils import (block, crop_data,
                                           one_hot_encoding)


class SampleCallback:
    """Pass through Callback on samples"""

    def __init__(self):
        pass

    def __call__(
            self, x_patch: np.ndarray, y_patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return x_patch, y_patch

    def reset(self):
        pass



class BatchCallback:
    """Pass through Callback on batches"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(
            self, x_batch: np.ndarray, y_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return x_batch, y_batch

    def reset(self):
        pass


class BlockShapedSampleCallback(SampleCallback):
    """Reshapes samples into blocks"""

    def __init__(self, nrows, ncols):
        self._nrows = nrows
        self._ncols = ncols

    def __call__(self, x_patch, y_patch):
        x_patches = block(x_patch, self._nrows, self._ncols)
        y_patches = block(y_patch, self._nrows, self._ncols)
        return x_patches, y_patches


class BackgroundMaskingCallback(SampleCallback):
    def __init__(self):
        super().__init__()

    def __call__(self, x_patch: np.ndarray, y_patch: np.ndarray):
        # Create a 3D mask from the 2D mask
        mask_3d = np.repeat(y_patch[:, :, np.newaxis], 3, axis=2)
        
        # Set image background to black or white based on the mask
        #x_patch[mask_3d == 0] = np.random.randint(2) * 255
        x_patch[mask_3d == 0] = 255
        
        # Set certain mask values to zero if needed
        y_patch[y_patch == 1] = 0
        
        return x_patch, y_patch

    def reset(self):
        pass


class OneHotEncodingSampleCallbackUpdate(SampleCallback):
    """One-hot encodes y sample"""

    def __init__(self, labels, ignore_zero=True):
        self._ignore_zero = ignore_zero
        self._label_map = {label.name: label.value for label in labels}

    def __call__(self, x_patch, y_patch):
        y_patch[y_patch == 0] = 1
        masks = [(y_patch == v) for v in range(1, 8)]
        y_patch = np.stack(masks, axis=-1).astype(np.float32)

        return x_patch, y_patch
    

class OneHotEncodingSampleCallback(SampleCallback):
    """One-hot encodes y sample"""

    def __init__(self, labels, ignore_zero=True):
        self._ignore_zero = ignore_zero
        self._label_map = {label.name: label.value for label in labels}

    def __call__(self, x_patch, y_patch):
        y_patch = one_hot_encoding(y_patch, self._label_map, self._ignore_zero)
        return x_patch, y_patch


class ReshapeSampleCallback(SampleCallback):
    """Flattens first and second dimension in y sample"""

    def __call__(self, x_patch, y_patch):
        shape = y_patch.shape
        y_patch = y_patch.reshape(shape[0] * shape[1], -1).squeeze()
        return x_patch, y_patch


class ChannelsFirstSampleCallback(SampleCallback):
    """Tranposes x sample from channels last to channels first"""

    def __call__(self, x_patch, y_patch):
        x_patch = x_patch.transpose(2, 0, 1)
        return x_patch, y_patch


class CropSampleCallback(SampleCallback):
    """Crops y patch to fit output shape"""

    def __init__(self, output_shape):
        self._output_shape = output_shape

    def __call__(self, x_patch, y_patch):
        y_patch = self._fit_data(y_patch)
        return x_patch, y_patch

    def _fit_data(self, y_patch):
        # cropping
        if y_patch.shape != self._output_shape:
            y_patch = crop_data(y_patch, self._output_shape)
        # Reshape
        return y_patch


class SwitchChannelsSampleCallback(SampleCallback):
    def __call__(self, x_patch, y_patch):
        x_patch = x_patch.transpose((2, 0, 1)).astype(np.float32)
        y_patch = y_patch.transpose((2, 0, 1)).astype(np.float32)
        return x_patch, y_patch
    

class NormalizeSampleCallback(BatchCallback):
    def __call__(self, x_patch, y_patch):
        return x_patch/255.0, y_patch
    

class NormalizeSampleCallback2(SampleCallback):
    def __call__(self, x_patch, y_patch):
        x_patch = x_patch.astype(np.float32)  # Explicitly cast to float32
        x_patch /= 255.0
        return x_patch, y_patch
    

class ImageNetNormalizeSampleCallback(SampleCallback):
    def __init__(self):
        super().__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __call__(self, x_patch: np.ndarray, y_patch: np.ndarray):
        # Normalize each channel of the image
        x_patch = x_patch.astype(np.float32) / 255.0  # Convert to float and scale
        x_patch -= self.mean[:, None, None]  # Subtract mean
        x_patch /= self.std[:, None, None]  # Divide by std deviation
        return x_patch, y_patch

    def reset(self):
        pass
    
    
class HedAugmentationBatchCallback(BatchCallback):
    def __init__(self, hem=0.02, eos=0.02, dab=0.02, probability=0.5):
        self._hem = hem
        self._eos = eos
        self._dab = dab
        self._probability = probability

    def __call__(self, x_batch: np.ndarray, y_batch: np.ndarray):
        batch_size = len(x_batch)
        
        probs = np.random.uniform(size=batch_size)
        if np.all(probs > self._probability):
            return x_batch, y_batch

        _type = type(x_batch)
        x_batch = np.array(x_batch)

        x_batch_hed = skimage.color.rgb2hed(x_batch / 255)

        h = np.random.uniform(low=-self._hem, high=self._hem, size=(batch_size,))
        e = np.random.uniform(low=-self._eos, high=self._eos, size=(batch_size,))
        d = np.random.uniform(low=-self._dab, high=self._dab, size=(batch_size,))

        h *= np.random.randint(2, size=(batch_size))
        e *= np.random.randint(2, size=(batch_size))
        d *= np.random.randint(2, size=(batch_size))

        for i, (hv, ev, db) in enumerate(zip(h, e, d)):
            if probs[i] > self._probability:
                continue
            x_batch_hed[i, ..., 0] += hv
            x_batch_hed[i, ..., 1] += ev
            x_batch_hed[i, ..., 2] += db

        ihc_rgb = skimage.color.hed2rgb(x_batch_hed)
        ihc = np.clip(a=ihc_rgb * 255, a_min=0, a_max=255)
        return _type(ihc), y_batch

class DeepSupervisionBatchCallback(BatchCallback):
    """ Generates downsampled representation from y_batch. 
        This Batch callback is not compatible with one-hot-encoding
    """

    def __init__(self, sizes):
        self._sizes = sizes

    def __call__(self, x_batch: np.ndarray, y_batch: np.ndarray):
        y_batch = np.array(y_batch)
        resized_y_batches = [self.resize(y_batch, size) for size in self._sizes]
        return x_batch, y_batch, *resized_y_batches
    

    def resize(self, y_batch, size):
        return cv2.resize(y_batch.transpose(1,2,0), 
                          size, 0, 0, 
                          interpolation = cv2.INTER_NEAREST).transpose(2,0,1)

class Resolver(BatchCallback):
    """Resolves shape of batch"""

    def __init__(self, return_dict=False):
        self._return_dict = return_dict

    def __call__(self, x_batch: np.ndarray, y_batch: np.ndarray):
        return self._resolve(x_batch, y_batch)

    def _resolve(self, x, y):
        x = list(map(self._resolve_samples, x))
        y = list(map(self._resolve_samples, y))
        if self._return_dict:
            return {"x": x, "y": y}
        return x, y

    def _resolve_samples(self, samples):
        out_samples = []
        for _, shapes in samples.items():
            for _, shape in shapes.items():
                out_samples.append(shape)

        if len(out_samples) == 1:
            return out_samples[0]
        return out_samples
