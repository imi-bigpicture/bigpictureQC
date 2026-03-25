import random
from typing import Tuple

import albumentations as A
import numpy as np

from wholeslidedata.samplers.callbacks import BatchCallback, SampleCallback



class AlbumentationsDetectionSampleCallback(SampleCallback):
    def __init__(self, augmentations):
        random.seed()
        self._augmentations = A.Compose(
            [
                getattr(A, class_name)(**params)
                for augmentation in augmentations
                for class_name, params in augmentation.items()
            ],
            bbox_params=A.BboxParams(format="pascal_voc"),
        )

    def __call__(
            self, x_patch: np.ndarray, y_patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_boxes = y_patch[~np.all(y_patch == 0, axis=-1)]
        augmented = self._augmentations(image=x_patch, bboxes=y_boxes)
        if y_boxes.size != 0:
            y_patch[~np.all(y_patch == 0, axis=-1)] = augmented["bboxes"]
        return augmented["image"], y_patch

    def reset(self):
        pass


class AlbumentationsSegmentationBatchCallback(BatchCallback):

    def __init__(self, augmentations):
        random.seed()
        self.augmentations = augmentations

    def __call__(
            self, x_batch: np.ndarray, y_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _type_x, _type_y = type(x_batch), type(y_batch)
        x_batch, y_batch = np.array(x_batch), np.array(y_batch)

        x_transformed, y_transformed = np.zeros(x_batch.shape), np.zeros(y_batch.shape)
        for batch_index in range(x_batch.shape[0]):
            if x_batch.ndim == 5:
                to_be_augmented = {"image": x_batch[batch_index][0], "mask": y_batch[batch_index][0]}
                additional_targets = {}
                for dim in range(1, x_batch.shape[1]):
                    additional_targets.update({f"image{dim}": "image", f"mask{dim}": "mask"})
                    to_be_augmented.update(
                        {f"image{dim}": x_batch[batch_index][dim], f"mask{dim}": y_batch[batch_index][dim]})
            else:
                additional_targets = None
                to_be_augmented = {"image": x_batch[batch_index], "mask": y_batch[batch_index]}

            self._augmentations = A.Compose(
                [getattr(A, class_name)(**params) for augmentation in self.augmentations for class_name, params in
                 augmentation.items()], additional_targets=additional_targets
            )
            augmented = self._augmentations(**to_be_augmented)
            if x_batch.ndim == 5:
                x_transformed[batch_index][0] = augmented[f"image"]
                y_transformed[batch_index][0] = augmented[f"mask"]
                for dim in range(1, x_batch.shape[1]):
                    x_transformed[batch_index][dim] = augmented[f"image{dim}"]
                    y_transformed[batch_index][dim] = augmented[f"mask{dim}"]
            else:
                x_transformed[batch_index] = augmented["image"]
                y_transformed[batch_index] = augmented["mask"]
        if _type_x != type(x_transformed):
            x_transformed = _type_x(x_transformed)
        if _type_y != type(y_transformed):
            y_transformed = _type_y(y_transformed)
        return x_transformed, y_transformed

    def reset(self):
        pass


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
        print("Masking background")
        x_patch[mask_3d == 0] = 255
        
        # Set certain mask values to zero if needed
        y_patch[y_patch == 1] = 0
        
        return x_patch, y_patch

    def reset(self):
        pass


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

    
class NormalizeSampleCallback(BatchCallback):
    def __call__(self, x_patch, y_patch):
        return x_patch/255.0, y_patch
    

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


class CustomAlbumentationsSampleCallback(SampleCallback):
    def __init__(self, patch_size):
        self.transform = A.Compose([
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.25),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0),
                A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=0, p=1.0)
            ], p=0.25),
            A.OneOf([
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=10, p=1.0)
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
                A.RandomGamma(p=1.0)
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=200, sigma=20, alpha_affine=20, interpolation=1, border_mode=0, value=0, mask_value=0, p=1.0),
                A.GridDistortion(num_steps=10, distort_limit=0.3, interpolation=1, border_mode=0, value=0, mask_value=0, p=1.0)
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.ImageCompression(quality_lower=60, quality_upper=100, compression_type=0, p=1.0)
            ], p=0.5),
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size)
        ])

    def __call__(self, x_patch, y_patch):
        augmented = self.transform(image=x_patch, mask=y_patch)
        return augmented['image'], augmented['mask']

    def reset(self):
        pass