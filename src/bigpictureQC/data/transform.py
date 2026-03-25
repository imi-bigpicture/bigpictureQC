"""Data augmentation transforms for histopathology images."""

from typing import Callable

import albumentations as A
from albumentations import Compose
import numpy as np


def load_train_transform(transform_type: str, patch_size: int = 512) -> Compose:
    """
    Load training augmentation pipeline based on preset name.
    
    Args:
        transform_type: Augmentation preset ('basic', 'light', 'medium', 'moderate', 'artifact')
        patch_size: Target patch size (height and width)
    
    Returns:
        Albumentations Compose transform pipeline
    """
    # Common geometric transforms
    geometric_base = [
        A.RandomRotate90(p=0.75),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    
    # Padding (always needed)
    padding = A.PadIfNeeded(patch_size, patch_size)
    
    transforms = {
        'basic': geometric_base + [padding],
        
        'light': geometric_base + [
            A.Rotate(limit=90, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.25),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=0, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.15, p=1.0),
            ], p=1.0),
            padding,
        ],
        
        'medium': geometric_base + [
            A.Rotate(limit=90, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.25),
            A.RandomScale(scale_limit=0.2, interpolation=2, p=1.0),  # Replaces deprecated RandomSizedCrop
            padding,
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.1),
            A.CoarseDropout(max_holes=12, max_height=12, max_width=12, fill_value=0, p=0.25),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=5, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.0, p=0.75),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=0.75),
                A.RandomGamma(gamma_limit=(75, 125), p=0.5),
            ], p=1.0),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.75),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.75),  # Replaces IAASharpen
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.75),
            ], p=1.0),
        ],
        
        'moderate': geometric_base + [
            A.Rotate(limit=90, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.25),
            A.RandomScale(scale_limit=0.2, interpolation=2, p=1.0),
            padding,
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0),
                A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=0, p=1.0),
            ], p=0.25),
            A.OneOf([
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=10, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=200, sigma=20, alpha_affine=20, p=1.0),
                A.GridDistortion(num_steps=10, distort_limit=0.3, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            ], p=0.5),
        ],
        
        'artifact': geometric_base + [
            A.Rotate(limit=90, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.25),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0),
                A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=0, p=1.0),
            ], p=0.25),
            A.OneOf([
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=10, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=200, sigma=20, alpha_affine=20, p=1.0),
                A.GridDistortion(num_steps=10, distort_limit=0.3, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            ], p=0.5),
            padding,
        ],
    }
    if transform_type is None:
        return A.Compose([padding])
    
    if isinstance(transform_type, str):
        key = transform_type.lower()
        if key in ("", "none", "no_aug", "no-aug", "noaugmentation", "no-augmentation"):
            return A.Compose([padding])
        transform_type = key 

    if transform_type not in transforms:
        raise ValueError(
            f"Unknown transform type: {transform_type}. "
            f"Choose from: {list(transforms.keys())}"
        )
    
    return A.Compose(transforms[transform_type])


def load_valid_transform(patch_size: int = 768) -> Compose:
    """Load validation/test transform (no augmentation, only padding)."""
    return A.Compose([A.PadIfNeeded(patch_size, patch_size)])


# ============================================================================
# Test-Time Augmentation (TTA) utilities
# ============================================================================

def augment_image_tta(image: np.ndarray) -> np.ndarray:
    """
    Apply 8-fold test-time augmentation (rotations + flips).
    
    Args:
        image: Input image of shape (H, W, C) or (C, H, W)
    
    Returns:
        Batch of 8 augmented images with shape (8, H, W, C) or (8, C, H, W)
    """
    # Determine format
    if image.shape[-1] in [1, 3, 4]:  # HWC format
        axes = (0, 1)
    else:  # CHW format
        axes = (1, 2)
    
    batch = np.empty((8,) + image.shape, dtype=image.dtype)
    batch[0] = image
    batch[1] = np.rot90(image, k=1, axes=axes)
    batch[2] = np.rot90(image, k=2, axes=axes)
    batch[3] = np.rot90(image, k=3, axes=axes)
    batch[4] = np.flip(image, axis=axes[1])  # Horizontal flip
    batch[5] = np.rot90(batch[4], k=1, axes=axes)
    batch[6] = np.rot90(batch[4], k=2, axes=axes)
    batch[7] = np.rot90(batch[4], k=3, axes=axes)
    
    return batch


def merge_tta_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    Merge TTA predictions by reverting augmentations and averaging.
    
    Args:
        predictions: Batch of 8 predictions with shape (8, H, W, C) or (8, C, H, W)
    
    Returns:
        Averaged prediction of shape (H, W, C) or (C, H, W)
    """
    # Determine format
    if predictions.shape[-1] in [1, 3, 4]:  # HWC format
        axes = (1, 2)
    else:  # CHW format
        axes = (2, 3)
    
    # Revert augmentations
    reverted = predictions.copy()
    reverted[1] = np.rot90(predictions[1], k=-1, axes=axes)
    reverted[2] = np.rot90(predictions[2], k=-2, axes=axes)
    reverted[3] = np.rot90(predictions[3], k=-3, axes=axes)
    reverted[4] = np.flip(predictions[4], axis=axes[1])
    reverted[5] = np.flip(np.rot90(predictions[5], k=-1, axes=axes), axis=axes[1])
    reverted[6] = np.flip(np.rot90(predictions[6], k=-2, axes=axes), axis=axes[1])
    reverted[7] = np.flip(np.rot90(predictions[7], k=-3, axes=axes), axis=axes[1])
    
    return np.mean(reverted, axis=0)