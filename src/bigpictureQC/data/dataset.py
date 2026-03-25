"""Dataset module for artifact detection in whole slide images."""
import numpy as np
import pandas as pd
import skimage.io
from torch.utils.data import Dataset
from utils.labels import class_labels
from typing import Optional, List, Tuple, Any

class NormalizationStrategy:
    """Encapsulates image normalization strategies."""
    
    @staticmethod
    def simple(image: np.ndarray) -> np.ndarray:
        """Simple [0, 1] normalization."""
        return (image / 255.0).astype(np.float32)
    
    @staticmethod
    def inception(image: np.ndarray) -> np.ndarray:
        """Inception-style normalization: (x/255 - 0.5) / 0.5."""
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return (image - mean) / std
    
    @staticmethod
    def imagenet(image: np.ndarray) -> np.ndarray:
        """ImageNet normalization with mean and std."""
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
        image = image.astype(np.float32)
        return (image - mean) / std


class ArtifactDataset(Dataset):
    """
    PyTorch Dataset for histopathology artifact detection.
    
    Loads image patches and corresponding segmentation masks from CSV file.

    Args:
        df_path: Path to CSV file with columns ['path', 'target'] pointing to images and masks
        classes: List of artifact class names to detect (e.g., ['Ink', 'Fold'])
        transform: Optional albumentations transform for data augmentation
        normalize: Normalization strategy: 'simple', 'inception', 'imagenet', or None
        ink_filters: Optional list of RGB threshold tuples for ink artifact filtering
    
    Returns:
        Tuple of (image, mask) where:
            - image: float32 tensor of shape (C, H, W)
            - mask: float32 tensor of shape (num_classes, H, W) with binary masks per class
    """

    def __init__(
        self,
        df_path: str,
        classes: List[str],
        transform: Optional[Any] = None,
        normalize: Optional[str] = None,
        ink_filters: Optional[List[Tuple]] = None
    ):
        self.df = pd.read_csv(df_path)
        self._validate_dataframe()
        
        self.classes = classes
        # For single class: binary segmentation (1 channel)
        # For multi-class: background + N classes (N+1 channels)
        self.num_classes = 1 if len(classes) == 1 else len(classes) + 1
        
        # Map class names to integer labels
        self.class_values = [class_labels[c] for c in classes]
        
        self.transform = transform
        self.normalize = normalize
        self.ink_filters = ink_filters
        
        self._normalizer = self._get_normalizer(normalize)

    def _validate_dataframe(self) -> None:
        """Validate that dataframe has required columns."""
        required_cols = {'path', 'target'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

    def _get_normalizer(self, normalize: Optional[str]):
        """Get normalization function based on strategy name."""
        if not normalize:
            return None
        
        normalizers = {
            'simple': NormalizationStrategy.simple,
            'inception': NormalizationStrategy.inception,
            'imagenet': NormalizationStrategy.imagenet,
        }
        
        norm_func = normalizers.get(normalize)
        if norm_func is None:
            raise ValueError(
                f"Unknown normalization: {normalize}. "
                f"Choose from: {list(normalizers.keys())}"
            )
        return norm_func

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process image-mask pair at index i."""
        img_path = self.df.iloc[i]['path']
        mask_path = self.df.iloc[i]['target']
        
        # Load images
        image = skimage.io.imread(img_path)
        mask = skimage.io.imread(mask_path)
        
        # Set background pixels in image based on mask
        if self.num_classes > 1:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            image[mask_3d == 0] = 255  # set unannotated pixels in patches to white
        
        mask[mask == 1] = 0

        # Apply ink filtering if specified
        if self.ink_filters:
            mask = self._apply_ink_filters(image, mask)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        # Convert mask to one-hot encoding
        mask = self._mask_to_onehot(mask)
        
        # Normalize image
        if self._normalizer:
            image = self._normalizer(image)
        else:
            image = image.astype(np.float32)
        
        # Convert to CHW format 
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        
        return image.astype(np.float32), mask.astype(np.float32)

    def _apply_ink_filters(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply RGB threshold filters for ink artifact detection."""
        maps = []
        
        for rgb_filter in self.ink_filters:
            # Each filter is ((r_min, r_max), (g_min, g_max), (b_min, b_max))
            r_match = (rgb_filter[0][0] <= image[:, :, 0]) & (image[:, :, 0] < rgb_filter[0][1])
            g_match = (rgb_filter[1][0] <= image[:, :, 1]) & (image[:, :, 1] < rgb_filter[1][1])
            b_match = (rgb_filter[2][0] <= image[:, :, 2]) & (image[:, :, 2] < rgb_filter[2][1])
            
            # Pixels must match RGB filter AND be annotated as Ink
            is_ink_annotation = (mask == class_labels['Ink'])
            matches_filter = r_match & g_match & b_match & is_ink_annotation
            maps.append(matches_filter)
        
        # Combine all filter results
        if maps:
            ink_mask_filtered = np.any(np.stack(maps, axis=-1), axis=-1)
            ink_annotation = (mask == class_labels['Ink'])
            
            # Remove ink pixels that don't match any filter
            mask[ink_annotation & ~ink_mask_filtered] = 0
        
        return mask

    def _mask_to_onehot(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert integer mask to one-hot encoded binary masks.
        
        For binary segmentation (num_classes=1): single channel with 0/1
        For multi-class (num_classes>1): N channels, one per class including background
        """
        if self.num_classes > 1:
            # Remap background from 0 to 1 for one-hot encoding
            # Class indices: background=1, class1=2, class2=3, ...
            mask = np.where(mask == 0, 1, mask)
            
            # Create binary mask for each class
            onehot_masks = []
            for class_idx in range(1, self.num_classes + 1):
                onehot_masks.append((mask == class_idx).astype(np.float32))
            
            return np.stack(onehot_masks, axis=-1)
        else:
            # Binary segmentation: single class
            binary_mask = (mask == self.class_values[0]).astype(np.float32)
            return binary_mask[:, :, np.newaxis]

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.df)