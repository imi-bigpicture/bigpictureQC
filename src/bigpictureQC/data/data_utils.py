"""Data loading utilities for training and validation."""

from typing import Any, Tuple, Optional
import logging

from torch.utils.data import DataLoader, RandomSampler
from wholeslidedata.iterators import create_batch_iterator

from bigpictureQC.data.dataset import ArtifactDataset
from bigpictureQC.data.transform import load_train_transform, load_valid_transform

logger = logging.getLogger(__name__)


def initialize_iterators(cfg) -> Tuple[Any, Any, Optional[Any]]:
    """
    Initialize WholeSlideData batch iterators for on-the-fly patch extraction.
    
    Args:
        cfg: Configuration object with batch_iterator_config section
    
    Returns:
        Tuple of (train_iterator, valid_iterator)
    """
    logger.info('Initializing data iterators...')
    
    user_config = cfg.get('batch_iterator_config.user_config_path')
    val_config = cfg.get('batch_iterator_config.val_config_path', user_config)
    
    train_iterator = create_batch_iterator(
        user_config=user_config,
        mode='training',
        number_of_batches=cfg.get('batch_iterator_config.train_batches')
    )
    logger.info(f"Train iterator size: {len(train_iterator)}")
    
    valid_iterator = create_batch_iterator(
        user_config=val_config,
        mode='validation',
        number_of_batches=cfg.get('batch_iterator_config.valid_batches')
    )
    logger.info(f"Valid iterator size: {len(valid_iterator)}")
    
    return train_iterator, valid_iterator


def initialize_dataset(cfg) -> Tuple[ArtifactDataset, ArtifactDataset, Optional[ArtifactDataset]]:
    """
    Initialize PyTorch datasets from CSV manifests.
    
    Args:
        cfg: Configuration object with data_config and model_config sections
    
    Returns:
        Tuple of (train_dataset, valid_dataset)
    """
    # Load transforms
    train_transform = load_train_transform(
        transform_type=cfg.data_config.get('transform'),
        patch_size=cfg.data_config.get('patch_size')
    )
    valid_transform = load_valid_transform(
        patch_size=cfg.data_config.get('patch_size')
    )
    
    # Common dataset parameters
    classes = cfg.model_config.get('classes')
    normalize = cfg.data_config.get('normalize')
    
    # Create datasets
    train_dataset = ArtifactDataset(
        df_path=cfg.data_config.get('train_data'),
        classes=classes,
        transform=train_transform,
        normalize=normalize,
        ink_filters=None  
    )
    
    valid_dataset = ArtifactDataset(
        df_path=cfg.data_config.get('valid_data'),
        classes=classes,
        transform=valid_transform,
        normalize=normalize,
        ink_filters=None
    )

    return train_dataset, valid_dataset


def initialize_dataloader(
    train_dataset: ArtifactDataset,
    valid_dataset: ArtifactDataset,
    cfg
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders with appropriate sampling strategies.
    
    Args:
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        cfg: Configuration object with data_config section
    
    Returns:
        Tuple of (train_loader, valid_loader)
    """
    batch_size = cfg.data_config.get('batch_size')
    workers = cfg.data_config.get('workers', 4)
    samples_per_epoch = cfg.data_config.get('train_samples_per_epoch')
    print(f"Samples per epoch: {samples_per_epoch}")
    
    # Training loader with optional epoch size limiting
    if samples_per_epoch:
        logger.info(f"Limiting training epoch to {samples_per_epoch} samples")
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=samples_per_epoch
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
    
    # Validation loader
    valid_batch_size = cfg.data_config.get('valid_batch_size', max(1, batch_size // 4))
    valid_samples = cfg.data_config.get('valid_samples_per_epoch')
    
    if valid_samples:
        logger.info(f"Limiting validation to {valid_samples} samples")
        val_sampler = RandomSampler(
            valid_dataset,
            replacement=True,
            num_samples=valid_samples
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            sampler=val_sampler,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
    else:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
    
    return train_loader, valid_loader


def get_data(cfg):
    """
    Main entry point for data loading.
    
    Returns either iterators (for on-the-fly extraction) or dataloaders (for pre-extracted patches).
    
    Args:
        cfg: Configuration object
    
    Returns:
        Tuple of (train, valid) data sources
    """
    if cfg.train_config.get('on_the_fly'):
        logger.info("Using on-the-fly patch extraction with iterators")
        return initialize_iterators(cfg)
    else:
        logger.info("Using pre-extracted patches with dataloaders")
        train_dataset, valid_dataset = initialize_dataset(cfg)
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Valid dataset size: {len(valid_dataset)}")
        
        return initialize_dataloader(train_dataset, valid_dataset, cfg)