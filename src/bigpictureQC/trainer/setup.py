"""Setup utilities for model initialization, loss functions, optimizers, and schedulers."""
import logging
from typing import Any, Dict, List, Tuple

import torch
from collections import OrderedDict
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric
from segmentation_models_pytorch import (
    Unet, UnetPlusPlus, MAnet, Linknet, FPN,
    PSPNet, PAN, DeepLabV3, DeepLabV3Plus
)
from segmentation_models_pytorch.utils.metrics import Fscore, IoU, Accuracy
from torch.nn import BCELoss
from torch.optim import Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, SequentialLR, LinearLR
)

from bigpictureQC.utils.config import Config

logger = logging.getLogger(__name__)


# Architecture name mapping for case-insensitive lookup
ARCHITECTURE_MAPPING = {
    'unet': 'Unet',
    'unetplusplus': 'UnetPlusPlus',
    'unet++': 'UnetPlusPlus',
    'manet': 'MAnet',
    'linknet': 'Linknet',
    'fpn': 'FPN',
    'pspnet': 'PSPNet',
    'pan': 'PAN',
    'deeplabv3': 'DeepLabV3',
    'deeplabv3plus': 'DeepLabV3Plus',
    'deeplabv3+': 'DeepLabV3Plus',
}

# Architectures that support decoder_use_batchnorm parameter
ARCHS_WITH_DECODER_BN = {'Unet', 'UnetPlusPlus', 'Linknet', 'MAnet'}


def setup_device(device_name: str) -> torch.device:
    """
    Setup CUDA device if available, otherwise CPU.
    
    Args:
        device_name: Requested device ('cuda' or 'cpu')
        
    Returns:
        Torch device object
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def normalize_architecture_name(name: str) -> str:
    """
    Normalize architecture name to match SMP naming convention.
    
    Args:
        name: Architecture name in any case format
        
    Returns:
        Normalized architecture name
        
    Raises:
        ValueError: If architecture name is not recognized
    
    Example:
        >>> normalize_architecture_name('unet++')
        'UnetPlusPlus'
    """
    normalized = name.replace('+', 'plus').lower()
    mapped_name = ARCHITECTURE_MAPPING.get(normalized)
    
    if mapped_name is None:
        raise ValueError(
            f"Unknown architecture: '{name}'. "
            f"Supported: {list(ARCHITECTURE_MAPPING.keys())}"
        )
    
    return mapped_name


def initialize_model(cfg: Config, device: torch.device) -> torch.nn.Module:
    """
    Initialize segmentation model from configuration.
    
    Args:
        cfg: Configuration object
        device: Device to place model on
        
    Returns:
        Initialized PyTorch model
        
    Raises:
        ValueError: If model configuration is invalid
    """
    model_cfg = cfg.get('model_config')
    
    # Determine number of classes
    classes = model_cfg.get('classes', [])
    if not classes:
        raise ValueError("No classes specified in model_config")
    
    num_classes = 1 if len(classes) == 1 else len(classes) + 1
    activation = 'sigmoid' if num_classes == 1 else 'softmax2d'
    
    logger.info(
        f"Initializing model for {len(classes)} artifact classes "
        f"(num_classes={num_classes}, activation={activation})"
    )
    
    # Setup auxiliary classification head if specified
    aux_params = None
    if model_cfg.get('aux_params'):
        aux_config = model_cfg.get('aux_params')
        aux_params = {
            'pooling': aux_config.get('pooling', 'max'),
            'dropout': aux_config.get('dropout', 0.4),
            'activation': 'sigmoid',  # Always sigmoid for multi-label classification
            'classes': num_classes,
        }
        logger.info(f"Using auxiliary classification head: {aux_params}")
    
    # Get architecture
    architecture = normalize_architecture_name(model_cfg.get('architecture'))
    
    # Common parameters for all SMP models
    common_kwargs = {
        'encoder_name': model_cfg.get('encoder_name'),
        'encoder_weights': model_cfg.get('encoder_weights'),
        'classes': num_classes,
        'activation': activation,
        'aux_params': aux_params,
    }
    
    # Add decoder_use_batchnorm for architectures that support it
    if architecture in ARCHS_WITH_DECODER_BN:
        common_kwargs['decoder_use_batchnorm'] = model_cfg.get('use_batchnorm', True)
    
    # Instantiate model
    model_classes = {
        'Unet': Unet,
        'UnetPlusPlus': UnetPlusPlus,
        'MAnet': MAnet,
        'Linknet': Linknet,
        'FPN': FPN,
        'PSPNet': PSPNet,
        'PAN': PAN,
        'DeepLabV3': DeepLabV3,
        'DeepLabV3Plus': DeepLabV3Plus,
    }
    
    model_class = model_classes.get(architecture)
    if model_class is None:
        raise ValueError(f"Architecture '{architecture}' not implemented")
    
    model = model_class(**common_kwargs)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model initialized: {architecture} with {model_cfg.get('encoder_name')} encoder. "
        f"Total params: {total_params:,}, Trainable: {trainable_params:,}"
    )
    
    return model


def initialize_artifacts_network(
    config: Config,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load pretrained artifact detection model from checkpoint.
    
    Used for inference or fine-tuning from a saved checkpoint.
    
    Args:
        config: Configuration object with 'artifact_network' key pointing to checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = initialize_model(config, device)
    
    checkpoint_path = config.get('artifact_network')
    if not checkpoint_path:
        raise ValueError("'artifact_network' path not specified in config")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded pretrained model from {checkpoint_path}")
    
    return model


def configure_loss(cfg: Config) -> torch.nn.Module:
    """
    Configure segmentation loss function.
    
    IMPORTANT: Assumes model outputs are already probabilities (via SMP activation),
    so we disable internal softmax/sigmoid in loss functions.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Configured loss function
        
    Raises:
        ValueError: If loss type is not supported
    """
    include_background = cfg.loss_config.get('include_background', True)
    batch = cfg.model_config.get('batch', True)
    
    losses = {
        'dice_loss': DiceLoss(
            include_background=include_background,
            softmax=False,  # Model already applies activation
            sigmoid=False,
            batch=batch
        ),
        'generalized_dice': GeneralizedDiceLoss(
            include_background=include_background,
            softmax=False,
            sigmoid=False,
            batch=batch
        ),
    }
    
    loss_type = cfg.loss_config.get('loss_type', 'dice_loss')
    if loss_type not in losses:
        raise ValueError(
            f"Unsupported loss: '{loss_type}'. "
            f"Available: {list(losses.keys())}"
        )
    
    logger.info(f"Using {loss_type} loss (include_background={include_background})")
    
    return losses[loss_type]


def configure_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Configure optimizer from configuration.
    
    Args:
        cfg: Configuration object
        model: Model to optimize
        
    Returns:
        Configured optimizer
        
    Raises:
        ValueError: If optimizer type is not supported
    """
    optimizer_type = cfg.train_config.get('optimizer_type', 'adam').lower()
    lr = float(cfg.train_config.get('lr', 1e-3))
    weight_decay = float(cfg.train_config.get('weight_decay', 0.0))
    
    optimizers = {
        'adam': lambda: Adam(model.parameters(), lr=lr),
        'adamw': lambda: AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
        'rmsprop': lambda: RMSprop(model.parameters(), lr=lr),
    }
    
    optimizer_func = optimizers.get(optimizer_type)
    if optimizer_func is None:
        raise ValueError(
            f"Unsupported optimizer: '{optimizer_type}'. "
            f"Available: {list(optimizers.keys())}"
        )
    
    optimizer = optimizer_func()
    logger.info(f"Using {optimizer_type} optimizer (lr={lr}, weight_decay={weight_decay})")
    
    return optimizer


def configure_scheduler(
    cfg: Config,
    optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Configure learning rate scheduler.
    
    Supports:
    - steplr: Step decay every N epochs
    - cosine: Cosine annealing
    - warmup_cosine: Linear warmup followed by cosine annealing
    
    Args:
        cfg: Configuration object
        optimizer: Optimizer to schedule
        
    Returns:
        Configured scheduler
        
    Raises:
        ValueError: If scheduler type is not supported
    """
    scheduler_type = cfg.scheduler_config.get('type', 'cosine').lower()
    epochs = cfg.train_config.get('epochs', 100)
    
    if scheduler_type == 'steplr':
        step_size = cfg.scheduler_config.get('step_size', 15)
        gamma = cfg.scheduler_config.get('gamma', 0.5)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"Using StepLR (step_size={step_size}, gamma={gamma})")
    
    elif scheduler_type == 'cosine':
        eta_min = float(cfg.scheduler_config.get('eta_min', 1e-6))
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        logger.info(f"Using CosineAnnealingLR (T_max={epochs}, eta_min={eta_min})")
    
    elif scheduler_type == 'warmup_cosine':
        warmup_epochs = cfg.scheduler_config.get('warmup_epochs', 5)
        eta_min = float(cfg.scheduler_config.get('eta_min', 1e-6))
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min),
            ],
            milestones=[warmup_epochs],
        )
        logger.info(
            f"Using warmup + cosine annealing "
            f"(warmup={warmup_epochs}, T_max={epochs}, eta_min={eta_min})"
        )
    
    else:
        raise ValueError(
            f"Unsupported scheduler: '{scheduler_type}'. "
            f"Available: ['steplr', 'cosine', 'warmup_cosine']"
        )
    
    return scheduler


def configure_metrics(cfg: Config) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    Configure metrics for training monitoring.
    
    IMPORTANT: For segmentation metrics expecting discrete masks (DiceMetric, IoU, Fscore),
    pass one-hot encoded predictions. For loss-like metrics, pass probabilities.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (metrics_list, metrics_dict) where:
            - metrics_list: List of (metric_instance, name, type) tuples
            - metrics_dict: Dictionary of name -> metric_instance
            
    Raises:
        ValueError: If metric name is not supported
    """
    include_background = cfg.get('include_background', True)
    ignore_channels = cfg.get('ignore_channels', [])
    threshold = float(cfg.get('rounding', 0.5))
    
    # Define available metrics
    metrics_registry = {
        # Stateful metric (accumulates across batches)
        'dice_score': lambda: DiceMetric(
            include_background=include_background,
            reduction='mean'
        ),
        
        # SMP metrics (per-batch)
        'dice_smp': lambda: Fscore(threshold=threshold, ignore_channels=ignore_channels),
        'iou_smp': lambda: IoU(threshold=threshold, ignore_channels=ignore_channels),
        'accuracy': lambda: Accuracy(ignore_channels=ignore_channels),
        
        # Loss-based metrics (probabilities as input)
        'generalized_dice': lambda: GeneralizedDiceLoss(
            include_background=include_background,
            softmax=False,  # Model outputs are probabilities
            batch=True
        ),
        'dice_loss': lambda: DiceLoss(
            include_background=include_background,
            softmax=False,
            batch=True
        ),
        
        # Classification metric
        'cross_entropy': lambda: BCELoss(reduction='mean'),
    }
    
    # Get requested metrics from config
    metrics_config = cfg.get('metrics', [])
    if not metrics_config:
        raise ValueError("No metrics specified in configuration")
    
    configured_metrics = []
    metrics_funcs = {}
    
    for metric_spec in metrics_config:
        metric_name = metric_spec.get('name')
        metric_type = metric_spec.get('type')  # 'segmentation' or 'classification'
        
        if metric_name not in metrics_registry:
            raise ValueError(
                f"Unsupported metric: '{metric_name}'. "
                f"Available: {list(metrics_registry.keys())}"
            )
        
        metric_instance = metrics_registry[metric_name]()
        configured_metrics.append((metric_instance, metric_name, metric_type))
        metrics_funcs[metric_name] = metric_instance
        
        logger.info(f"Configured {metric_type} metric: {metric_name}")
    
    return configured_metrics, metrics_funcs