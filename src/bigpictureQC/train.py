"""Main training script for BigPicture QC artifact detection."""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Fix import path issues
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bigpictureQC.utils.config import Config
from bigpictureQC.utils.common import seed_everything
from bigpictureQC.utils.tracking import initialize_wandb
from bigpictureQC.trainer.setup import (
    setup_device,
    initialize_model,
    configure_loss,
    configure_optimizer,
    configure_scheduler,
)
from bigpictureQC.trainer.trainer import Trainer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.info("Root logger test: this should be visible")


def main(cfg: Config) -> None:
    """
    Main training routine.
    
    Args:
        cfg: Configuration object containing all training parameters.
        
    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If training fails.
    """
    # Set random seeds for reproducibility
    seed = cfg.get('seed', 42)
    seed_everything(seed=seed)
    print(f"Random seed set to: {seed}")
    
    # Initialize experiment tracking
    tracker: Optional[object] = None
    if cfg.get('wandb_config.enable', False):
        wandb_key = os.environ.get("WANDB_API_KEY")
        if not wandb_key:
            print("Warning: WANDB_API_KEY not found in environment. Skipping tracking.")
        else:
            tracker = initialize_wandb(cfg, key=wandb_key, fold=0)
    
    print('Preparing training components...')
    
    # Setup device
    device_name = cfg.model_config.get('device', 'cuda')
    device = setup_device(device_name)
    print(f"Using device: {device}")
    
    # Initialize model
    model = initialize_model(cfg, device)
    print(f"Model initialized: {cfg.model_config.get('architecture')}")
    
    # Configure training components
    loss_fn = configure_loss(cfg)
    optimizer = configure_optimizer(cfg, model)
    scheduler = configure_scheduler(cfg, optimizer)
    
    # Calculate number of output classes
    classes = cfg.model_config.get('classes', [])
    num_classes = 1 if len(classes) == 1 else len(classes) + 1
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        cfg=cfg,
        tracker=tracker
    )
    
    # Compile trainer with loss and optimizer
    trainer.compile(
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_classes=num_classes
    )
    
    # Start training
    print('Starting training...')
    try:
        trainer.fit(
            scheduler=scheduler,
            cfg=cfg,
            verbose=True,
            tracker=tracker
        )
        print('Training completed successfully!')
    except Exception as e:
        print(f'Training failed with error: {e}')
        raise
    finally:
        if tracker is not None:
            tracker.finish()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train segmentation model for artifact detection.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load config and run training
    config = Config(str(config_path))
    main(config)