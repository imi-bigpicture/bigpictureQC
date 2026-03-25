"""Training module for artifact segmentation models."""
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import wandb
import numpy as np
import time
from tqdm import tqdm

from bigpictureQC.data.data_utils import get_data
from bigpictureQC.trainer.setup import configure_metrics
from bigpictureQC.utils.common import class_count
from monai.networks.utils import one_hot


logger = logging.getLogger(__name__)


class AverageMetricTracker:
    """Welford's online algorithm for computing running mean and variance.
    
    Provides numerically stable computation of mean and standard deviation
    over streaming data without storing all values.
    
    References:
        Welford, B. P. (1962). "Note on a method for calculating corrected 
        sums of squares and products". Technometrics. 4 (3): 419–420.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all statistics to initial state."""
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0  # Sum of squared differences from mean
        self.val: float = 0.0  # Last added value

    def add(self, value: float) -> None:
        """Add a new value and update statistics.
        
        Args:
            value: New observation to add
        """
        x = float(value)
        self.val = x
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        """Return sample standard deviation."""
        if self.n < 2:
            return float("nan")
        return (self.M2 / (self.n - 1)) ** 0.5

    def value(self) -> Tuple[float, float]:
        """Return (mean, std) tuple."""
        return self.mean, self.std


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
        verbose: Whether to print early stopping messages
        delta: Minimum change to qualify as improvement
        mode: 'min' for metrics where lower is better (e.g., loss),
              'max' for metrics where higher is better (e.g., dice)
        trace_func: Function for printing messages (default: print)
    
    Example:
        >>> early_stopping = EarlyStopping(patience=10, mode='max')
        >>> for epoch in range(100):
        ...     val_dice = validate()
        ...     early_stopping(val_dice)
        ...     if early_stopping.early_stop:
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        verbose: bool = False,
        delta: float = 0.0,
        mode: str = 'min',
        trace_func = print
    ):
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got: {mode}")
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
        self.mode = mode

    def __call__(self, metric_value: float) -> None:
        """Check if metric has improved and update early stopping state.
        
        Args:
            metric_value: Current validation metric value
        """
        if self.best_score is None:
            self.best_score = metric_value
            return
        
        # Check if metric improved based on mode
        improved = (
            (self.mode == 'min' and metric_value < self.best_score - self.delta) or
            (self.mode == 'max' and metric_value > self.best_score + self.delta)
        )
        
        if improved:
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func(
                        f'Early stopping activated. No improvement for {self.patience} epochs.'
                    )


class Trainer:
    """
    Trainer for artifact segmentation models with multi-task learning.
    
    Supports:
    - Multi-class semantic segmentation
    - Auxiliary classification head (image-level labels)
    - Gradient accumulation
    - Mixed precision training (AMP)
    - Gradient clipping
    - Early stopping
    - Checkpoint management
    - Weights & Biases tracking
    
    The model should have two outputs:
    - Segmentation head: (B, C, H, W) probabilities
    - Classification head: (B, C) probabilities
    
    Args:
        model: PyTorch model with segmentation and classification heads
        device: Device to train on (cuda/cpu)
        cfg: Configuration object
        tracker: Optional W&B run object for logging
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        cfg: Any,
        tracker: Optional[Any] = None
    ):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.tracker = tracker
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(cfg.directories.get('checkpoint_dir'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = cfg.directories.get('checkpoint_name')
        if not checkpoint_name.endswith('.pth'):
            checkpoint_name += '.pth'
        self.checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Initialize metrics
        self.metrics, self.metrics_funcs = configure_metrics(cfg)
        
        # Training mode settings
        self.on_the_fly = cfg.train_config.get('on_the_fly', False)
        self.use_early_stopping = cfg.train_config.get('early_stopping', False)
        
        # Setup W&B metrics if tracking enabled
        if tracker is not None:
            tracker.define_metric("epoch", summary="max")
            tracker.define_metric("lr", step_metric="epoch")
        
        # Compilation state (set by compile() method)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss: Optional[torch.nn.Module] = None
        self.num_classes: Optional[int] = None
        self.scheduler: Optional[Any] = None
        
        # Training state
        self.history: Dict[str, Dict] = {'train': {}, 'valid': {}}
        self.best_score: Optional[float] = None
        self.verbose: bool = True
        
        # Classification loss for auxiliary head
        self.criterion = torch.nn.BCELoss(reduction='mean')
        
        logger.info("Trainer initialized")

    def compile(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        num_classes: int
    ) -> None:
        """
        Prepare the trainer with necessary components before training.
        
        Args:
            optimizer: Optimizer for parameter updates
            loss_fn: Loss function for segmentation task
            num_classes: Number of output classes (including background)
        """
        self.optimizer = optimizer
        self.loss = loss_fn
        self.num_classes = num_classes
        self.binary = (num_classes == 1)
        self.model.to(self.device)
        
        logger.info(
            f"Trainer compiled with optimizer={type(optimizer).__name__}, "
            f"loss={type(loss_fn).__name__}, num_classes={num_classes}"
        )

    def fit(
        self,
        scheduler: Any,
        cfg: Any,
        verbose: bool = True,
        tracker: Optional[Any] = None
    ) -> None:
        """
        Train the model for specified number of epochs.
        
        Args:
            scheduler: Learning rate scheduler
            cfg: Configuration object
            verbose: Whether to print training progress
            tracker: Optional W&B tracker
        """
        # Training hyperparameters
        self.epochs = cfg.train_config.get('epochs')
        self.scheduler = scheduler
        self.loss_weight = cfg.loss_config.get('loss_weight')
        self.accumulation_steps = max(1, int(cfg.train_config.get('accumulation_steps', 1)))
        self.use_amp = bool(cfg.train_config.get('amp', False))
        self.grad_clip = float(cfg.train_config.get('grad_clip', 0.0))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.verbose = verbose
        
        # Validation metric for model selection
        self.valid_metric = cfg.train_config.get('valid_metric')
        self.metric_mode = next(
            (m['mode'] for m in cfg.get('metrics') if m['name'] == self.valid_metric),
            'max'
        )
        
        # Initialize best score based on metric mode
        self.best_score = float('-inf') if self.metric_mode == 'max' else float('inf')
        
        # Early stopping
        early_stopping: Optional[EarlyStopping] = None
        if self.use_early_stopping:
            patience = cfg.train_config.get('early_stopping_patience', 20)
            early_stopping = EarlyStopping(
                mode=self.metric_mode,
                patience=patience,
                verbose=verbose
            )
        
        # Load checkpoint or pretrained weights
        start_epoch = self._load_or_initialize_weights(cfg)
        
        # Get data loaders
        self.train_data, self.valid_data = get_data(cfg)
        
        logger.info(f"Starting training from epoch {start_epoch} to {self.epochs}")
        
        # Training loop
        for epoch in range(start_epoch, self.epochs):
            # Log epoch number
            if self.tracker is not None:
                self.tracker.log({"epoch": epoch + 1})
                if hasattr(self.optimizer, 'param_groups'):
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.tracker.log({"lr": current_lr})
            
            if self.verbose:
                logger.info(f'Epoch {epoch + 1}/{self.epochs}')
            
            # Train and validate
            train_logs = self._process_epoch(iterator=self.train_data, training=True)
            valid_logs = self._process_epoch(iterator=self.valid_data, training=False)
            
            # Store history
            self.history['train'][epoch] = train_logs
            self.history['valid'][epoch] = valid_logs
            
            # Check if validation metric improved
            current_score = valid_logs[self.valid_metric]
            improved = self._check_improvement(current_score)
            
            if improved:
                self.best_score = current_score
                self._save_checkpoint(epoch, tracker=tracker)
                logger.info(f"New best {self.valid_metric}: {current_score:.4f}")
            
            # Early stopping check
            if early_stopping is not None:
                early_stopping(current_score)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered")
                    break
            
            # Step scheduler
            self._step_scheduler(current_score)
        
        logger.info("Training completed")

    def _check_improvement(self, current_score: float) -> bool:
        """Check if current score is better than best score."""
        if self.metric_mode == 'max':
            return current_score > self.best_score
        else:
            return current_score < self.best_score

    def _step_scheduler(self, metric_value: float) -> None:
        """Step the learning rate scheduler."""
        if self.scheduler is None:
            return
        
        # ReduceLROnPlateau needs the metric value
        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(metric_value)
        # Other schedulers just step
        elif hasattr(self.scheduler, 'step'):
            self.scheduler.step()

    def _load_or_initialize_weights(self, cfg: Any) -> int:
        """Load checkpoint or pretrained weights. Returns start epoch."""
        # Check if resuming from checkpoint
        if self.checkpoint_path.exists():
            start_epoch = self._load_checkpoint()
            logger.info(f"Resuming from checkpoint at epoch {start_epoch}")
            return start_epoch
        
        # Load pretrained weights if specified
        finetune_from = cfg.train_config.get('finetune_from')
        if finetune_from and Path(finetune_from).exists():
            self._load_pretrained_weights(finetune_from)
            logger.info(f"Loaded pretrained weights from {finetune_from}")
        
        return 0

    def _load_pretrained_weights(self, checkpoint_path: str) -> None:
        """Load pretrained model weights (not optimizer state)."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:] if k.startswith('module.') else k
            new_state[name] = v
        
        self.model.load_state_dict(new_state, strict=True)

    def _format_logs(self, logs: Dict[str, float]) -> str:
        return ' - '.join([f'{k}: {round(float(v), 4)}' for k, v in logs.items()])

    def _show_progress(self, iterator, duration, logs, stage='train'):
        try:
            steps = len(iterator)
        except TypeError:
            steps = "?"
        if self.verbose:
            duration = int(round(duration, 0))
            print(f'{stage} {steps}/{steps} [==========] - {duration}s - {self._format_logs(logs)}')

    def _process_epoch(self, iterator, training=True):
        """
        One full epoch over `iterator` with:
        - gradient accumulation (self.accumulation_steps)
        - optional AMP (self.use_amp + self.scaler)
        - optional grad clipping (self.grad_clip)
        - stateful Dice aggregation; simple running avgs for others
        """

        # ---- settings / fallbacks ----
        accumulation_steps = max(1, int(getattr(self, "accumulation_steps", 1)))
        use_amp            = bool(getattr(self, "use_amp", False))
        scaler             = getattr(self, "scaler", None)
        grad_clip          = float(getattr(self, "grad_clip", 0.0))
        step_sched_per_iter = bool(getattr(self, "scheduler_per_iter", False))  # if you want per-iteration schedulers

        start = time.time()
        logs = {}
        loss_meter = AverageMetricTracker()

        # Split metrics into stateful (e.g., monai DiceMetric) vs simple running averages
        stateful_metrics = {}
        simple_trackers = {}
        for metric, name, type_ in self.metrics:
            if type_ == 'segmentation' and name == 'dice_score':
                stateful_metrics[name] = metric
            else:
                simple_trackers[name] = AverageMetricTracker()

        # Set model mode
        self.model.train() if training else self.model.eval()
        # If encoder is frozen, keep its BN stats fixed while training
        if training and hasattr(self.model, "encoder"):
            if all(not p.requires_grad for p in self.model.encoder.parameters()):
                self.model.encoder.eval()


        # Zero grads once before the first accumulation window
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        # reset stateful metrics at the start of each epoch
        if 'dice_score' in stateful_metrics:
            try:
                stateful_metrics['dice_score'].reset()
            except Exception:
                pass

        for batch_idx, batch in tqdm(enumerate(iterator)):
            # ----- load batch -----
            if self.on_the_fly:
                x, y, _ = batch
                x = x.transpose((0, 3, 1, 2)).astype(np.float32)
                y = y.transpose((0, 3, 1, 2)).astype(np.float32)
                x = x / 255.0
                x = torch.tensor(x).to(self.device)
                y = torch.tensor(y).to(self.device)
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

            # Forward + loss (with optional AMP)
            with torch.set_grad_enabled(training):
                if use_amp:
                    assert scaler is not None, "GradScaler not initialized; set self.scaler in fit()."
                    with torch.cuda.amp.autocast():
                        out1, out2 = self.model.forward(x)
                        classes = class_count(y, self.num_classes)  # if this returns counts, binarize externally if desired
                        clf_loss = self.criterion(out2, classes)
                        sgm_loss = self.loss(out1, y)
                        total_loss = ((1.0 - self.loss_weight) * clf_loss) + (self.loss_weight * sgm_loss)
                        loss = total_loss / accumulation_steps
                    if training:
                        scaler.scale(loss).backward()
                else:
                    out1, out2 = self.model.forward(x)
                    classes = class_count(y, self.num_classes)
                    clf_loss = self.criterion(out2, classes)
                    sgm_loss = self.loss(out1, y)
                    total_loss = ((1.0 - self.loss_weight) * clf_loss) + (self.loss_weight * sgm_loss)
                    loss = total_loss / accumulation_steps
                    if training:
                        loss.backward()

            # ----- loss logging (running avg on the *unscaled* total loss) -----
            loss_value = float(total_loss.detach().cpu().numpy())
            loss_meter.add(loss_value)
            logs.update({'loss': loss_meter.mean})

            if self.tracker is not None and training:
                self.tracker.log({f"train/loss_step": loss_value})


            # ----- metrics update -----
            for metric, name, type_ in self.metrics:
                if type_ == 'classification':
                    val = metric(out2, classes)
                    val = float(val.detach().cpu().numpy()) if hasattr(val, "detach") else float(val)
                    simple_trackers[name].add(val)

                elif type_ == 'segmentation':
                    if name in ['dice_loss', 'generalized_dice']:
                        # loss-like metrics computed on probabilities/logits
                        val = metric(out1, y)
                        val = val[0][0] if isinstance(val, tuple) else val
                        val = float(val.detach().cpu().numpy()) if hasattr(val, "detach") else float(val)
                        simple_trackers[name].add(val)
                    else:
                        # score metrics (DiceMetric/IoU/Fscore) on discrete masks
                        d = out1.device
                        if not self.binary:
                            y_pred = torch.argmax(out1, dim=1, keepdim=True).to(d)
                            y_pred = one_hot(y_pred, num_classes=out1.shape[1])
                        else:
                            # if binary and model outputs logits, sigmoid before rounding
                            # y_pred = torch.round(torch.sigmoid(out1)).to(d)
                            y_pred = torch.round(out1).to(d)

                        if name in ['dice_score']:  # stateful metric: update buffers now, aggregate later
                            metric(y_pred, y)
                        else:
                            val = metric(y_pred, y)
                            val = val[0][0] if isinstance(val, tuple) else val
                            val = float(val.detach().cpu().numpy()) if hasattr(val, "detach") else float(val)
                            simple_trackers[name].add(val)


            # ----- optimizer update every N micro-batches (or at epoch end) -----
            do_update = training and (
                ((batch_idx + 1) % accumulation_steps == 0) or
                ((batch_idx + 1) == len(iterator))
            )
            if do_update:
                if grad_clip > 0.0:
                    if use_amp:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                if use_amp:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                # Optional per-iteration scheduler step (e.g., OneCycleLR)
                if step_sched_per_iter and self.scheduler is not None and hasattr(self.scheduler, "step"):
                    self.scheduler.step()

        # ----- finalize epoch metrics -----
        # 1) simple averages
        for n, tracker in simple_trackers.items():
            logs[n] = tracker.mean

        # 2) aggregate stateful metrics once per epoch
        if 'dice_score' in stateful_metrics:
            dice_val = stateful_metrics['dice_score'].aggregate().item()
            stateful_metrics['dice_score'].reset()
            logs['dice_score'] = dice_val

        # tracker (per-epoch)
        mode = 'train' if training else 'valid'
        if self.tracker is not None:
            for k, v in logs.items():
                self.tracker.log({f"{mode}/{k}_epoch": float(v)})

        # pretty print
        duration = time.time() - start
        if self.verbose:
            self._show_progress(iterator, duration, logs, stage=mode)

        return logs

    def _save_checkpoint(
        self,
        epoch: int,
        tracker: Optional[Any] = None,
        suffix: Optional[str] = None
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            tracker: Optional W&B tracker
            suffix: Optional suffix for checkpoint filename
        """
        if suffix:
            filename = f"{self.checkpoint_path.stem}_{suffix}.pth"
        else:
            filename = self.checkpoint_path.name
        
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_score': self.best_score,
        }, checkpoint_path)
        
        if tracker:
            wandb.save(str(checkpoint_path))
        
        logger.info(f'Checkpoint saved to {checkpoint_path}')

    def _load_checkpoint(self) -> int:
        """
        Load model state from checkpoint.
        
        Returns:
            Epoch number to resume from
        """
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load model state (handle DataParallel prefix)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        
        # Load optimizer state
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            sched_state = checkpoint.get('scheduler_state_dict')
            if sched_state:
                self.scheduler.load_state_dict(sched_state)
        
        # Load best score
        self.best_score = checkpoint.get('best_score', self.best_score)
        
        return checkpoint.get('epoch', 0)