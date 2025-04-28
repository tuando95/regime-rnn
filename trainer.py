## trainer.py

"""
trainer.py

Training logic for ModularRegimeRNN and other models. Handles:
  - mixed-precision training
  - inverse-sqrt learning rate scheduling
  - MSE + L2 + entropy loss
  - gradient clipping
  - early stopping
  - TensorBoard logging
"""

import math
import copy
from typing import Any, Dict, Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

from torch.utils.tensorboard import SummaryWriter

import utils
import tqdm # For epoch progress bar


class Trainer:
    """
    Trainer orchestrates model training with configurable hyperparameters.

    Args:
        model: nn.Module to train (e.g., ModularRegimeRNN).
        datasets: Dict with keys 'train' and 'val' mapping to TensorDataset.
        config: Configuration dictionary loaded from config.yaml.
        epoch_callback: Optional callback function to call after each epoch.
    """

    def __init__(self,
                 model: nn.Module,
                 datasets: Dict[str, TensorDataset],
                 config: Dict[str, Any],
                 epoch_callback: Optional[Callable[[int, float], None]] = None) -> None:
        # Save references
        self.model = model
        self.datasets = datasets
        self.config = config
        self.epoch_callback = epoch_callback

        # Reproducibility and device setup
        seed = int(config.get("seed", 42))
        utils.seed_everything(seed)
        self.device = utils.get_device()
        self.model.to(self.device)

        # Parse training configuration
        train_cfg = config.get("training", {})

        # Optimizer hyperparameters
        opt_cfg = train_cfg.get("optimizer", {})
        self.beta1: float = float(opt_cfg.get("beta1", 0.9))
        self.beta2: float = float(opt_cfg.get("beta2", 0.999))

        # Learning rate schedule
        lr_cfg = train_cfg.get("lr_schedule", {})
        self.lr_schedule_type: str = lr_cfg.get("type", "inverse_sqrt")
        init_opts = lr_cfg.get("init_options", [1e-4])
        self.lr0: float = float(init_opts[0])
        decay_cfg = lr_cfg.get("decay_steps", None)
        if isinstance(decay_cfg, int) and decay_cfg > 0:
            self.T_decay: int = decay_cfg
        else:
            # Fallback to sequence length from data config
            data_syn = config.get("data", {}).get("synthetic", {})
            self.T_decay = int(data_syn.get("sequence_length", 1))

        # Gradient clipping
        self.grad_clip_norm: float = float(
            train_cfg.get("gradient_clipping_norm", 5.0)
        )

        # Regularization penalties
        reg_cfg = train_cfg.get("regularization", {})
        l2_opts = reg_cfg.get("lambda_l2_options", [1e-6])
        self.lambda_l2: float = float(l2_opts[0])
        ent_opts = reg_cfg.get("lambda_entropy_options", [1e-4])
        self.lambda_ent: float = float(ent_opts[0])
        # Add regime loss weight
        regime_opts = reg_cfg.get("lambda_regime_options", [10.0]) # Default to 1.0 initially
        self.lambda_regime: float = float(regime_opts[1])

        # Early stopping
        self.early_stopping_patience: int = int(
            train_cfg.get("early_stopping_patience", 50)
        )

        # Batch size and number of epochs (with sensible defaults)
        self.batch_size: int = int(train_cfg.get("batch_size", 32))
        self.max_epochs: int = int(train_cfg.get("max_epochs", 150))

        # DataLoaders
        self.train_loader = DataLoader(
            datasets["train"], batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            datasets["val"], batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Mixed-precision training
        precision_cfg = config.get("precision", {})
        self.use_amp: bool = bool(precision_cfg.get("mixed_fp16", False))
        self.scaler: Optional[GradScaler] = (
            torch.amp.GradScaler('cuda', enabled=self.use_amp) if self.use_amp and torch.cuda.is_available() else None
        )

        # Losses
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean") # Add CE loss

        # TensorBoard writer
        log_dir = train_cfg.get("log_dir", None)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self) -> nn.Module:
        """
        Run the training loop, perform early stopping, and return the best model.

        Returns:
            Tuple[nn.Module, int]: The trained model and the final epoch number.
        """
        # Optimizer (L2 handled manually in loss, so weight_decay=0)
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr0,
            betas=(self.beta1, self.beta2),
            weight_decay=0.0
        )

        # Learning rate scheduler
        if self.lr_schedule_type == "inverse_sqrt":
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda step: 1.0 / math.sqrt(
                    1.0 + step / float(self.T_decay)
                )
            )
        else:
            raise NotImplementedError(
                f"LR schedule '{self.lr_schedule_type}' not supported"
            )

        # Early stopping state
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = copy.deepcopy(self.model.state_dict())

        final_epoch = self.max_epochs # Initialize final epoch
        # Main training loop
        epoch_pbar = tqdm.trange(1, self.max_epochs + 1, desc="Training Epochs")
        for epoch in epoch_pbar:
            # === Training Phase ===
            self.model.train()
            train_loss_sum = 0.0
            train_mse_sum = 0.0
            train_ent_sum = 0.0
            train_ce_sum = 0.0 # Accumulator for CE loss
            train_samples = 0

            # Add tqdm for batch progress within an epoch
            batch_pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch} Batches", leave=False)
            # Get regimes (z_batch) from the loader
            for X_batch, y_batch, z_batch in batch_pbar:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                z_batch = z_batch.to(self.device).long() # Ensure regimes are Long type for CE
                B, T_seq, _ = X_batch.shape
                # Loop over the entire sequence length T_seq
                # since y_batch[:, t] is the target for x_batch[:, t]

                # Initialize hidden states (now returns a list)
                if hasattr(self.model, "init_hidden"):
                    h_prev = self.model.init_hidden(B, device=self.device) # h_prev is List[Tensor]
                else:
                    h_prev = None

                optimizer.zero_grad()
                sum_mse = 0.0
                sum_ent = 0.0
                sum_ce = 0.0 # Accumulator for CE loss in this batch

                # Forward and loss accumulation
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                        for t in range(T_seq):
                            x_t = X_batch[:, t, :]
                            y_t = y_batch[:, t, :]
                            z_t = z_batch[:, t]

                            # Model forward returns 4 values now
                            h_prev, y_hat, g_t, logits = self.model(x_t, h_prev)

                            mse_t = self.mse_loss(y_hat, y_t)
                            ent_t = -(g_t * torch.log(g_t + utils.EPS)).sum(dim=1).mean()
                            ce_t = self.ce_loss(logits, z_t)

                            sum_mse = sum_mse + mse_t
                            sum_ent = sum_ent + ent_t
                            sum_ce = sum_ce + ce_t

                        # normalize losses by number of steps
                        # Normalize by sequence length T_seq, matching paper's 1/(NT) average
                        sum_mse = sum_mse / float(T_seq)
                        sum_ent = sum_ent / float(T_seq)
                        sum_ce = sum_ce / float(T_seq)

                        # L2 penalty
                        l2_norm = sum(
                            p.pow(2).sum() for p in self.model.parameters()
                        )
                        loss = sum_mse + self.lambda_l2 * l2_norm \
                               + self.lambda_ent * sum_ent \
                               + self.lambda_regime * sum_ce

                    # Backprop with scaler
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_norm
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    # Step LR scheduler *after* optimizer step
                    scheduler.step()

                else:
                    # FP32 training
                    for t in range(T_seq):
                        x_t = X_batch[:, t, :]
                        y_t = y_batch[:, t, :]
                        z_t = z_batch[:, t]

                        # Model forward returns 4 values now
                        h_prev, y_hat, g_t, logits = self.model(x_t, h_prev)

                        mse_t = self.mse_loss(y_hat, y_t)
                        ent_t = -(g_t * torch.log(g_t + utils.EPS)).sum(dim=1).mean()
                        ce_t = self.ce_loss(logits, z_t)

                        sum_mse = sum_mse + mse_t
                        sum_ent = sum_ent + ent_t
                        sum_ce = sum_ce + ce_t

                    # normalize losses by number of steps
                    sum_mse = sum_mse / float(T_seq)
                    sum_ent = sum_ent / float(T_seq)
                    sum_ce = sum_ce / float(T_seq)
                    l2_norm = sum(
                        p.pow(2).sum() for p in self.model.parameters()
                    )
                    loss = sum_mse + self.lambda_l2 * l2_norm \
                           + self.lambda_ent * sum_ent \
                           + self.lambda_regime * sum_ce

                    loss.backward()
                    clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_norm
                    )
                    optimizer.step()

                    # Step LR scheduler *after* optimizer step
                    scheduler.step()

                # Accumulate metrics
                train_samples += B
                train_mse_sum += sum_mse.item() * B
                train_ent_sum += sum_ent.item() * B
                train_ce_sum += sum_ce.item() * B
                train_loss_sum += loss.item() * B

            avg_train_mse = train_mse_sum / max(train_samples, 1)
            avg_train_ent = train_ent_sum / max(train_samples, 1)
            avg_train_ce = train_ce_sum / max(train_samples, 1)
            avg_train_loss = train_loss_sum / max(train_samples, 1)

            # === Validation Phase ===
            avg_val_loss, avg_val_mse, avg_val_ent, avg_val_ce = self._validate()

            # === Logging ===
            self.writer.add_scalar("train/mse", avg_train_mse, epoch)
            self.writer.add_scalar("train/entropy", avg_train_ent, epoch)
            self.writer.add_scalar("train/loss", avg_train_loss, epoch)
            self.writer.add_scalar("val/mse", avg_val_mse, epoch)
            self.writer.add_scalar("val/entropy", avg_val_ent, epoch)
            self.writer.add_scalar("val/loss", avg_val_loss, epoch)
            self.writer.add_scalar("CrossEntropy/Train", avg_train_ce, epoch)
            self.writer.add_scalar("CrossEntropy/Val", avg_val_ce, epoch)
            self.writer.add_scalar(
                "learning_rate", optimizer.param_groups[0]["lr"], epoch
            )

            # Update tqdm postfix with latest metrics
            epoch_pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}", 
                "val_loss": f"{avg_val_loss:.4f}",
                "best_val_loss": f"{best_val_loss:.4f}",
                "val_mse": f"{avg_val_mse:.4f}",
                "val_ce": f"{avg_val_ce:.4f}"
            })

            # === Optuna Pruning Callback ===
            if self.epoch_callback:
                self.epoch_callback(epoch, avg_val_loss)

            # === Early Stopping Check ===
            if avg_val_loss < best_val_loss - utils.EPS:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    final_epoch = epoch
                    epoch_pbar.set_description(f"Early stopping at epoch {epoch} based on val_loss")
                    epoch_pbar.close()
                    break

        # Load best model and finalize
        self.model.load_state_dict(best_state)
        self.writer.close()
        self.model.eval()
        return self.model, final_epoch

    def _validate(self) -> tuple[float, float, float, float]:
        """Run validation loop and return average losses."""
        self.model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_ent_sum = 0.0
        val_ce_sum = 0.0 # Accumulator for CE loss
        val_samples = 0

        with torch.no_grad():
            # Get regimes (z_batch) from the loader
            for X_batch, y_batch, z_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                z_batch = z_batch.to(self.device).long() # Ensure Long type
                B, T_seq, _ = X_batch.shape

                if hasattr(self.model, "init_hidden"):
                    h_prev = self.model.init_hidden(B, device=self.device) # h_prev is list
                else:
                    h_prev = None

                sum_mse = 0.0
                sum_ent = 0.0
                sum_ce = 0.0 # Accumulator for CE loss in this batch

                # Need to handle AMP context for validation too if enabled
                amp_context = torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp and self.scaler is not None)
                with amp_context:
                    for t in range(T_seq):
                        x_t = X_batch[:, t, :]
                        y_t = y_batch[:, t, :]
                        z_t = z_batch[:, t]

                        # Model forward returns 4 values
                        h_prev, y_hat, g_t, logits = self.model(x_t, h_prev)

                        mse_t = self.mse_loss(y_hat, y_t)
                        ent_t = -(g_t * torch.log(g_t + utils.EPS)).sum(dim=1).mean()
                        ce_t = self.ce_loss(logits, z_t)

                        sum_mse = sum_mse + mse_t
                        sum_ent = sum_ent + ent_t
                        sum_ce = sum_ce + ce_t

                sum_mse = sum_mse / float(T_seq)
                sum_ent = sum_ent / float(T_seq)
                sum_ce = sum_ce / float(T_seq)
                l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
                # Calculate combined loss for reporting, mirroring training loss structure
                loss = sum_mse + self.lambda_l2 * l2_norm \
                       + self.lambda_ent * sum_ent \
                       + self.lambda_regime * sum_ce

                val_loss_sum += loss.item() * B
                val_mse_sum += sum_mse.item() * B
                val_ent_sum += sum_ent.item() * B
                val_ce_sum += sum_ce.item() * B # Accumulate CE loss for epoch avg
                val_samples += B

        avg_val_loss = val_loss_sum / max(val_samples, 1)
        avg_val_mse = val_mse_sum / max(val_samples, 1)
        avg_val_ent = val_ent_sum / max(val_samples, 1)
        avg_val_ce = val_ce_sum / max(val_samples, 1) # Avg CE loss

        return avg_val_loss, avg_val_mse, avg_val_ent, avg_val_ce
