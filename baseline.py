## baseline.py

"""
baseline.py

Implements baseline forecasting models for time series:
  - MonolithicRNN: single-layer GRU with linear readout
  - MonolithicLSTM: single-layer LSTM with linear readout
  - TransformerModel: vanilla Transformer encoder-decoder
  - MarkovSwitchingAR: switching AR(p) via EM

Each class follows the BaselineModel interface:
    __init__(params: dict)
    train_model(datasets: Dict[str, TensorDataset]) -> None
    predict(x: torch.Tensor) -> torch.Tensor
"""

import math
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import tqdm # For progress bars

import utils

# Configure logger
logger = utils.configure_logging()


class BaselineModel:
    """
    Interface for baseline models.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Args:
            params: Configuration parameters for the baseline.
        """
        self.params = params or {}
        self.device = utils.get_device()

    def train_model(self, datasets: Dict[str, TensorDataset]) -> None:
        """
        Train the model on the provided datasets.

        Args:
            datasets: Dict with keys 'train', 'val', 'test', each a TensorDataset
                      of (X:FloatTensor[B,T,d], y:FloatTensor[B,T,m], regimes:LongTensor).
        """
        raise NotImplementedError("train_model must be implemented by subclass")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict outputs for input x.

        Args:
            x: Tensor of shape (B, T, d)

        Returns:
            y_pred: Tensor of shape (B, T, m)
        """
        raise NotImplementedError("predict must be implemented by subclass")


class MonolithicRNN(nn.Module, BaselineModel):
    """
    Monolithic single-layer GRU baseline.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        # Call nn.Module init first
        nn.Module.__init__(self)
        BaselineModel.__init__(self, params)
        # Model dimensions: must be provided in params
        self.input_dim = int(self.params.get("input_dim", 1))
        self.hidden_dim = int(self.params.get("hidden_dim", self.input_dim))
        self.output_dim = int(self.params.get("output_dim", self.input_dim))
        # Build modules
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        ).to(self.device)
        self.readout = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        # Initialize weights
        self._init_weights()
        # Check parameter budget
        self._check_param_budget()

    def _init_weights(self) -> None:
        # GRU weight init: Xavier uniform, biases zero
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Readout: Kaiming normal, bias zero
        nn.init.kaiming_normal_(self.readout.weight, nonlinearity="linear")
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)

    def _check_param_budget(self) -> None:
        param_count = sum(p.numel() for p in self.rnn.parameters()) + \
                      sum(p.numel() for p in self.readout.parameters())
        budget = int(utils._CONFIG.get("model", {})
                           .get("budget_params", 25000000))
        if param_count > budget:
            logger.warning(
                f"MonolithicRNN params ({param_count}) exceed budget ({budget})"
            )

    def train_model(self, datasets: Dict[str, TensorDataset]) -> None:
        # Hyperparameters from global config
        cfg = utils._CONFIG.get("training", {})
        lr0 = float(utils._CONFIG.get("training", {})
                              .get("lr_schedule", {})
                              .get("init_options", [1e-4])[0])
        weight_decay = float(cfg.get("regularization", {})
                                 .get("lambda_l2_options", [1e-6])[0])
        clip_norm = float(cfg.get("gradient_clipping_norm", 5.0))
        patience = int(cfg.get("early_stopping_patience", 10))
        # DataLoader settings
        batch_size = int(self.params.get("batch_size", 32))
        train_ds = datasets["train"]
        val_ds = datasets["val"]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        # Loss
        mse_loss = nn.MSELoss(reduction="mean")
        # Optimizer and schedule
        optimizer = Adam(
            list(self.rnn.parameters()) + list(self.readout.parameters()),
            lr=lr0,
            betas=(cfg.get("optimizer", {}).get("beta1", 0.9),
                   cfg.get("optimizer", {}).get("beta2", 0.999)),
            weight_decay=weight_decay,
        )
        T_decay = int(utils._CONFIG.get("data", {})
                             .get("synthetic", {})
                             .get("sequence_length", 1))
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0 / math.sqrt(1.0 + step / float(T_decay))
        )
        best_val = float("inf")
        epochs_no_improve = 0
        epoch = 0
        # Training loop
        while True:
            epoch += 1
            # Train phase
            self.rnn.train()
            train_loss = 0.0
            count = 0
            for X_batch, y_batch, _ in train_loader:
                # X_batch: (B,T,d), y_batch: (B,T,m)
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # Since y_batch[t] is target for X_batch[t] (y_t=x_{t+1})
                # We feed the whole X_batch sequence to the RNN
                # And compare the output with the whole y_batch sequence.
                # The last output h_seq[:, -1] prediction is ignored implicitly 
                # if y_batch[:, -1] is padding/zeros, or explicitly if needed.
                x_in = X_batch 
                y_true = y_batch
                optimizer.zero_grad()
                h_seq, _ = self.rnn(x_in)
                y_pred = self.readout(h_seq)
                loss = mse_loss(y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.rnn.parameters()) + list(self.readout.parameters()),
                    max_norm=clip_norm
                )
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * x_in.size(0)
                count += x_in.size(0)
            train_loss /= max(count, 1)
            # Validation phase
            self.rnn.eval()
            val_loss = 0.0
            count = 0
            with torch.no_grad():
                for X_batch, y_batch, _ in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    x_in = X_batch # Use full sequence
                    y_true = y_batch # Targets are already aligned
                    h_seq, _ = self.rnn(x_in)
                    y_pred = self.readout(h_seq)
                    loss = mse_loss(y_pred, y_true)
                    val_loss += loss.item() * x_in.size(0)
                    count += x_in.size(0)
            val_loss /= max(count, 1)
            logger.info(
                f"[MonolithicRNN] Epoch {epoch} TrainLoss={train_loss:.6f} "
                f"ValLoss={val_loss:.6f}"
            )
            # Early stopping
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {
                    'rnn': self.rnn.state_dict(),
                    'readout': self.readout.state_dict()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f"[MonolithicRNN] Early stopping at epoch {epoch}"
                    )
                    break
        # load best
        self.rnn.load_state_dict(best_state['rnn'])
        self.readout.load_state_dict(best_state['readout'])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.rnn.eval()
        with torch.no_grad():
            x = x.to(self.device)
            # full sequence prediction
            h_seq, _ = self.rnn(x)
            y_pred = self.readout(h_seq)
        return y_pred.cpu()


class MonolithicLSTM(nn.Module, BaselineModel):
    """
    Monolithic single-layer LSTM baseline.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        # Call nn.Module init first
        nn.Module.__init__(self)
        BaselineModel.__init__(self, params)
        self.input_dim = int(self.params.get("input_dim", 1))
        self.hidden_dim = int(self.params.get("hidden_dim", self.input_dim))
        self.output_dim = int(self.params.get("output_dim", self.input_dim))
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        ).to(self.device)
        self.readout = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        self._init_weights()
        self._check_param_budget()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.kaiming_normal_(self.readout.weight, nonlinearity="linear")
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)

    def _check_param_budget(self) -> None:
        param_count = sum(p.numel() for p in self.lstm.parameters()) + \
                      sum(p.numel() for p in self.readout.parameters())
        budget = int(utils._CONFIG.get("model", {})
                           .get("budget_params", 25000000))
        if param_count > budget:
            logger.warning(
                f"MonolithicLSTM params ({param_count}) exceed budget ({budget})"
            )

    def train_model(self, datasets: Dict[str, TensorDataset]) -> None:
        # identical to MonolithicRNN training but with LSTM
        cfg = utils._CONFIG.get("training", {})
        lr0 = float(utils._CONFIG.get("training", {})
                              .get("lr_schedule", {})
                              .get("init_options", [1e-4])[0])
        weight_decay = float(cfg.get("regularization", {})
                                 .get("lambda_l2_options", [1e-6])[0])
        clip_norm = float(cfg.get("gradient_clipping_norm", 5.0))
        patience = int(cfg.get("early_stopping_patience", 10))
        batch_size = int(self.params.get("batch_size", 32))
        train_ds = datasets["train"]
        val_ds = datasets["val"]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        mse_loss = nn.MSELoss(reduction="mean")
        optimizer = Adam(
            list(self.lstm.parameters()) + list(self.readout.parameters()),
            lr=lr0,
            betas=(cfg.get("optimizer", {}).get("beta1", 0.9),
                   cfg.get("optimizer", {}).get("beta2", 0.999)),
            weight_decay=weight_decay,
        )
        T_decay = int(utils._CONFIG.get("data", {})
                             .get("synthetic", {})
                             .get("sequence_length", 1))
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0 / math.sqrt(1.0 + step / float(T_decay))
        )
        best_val = float("inf")
        epochs_no_improve = 0
        epoch = 0
        while True:
            epoch += 1
            self.lstm.train()
            train_loss = 0.0
            count = 0
            for X_batch, y_batch, _ in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # y_batch[t] is target for X_batch[t]
                x_in = X_batch
                y_true = y_batch
                optimizer.zero_grad()
                h_seq, _ = self.lstm(x_in)
                y_pred = self.readout(h_seq)
                loss = mse_loss(y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.lstm.parameters()) + list(self.readout.parameters()),
                    max_norm=clip_norm
                )
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * x_in.size(0)
                count += x_in.size(0)
            train_loss /= max(count, 1)
            self.lstm.eval()
            val_loss = 0.0
            count = 0
            with torch.no_grad():
                for X_batch, y_batch, _ in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    x_in = X_batch # Use full sequence
                    y_true = y_batch # Targets are already aligned
                    h_seq, _ = self.lstm(x_in)
                    y_pred = self.readout(h_seq)
                    loss = mse_loss(y_pred, y_true)
                    val_loss += loss.item() * x_in.size(0)
                    count += x_in.size(0)
            val_loss /= max(count, 1)
            logger.info(
                f"[MonolithicLSTM] Epoch {epoch} TrainLoss={train_loss:.6f} "
                f"ValLoss={val_loss:.6f}"
            )
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {
                    'lstm': self.lstm.state_dict(),
                    'readout': self.readout.state_dict()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f"[MonolithicLSTM] Early stopping at epoch {epoch}"
                    )
                    break
        self.lstm.load_state_dict(best_state['lstm'])
        self.readout.load_state_dict(best_state['readout'])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.eval()
        with torch.no_grad():
            x = x.to(self.device)
            h_seq, _ = self.lstm(x)
            y_pred = self.readout(h_seq)
        return y_pred.cpu()


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / float(d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module, BaselineModel):
    """
    Transformer encoder-decoder baseline.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        # Call nn.Module init first
        nn.Module.__init__(self)
        BaselineModel.__init__(self, params)
        # Dimensions
        self.input_dim = int(self.params.get("input_dim", 1))
        self.output_dim = int(self.params.get("output_dim", 1))
        # Architecture hyperparams
        self.d_model = int(self.params.get("d_model", max(32, self.input_dim)))
        self.nhead = int(self.params.get("nhead", 8))
        self.num_layers = int(self.params.get("num_layers", 6))
        self.dim_feedforward = int(self.params.get("dim_feedforward", 4 * self.d_model))
        self.dropout = float(self.params.get("dropout", 0.1))
        # Build modules
        self.embedding = nn.Linear(self.input_dim, self.d_model).to(self.device)
        self.pos_enc = PositionalEncoding(self.d_model, self.dropout).to(self.device)
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=False,
        ).to(self.device)
        self.readout = nn.Linear(self.d_model, self.output_dim).to(self.device)
        # Initialize
        nn.init.kaiming_normal_(self.embedding.weight, nonlinearity="linear")
        if self.embedding.bias is not None:
            nn.init.zeros_(self.embedding.bias)
        nn.init.kaiming_normal_(self.readout.weight, nonlinearity="linear")
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)
        # Check budget
        param_count = sum(p.numel() for p in self.embedding.parameters()) + \
                      sum(p.numel() for p in self.pos_enc.parameters()) + \
                      sum(p.numel() for p in self.transformer.parameters()) + \
                      sum(p.numel() for p in self.readout.parameters())
        budget = int(utils._CONFIG.get("model", {})
                           .get("budget_params", 25000000))
        if param_count > budget:
            logger.warning(
                f"TransformerModel params ({param_count}) exceed budget ({budget})"
            )

    def train_model(self, datasets: Dict[str, TensorDataset]) -> None:
        cfg = utils._CONFIG.get("training", {})
        lr0 = float(utils._CONFIG.get("training", {})
                              .get("lr_schedule", {})
                              .get("init_options", [1e-4])[0])
        weight_decay = float(cfg.get("regularization", {})
                                 .get("lambda_l2_options", [1e-6])[0])
        clip_norm = float(cfg.get("gradient_clipping_norm", 5.0))
        patience = int(cfg.get("early_stopping_patience", 10))
        batch_size = int(self.params.get("batch_size", 32))
        train_ds = datasets["train"]
        val_ds = datasets["val"]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        mse_loss = nn.MSELoss(reduction="mean")
        optimizer = Adam(
            list(self.embedding.parameters()) +
            list(self.pos_enc.parameters()) +
            list(self.transformer.parameters()) +
            list(self.readout.parameters()),
            lr=lr0,
            betas=(cfg.get("optimizer", {}).get("beta1", 0.9),
                   cfg.get("optimizer", {}).get("beta2", 0.999)),
            weight_decay=weight_decay,
        )
        T_decay = int(utils._CONFIG.get("data", {})
                             .get("synthetic", {})
                             .get("sequence_length", 1))
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0 / math.sqrt(1.0 + step / float(T_decay))
        )
        best_val = float("inf")
        epochs_no_improve = 0
        epoch = 0
        while True:
            epoch += 1
            # Train
            self.embedding.train()
            self.pos_enc.train()
            self.transformer.train()
            train_loss = 0.0
            count = 0
            for X_batch, y_batch, _ in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                x_in = X_batch[:, :-1, :]    # (B, T-1, d)
                y_true = y_batch[:, 1:, :]   # (B, T-1, m)
                optimizer.zero_grad()
                # embed + pos
                src = self.embedding(x_in)   # (B, T-1, d_model)
                src = self.pos_enc(src)
                # transformer expects (T-1, B, d_model)
                seq_len = src.size(1)
                mask = self.transformer.generate_square_subsequent_mask(seq_len).to(self.device)
                out = self.transformer(
                    src.transpose(0, 1), src.transpose(0, 1), src_mask=mask
                ).transpose(0, 1)
                y_pred = self.readout(out)
                loss = mse_loss(y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.embedding.parameters()) +
                    list(self.pos_enc.parameters()) +
                    list(self.transformer.parameters()) +
                    list(self.readout.parameters()),
                    max_norm=clip_norm
                )
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * x_in.size(0)
                count += x_in.size(0)
            train_loss /= max(count, 1)
            # Validate
            self.embedding.eval()
            self.pos_enc.eval()
            self.transformer.eval()
            val_loss = 0.0
            count = 0
            with torch.no_grad():
                for X_batch, y_batch, _ in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    x_in = X_batch[:, :-1, :]
                    y_true = y_batch[:, 1:, :]
                    src = self.embedding(x_in)
                    src = self.pos_enc(src)
                    # transformer expects (T-1, B, d_model)
                    seq_len = src.size(1)
                    mask = self.transformer.generate_square_subsequent_mask(seq_len).to(self.device)
                    out = self.transformer(
                        src.transpose(0, 1), src.transpose(0, 1), src_mask=mask
                    ).transpose(0, 1)
                    y_pred = self.readout(out)
                    loss = mse_loss(y_pred, y_true)
                    val_loss += loss.item() * x_in.size(0)
                    count += x_in.size(0)
            val_loss /= max(count, 1)
            logger.info(
                f"[Transformer] Epoch {epoch} TrainLoss={train_loss:.6f} "
                f"ValLoss={val_loss:.6f}"
            )
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {
                    'emb': self.embedding.state_dict(),
                    'pos': self.pos_enc.state_dict(),
                    'trans': self.transformer.state_dict(),
                    'readout': self.readout.state_dict(),
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f"[Transformer] Early stopping at epoch {epoch}"
                    )
                    break
        # load best
        self.embedding.load_state_dict(best_state['emb'])
        self.pos_enc.load_state_dict(best_state['pos'])
        self.transformer.load_state_dict(best_state['trans'])
        self.readout.load_state_dict(best_state['readout'])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.embedding.eval()
        self.pos_enc.eval()
        self.transformer.eval()
        with torch.no_grad():
            X_batch = x.to(self.device)
            # full sequence
            src = self.embedding(X_batch)
            src = self.pos_enc(src)
            # transformer expects (T, B, d_model)
            seq_len = src.size(1)
            mask = self.transformer.generate_square_subsequent_mask(seq_len).to(self.device)
            out = self.transformer(
                src.transpose(0, 1), src.transpose(0, 1), src_mask=mask
            ).transpose(0, 1)
            y_pred = self.readout(out)
        return y_pred.cpu()


class MarkovSwitchingAR(BaselineModel):
    """
    Markov-switching AR(p) baseline via EM algorithm.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        # AR order
        self.p = int(utils._CONFIG.get("data", {})
                         .get("synthetic", {})
                         .get("ar_order", 1))
        # EM settings
        self.max_iters = int(self.params.get("max_iters", 100))
        self.tol = float(self.params.get("tol", 1e-4))
        # these will be set in train_model:
        self.R: Optional[int] = None
        self.d: Optional[int] = None
        self.pi: Optional[np.ndarray] = None
        self.T_mat: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None
        self.Sigma_inv: Optional[np.ndarray] = None
        self.Sigma_logdet: Optional[np.ndarray] = None

    def train_model(self, datasets: Dict[str, TensorDataset]) -> None:
        # Extract training data
        train_ds = datasets["train"]
        X_t, Y_t, regimes_t = train_ds.tensors
        # Convert to numpy
        Y = Y_t.cpu().numpy()      # (N, T, d)
        regimes = regimes_t.numpy()  # (N, T)
        N, T, d = Y.shape
        self.d = d
        # Infer number of regimes from true regimes
        self.R = int(regimes.max() + 1)
        # Build design matrices for each sequence
        X_design = np.zeros((N, T, self.p * d), dtype=float)
        for i in range(N):
            for t in range(T):
                if t < self.p:
                    continue
                # stack y_{t-1},...,y_{t-p}
                lagged = [Y[i, t - 1 - k, :] for k in range(self.p)]
                X_design[i, t, :] = np.concatenate(lagged, axis=0)
        # Initialize parameters
        # pi uniform
        self.pi = np.full(self.R, 1.0 / self.R, dtype=float)
        # transition matrix abrupt style
        spec = utils._CONFIG.get("data", {}).get("synthetic", {}) \
                      .get("regimes", {}) \
                      .get("abrupt", {})
        alpha = float(spec.get("self_transition_prob", 0.9))
        off = (1.0 - alpha) / float(max(self.R - 1, 1))
        self.T_mat = np.full((self.R, self.R), off, dtype=float)
        np.fill_diagonal(self.T_mat, alpha)
        # AR coeffs and covariances
        rng = np.random.default_rng(int(utils._CONFIG.get("seed", 42)))
        self.A = np.zeros((self.R, d, self.p * d), dtype=float)
        self.Sigma = np.zeros((self.R, d, d), dtype=float)
        for k in range(self.R):
            # random A
            self.A[k] = rng.uniform(-0.5, 0.5, size=(d, self.p * d))
            # init Sigma
            self.Sigma[k] = np.eye(d)
        # precompute inverses and log-dets
        self.Sigma_inv = np.zeros_like(self.Sigma)
        self.Sigma_logdet = np.zeros(self.R, dtype=float)
        for k in range(self.R):
            try:
                inv = np.linalg.inv(self.Sigma[k])
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(self.Sigma[k])
            self.Sigma_inv[k] = inv
            sign, logdet = np.linalg.slogdet(self.Sigma[k])
            self.Sigma_logdet[k] = logdet if sign > 0 else np.inf
        # EM loop
        prev_ll = None
        # Add tqdm progress bar for EM iterations
        em_pbar = tqdm.trange(self.max_iters, desc="MSAR EM Iterations")
        for itr in em_pbar:
            # accumulators
            sum_pi = np.zeros(self.R, dtype=float)
            sum_xi = np.zeros((self.R, self.R), dtype=float)
            sum_gamma = np.zeros(self.R, dtype=float)
            Sx = [np.zeros((self.p * d, self.p * d), dtype=float)
                  for _ in range(self.R)]
            Sc = [np.zeros((self.p * d, d), dtype=float)
                  for _ in range(self.R)]
            Sr = [np.zeros((d, d), dtype=float) for _ in range(self.R)]
            total_ll = 0.0
            # E-step per sequence
            for i in range(N):
                Y_seq = Y[i]           # (T, d)
                X_seq = X_design[i]    # (T, p*d)
                # compute log-emission matrix
                logE = np.zeros((T, self.R), dtype=float)
                for t in range(T):
                    if t < self.p:
                        continue
                    x_t = X_seq[t]
                    y_t = Y_seq[t]
                    for k in range(self.R):
                        diff = y_t - (self.A[k] @ x_t)
                        # log Gaussian
                        logE[t, k] = -0.5 * (
                            self.Sigma_logdet[k]
                            + diff @ self.Sigma_inv[k] @ diff
                            + d * math.log(2 * math.pi)
                        )
                # forward-backward in log-domain
                # alpha
                log_pi = np.log(self.pi + 1e-12)
                log_T = np.log(self.T_mat + 1e-12)
                alpha = np.zeros((T, self.R), dtype=float)
                alpha[0] = log_pi + logE[0]
                for t in range(1, T):
                    for j in range(self.R):
                        prev = alpha[t - 1] + log_T[:, j]
                        alpha[t, j] = np.logaddexp.reduce(prev) + logE[t, j]
                # beta
                beta = np.zeros((T, self.R), dtype=float)
                for t in range(T - 2, -1, -1):
                    for i0 in range(self.R):
                        nxt = log_T[i0, :] + logE[t + 1] + beta[t + 1]
                        beta[t, i0] = np.logaddexp.reduce(nxt)
                # log-likelihood
                ll_seq = np.logaddexp.reduce(alpha[-1])
                total_ll += ll_seq
                # gamma and xi
                gamma = np.exp(alpha + beta - ll_seq)
                xi = np.zeros((T - 1, self.R, self.R), dtype=float)
                for t in range(T - 1):
                    # vectorized xi
                    mat = (alpha[t][:, None]
                           + log_T
                           + logE[t + 1][None, :]
                           + beta[t + 1][None, :])
                    xi[t] = np.exp(mat - np.logaddexp.reduce(alpha[t]))
                # accumulate stats
                sum_pi += gamma[0]
                sum_xi += xi.sum(axis=0)
                for k in range(self.R):
                    for t in range(self.p, T):
                        w = gamma[t, k]
                        sum_gamma[k] += w
                        x_t = X_seq[t]
                        y_t = Y_seq[t]
                        Sx[k] += w * np.outer(x_t, x_t)
                        Sc[k] += w * np.outer(x_t, y_t)
                        res = y_t - (self.A[k] @ x_t)
                        Sr[k] += w * np.outer(res, res)
            # M-step updates
            # pi
            self.pi = sum_pi / sum_pi.sum()
            # transitions
            row_sum = sum_xi.sum(axis=1, keepdims=True)
            self.T_mat = sum_xi / np.where(row_sum > 0, row_sum, 1.0)
            # AR and Sigma
            for k in range(self.R):
                # update A[k]
                if np.linalg.matrix_rank(Sx[k]) < Sx[k].shape[0]:
                    A_flat = np.linalg.pinv(Sx[k]) @ Sc[k]
                else:
                    A_flat = np.linalg.solve(Sx[k], Sc[k])
                # A_flat: (p*d, d) => transpose
                self.A[k] = A_flat.T
                # update Sigma
                denom = sum_gamma[k] if sum_gamma[k] > 0 else 1.0
                self.Sigma[k] = Sr[k] / denom
                try:
                    inv = np.linalg.inv(self.Sigma[k])
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(self.Sigma[k])
                self.Sigma_inv[k] = inv
                sign, logdet = np.linalg.slogdet(self.Sigma[k])
                self.Sigma_logdet[k] = logdet if sign > 0 else np.inf
            # check convergence
            if prev_ll is not None and abs(total_ll - prev_ll) < self.tol:
                logger.info(
                    f"[MS-AR] EM converged at iter {itr} ΔLL={abs(total_ll - prev_ll):.6f}"
                )
                break
            # Update progress bar postfix with log-likelihood
            em_pbar.set_postfix({"LogLikelihood": f"{total_ll:.2f}", "DeltaLL": f"{abs(total_ll - (prev_ll if prev_ll is not None else 0.0)):.4f}"})
            prev_ll = total_ll
        logger.info(f"[MS-AR] Finished EM after {itr+1} iterations")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict using learned MS-AR model on new sequences.
        Args:
            x: Tensor (B, T, d)  -- here d must match training
        Returns:
            Tuple[torch.Tensor, np.ndarray]: 
                y_pred: Tensor (B, T, d)
                gamma_all: Numpy array (B, T, R) of regime probabilities
        """
        X_np = x.cpu().numpy()
        B, T, d = X_np.shape
        p, R = self.p, self.R
        # build design
        X_design = np.zeros((T, p * d), dtype=float)
        y_pred = np.zeros((B, T, d), dtype=float)
        gamma_all = np.zeros((B, T, R), dtype=float) # Store gamma for all sequences
        for i in range(B):
            Y_seq = X_np[i]  # for AR data X==Y
            # forward-back to get gamma
            # compute logE
            logE = np.zeros((T, R), dtype=float)
            for t in range(T):
                if t < p:
                    continue
                x_t = np.concatenate([Y_seq[t - 1 - k] for k in range(p)], axis=0)
                for k in range(R):
                    diff = Y_seq[t] - (self.A[k] @ x_t)
                    logE[t, k] = -0.5 * (
                        self.Sigma_logdet[k]
                        + diff @ self.Sigma_inv[k] @ diff
                        + d * math.log(2 * math.pi)
                    )
            # forward-backward
            log_pi = np.log(self.pi + 1e-12)
            log_T = np.log(self.T_mat + 1e-12)
            alpha = np.zeros((T, R), dtype=float)
            alpha[0] = log_pi + logE[0]
            for t in range(1, T):
                for j in range(R):
                    alpha[t, j] = (
                        np.logaddexp.reduce(alpha[t - 1] + log_T[:, j])
                        + logE[t, j]
                    )
            beta = np.zeros((T, R), dtype=float)
            for t in range(T - 2, -1, -1):
                for i0 in range(R):
                    beta[t, i0] = np.logaddexp.reduce(
                        log_T[i0, :] + logE[t + 1] + beta[t + 1]
                    )
            ll_seq = np.logaddexp.reduce(alpha[-1])
            gamma = np.exp(alpha + beta - ll_seq)
            gamma_all[i] = gamma # Store gamma for this sequence
            # predict
            for t in range(T):
                if t < p:
                    # copy input
                    y_pred[i, t] = Y_seq[t]
                else:
                    x_t = np.concatenate(
                        [Y_seq[t - 1 - k] for k in range(p)], axis=0
                    )
                    pred_k = np.array([self.A[k] @ x_t for k in range(R)])
                    # weighted
                    y_pred[i, t] = (gamma[t][:, None] * pred_k).sum(axis=0)
        return torch.tensor(y_pred, dtype=torch.float32), gamma_all
