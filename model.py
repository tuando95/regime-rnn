"""
model.py

Defines the abstract Model base class and the ModularRegimeRNN model
with K expert GRU modules and a lightweight gating MLP.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch import Tensor


class Model(nn.Module):
    """
    Abstract base class for all models.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the model with given parameters.

        Args:
            params: A dictionary of model configuration parameters.
        """
        super().__init__()
        self.params = params

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward computation to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_num_params(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            int: Total count of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ExpertModule(nn.Module):
    """
    Single expert module: a GRUCell followed by LayerNorm.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """
        Initialize one expert.

        Args:
            input_dim: Dimension of input vector x_t.
            hidden_dim: Dimension of the hidden state.
        """
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, h_prev: Tensor) -> Tensor:
        """
        Compute one GRUCell step and apply LayerNorm.

        Args:
            x: Tensor of shape (batch_size, input_dim).
            h_prev: Tensor of shape (batch_size, hidden_dim).

        Returns:
            h_new: Tensor of shape (batch_size, hidden_dim).
        """
        h_new = self.gru(x, h_prev)
        h_new = self.ln(h_new)
        return h_new


class ModularRegimeRNN(Model):
    """
    Modular Regime RNN with K expert GRUs and a gating MLP that
    computes a mixture of expert hidden states for prediction.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the ModularRegimeRNN.

        Args:
            params: Configuration dictionary containing:
                - input_dim (int): Dimension of x_t.
                - output_dim (int): Dimension of y_hat_t.
                - experts (dict): {
                      'K': int, number of experts,
                      'hidden_dim': int, GRU hidden size
                  }
                - gating (dict): {
                      'depth': int, number of MLP layers,
                      'width': Optional[int], hidden width (defaults to hidden_dim),
                      'dropout': float, dropout probability
                  }
        """
        super().__init__(params)

        # Parse core dimensions
        input_dim = int(params.get("input_dim", 0))
        output_dim = int(params.get("output_dim", 0))
        experts_cfg = params.get("experts", {})
        gating_cfg = params.get("gating", {})

        # Expert settings
        self.K = int(experts_cfg.get("K", 1))
        self.hidden_dim = int(experts_cfg.get("hidden_dim", 1))

        # Gating settings
        self.gating_depth = int(gating_cfg.get("depth", 1))
        width_opt = gating_cfg.get("width", None)
        # Default gating width to hidden_dim if not set
        self.gating_width = (int(width_opt)
                             if isinstance(width_opt, int) and width_opt > 0
                             else self.hidden_dim)
        self.gating_dropout_p = float(gating_cfg.get("dropout", 0.0))

        # Build experts and gating
        self._build_experts(input_dim, self.hidden_dim, self.K)
        self._build_gating(input_dim,
                           num_experts=self.K,
                           hidden_dim=self.hidden_dim,
                           depth=self.gating_depth,
                           width=self.gating_width,
                           dropout_p=self.gating_dropout_p)

        # Readout layer
        self.readout = nn.Linear(self.hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _build_experts(self,
                       input_dim: int,
                       hidden_dim: int,
                       num_experts: int) -> None:
        """
        Construct expert GRUCell modules.

        Args:
            input_dim: Dimension of input x_t.
            hidden_dim: Dimension of GRU hidden state.
            num_experts: Number of experts K.
        """
        experts = [
            ExpertModule(input_dim=input_dim, hidden_dim=hidden_dim)
            for _ in range(num_experts)
        ]
        self.experts = nn.ModuleList(experts)

    def _build_gating(self,
                      input_dim: int,
                      num_experts: int,
                      hidden_dim: int,
                      depth: int,
                      width: int,
                      dropout_p: float) -> None:
        """
        Construct the gating MLP network.

        Args:
            input_dim: Dimension of x_t.
            num_experts: Number of experts K.
            hidden_dim: Hidden dimension of each expert.
            depth: Number of MLP layers.
            width: Width of hidden layers.
            dropout_p: Dropout probability.
        """
        in_features = num_experts * hidden_dim + input_dim
        layers: List[nn.Module] = []

        if depth == 1:
            # Single-layer gating: apply weight_norm
            ln = nn.Linear(in_features, num_experts)
            layers.append(weight_norm(ln))
            self.gating_layers = nn.ModuleList(layers)
            self.gating_output = None
        else:
            # Hidden MLP layers with weight_norm
            first = nn.Linear(in_features, width)
            layers.append(weight_norm(first))
            for _ in range(depth - 1):
                lin = nn.Linear(width, width)
                layers.append(weight_norm(lin))
            self.gating_layers = nn.ModuleList(layers)
            # Final projection to K logits with weight_norm
            self.gating_output = weight_norm(nn.Linear(width, num_experts))

        self.gating_dropout = nn.Dropout(p=dropout_p)

    def forward(self,
                x_t: Tensor,
                h_prev: List[Tensor]) -> Any:
        """
        Forward pass for one time step.

        Args:
            x_t: Input tensor of shape (batch_size, input_dim).
            h_prev: List of K hidden-state tensors, each
                    of shape (batch_size, hidden_dim).

        Returns:
            h_new: List of K updated hidden states.
            y_hat: Tensor of shape (batch_size, output_dim).
            g_t: Gating probabilities tensor of shape (batch_size, K).
        """
        # 1) Expert updates
        h_new: List[Tensor] = []
        for k, expert in enumerate(self.experts):
            h_k = expert(x_t, h_prev[k])
            h_new.append(h_k)

        # 2) Gating MLP
        gating_input = torch.cat(h_new + [x_t], dim=1)
        out = gating_input
        if self.gating_output is not None:
            # Multiple-layer MLP
            for layer in self.gating_layers:
                out = F.relu(layer(out))
                out = self.gating_dropout(out)
            logits = self.gating_output(out)
        else:
            # Single-layer gating: apply layer, dropout
            out = self.gating_layers[0](out)
            out = self.gating_dropout(out)
            logits = out
        g_t = F.softmax(logits, dim=-1)

        # 3) Mixture of expert states
        # Stack: (batch_size, K, hidden_dim)
        H = torch.stack(h_new, dim=1)
        # Weighted sum: (batch_size, hidden_dim)
        mixed_h = torch.sum(g_t.unsqueeze(-1) * H, dim=1)

        # 4) Readout to predictions
        y_hat = self.readout(mixed_h)

        return h_new, y_hat, g_t

    def _init_weights(self) -> None:
        """
        Initialize module weights:
          - GRUCell weights: Xavier uniform, biases to zero.
          - Linear weights (MLP + readout): Kaiming normal, biases to zero.
          - LayerNorm left at default (weight=1, bias=0).
        """
        for module in self.modules():
            if isinstance(module, nn.GRUCell):
                nn.init.xavier_uniform_(module.weight_ih)
                nn.init.xavier_uniform_(module.weight_hh)
                if module.bias_ih is not None:
                    nn.init.zeros_(module.bias_ih)
                if module.bias_hh is not None:
                    nn.init.zeros_(module.bias_hh)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def init_hidden(self,
                    batch_size: int,
                    device: Optional[torch.device] = None) -> List[Tensor]:
        """
        Create initial zero hidden states for all expert modules.

        Args:
            batch_size: Number of sequences in batch.
            device: Torch device for the tensors; if None, inferred
                    from module parameters.

        Returns:
            List of K zero tensors of shape (batch_size, hidden_dim).
        """
        if device is None:
            # Infer device from parameters
            device = next(self.parameters()).device
        return [
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.K)
        ]
