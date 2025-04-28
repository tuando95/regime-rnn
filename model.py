"""
model.py

Defines the abstract Model base class and the ModularRegimeRNN model
with K expert GRU modules and a lightweight gating MLP.
"""

from typing import Any, Dict, List, Optional, Tuple

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
        Initialize the ModularRegimeRNN with Attention Gating.

        Args:
            params: Configuration dictionary containing:
                - input_dim (int): Dimension of x_t.
                - output_dim (int): Dimension of y_hat_t.
                - experts (dict): { 'K': int, 'hidden_dim': int }
                - gating (dict): {
                      'attention_heads': int, # Num heads for MultiheadAttention
                      'dropout': float
                  }
        """
        super().__init__(params)

        # Parse core dimensions
        self.input_dim = int(params.get("input_dim", 0))
        output_dim = int(params.get("output_dim", 0))
        experts_cfg = params.get("experts", {})
        gating_cfg = params.get("gating", {})

        # Expert settings
        self.K = int(experts_cfg.get("K", 1))
        self.hidden_dim = int(experts_cfg.get("hidden_dim", 1))

        # Gating settings
        self.attention_heads = int(gating_cfg.get("attention_heads", 4))
        self.gating_dropout_p = float(gating_cfg.get("dropout", 0.0))

        # Ensure hidden_dim is divisible by number of heads for MultiheadAttention
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError(
                f"Expert hidden_dim ({self.hidden_dim}) must be divisible by "
                f"attention_heads ({self.attention_heads})"
            )

        # Build experts and gating
        self._build_experts(self.input_dim, self.hidden_dim, self.K)
        self._build_gating(self.input_dim, self.hidden_dim, self.K, self.attention_heads, self.gating_dropout_p)

        # Readout layer
        self.readout = nn.Linear(self.hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _build_experts(self, input_dim: int, hidden_dim: int, num_experts: int) -> None:
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

    def _build_gating(self, input_dim: int, hidden_dim: int, num_experts: int, num_heads: int, dropout_p: float) -> None:
        """
        Construct the attention-based gating mechanism.
        Uses x_t to generate query, h_expert_new as keys/values.
        """
        # Linear layer to transform input x_t into a query vector of hidden_dim
        self.query_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-head Attention layer
        # embed_dim = hidden_dim (operating on expert states)
        # kdim, vdim defaults to embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_p, batch_first=True
        )
        # LayerNorm after attention
        self.attention_ln = nn.LayerNorm(hidden_dim)

        # Final linear layer to project attention output context to K logits
        self.gating_output = nn.Linear(hidden_dim, num_experts)

        # Separate dropout for the final layer? Maybe not needed if attention has it.
        # self.gating_dropout = nn.Dropout(p=dropout_p) # Re-using attention dropout for now

    def forward(self,
                x_t: Tensor,
                h_expert_prev: List[Tensor] # Only expert states needed now
               ) -> Tuple[List[Tensor], Tensor, Tensor, Tensor]: # Back to 4 return values
        """
        Forward pass for one time step using attention gating.

        Args:
            x_t: Input tensor of shape (batch_size, input_dim).
            h_expert_prev: List of K expert hidden-state tensors (batch_size, hidden_dim).

        Returns:
            h_expert_new: List of K updated expert hidden states.
            y_hat: Prediction tensor of shape (batch_size, output_dim).
            g_t: Gating probabilities tensor of shape (batch_size, K).
            logits: Gating logits tensor of shape (batch_size, K).
        """
        # 1) Expert updates
        h_expert_new: List[Tensor] = []
        for k, expert in enumerate(self.experts):
            h_k = expert(x_t, h_expert_prev[k])
            h_expert_new.append(h_k)

        # 2) Attention Gating
        # Project x_t to query vector (batch_size, 1, hidden_dim)
        # Note: MHA expects sequence dimension, so unsqueeze
        query = self.query_proj(x_t).unsqueeze(1)

        # Stack expert states to form Key and Value sequence (batch_size, K, hidden_dim)
        expert_states_stacked = torch.stack(h_expert_new, dim=1)
        keys = expert_states_stacked
        values = expert_states_stacked

        # Apply multi-head attention
        # attn_output: (batch_size, 1, hidden_dim) - context vector based on query
        # attn_weights: (batch_size, 1, K) - alignment scores (optional)
        attn_output, _ = self.attention(query=query, key=keys, value=values, need_weights=False)

        # Squeeze sequence dimension, apply LayerNorm
        # Residual connection? Maybe add query + attn_output before LN
        attn_output_squeezed = attn_output.squeeze(1)
        context_vector = self.attention_ln(query.squeeze(1) + attn_output_squeezed) # Added residual connection

        # Compute logits from the attention context vector
        logits = self.gating_output(context_vector)
        # Compute probabilities
        g_t = F.softmax(logits, dim=-1)

        # 3) Mixture of expert states using g_t
        mixed_h = torch.sum(g_t.unsqueeze(-1) * expert_states_stacked, dim=1)

        # 4) Readout to predictions
        y_hat = self.readout(mixed_h)

        # Return new expert states, prediction, gating probs, gating logits
        # Note: No separate gate state to return
        return h_expert_new, y_hat, g_t, logits

    def _init_weights(self) -> None:
        """
        Initialize module weights:
          - GRUCell weights: Xavier uniform, biases to zero.
          - Linear weights (readout + gating output): Kaiming normal, biases zero.
          - LayerNorm left at default.
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
                # Kaiming normal is okay for query_proj, readout, gating_output
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # MultiheadAttention weights are initialized reasonably by default
            # LayerNorm weights/biases default to 1/0

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> List[Tensor]:
        """
        Create initial zero hidden states for experts only.

        Returns:
            List of K zero tensors for experts (batch_size, hidden_dim).
        """
        if device is None:
            device = next(self.parameters()).device
        # Only expert states now
        expert_hidden_states = [
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.K)
        ]
        return expert_hidden_states
