"""
evaluator.py

Evaluator for trained time series models. Computes:
  - Mean Squared Error (MSE)
  - Regime identification accuracy
  - Specialization Index (SI) per expert/regime
  - Parameter efficiency (MSE per million parameters)

Interface:
    evaluator = Evaluator(model, datasets, config)
    metrics = evaluator.evaluate()
"""

from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, accuracy_score


class Evaluator:
    """
    Evaluator orchestrates inference on the test set and computes
    evaluation metrics for ModularRegimeRNN models.

    Attributes:
        model: Trained time series model with a forward() returning
               (h_prev_list, y_hat, g_t) and get_num_params().
        datasets: Dict[str, TensorDataset] containing 'test' split.
        config: Configuration dictionary loaded from YAML.
        device: torch.device for computation.
        use_amp: Whether to use mixed-precision inference.
        test_loader: DataLoader for the test set.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        datasets: Dict[str, TensorDataset],
        config: Dict[str, Any]
    ) -> None:
        """
        Args:
            model: Trained model instance.
            datasets: Dictionary with keys 'train', 'val', 'test'.
            config: Full configuration dict.

        Raises:
            ValueError: If 'test' split is missing in datasets.
        """
        self.model = model
        self.datasets = datasets
        self.config = config

        if "test" not in datasets:
            raise ValueError("Test dataset ('test') not found in datasets")

        # Determine device
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            # Model has no parameters
            self.device = torch.device("cpu")

        # Mixed precision flag
        self.use_amp = bool(config.get("precision", {})
                                   .get("mixed_fp16", False))

        # Prepare test DataLoader
        self.test_loader = DataLoader(
            datasets["test"], batch_size=1, shuffle=False
        )

    def evaluate(self) -> Dict[str, Any]:
        """
        Run the model on the test set and compute metrics.

        Returns:
            A dict containing:
              - 'mse': float, mean squared error over all time points.
              - 'regime_accuracy': float, regime‐ID accuracy.
              - 'parameter_efficiency': float, MSE per million params.
              - 'num_parameters': int, total trainable parameters.
              - 'specialization_index': np.ndarray of shape (K, R).
        """
        self.model.eval()

        # Containers for concatenated results
        y_trues = []
        y_preds = []
        regimes_all = []
        gating_all = []

        with torch.no_grad():
            for batch in self.test_loader:
                x_batch, y_batch, regimes_batch = batch
                x_seq = x_batch[0]           
                y_seq = y_batch[0]           
                regimes_seq = regimes_batch[0].cpu().numpy()

                x_seq = x_seq.to(self.device)
                y_seq = y_seq.to(self.device)

                # Initialize hidden states (returns list)
                if hasattr(self.model, "init_hidden"):
                    try:
                        h_prev = self.model.init_hidden(
                            batch_size=1, device=self.device
                        ) # h_prev is List[Tensor]
                    except Exception:
                        h_prev = None
                else:
                    h_prev = None

                seq_preds = []
                seq_gatings = []
                T_seq = x_seq.size(0)

                for t in range(T_seq):
                    x_t = x_seq[t].unsqueeze(0) 
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(x_t, h_prev)
                    else:
                        outputs = self.model(x_t, h_prev)

                    # Expect 4 return values now
                    if not (isinstance(outputs, (tuple, list)) and len(outputs) == 4):
                        raise RuntimeError(
                            "Model.forward must return (h_expert_new, y_hat, g_t, logits)"
                        )
                    # Unpack, assign new states back to h_prev, ignore logits
                    h_prev, y_hat, g_t, _ = outputs 

                    # Collect prediction and gating (g_t)
                    seq_preds.append(y_hat.detach().cpu().numpy().reshape(-1))
                    seq_gatings.append(g_t.detach().cpu().numpy().reshape(-1))

                # Store sequence results
                y_trues.append(y_seq.detach().cpu().numpy())        # (T, m)
                y_preds.append(np.vstack(seq_preds))                # (T, m)
                regimes_all.append(regimes_seq)                     # (T,)
                gating_all.append(np.vstack(seq_gatings))          # (T, K)

        # Flatten across all sequences
        y_true_flat = np.concatenate(y_trues, axis=0)      # (N_total, m)
        y_pred_flat = np.concatenate(y_preds, axis=0)      # (N_total, m)
        regimes_flat = np.concatenate(regimes_all, axis=0) # (N_total,)
        gating_flat = np.concatenate(gating_all, axis=0)   # (N_total, K)

        # Compute MSE
        mse_val = float(mean_squared_error(y_true_flat, y_pred_flat))

        # Compute regime‐ID accuracy
        regime_preds = np.argmax(gating_flat, axis=1)
        regime_acc = float(accuracy_score(regimes_flat, regime_preds))

        # Compute Specialization Index (SI)
        unique_regs = np.unique(regimes_flat)
        K = gating_flat.shape[1]
        R = unique_regs.shape[0]
        SI = np.zeros((K, R), dtype=float)
        for k in range(K):
            total_gk = gating_flat[:, k].sum()
            if total_gk <= 0.0:
                total_gk = 1.0
            for idx, r in enumerate(unique_regs):
                mask = (regimes_flat == r)
                SI[k, idx] = gating_flat[mask, k].sum() / total_gk

        # Parameter efficiency: MSE per million parameters
        num_params = int(self.model.get_num_params())
        params_millions = num_params / 1e6 if num_params > 0 else 1.0
        param_eff = mse_val / params_millions

        # Compile all metrics
        metrics: Dict[str, Any] = {
            "mse": mse_val,
            "regime_accuracy": regime_acc,
            "parameter_efficiency": param_eff,
            "num_parameters": num_params,
            "specialization_index": SI,
        }
        return metrics
