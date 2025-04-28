## dataset_loader.py

"""
DatasetLoader

Splits raw time-series data into train/validation/test sets, applies
interpolation and normalization preprocessing, and wraps each split
into a torch.utils.data.TensorDataset containing (features, targets, regimes).

Interface:
    loader = DatasetLoader((X, y, regimes), split_cfg)
    datasets = loader.load()
    # datasets is a dict with keys 'train', 'val', 'test'
    # each value is a TensorDataset of (X_tensor, y_tensor, regimes_tensor)
"""

from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import TensorDataset

import utils


class DatasetLoader:
    """
    DatasetLoader handles splitting and preprocessing of raw sequence data.

    Args:
        raw: Tuple containing
            - X: np.ndarray of shape (N, T, d), input time series
            - y: np.ndarray of shape (N, T, m), target time series
            - regimes: np.ndarray of shape (N, T), integer regime labels
        split_cfg: Dict with keys 'train', 'val', 'test' giving
            fractional splits that sum to 1.0.
    """

    def __init__(
        self,
        raw: Tuple[np.ndarray, np.ndarray, np.ndarray],
        split_cfg: Dict[str, float]
    ) -> None:
        # Unpack raw data
        X, y, regimes = raw
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (N,T,d), got shape {X.shape}")
        if y.ndim != 3:
            raise ValueError(f"y must be 3D (N,T,m), got shape {y.shape}")
        if regimes.ndim != 2:
            raise ValueError(f"regimes must be 2D (N,T), got shape {regimes.shape}")

        N, T, d = X.shape
        N_y, T_y, m = y.shape
        N_r, T_r = regimes.shape

        if N != N_y or N != N_r:
            raise ValueError("Number of sequences must match in X, y, and regimes")
        if T != T_y or T != T_r:
            raise ValueError("Sequence length must match in X, y, and regimes")

        # Store raw arrays
        self.X_raw = X.astype(float)
        self.y_raw = y.astype(float)
        self.regimes_raw = regimes.astype(int)

        # Problem dimensions
        self.N = N
        self.T = T
        self.d = d
        self.m = m

        # Split configuration
        train_frac = float(split_cfg.get("train", 0.8))
        val_frac = float(split_cfg.get("val", 0.1))
        test_frac = float(split_cfg.get("test", 0.1))
        total = train_frac + val_frac + test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split fractions must sum to 1.0, got sum={total:.6f}"
            )
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

        # Preprocessing settings from global config
        pre_cfg = utils._CONFIG.get("preprocessing", {})
        self.interpolation = pre_cfg.get("interpolation", "linear")
        self.normalization = pre_cfg.get("normalization", "zscore")

    def load(self) -> Dict[str, TensorDataset]:
        """
        Split the raw data into train/val/test, apply preprocessing,
        and return a dict of TensorDatasets.

        Returns:
            A dict with keys 'train', 'val', 'test', each mapping to
            a TensorDataset of (X_tensor, y_tensor, regimes_tensor).
        """
        # Shuffle indices for splitting
        indices = np.arange(self.N)
        np.random.shuffle(indices)

        # Compute split sizes
        n_train = int(self.N * self.train_frac)
        n_val = int(self.N * self.val_frac)
        n_test = self.N - n_train - n_val

        # Partition indices
        idx_train = indices[:n_train]
        idx_val = indices[n_train : n_train + n_val]
        idx_test = indices[n_train + n_val :]

        datasets: Dict[str, TensorDataset] = {}
        for split_name, idx_split in [
            ("train", idx_train),
            ("val", idx_val),
            ("test", idx_test),
        ]:
            # Slice raw data
            X_split = self.X_raw[idx_split]      # shape (Ni, T, d)
            y_split = self.y_raw[idx_split]      # shape (Ni, T, m)
            regimes_split = self.regimes_raw[idx_split]  # shape (Ni, T)

            # Interpolation (if requested and if NaNs present)
            if self.interpolation == "linear" and np.isnan(X_split).any():
                X_split = self._interpolate_sequences(X_split)
            # else: leave X_split unchanged

            # Normalization
            if self.normalization == "zscore":
                X_split = self._zscore_normalize(X_split)
                # Normalize y independently
                y_split = self._zscore_normalize(y_split)

            # Convert to torch tensors
            X_tensor = torch.tensor(X_split, dtype=torch.float32)
            y_tensor = torch.tensor(y_split, dtype=torch.float32)
            regimes_tensor = torch.tensor(regimes_split, dtype=torch.long)

            # Wrap into TensorDataset
            datasets[split_name] = TensorDataset(
                X_tensor, y_tensor, regimes_tensor
            )

        return datasets

    def _interpolate_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Linearly interpolate NaNs in each sequence and feature channel.

        Args:
            X: np.ndarray of shape (Ni, T, D)

        Returns:
            X_imputed: np.ndarray of same shape with NaNs filled.
        """
        Ni, T, D = X.shape
        X_imputed = X.copy()
        for i in range(Ni):
            for d in range(D):
                seq = X_imputed[i, :, d]
                mask = np.isnan(seq)
                if not mask.any():
                    continue
                valid_idx = np.where(~mask)[0]
                if valid_idx.size >= 2:
                    valid_vals = seq[valid_idx]
                    # np.interp fills NaNs by linear interpolation
                    interp_vals = np.interp(
                        x=np.arange(T), xp=valid_idx, fp=valid_vals
                    )
                    X_imputed[i, :, d] = interp_vals
                elif valid_idx.size == 1:
                    # Only one valid point: fill all with that value
                    seq.fill(seq[valid_idx[0]])
                    X_imputed[i, :, d] = seq
                else:
                    # All values NaN: fill zeros
                    X_imputed[i, :, d] = 0.0
        return X_imputed

    def _zscore_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Apply per-sequence, per-feature z-score normalization.

        Args:
            X: np.ndarray of shape (Ni, T, D)

        Returns:
            X_norm: np.ndarray of same shape, zero mean unit variance.
        """
        Ni, T, D = X.shape
        X_norm = np.empty_like(X, dtype=float)
        eps = getattr(utils, "EPS", 1e-8)
        for i in range(Ni):
            seq = X[i]  # shape (T, D)
            mean = seq.mean(axis=0)    # (D,)
            std = seq.std(axis=0)      # (D,)
            std_adj = np.where(std < eps, 1.0, std)
            X_norm[i] = (seq - mean) / std_adj
        return X_norm
