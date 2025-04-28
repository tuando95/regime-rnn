"""
utils.py

Utility functions for experiment reproducibility, device setup,
matrix operations, statistical sampling, clustering metrics, and logging.
"""

import os
import random
import logging

import numpy as np
import torch
import yaml
from scipy.stats import wishart
from sklearn.metrics import silhouette_score

# Numeric tolerance constant
EPS: float = 1e-8

# Attempt to load project configuration
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(_CONFIG_PATH, "r") as _f:
        _CONFIG = yaml.safe_load(_f) or {}
except Exception:
    _CONFIG = {}


def seed_everything(seed: int) -> None:
    """
    Seed all random number generators for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure cuDNN for reproducibility/performance
    precision_cfg = _CONFIG.get("precision", {})
    cudnn_deterministic = precision_cfg.get("cudnn_deterministic", False)
    cudnn_benchmark = precision_cfg.get("cudnn_benchmark", True)

    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def get_device() -> torch.device:
    """
    Get the available computation device.

    Returns:
        torch.device: 'cuda' if available, else 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spectral_radius_rescale(A: np.ndarray, tol: float) -> np.ndarray:
    """
    Rescale a square matrix A so that its spectral radius <= tol.

    Args:
        A (np.ndarray): Square matrix of shape (d, d).
        tol (float): Maximum allowed spectral radius.

    Returns:
        np.ndarray: Rescaled matrix if needed, else original.
    """
    A_arr = np.asarray(A, dtype=float)
    eigs = np.linalg.eigvals(A_arr)
    spec_rad = float(np.max(np.abs(eigs)))
    if spec_rad > tol and spec_rad > 0:
        scale = spec_rad / tol
        return A_arr / scale
    return A_arr


def wishart_sample(d: int,
                   cov: np.ndarray,
                   df: int = None) -> np.ndarray:
    """
    Sample a positive-definite matrix from a Wishart distribution and
    enforce a maximum condition number.

    Args:
        d (int): Dimension of the matrix.
        cov (np.ndarray): Scale matrix (d x d), positive-definite.
        df (int, optional): Degrees of freedom. If None or < d, defaults to d.

    Returns:
        np.ndarray: Sampled covariance matrix (d x d).
    """
    # Degrees of freedom
    df_eff = df if isinstance(df, int) and df >= d else d

    # Sample Wishart
    W = wishart(df=df_eff, scale=cov).rvs()

    # Enforce maximum condition number if configured
    multivariate_cfg = _CONFIG.get("data", {}).get("multivariate", {})
    max_cond = multivariate_cfg.get("max_condition_number", None)

    try:
        cond = np.linalg.cond(W)
    except Exception:
        cond = np.inf

    if isinstance(max_cond, (int, float)) and cond > max_cond and cond != np.inf:
        # Regularize W by adding epsilon * I to reduce condition number
        # eps ≈ (cond/max_cond - 1) * (trace(W)/d)
        eps = (cond / float(max_cond) - 1.0) * (np.trace(W) / float(d))
        W = W + eps * np.eye(d)

    return W


def compute_silhouette(features: np.ndarray,
                       labels: np.ndarray) -> float:
    """
    Compute the silhouette score for clustering quality.

    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features).
        labels (np.ndarray): 1D array of integer labels (n_samples,).

    Returns:
        float: Silhouette score in [-1, 1], or NaN if undefined.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got ndim={features.ndim}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got ndim={labels.ndim}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Number of samples in features and labels must match")

    unique_labels = np.unique(labels)
    if unique_labels.shape[0] < 2:
        # Silhouette is undefined for a single cluster
        return float("nan")

    return silhouette_score(features, labels, metric="euclidean")


def configure_logging(level: str = None) -> logging.Logger:
    """
    Configure the root logger with a standard format and level.

    Args:
        level (str, optional): Logging level (e.g., 'INFO', 'DEBUG').
                               If None, defaults to 'INFO'.

    Returns:
        logging.Logger: Configured root logger.
    """
    cfg_level = level or _CONFIG.get("logging", {}).get("level", "INFO")
    if isinstance(cfg_level, str):
        cfg_level = cfg_level.upper()
    numeric_level = getattr(logging, cfg_level, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s [%(module)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger()
