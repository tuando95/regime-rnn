"""
ablation_studies.py

Functions to run ablation studies as described in the paper:
1. Module Count vs Size (K vs h)
2. Gating Complexity
3. Regularization (lambda_ent)
"""

import os
import copy
import logging
import json
import time
from typing import Dict, Any, List

import torch
import numpy as np
# Placeholder for potential future gating mechanisms
# from torch import nn 

import utils
from model import ModularRegimeRNN
from trainer import Trainer
from evaluator import Evaluator

logger = utils.configure_logging()

def _run_single_ablation_trial(
    model_params: Dict[str, Any],
    train_params: Dict[str, Any],
    datasets: Dict[str, torch.utils.data.TensorDataset],
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Instantiates, trains, and evaluates a single ModularRegimeRNN configuration.

    Args:
        model_params: Constructor arguments for ModularRegimeRNN.
        train_params: Overrides for the training section of the config 
                      (e.g., regularization).
        datasets: Dictionary containing 'train', 'val', 'test' TensorDatasets.
        base_config: The base configuration dictionary.

    Returns:
        Dictionary containing evaluation metrics. Returns empty if budget exceeded.
    """
    trial_config = copy.deepcopy(base_config)
    
    # Apply training parameter overrides
    trial_config.setdefault("training", {}).update(train_params)
    
    # Ensure essential keys are present
    for key in ['input_dim', 'output_dim']:
        if key not in model_params:
            logger.error(f"Missing required key '{key}' in model_params.")
            # Get dims from data if possible
            try:
                X_train, y_train, _ = datasets['train'].tensors
                model_params['input_dim'] = X_train.shape[-1]
                model_params['output_dim'] = y_train.shape[-1]
                logger.warning(f"Inferred {key} from dataset.")
            except Exception as e:
                 logger.error(f"Could not infer dimensions from dataset: {e}")
                 return {"error": f"Missing {key}"}
                 
    logger.info(f"Running trial with model_params: {model_params}")
    logger.info(f"and train_params: {train_params}")

    # Instantiate model
    try:
        model = ModularRegimeRNN(model_params)
    except Exception as e:
        logger.error(f"Error instantiating model: {e}")
        return {"error": str(e)}

    # Check parameter budget
    param_count = model.get_num_params()
    budget = int(base_config.get("model", {}).get("budget_params", 0))
    if budget > 0 and param_count > budget:
        logger.warning(
            f"Skipping trial: params={param_count} exceeds budget={budget}"
        )
        # Return special value or empty dict to indicate skipped trial
        return {"skipped": "param_budget_exceeded", "params": param_count}

    # Train
    try:
        trainer = Trainer(
            model,
            {"train": datasets["train"], "val": datasets["val"]},
            trial_config
        )
        start_time = time.time()
        trained_model = trainer.train()
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f}s")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return {"error": f"Training failed: {e}"}

    # Evaluate
    try:
        evaluator = Evaluator(
            trained_model,
            {"test": datasets["test"]}, # Evaluate on test set
            trial_config
        )
        metrics = evaluator.evaluate()
        metrics["params"] = param_count
        metrics["train_time_s"] = train_time
         # Convert numpy array to list for JSON serialization
        if "specialization_index" in metrics and isinstance(metrics["specialization_index"], np.ndarray):
             metrics["specialization_index"] = metrics["specialization_index"].tolist()
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return {"error": f"Evaluation failed: {e}"}

    return metrics

def run_ablation_k_vs_h(
    config: Dict[str, Any],
    datasets: Dict[str, torch.utils.data.TensorDataset]
) -> Dict[str, Any]:
    """
    Ablation Study 1: Module Count (K) vs Hidden Size (h).
    Varies K and h according to options in config['model']['experts']
    while respecting config['model']['budget_params'].
    """
    logger.info("Starting Ablation Study: K vs h")
    results = {}
    
    model_cfg = config.get("model", {})
    experts_cfg = model_cfg.get("experts", {})
    gating_cfg = model_cfg.get("gating", {})
    
    k_options = experts_cfg.get("K_options", [3, 5, 7])
    h_options = experts_cfg.get("hidden_dim_options", [64, 128, 256])
    
    # Use default gating params from config if not overridden by search later
    depth_opts = gating_cfg.get("depth_options", [1])
    dropout_opts = gating_cfg.get("dropout_options", [0.1]) # Use paper's default if not in config
    gating_depth = depth_opts[0]
    gating_dropout = dropout_opts[0]
    width_cfg = gating_cfg.get("width", None)

    # Get dims from data
    X_train, y_train, _ = datasets['train'].tensors
    input_dim = X_train.shape[-1]
    output_dim = y_train.shape[-1]

    for k in k_options:
        for h in h_options:
            trial_key = f"K={k}_h={h}"
            logger.info(f"--- Running trial: {trial_key} ---")
            
            # Determine gating width (defaults to h if null)
            gating_width = int(width_cfg) if isinstance(width_cfg, int) and width_cfg > 0 else h
            
            model_params = {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "experts": {"K": k, "hidden_dim": h},
                "gating": {
                    "depth": gating_depth,
                    "width": gating_width,
                    "dropout": gating_dropout
                }
            }
            
            # Use default training params for this ablation
            train_params = {} 
            
            metrics = _run_single_ablation_trial(model_params, train_params, datasets, config)
            results[trial_key] = metrics

    logger.info("Completed Ablation Study: K vs h")
    return results

def run_ablation_gating(
    config: Dict[str, Any],
    datasets: Dict[str, torch.utils.data.TensorDataset]
) -> Dict[str, Any]:
    """
    Ablation Study 2: Gating Complexity.
    Compares different gating mechanisms: linear, MLP(1-3), [Self-Attention, 1D-CNN placeholders].
    Uses the 'best' K and h (e.g., first option in config or from hyperparam search if available).
    """
    logger.info("Starting Ablation Study: Gating Complexity")
    results = {}

    model_cfg = config.get("model", {})
    experts_cfg = model_cfg.get("experts", {})
    gating_cfg = model_cfg.get("gating", {})

    # Use first K/h option as base unless specified otherwise
    k = experts_cfg.get("K_options", [5])[0] # Default K=5 if not in config
    h = experts_cfg.get("hidden_dim_options", [128])[0] # Default h=128

    # Gating types to test
    # Paper mentions: linear, MLP(1–3), self-attention, 1D-CNN
    gating_options = {
        "linear": {"depth": 1}, 
        "mlp_2": {"depth": 2},
        "mlp_3": {"depth": 3},
    }

    # Default dropout and width
    dropout = gating_cfg.get("dropout_options", [0.1])[0]
    width_cfg = gating_cfg.get("width", None)
    gating_width = int(width_cfg) if isinstance(width_cfg, int) and width_cfg > 0 else h

    # Get dims from data
    X_train, y_train, _ = datasets['train'].tensors
    input_dim = X_train.shape[-1]
    output_dim = y_train.shape[-1]
    
    for name, gate_params in gating_options.items():
        trial_key = f"Gating={name}"
        logger.info(f"--- Running trial: {trial_key} ---")

        model_params = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "experts": {"K": k, "hidden_dim": h},
            "gating": {
                "depth": gate_params["depth"],
                "width": gating_width,
                "dropout": dropout
                # Potentially add 'type': name here if model is adapted
            }
        }
        
        # Use default training params
        train_params = {} 

        metrics = _run_single_ablation_trial(model_params, train_params, datasets, config)
        results[trial_key] = metrics

    logger.info("Completed Ablation Study: Gating Complexity")
    return results


def run_ablation_regularization(
    config: Dict[str, Any],
    datasets: Dict[str, torch.utils.data.TensorDataset]
) -> Dict[str, Any]:
    """
    Ablation Study 3: Regularization (Entropy).
    Varies lambda_ent over {0, 1e-5, 1e-4}.
    Uses the 'best' K, h, and gating (e.g., first option or from search).
    """
    logger.info("Starting Ablation Study: Regularization (lambda_ent)")
    results = {}

    model_cfg = config.get("model", {})
    experts_cfg = model_cfg.get("experts", {})
    gating_cfg = model_cfg.get("gating", {})
    
    # Base model params (using defaults or first option)
    k = experts_cfg.get("K_options", [5])[0]
    h = experts_cfg.get("hidden_dim_options", [128])[0]
    depth = gating_cfg.get("depth_options", [2])[0] # Default to MLP depth 2
    dropout = gating_cfg.get("dropout_options", [0.1])[0]
    width_cfg = gating_cfg.get("width", None)
    gating_width = int(width_cfg) if isinstance(width_cfg, int) and width_cfg > 0 else h
    
    # Get dims from data
    X_train, y_train, _ = datasets['train'].tensors
    input_dim = X_train.shape[-1]
    output_dim = y_train.shape[-1]

    model_params = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "experts": {"K": k, "hidden_dim": h},
        "gating": {
            "depth": depth,
            "width": gating_width,
            "dropout": dropout
        }
    }
    
    # Lambda values to test
    lambda_ent_values = [0.0, 1e-5, 1e-4] # As specified in paper section 6.1

    for l_ent in lambda_ent_values:
        trial_key = f"lambda_ent={l_ent}"
        logger.info(f"--- Running trial: {trial_key} ---")
        
        # Override training params
        train_params = {
            "regularization": {
                 # Keep default L2 from config, override only entropy
                 "lambda_l2_options": config.get("training",{}).get("regularization",{}).get("lambda_l2_options", [1e-6]),
                 "lambda_entropy_options": [l_ent] 
            }
        }
            
        metrics = _run_single_ablation_trial(model_params, train_params, datasets, config)
        results[trial_key] = metrics
            
    logger.info("Completed Ablation Study: Regularization (lambda_ent)")
    return results


def run_all_ablations(
    config: Dict[str, Any],
    datasets: Dict[str, torch.utils.data.TensorDataset],
    output_dir: str
) -> None:
    """Runs all ablation studies and saves results to JSON files."""
    logger.info("===== Starting All Ablation Studies =====")
    os.makedirs(output_dir, exist_ok=True)

    results_k_vs_h = run_ablation_k_vs_h(config, datasets)
    k_vs_h_path = os.path.join(output_dir, "ablation_k_vs_h_results.json")
    with open(k_vs_h_path, "w") as f:
        json.dump(results_k_vs_h, f, indent=2)
    logger.info(f"Saved K vs h ablation results to {k_vs_h_path}")

    results_gating = run_ablation_gating(config, datasets)
    gating_path = os.path.join(output_dir, "ablation_gating_results.json")
    with open(gating_path, "w") as f:
        json.dump(results_gating, f, indent=2)
    logger.info(f"Saved gating ablation results to {gating_path}")

    results_reg = run_ablation_regularization(config, datasets)
    reg_path = os.path.join(output_dir, "ablation_regularization_results.json")
    with open(reg_path, "w") as f:
        json.dump(results_reg, f, indent=2)
    logger.info(f"Saved regularization ablation results to {reg_path}")

    logger.info("===== Completed All Ablation Studies =====")

# Example usage (if run directly, though intended to be called from main.py)
# if __name__ == '__main__':
#     # This requires setting up a dummy config and datasets
#     # For testing purposes only
#     print("Running ablation studies module directly (requires dummy setup)")
#     # config = ... load dummy config ...
#     # datasets = ... create dummy datasets ...
#     # run_all_ablations(config, datasets, "ablation_results") 