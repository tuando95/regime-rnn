"""
main.py

Entry point for Modular Regime RNN experiments:
  - Synthetic data generation
  - Dataset preparation
  - Hyperparameter search (Optuna)
  - Model training
  - Evaluation and metrics reporting
  - Checkpointing and metric saving
"""

import os
import argparse
import yaml
import logging
import json
import copy
import numpy as np
import time

import torch

import utils
from synthetic_data_generator import SyntheticDataGenerator
from dataset_loader import DatasetLoader
from hyperparameter_search import HyperparameterSearch
from model import ModularRegimeRNN
from trainer import Trainer
from evaluator import Evaluator
from ablation_studies import run_all_ablations
from comparative_studies import run_comparative_analysis


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Modular Regime RNN Experiment Pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["all", "search", "train", "eval"],
        default="all",
        help=(
            "Mode of operation: "
            "'all' = search → train → eval; "
            "'search' = hyperparameter search only; "
            "'train' = train (uses defaults or last search) + eval; "
            "'eval' = load checkpoint + eval only"
        )
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save logs, checkpoints, and metrics."
    )
    parser.add_argument(
        "--checkpoint", "-ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation (required in eval mode)."
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run ablation studies instead of standard train/eval."
    )
    parser.add_argument(
        "--run-comparative",
        action="store_true",
        help="Run comparative analysis instead of standard train/eval."
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load YAML configuration from file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def main():
    # 1. Parse arguments
    args = parse_args()

    # Handle mutually exclusive study flags
    if args.run_ablation and args.run_comparative:
        logger.error("--run-ablation and --run-comparative cannot be used together.")
        return
    # If running studies, certain modes like 'eval' or 'search' alone might not make sense
    if (args.run_ablation or args.run_comparative) and args.mode == "eval":
        logger.warning("--mode=eval requested with study flag; study will run using base/default parameters.")
        # Mode is not changed; study logic will proceed.
    if (args.run_ablation or args.run_comparative) and args.mode == "search":
        logger.warning("--mode=search requested with study flag; only search will run, not the study itself unless mode is 'all'.")
        # Mode is not changed; search block will execute, but study block might not.

    # 2. Load configuration
    config = load_config(args.config)

    # Override log_dir if provided
    if args.log_dir:
        config.setdefault("training", {})["log_dir"] = args.log_dir

    # 3. Configure logging
    logger = utils.configure_logging()
    logger.info(f"Using configuration file: {args.config}")

    # 4. Set random seeds and device
    seed = int(config.get("seed", 42))
    utils.seed_everything(seed)
    device = utils.get_device()
    logger.info(f"Random seed set to {seed}; device = {device}")

    # 5. Generate synthetic data
    logger.info("Generating synthetic data...")
    data_cfg = config.get("data", {})
    generator = SyntheticDataGenerator(data_cfg)
    X, y, regimes = generator.generate()
    logger.info(
        f"Generated data shapes: X={X.shape}, y={y.shape}, regimes={regimes.shape}"
    )

    # 6. Prepare datasets
    logger.info("Preparing train/val/test splits and preprocessing...")
    split_cfg = config.get("split", {"train": 0.6, "val": 0.2, "test": 0.2})
    loader = DatasetLoader((X, y, regimes), split_cfg)
    datasets = loader.load()
    logger.info(
        f"Dataset sizes — train: {len(datasets['train'])}, "
        f"val: {len(datasets['val'])}, test: {len(datasets['test'])}"
    )

    # 7. Attach datasets for hyperparameter search
    config["datasets"] = {
        "train": datasets["train"],
        "val": datasets["val"],
        "test": datasets["test"] # Add test set for convenience in studies
    }

    best_model_params = None

    # 8. Hyperparameter search
    if args.mode in ("all", "search"):
        logger.info("Starting hyperparameter search with Optuna...")
        search_space = config.get("hyperparameter_search", {})
        hps = HyperparameterSearch(search_space, config)
        best_model_params = hps.run()
        logger.info(f"Hyperparameter search completed. Best params:\n{best_model_params}")
        if args.mode == "search":
            # Only search requested; exit here
            return

    # 9. Build final model parameters
    if best_model_params:
        model_params = best_model_params
    else:
        # No search run; fallback to defaults from config options
        logger.info("No search results found; using default hyperparameters from config.")
        # Infer input/output dims from training data
        X_train, y_train, _ = datasets["train"].tensors
        input_dim = int(X_train.shape[-1])
        output_dim = int(y_train.shape[-1])
        # Experts defaults (using first option)
        experts_cfg = config.get("model", {}).get("experts", {})
        K_opts = experts_cfg.get("K_options", [1])
        hidden_opts = experts_cfg.get("hidden_dim_options", [input_dim])
        K = int(K_opts[-1]) # Use first option
        hidden_dim = int(hidden_opts[-1]) # Use first option
        # Gating defaults (Attention mechanism - using first option)
        gating_cfg = config.get("model", {}).get("gating", {})
        attn_head_opts = gating_cfg.get("attention_heads_options", [4]) # Default to 4 heads if missing
        dropout_opts = gating_cfg.get("dropout_options", [0.0])
        attention_heads = int(attn_head_opts[-1]) # Use first option
        dropout = float(dropout_opts[-1]) # Use first option

        # Construct parameters for Attention Gating model
        model_params = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "experts": {"K": K, "hidden_dim": hidden_dim},
            "gating": {"attention_heads": attention_heads, "dropout": dropout}
        }
        logger.info(f"Default model params (Attention Gate):\n{model_params}")

    # 10. Instantiate model (needed for studies base case)
    logger.info("Initializing base ModularRegimeRNN model configuration...")
    model = ModularRegimeRNN(model_params).to(device)
    num_params = model.get_num_params()
    logger.info(f"Base model config has {num_params:,} trainable parameters.")

    # Determine default log/output directory
    log_dir = config.get("training", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # === Run Ablation Studies ===
    if args.run_ablation:
        ablation_output_dir = os.path.join(log_dir, "ablation_studies")
        run_all_ablations(config, datasets, ablation_output_dir)

    # === Run Comparative Analysis ===
    elif args.run_comparative:
        comparative_output_dir = os.path.join(log_dir, "comparative_analysis")
        # Pass the determined model_params (best from search or default)
        run_comparative_analysis(config, datasets, comparative_output_dir, model_params)

    # === Standard Train/Eval Workflow ===
    else:
        # 11. Training
        if args.mode in ("all", "train"):
            logger.info("Starting training phase...")
            # Re-instantiate model with the chosen params for a clean train run
            model = ModularRegimeRNN(model_params).to(device)
            trainer = Trainer(
                model,
                {"train": datasets["train"], "val": datasets["val"]},
                config
            )
            logger.info("Starting training...")
            start_time = time.time()
            trained_model, _ = trainer.train() # Unpack the model
            train_time = time.time() - start_time
            logger.info(f"Training completed in {train_time:.2f} seconds.")

            # 12. Save checkpoint if we trained
            ckpt_path = os.path.join(log_dir, "model_checkpoint.pt")
            torch.save(trained_model.state_dict(), ckpt_path)
            logger.info(f"Saved trained model checkpoint to: {ckpt_path}")

        elif args.mode == "eval":
             # Load from checkpoint for eval mode
            if not args.checkpoint:
                logger.error("Checkpoint path must be provided in eval mode.")
                return
            logger.info(f"Loading model state from checkpoint: {args.checkpoint}")
            state = torch.load(args.checkpoint, map_location=device)
            # Instantiate model first, then load state
            model = ModularRegimeRNN(model_params).to(device)
            model.load_state_dict(state)
            trained_model = model # Use loaded model for evaluation
        else:
            # Mode is 'search' - no training/eval needed here
            logger.info("Search mode selected - skipping training and evaluation.")
            return

        # 13. Evaluation (only if training occurred or eval mode selected)
        if args.mode in ("all", "train", "eval"):
            logger.info("Starting evaluation phase...")
            evaluator = Evaluator(trained_model, datasets, config)
            metrics = evaluator.evaluate()

            # Log and print metrics
            logger.info("Evaluation results:")
            # Custom print for numpy array
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"  {key}: array shape {value.shape}")
                else:
                    logger.info(f"  {key}: {value}")

            # Save metrics to JSON
            metrics_path = os.path.join(log_dir, "metrics.json")
            try:
                # Convert numpy arrays for JSON
                metrics_save = copy.deepcopy(metrics)
                if isinstance(metrics_save.get("specialization_index"), np.ndarray):
                    metrics_save["specialization_index"] = metrics_save["specialization_index"].tolist()
                with open(metrics_path, "w") as mf:
                    json.dump(metrics_save, mf, indent=2)
                logger.info(f"Saved evaluation metrics to: {metrics_path}")
            except Exception as e:
                 logger.error(f"Failed to save metrics: {e}")


if __name__ == "__main__":
    main()
