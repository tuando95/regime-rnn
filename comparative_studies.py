"""
comparative_studies.py

Functions to run the comparative analysis against baseline models as described 
in the paper:
- ModularRegimeRNN
- MonolithicLSTM
- TransformerModel
- MarkovSwitchingAR

Collects metrics: Params (M), MSE, Regime Acc, Param-efficiency.
Performs paired t-tests for significance.
"""

import os
import copy
import logging
import json
import time
from typing import Dict, Any

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy import stats

import utils
from model import ModularRegimeRNN
from baseline import MonolithicLSTM, TransformerModel, MarkovSwitchingAR
from trainer import Trainer
from evaluator import Evaluator # May need adaptation for baselines

logger = utils.configure_logging()

def _get_baseline_params(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Constructs parameters for baseline models based on config."""
    model_cfg = config.get("model", {})
    baselines_cfg = config.get("baselines", {}) # Read baselines config
    budget = int(model_cfg.get("budget_params", 25000000))
    
    # Get dims from data config (assuming synthetic for now)
    data_syn = config.get("data", {}).get("synthetic", {})
    input_dim = int(data_syn.get("dimension", 1))
    output_dim = input_dim # Assuming prediction task matches input dim
    
    # Base params
    params = {"input_dim": input_dim, "output_dim": output_dim}

    # Iteratively find largest hidden dim h within budget for RNN/LSTM
    if model_name in ["MonolithicLSTM", "MonolithicRNN"]:
        baseline_specific_cfg = baselines_cfg.get(model_name.lower().replace("monolithic", ""), {}) # Get e.g., baselines.lstm
        d = input_dim
        m = output_dim
        best_h = 1 # Default small value
        if budget > 0:
            # Iterate through potential hidden sizes
            for h_candidate in range(8, 1025, 8): # Check sensible range with steps
                if model_name == "MonolithicLSTM":
                     # LSTM: 4 * (d*h + h*h + 2*h) + (h*m + m)
                     num_params = 4 * (d * h_candidate + h_candidate**2 + 2 * h_candidate) + (h_candidate * m + m)
                else: # MonolithicRNN (GRU)
                     # GRU: 3 * (d*h + h*h + 2*h) + (h*m + m)
                     num_params = 3 * (d * h_candidate + h_candidate**2 + 2 * h_candidate) + (h_candidate * m + m)

                if num_params <= budget:
                    best_h = h_candidate
                else:
                    # Stop searching once budget is exceeded
                    break 
            logger.info(f"Setting {model_name} hidden_dim to {best_h} ({num_params} params) based on budget {budget}")
        else:
             best_h = baseline_specific_cfg.get("hidden_dim_fallback", 64) # Use fallback from config
             logger.warning(f"No budget specified, setting {model_name} hidden_dim to default {best_h}")
        params["hidden_dim"] = best_h
    elif model_name == "TransformerModel":
        tf_cfg = baselines_cfg.get("transformer", {}) # Read transformer config
        # Use values from config, falling back to reasonable defaults if keys missing
        params["d_model"] = tf_cfg.get("d_model", 512) 
        params["nhead"] = tf_cfg.get("nhead", 8)
        params["num_layers"] = tf_cfg.get("num_layers", 6)
        # Default dim_feedforward to 4*d_model if not specified
        params["dim_feedforward"] = tf_cfg.get("dim_feedforward", 4 * params["d_model"])
        params["dropout"] = tf_cfg.get("dropout", 0.1)
        logger.info(f"Using Transformer config: { {k: params[k] for k in ['d_model', 'nhead', 'num_layers']} }")
    elif model_name == "MarkovSwitchingAR":
         msar_cfg = baselines_cfg.get("markov_switching_ar", {}) # Read MSAR config
         params.update(msar_cfg) # Pass specific MSAR config
         logger.info(f"Using MSAR config: {msar_cfg}")

    return params

def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MSE for baseline predictions."""
    # Ensure inputs are flat for sklearn metrics
    if y_true.ndim > 2: y_true = y_true.reshape(-1, y_true.shape[-1])
    if y_pred.ndim > 2: y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    if y_true.shape[0] != y_pred.shape[0]:
         # Handle potential off-by-one from prediction shift
         min_len = min(y_true.shape[0], y_pred.shape[0])
         y_true = y_true[:min_len]
         y_pred = y_pred[:min_len]
         
    mse = float(mean_squared_error(y_true, y_pred))
    return {"mse": mse}

def run_comparative_analysis(
    config: Dict[str, Any],
    datasets: Dict[str, torch.utils.data.TensorDataset],
    output_dir: str,
    mod_rnn_params: Dict[str, Any] # Best params from search or defaults
) -> None:
    """
    Runs the comparative analysis: ModRegimeRNN vs Baselines.
    Saves results to JSON.
    """
    logger.info("===== Starting Comparative Analysis =====")
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    device = utils.get_device()
    
    # Prepare test data (needed for baselines)
    X_test, y_test, regimes_test = datasets['test'].tensors
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size=1, shuffle=False)
    y_test_np = y_test.cpu().numpy()

    # --- 1. Modular Regime RNN --- 
    model_name = "ModularRegimeRNN"
    logger.info(f"--- Running Model: {model_name} ---")
    try:
        model = ModularRegimeRNN(mod_rnn_params).to(device)
        trainer = Trainer(model, {"train": datasets["train"], "val": datasets["val"]}, config)
        start_time = time.time()
        trained_model, _ = trainer.train()
        train_time = time.time() - start_time
        
        evaluator = Evaluator(trained_model, {"test": datasets["test"]}, config)
        metrics = evaluator.evaluate()
        metrics["train_time_s"] = train_time
        if "specialization_index" in metrics and isinstance(metrics["specialization_index"], np.ndarray):
             metrics["specialization_index"] = metrics["specialization_index"].tolist()
        results[model_name] = metrics
        logger.info(f"{model_name} results: {metrics}")
        # Store predictions for t-test
        # Re-run prediction part of evaluator to get flat preds
        y_preds_modrnn = []
        with torch.no_grad():
             for x_batch, _, _ in test_loader:
                 x_seq = x_batch[0].to(device)
                 h_prev = trained_model.init_hidden(1, device=device)
                 seq_preds = []
                 for t in range(x_seq.size(0)):
                      # Correctly update h_prev with the output from the previous step
                      h_prev, y_hat, _ = trained_model(x_seq[t].unsqueeze(0), h_prev)
                      seq_preds.append(y_hat.cpu().numpy())
                 y_preds_modrnn.append(np.vstack(seq_preds))
        y_pred_modrnn_flat = np.concatenate(y_preds_modrnn, axis=0)
        
    except Exception as e:
        logger.error(f"Error running {model_name}: {e}", exc_info=True)
        results[model_name] = {"error": str(e)}

    # --- 2. Baselines --- 
    baseline_classes = {
        "MonolithicLSTM": MonolithicLSTM,
        "TransformerModel": TransformerModel,
        "MarkovSwitchingAR": MarkovSwitchingAR
    }
    
    baseline_preds = {}

    for model_name, BaselineClass in baseline_classes.items():
        logger.info(f"--- Running Model: {model_name} ---")
        try:
            baseline_params = _get_baseline_params(model_name, config)
            model = BaselineClass(baseline_params) # Already moves to device in init
            
            start_time = time.time()
            # Baselines have their own train method
            model.train_model({"train": datasets["train"], "val": datasets["val"]})
            train_time = time.time() - start_time
            logger.info(f"Training completed in {train_time:.2f}s")

            # Baselines have their own predict method
            gamma_all = None # Initialize gamma_all
            if model_name == "MarkovSwitchingAR":
                y_pred_tensor, gamma_all = model.predict(X_test)
                # Ensure gamma_all is a numpy array if it comes from the model
                if isinstance(gamma_all, torch.Tensor):
                    gamma_all = gamma_all.cpu().numpy()
            else:
                 y_pred_tensor = model.predict(X_test)

            y_pred_np = y_pred_tensor.cpu().numpy()
            baseline_preds[model_name] = y_pred_np.reshape(-1, y_pred_np.shape[-1]) # Store flat predictions
            
            # Simple evaluation (MSE)
            eval_metrics = _evaluate_predictions(y_test_np, y_pred_np)
            
            # Calculate regime accuracy for MSAR
            regime_accuracy = None
            if model_name == "MarkovSwitchingAR" and gamma_all is not None:
                try:
                    # gamma_all shape: (n_sequences, sequence_length, n_regimes)
                    # regimes_test shape: (n_sequences, sequence_length) or similar
                    z_true = regimes_test.cpu().numpy().flatten() # Flatten true regimes
                    
                    # Predict regimes from gamma probabilities
                    z_pred = np.argmax(gamma_all, axis=-1).flatten() # Flatten predicted regimes
                    
                    # Ensure lengths match (due to potential flattening differences or off-by-one)
                    min_len = min(len(z_true), len(z_pred))
                    z_true = z_true[:min_len]
                    z_pred = z_pred[:min_len]

                    regime_accuracy = accuracy_score(z_true, z_pred)
                    logger.info(f"Calculated Regime Accuracy for MSAR: {regime_accuracy:.4f}")
                except Exception as acc_e:
                    logger.error(f"Could not calculate regime accuracy for {model_name}: {acc_e}")
                    regime_accuracy = None # Fallback if calculation fails

            # Get param count (requires implementation in baselines or manual calc)
            if hasattr(model, 'get_num_params'): # Ideal case
                 param_count = model.get_num_params()
            elif hasattr(model, 'parameters'): # PyTorch module case
                 param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            elif model_name == "MarkovSwitchingAR": # Non-pytorch
                 # Calculate params for MSAR based on fitted parameters
                 if model.R is not None and model.d is not None and model.p is not None:
                     R = model.R
                     d = model.d
                     p_ar = model.p
                     # Pi (initial state probs): R (or R-1 if sum-to-1 constraint)
                     pi_params = R - 1 
                     # Transition Matrix T: R * (R-1) free params
                     t_params = R * (R - 1)
                     # AR Coeffs A: R * d * (p * d)
                     a_params = R * d * p_ar * d
                     # Covariance Sigma: R * d*(d+1)/2 unique params per matrix
                     sigma_params = R * d * (d + 1) // 2
                     param_count = pi_params + t_params + a_params + sigma_params
                     logger.info(f"Calculated MSAR params: R={R}, d={d}, p={p_ar} -> {param_count}")
                 else:
                     logger.warning("Could not calculate MSAR params: R, d, or p not set after training.")
                     param_count = -1
            else:
                 param_count = -1
            
            eval_metrics["params"] = param_count
            eval_metrics["num_parameters"] = param_count # Match evaluator output key
            if param_count > 0:
                 eval_metrics["parameter_efficiency"] = eval_metrics["mse"] / (param_count / 1e6)
            else: 
                 eval_metrics["parameter_efficiency"] = float('inf')
            eval_metrics["train_time_s"] = train_time
            
            # Regime accuracy is not directly applicable to all baselines
            eval_metrics["regime_accuracy"] = regime_accuracy # Use calculated value for MSAR, None otherwise

            results[model_name] = eval_metrics
            logger.info(f"{model_name} results: {eval_metrics}")

        except Exception as e:
            logger.error(f"Error running {model_name}: {e}", exc_info=True)
            results[model_name] = {"error": str(e)}
            
    # --- 3. Statistical Significance (Paired t-tests) --- 
    t_test_results = {}
    if "ModularRegimeRNN" in results and "error" not in results["ModularRegimeRNN"]:
        # Calculate squared errors per time step for ModRNN
        # Need to handle shape mismatches carefully (e.g., one-step-ahead)        
        # Assuming y_test_np shape is (n_sequences, sequence_length, n_features)
        y_test_flat = y_test_np.reshape(-1, y_test_np.shape[-1])
        min_len_modrnn = min(y_test_flat.shape[0], y_pred_modrnn_flat.shape[0]) 

        # Adjust target based on prediction type (one-step ahead vs full sequence)
        # Let's assume prediction lengths might differ, align them
        y_true_modrnn = y_test_flat[:min_len_modrnn]
        y_pred_modrnn = y_pred_modrnn_flat[:min_len_modrnn] 
        squared_errors_modrnn = np.sum((y_true_modrnn - y_pred_modrnn)**2, axis=1)


        for baseline_name, y_pred_baseline_flat in baseline_preds.items():
             logger.info(f"Running t-test: ModularRegimeRNN vs {baseline_name}")
             try:
                 min_len_baseline = min(y_test_flat.shape[0], y_pred_baseline_flat.shape[0])
                 y_true_baseline = y_test_flat[:min_len_baseline]
                 y_pred_baseline = y_pred_baseline_flat[:min_len_baseline]
                 
                 # Paired t-test on the squared errors, ensure equal length for comparison
                 common_len = min(len(squared_errors_modrnn), len(y_true_baseline))
                 squared_errors_baseline = np.sum((y_true_baseline[:common_len] - y_pred_baseline[:common_len])**2, axis=1)
                 
                 t_stat, p_value = stats.ttest_rel(squared_errors_modrnn[:common_len], squared_errors_baseline)
                 t_test_results[f"ModRNN_vs_{baseline_name}"] = {
                     "t_statistic": t_stat,
                     "p_value": p_value,
                     "significant_p_0_01": p_value < 0.01
                 }
             except Exception as e:
                 logger.error(f"Error during t-test vs {baseline_name}: {e}")
                 t_test_results[f"ModRNN_vs_{baseline_name}"] = {"error": str(e)}
    else:
         logger.warning("Skipping t-tests as ModularRegimeRNN results are missing or errored.")

    results["_t_tests"] = t_test_results
    
    # --- 4. Save Results --- 
    results_path = os.path.join(output_dir, "comparative_analysis_results.json")
    try:
        with open(results_path, "w") as f:
            # Custom encoder to handle potential numpy types if conversion failed
            class NpEncoder(json.JSONEncoder):
                 def default(self, obj):
                      if isinstance(obj, np.integer):
                           return int(obj)
                      if isinstance(obj, np.floating):
                           return float(obj)
                      if isinstance(obj, np.bool_):
                           return bool(obj) # Convert numpy bool_ to standard bool
                      if isinstance(obj, np.ndarray):
                           return obj.tolist()
                      return super(NpEncoder, self).default(obj)
            json.dump(results, f, indent=2, cls=NpEncoder)
        logger.info(f"Saved comparative analysis results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save comparative results: {e}")

    logger.info("===== Completed Comparative Analysis =====")

# Example usage (if run directly)
# if __name__ == '__main__':
#     # Requires dummy config, datasets, and mod_rnn_params
#     print("Running comparative studies module directly (requires dummy setup)")
#     # config = ...
#     # datasets = ...
#     # mod_rnn_params = ... # Typically from hyperparam search
#     # run_comparative_analysis(config, datasets, "comparative_results", mod_rnn_params) 