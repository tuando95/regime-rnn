"""
hyperparameter_search.py

Performs hyperparameter search for the ModularRegimeRNN model
using Optuna. Samples over the number of experts, hidden dimension,
gating network depth, dropout rate, and regularization coefficients,
while enforcing a global parameter budget constraint.
"""

import copy
import logging
from typing import Any, Dict

import tqdm
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned

import utils
from model import ModularRegimeRNN
from trainer import Trainer
from evaluator import Evaluator

# Configure module‐level logger
logger = utils.configure_logging()


class HyperparameterSearch:
    """
    Encapsulates an Optuna study to tune hyperparameters for
    the ModularRegimeRNN model on a validation split.
    """

    def __init__(self,
                 search_space: Dict[str, Any],
                 config: Dict[str, Any]) -> None:
        """
        Initialize the hyperparameter searcher.

        Args:
            search_space: Dictionary with keys 'grid', 'random', 'bayesian'
                          specifying the hyperparameter domains.
            config: Full experiment configuration dict (parsed from YAML),
                    must include a 'datasets' entry with 'train' and 'val'.
        """
        self.search_space = search_space or {}
        self.config = config or {}

        # Validate search_space keys
        for section in ("grid", "random", "bayesian"):
            if section not in self.search_space:
                raise ValueError(f"Hyperparameter search space missing '{section}' section")

        self.grid_space = self.search_space["grid"]
        self.random_space = self.search_space["random"]
        self.bayesian_space = self.search_space["bayesian"]

        # Determine number of trials (default=50 if not set)
        hp_cfg = self.config.get("hyperparameter_search", {})
        self.n_trials = int(hp_cfg.get("num_trials", 50))
        if "num_trials" not in hp_cfg:
            logger.warning("hyperparameter_search.num_trials not specified; defaulting to 50")

        # Global parameter budget
        self.budget_params = int(
            self.config.get("model", {}).get("budget_params", 0)
        )
        if self.budget_params <= 0:
            raise ValueError("model.budget_params must be a positive integer in config")

        # Datasets for training/validation
        self.datasets = self.config.get("datasets", {})
        if "train" not in self.datasets or "val" not in self.datasets:
            raise ValueError("Config must include 'datasets' with keys 'train' and 'val'")

        # Initialize Optuna study
        sampler = TPESampler(seed=int(self.config.get("seed", 42)))
        pruner = MedianPruner()
        self.study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner
        )
        logger.info("Optuna study created for hyperparameter search")

    def run(self) -> Dict[str, Any]:
        """
        Run the Optuna optimization loop.

        Returns:
            A dict of model constructor parameters corresponding
            to the best trial (to pass into ModularRegimeRNN).
        """
        logger.info(f"Starting hyperparameter search ({self.n_trials} trials)")

        # Add TQDM progress bar for trials
        with tqdm.tqdm(total=self.n_trials, desc="Optuna Trials") as pbar:
            self.study.optimize(
                lambda trial: self._objective_with_pbar(trial, pbar),
                n_trials=self.n_trials
            )

        best = self.study.best_trial
        logger.info(
            f"Best trial #{best.number} — val_mse={best.value:.6f}, params={best.params}"
        )

        # Retrieve the saved model_params or rebuild if missing
        best_model_params = best.user_attrs.get("model_params")
        if best_model_params is None:
            best_model_params = self._reconstruct_model_params(best.params)
            logger.debug("Reconstructed model_params from trial params")

        return best_model_params

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for a single trial: sample hyperparameters,
        train a ModularRegimeRNN, evaluate on validation set, return MSE.
        Now uses attention gating.

        Args:
            trial: Optuna trial object.

        Returns:
            Validation MSE (lower is better).
        Raises:
            TrialPruned: to indicate that the trial should be pruned.
        """
        try:
            # 1) Sample hyperparameters
            K = trial.suggest_categorical("K", self.grid_space["K"])
            hidden_dim = trial.suggest_categorical(
                "hidden_dim", self.grid_space["hidden_dim"]
            )
            # Sample attention heads instead of gating layers
            attention_heads = trial.suggest_categorical(
                "attention_heads", self.random_space["attention_heads"]
            )
            # Check for divisibility constraint early
            if hidden_dim % attention_heads != 0:
                # If not divisible, this combination is invalid, prune the trial
                raise TrialPruned(f"hidden_dim ({hidden_dim}) not divisible by attention_heads ({attention_heads})")
            
            dropout = trial.suggest_float(
                "dropout", 
                low=float(self.random_space["dropout"][0]),
                high=float(self.random_space["dropout"][1])
            )
            lambda_l2 = trial.suggest_float(
                "lambda_l2", 
                low=float(self.random_space["lambda_l2"][0]),
                high=float(self.random_space["lambda_l2"][1]),
                log=True
            )
            lambda_entropy = trial.suggest_float(
                "lambda_entropy", 
                low=float(self.random_space["lambda_entropy"][0]),
                high=float(self.random_space["lambda_entropy"][1]),
                log=True
            )
            # Sample regime loss weight
            lambda_regime = trial.suggest_float(
                "lambda_regime", 
                low=float(self.random_space["lambda_regime"][0]),
                high=float(self.random_space["lambda_regime"][1]),
                log=False
            ) # Use log=False or suggest_categorical if discrete steps are better
            
            learning_rate_init = trial.suggest_float(
                "learning_rate_init",
                low=float(self.bayesian_space["learning_rate_init"][0]),
                high=float(self.bayesian_space["learning_rate_init"][1]),
                log=True
            )

            # 2) Extract data dimensions
            train_ds = self.datasets["train"]
            X_train, y_train, _ = train_ds.tensors
            input_dim = int(X_train.shape[-1])
            output_dim = int(y_train.shape[-1])

            # 3) Build model constructor args for Attention Gating
            model_params: Dict[str, Any] = {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "experts": {"K": int(K), "hidden_dim": int(hidden_dim)},
                "gating": {
                    "attention_heads": int(attention_heads),
                    "dropout": float(dropout)
                }
            }
            trial.set_user_attr("model_params", model_params)

            # 4) Instantiate model and check parameter budget
            # (Instantiation might fail if hidden_dim % attention_heads != 0, but we checked above)
            model = ModularRegimeRNN(model_params)
            param_count = model.get_num_params()
            if param_count > self.budget_params:
                logger.warning(
                    f"Trial {trial.number}: params={param_count} exceeds budget={self.budget_params}"
                )
                raise TrialPruned(f"Param budget exceeded: {param_count} > {self.budget_params}")

            # 5) Prepare trial config with learning/regularization settings
            cfg_trial = copy.deepcopy(self.config)
            cfg_trial.setdefault("training", {}).setdefault("lr_schedule", {})['init_options'] = [learning_rate_init]
            reg_cfg = cfg_trial.setdefault("training", {}).setdefault("regularization", {})
            reg_cfg["lambda_l2_options"] = [lambda_l2]
            reg_cfg["lambda_entropy_options"] = [lambda_entropy]
            reg_cfg["lambda_regime_options"] = [lambda_regime] # Use sampled regime weight

            # 6) Train the model
            # (Trainer needs to be updated to handle list hidden state and 4 returns)
            trainer = Trainer(
                model,
                {"train": self.datasets["train"], "val": self.datasets["val"]},
                cfg_trial,
                epoch_callback=lambda epoch, metric: trial.report(metric, epoch) # Pass metric used for early stopping
            )
            trained_model, final_epoch = trainer.train()
            trial.set_user_attr("final_epoch", final_epoch)

            # 7) Evaluate on validation set
            # (Evaluator needs to be updated for list hidden state and 4 returns)
            evaluator = Evaluator(
                trained_model,
                {"test": self.datasets["val"]}, # Evaluate on VAL set for HPO
                cfg_trial # Use trial config in case evaluator needs settings?
            )
            metrics = evaluator.evaluate()
            # Objective: Use total validation loss (includes weighted CE)
            # Make sure Trainer._validate returns it or calculate here
            # Assuming evaluator.evaluate doesn't give total loss, let's just use MSE for now
            # TODO: Ideally optimize based on val_loss or weighted combo of mse/regime_acc
            objective_value = float(metrics.get("mse", float("inf"))) 
            trial.set_user_attr("val_mse", metrics.get("mse"))
            trial.set_user_attr("val_regime_accuracy", metrics.get("regime_accuracy"))

            # 8) Report and possibly prune based on the objective value
            trial.report(objective_value, step=final_epoch) # Report at final epoch
            if trial.should_prune():
                raise TrialPruned()

            return objective_value

        except TrialPruned:
            # Re‐raise to tell Optuna this trial is pruned
            raise
        except Exception as exc:
            # Unhandled exception: prune this trial
            logger.exception(f"Trial {trial.number} failed with exception: {exc}")
            trial.report(float("inf"), step=0)
            raise TrialPruned() from exc

    def _objective_with_pbar(self, trial: optuna.Trial, pbar: tqdm.tqdm) -> float:
        """Wrapper for the objective function to update the progress bar."""
        final_epoch_str = ""
        try:
            result = self._objective(trial)
            # Update progress bar upon successful completion or pruning
            pbar.update(1)
            # Add best value so far to pbar description
            postfix_dict = {}
            if self.study.best_value is not None:
                postfix_dict["best_mse"] = f"{self.study.best_value:.4f}"
            # Get final epoch from trial attributes if available
            final_epoch = trial.user_attrs.get("final_epoch")
            if final_epoch is not None:
                postfix_dict["last_epoch"] = final_epoch
            pbar.set_postfix(postfix_dict)
            return result
        except TrialPruned as e:
            # Update progress bar if trial is pruned
            pbar.update(1)
            raise e # Re-raise TrialPruned
        except Exception as e: 
            # Also update progress bar if trial fails
            pbar.update(1)
            # Log failure
            pbar.set_postfix({"status": "failed"})
            raise e # Re-raise the exception

    def _reconstruct_model_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback method to rebuild the model_params dict for Attention Gating.
        """
        # Re-extract dims
        train_ds = self.datasets["train"]
        X_train, y_train, _ = train_ds.tensors
        input_dim = int(X_train.shape[-1])
        output_dim = int(y_train.shape[-1])
        return {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "experts": {"K": int(params["K"]), "hidden_dim": int(params["hidden_dim"])},
            "gating": {
                "attention_heads": int(params["attention_heads"]),
                "dropout": float(params["dropout"])
            }
        }
