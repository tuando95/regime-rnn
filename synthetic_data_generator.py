# synthetic_data_generator.py

"""
SyntheticDataGenerator

Generates synthetic non-stationary time series data with regime-switching AR(p)
dynamics for use in training and evaluating the Modular Regime RNN and baselines.

Interface:
    generator = SyntheticDataGenerator(config.data)
    X, y, regimes = generator.generate()
"""

import math
from typing import Tuple, Dict, List, Any, Union, Optional

import numpy as np

import utils


class SyntheticDataGenerator:
    """
    Synthetic data generator for regime-switching AR processes.

    Args:
        params (dict): Dictionary containing synthetic data configuration under key
            'synthetic', e.g.:
            {
              'synthetic': {
                  'num_sequences': int,
                  'sequence_length': int,
                  'regimes': { ... },
                  'num_regimes': int,              # optional, default=3
                  'regime_type': str,              # 'abrupt'|'gradual'|'hierarchical', default='abrupt'
                  'ar_order': int,
                  'ar_coeff_range': [float, float],
                  'spectral_radius_max': float,
                  'sigma_range': [float, float],
                  'multivariate': {
                      'wishart_df': int,
                      'max_condition_number': float
                  },
                  'dimension': int                # optional, default=1
              }
            }
    """

    def __init__(self, params: Dict[str, Any]):
        # Extract synthetic configuration
        if "synthetic" in params and isinstance(params["synthetic"], dict):
            cfg = params["synthetic"]
        else:
            cfg = params

        # Reproducibility: seed global RNGs and our own Generator
        seed = utils._CONFIG.get("seed", None)
        if seed is not None:
            utils.seed_everything(int(seed))
        self.rng = np.random.default_rng(int(seed) if seed is not None else None)

        # Basic dimensions
        self.N: int = int(cfg.get("num_sequences", 1))
        self.T: int = int(cfg.get("sequence_length", 1))
        self.d: int = int(cfg.get("dimension", 1))

        # Regime configuration
        self.regime_specs: Dict[str, Any] = cfg.get("regimes", {})
        self.regime_type: str = cfg.get("regime_type", "abrupt")
        if self.regime_type not in self.regime_specs:
            raise ValueError(f"Unknown regime_type '{self.regime_type}'. "
                             f"Must be one of {list(self.regime_specs.keys())}.")
        # Number of regimes (for hierarchical, may be overridden later)
        self.R: int = int(cfg.get("num_regimes", 3))

        # AR process configuration
        self.p: int = int(cfg.get("ar_order", 1))
        # Handle new dict format for ar_coeff_range
        ar_coeffs_cfg = cfg.get("ar_coeff_range", [-0.8, 0.8])
        if isinstance(ar_coeffs_cfg, dict):
            self.ar_coeff_range_dict: Optional[Dict[str, List[float]]] = ar_coeffs_cfg
            self.ar_coeff_range: Optional[Tuple[float, float]] = None
            # Validate dict keys match expected number of regimes (R determined later if hierarchical)
            # We'll validate inside generate_ar_coeffs after R is finalized
        elif isinstance(ar_coeffs_cfg, list) and len(ar_coeffs_cfg) == 2:
            a_min, a_max = ar_coeffs_cfg
            self.ar_coeff_range = (float(a_min), float(a_max))
            self.ar_coeff_range_dict = None
        else:
            raise ValueError("Invalid format for ar_coeff_range")
        self.spectral_radius_max: float = float(cfg.get("spectral_radius_max", 0.95))

        # Noise configuration
        s_min, s_max = cfg.get("sigma_range", [0.01, 0.1])
        self.sigma_range: Tuple[float, float] = (float(s_min), float(s_max))

        # Multivariate (optional)
        mv_cfg = cfg.get("multivariate", {})
        self.wishart_df: int = mv_cfg.get("wishart_df", self.d)
        self.max_cond: float = mv_cfg.get("max_condition_number", None)

        # Regime-specific Frequencies (New)
        self.regime_frequencies: Optional[Dict[str, List[float]]] = cfg.get("regime_frequencies", None)
        # Regime-specific Oscillation Amplitude (New)
        self.oscillation_amplitude_range: List[float] = cfg.get("oscillation_amplitude_range", [0.05, 0.15])

        # --- Ensure R is set before validating dicts ---
        # Initial validation if not hierarchical (R won't change)
        if self.regime_type != "hierarchical":
            self._validate_regime_specific_configs()

    def _validate_regime_specific_configs(self):
        """Validate regime-specific dicts after R is finalized."""
        if self.ar_coeff_range_dict is not None:
            expected_keys = {f"regime{r+1}" for r in range(self.R)}
            if set(self.ar_coeff_range_dict.keys()) != expected_keys:
                raise ValueError(f"ar_coeff_range dict keys {list(self.ar_coeff_range_dict.keys())} "
                                 f"do not match expected keys {list(expected_keys)} for R={self.R}")
        if self.regime_frequencies is not None:
            expected_keys = {f"regime{r+1}" for r in range(self.R)}
            if set(self.regime_frequencies.keys()) != expected_keys:
                 raise ValueError(f"regime_frequencies dict keys {list(self.regime_frequencies.keys())} "
                                 f"do not match expected keys {list(expected_keys)} for R={self.R}")

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic dataset.

        Returns:
            X: np.ndarray of shape (N, T, d) -- features/time series.
            y: np.ndarray of shape (N, T, d) -- targets (same as X for AR data).
            regimes: np.ndarray of shape (N, T), dtype int -- regime indices [0..R-1].
        """
        regimes = self.generate_regime_chain()
        # Final validation of regime-specific configs now that R is set
        self._validate_regime_specific_configs()
        ar_params = self.generate_ar_coeffs()
        X, y = self.generate_emissions(regimes, ar_params)
        # Create one-step-ahead targets: y_t = x_{t+1}; last step has no future, fill zeros
        y[:, :-1, :] = X[:, 1:, :]
        y[:, -1, :] = 0.0
        return X, y, regimes

    def generate_regime_chain(self) -> np.ndarray:
        """
        Simulate regime indices as a first-order Markov chain.

        Returns:
            regimes: np.ndarray of shape (N, T), dtype=int
        """
        # Build transition matrix P of shape (R, R)
        R = self.R
        P = np.zeros((R, R), dtype=float)

        spec = self.regime_specs[self.regime_type]
        if self.regime_type == "abrupt":
            alpha = float(spec.get("self_transition_prob", 0.9))
            off = (1.0 - alpha) / float(max(R - 1, 1))
            P.fill(off)
            np.fill_diagonal(P, alpha)

        elif self.regime_type == "gradual":
            tau = float(spec.get("tau", 2.0))
            for i in range(R):
                for j in range(R):
                    P[i, j] = math.exp(-abs(i - j) / tau)
                P[i, :] /= P[i, :].sum()

        elif self.regime_type == "hierarchical":
            # Parent HMM
            num_parents = int(spec.get("num_parents", 2))
            # Sample children counts per parent
            counts = self.rng.integers(2, 4, size=num_parents)  # each in {2,3}
            children_idxs: List[np.ndarray] = []
            idx0 = 0
            for c in counts:
                children_idxs.append(np.arange(idx0, idx0 + c, dtype=int))
                idx0 += c
            R = idx0
            self.R = R  # override
            # Parent-level transition (abrupt style with alpha=0.9)
            alpha_p = float(spec.get("self_transition_prob", 0.9))
            off_p = (1.0 - alpha_p) / float(max(num_parents - 1, 1))
            Pp = np.full((num_parents, num_parents), off_p)
            np.fill_diagonal(Pp, alpha_p)
            # Expand to child-level P
            P = np.zeros((R, R), dtype=float)
            for i in range(num_parents):
                for j in range(num_parents):
                    prob_ij = Pp[i, j] / float(len(children_idxs[j]))
                    P[np.ix_(children_idxs[i], children_idxs[j])] = prob_ij

        else:
            raise ValueError(f"Unsupported regime_type: {self.regime_type}")

        # Simulate for each sequence
        regimes = np.zeros((self.N, self.T), dtype=int)
        # initial regimes: uniform
        regimes[:, 0] = self.rng.integers(0, self.R, size=self.N)
        for t in range(1, self.T):
            prev = regimes[:, t - 1]
            # for each possible previous state, sample transitions
            for r in range(self.R):
                idx = np.where(prev == r)[0]
                if idx.size > 0:
                    regimes[idx, t] = self.rng.choice(
                        self.R, size=idx.size, p=P[r]
                    )
        return regimes

    def generate_ar_coeffs(self) -> Dict[str, Any]:
        """
        Generate AR(p) coefficients and noise covariances for each regime.

        Returns:
            ar_params: {
                'coeffs': Dict[int, List[np.ndarray or float]],
                'noise_cov': Dict[int, np.ndarray or float]
            }
        """
        coeffs: Dict[int, List[Any]] = {}
        noise_cov: Dict[int, Any] = {}

        for r in range(self.R):
            # Determine the AR coefficient range for this regime
            if self.ar_coeff_range_dict:
                regime_key = f"regime{r+1}" # Assumes keys like "regime1", "regime2", ...
                current_ar_range = self.ar_coeff_range_dict[regime_key]
            else:
                current_ar_range = self.ar_coeff_range # Use single range
            
            coeff_min, coeff_max = current_ar_range

            # sample raw A_i using the specific range
            A_list: List[Any] = []
            for _ in range(self.p):
                if self.d == 1:
                    a = self.rng.uniform(coeff_min, coeff_max)
                    A_list.append(float(a))
                else:
                    A_mat = self.rng.uniform(
                        coeff_min, coeff_max, size=(self.d, self.d)
                    )
                    A_list.append(A_mat)

            # enforce spectral radius constraint via companion matrix
            F = self._build_companion(A_list, self.d)
            F_rescaled = utils.spectral_radius_rescale(F, self.spectral_radius_max)
            A_list = self._extract_from_companion(F_rescaled, self.d, self.p)

            # sample noise covariance
            if self.d == 1:
                sigma = self.rng.uniform(self.sigma_range[0], self.sigma_range[1])
                cov = sigma * sigma
            else:
                cov = utils.wishart_sample(self.d, np.eye(self.d), df=self.wishart_df)
                # Optional: Add condition number check/rescaling for covariance here
            
            coeffs[r] = A_list
            noise_cov[r] = cov

        return {"coeffs": coeffs, "noise_cov": noise_cov}

    def generate_emissions(
        self, regimes: np.ndarray, ar_params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series X and targets y following AR dynamics.

        Args:
            regimes: np.ndarray of shape (N, T), regime indices.
            ar_params: dict from generate_ar_coeffs()

        Returns:
            X: np.ndarray (N, T, d)
            y: np.ndarray (N, T, d) same as X for AR
        """
        N, T, d = self.N, self.T, self.d
        p = self.p
        coeffs = ar_params["coeffs"]
        noise_cov = ar_params["noise_cov"]

        X = np.zeros((N, T, d), dtype=float)

        # Pre-sample frequencies and amplitudes per sequence per regime if they exist
        # This assumes frequencies/amplitudes are constant *within* a regime for a given sequence
        seq_frequencies = np.zeros((N, self.R)) if self.regime_frequencies else None
        seq_amplitudes = np.zeros((N, self.R, d))
        if self.regime_frequencies:
            for i in range(N):
                for r in range(self.R):
                    regime_key = f"regime{r+1}"
                    f_min, f_max = self.regime_frequencies[regime_key]
                    seq_frequencies[i, r] = self.rng.uniform(f_min, f_max)
                    # Sample amplitude - using oscillation_amplitude_range
                    amp_min, amp_max = self.oscillation_amplitude_range
                    # Sample one amplitude per dimension per regime per sequence
                    seq_amplitudes[i, r, :] = self.rng.uniform(amp_min, amp_max, size=d)

        for t in range(T):
            for r in range(self.R):
                idx = np.where(regimes[:, t] == r)[0]
                if idx.size == 0:
                    continue

                # Get AR parameters for this regime
                A_list_r = coeffs[r]
                cov_r = noise_cov[r]

                # Calculate AR component
                ar_val = np.zeros((idx.size, d))
                if t >= p:
                    for k_lag in range(p):
                        A_k = A_list_r[k_lag]
                        if d == 1:
                            ar_val += A_k * X[idx, t - 1 - k_lag]
                        else:
                            # Need einsum or loop for batch matrix multiply
                            # X[idx, t - 1 - k_lag] is (batch, d)
                            # A_k is (d, d)
                            # einsum: 'bd,dd->bd'
                            ar_val += np.einsum('bd,dd->bd', X[idx, t - 1 - k_lag], A_k)
                
                # Sample noise
                if d == 1:
                    noise = self.rng.normal(0, np.sqrt(cov_r), size=(idx.size, 1))
                else:
                    noise = self.rng.multivariate_normal(np.zeros(d), cov_r, size=idx.size)

                # Calculate Oscillation component (if configured)
                oscillation = np.zeros((idx.size, d))
                if self.regime_frequencies is not None:
                    # Use pre-sampled frequencies/amplitudes for these sequences in this regime
                    freqs = seq_frequencies[idx, r] # (batch,) -> (batch, 1)
                    amps = seq_amplitudes[idx, r, :] # (batch, d)
                    # Add time dimension: sin(2 * pi * freq * t)
                    time_vec = 2 * np.pi * freqs * t 
                    # Apply sin element-wise for each dim based on amps
                    oscillation = amps * np.sin(time_vec[:, np.newaxis]) 

                # Combine components
                X[idx, t] = ar_val + noise + oscillation

        # Targets (y) are generated after this function returns in self.generate
        return X, X # Return X for both features and potential base for target shifting

    @staticmethod
    def _build_companion(A_list: List[Any], d: int) -> np.ndarray:
        """
        Build companion matrix for AR coefficients.

        Args:
            A_list: list of p floats (d=1) or p arrays (d,d).
            d: dimension of each A_i.

        Returns:
            F: companion matrix of shape (p*d, p*d) or (p, p) if d==1.
        """
        p = len(A_list)
        if d == 1:
            F = np.zeros((p, p), dtype=float)
            # first row = coefficients
            for j in range(p):
                F[0, j] = float(A_list[j])
            # subdiagonal ones
            for i in range(1, p):
                F[i, i - 1] = 1.0
            return F
        else:
            F = np.zeros((p * d, p * d), dtype=float)
            # top row blocks
            for j in range(p):
                F[0:d, j * d : (j + 1) * d] = A_list[j]
            # identity sub-blocks
            for i in range(1, p):
                F[i * d : (i + 1) * d, (i - 1) * d : i * d] = np.eye(d)
            return F

    @staticmethod
    def _extract_from_companion(
        F: np.ndarray, d: int, p: int
    ) -> List[Any]:
        """
        Extract AR coefficients from rescaled companion matrix.

        Args:
            F: companion matrix (p*d, p*d) or (p, p).
            d: dimension.
            p: AR order.

        Returns:
            A_list: list of p floats or p (d x d) arrays.
        """
        if d == 1:
            # F is (p, p)
            return [float(F[0, j]) for j in range(p)]
        else:
            A_list: List[np.ndarray] = []
            for j in range(p):
                block = F[0:d, j * d : (j + 1) * d]
                A_list.append(block.copy())
            return A_list
