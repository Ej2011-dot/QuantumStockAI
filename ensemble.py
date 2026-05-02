# models/ensemble.py
#
# Ensemble of multiple VQNNs.
#
# WHY ENSEMBLE?
# A single VQC can get stuck in local minima (especially with barren plateaus).
# Training multiple circuits with different random initialisations and averaging
# their outputs gives more robust, less noisy predictions — exactly like
# classical ensemble methods (random forests, XGBoost, etc).
#
# Each member uses a different ansatz or random seed, capturing
# different features of the data.

import numpy as np
from typing import List, Tuple, Optional
import logging
import os

from models.vqnn import VQNN
from config import QuantumConfig

logger = logging.getLogger(__name__)


class VQNNEnsemble:
    """
    Ensemble of VQNN models. Predictions are averaged across members.

    Supports:
      - Uniform averaging
      - Confidence-weighted averaging (members with better val performance weighted higher)
      - Majority voting (for discrete buy/sell/hold signals)

    Usage:
        ensemble = VQNNEnsemble(n_members=3, config=config.quantum)
        ensemble.fit(X_train, y_train, X_val, y_val)
        signal = ensemble.predict(x)
    """

    ANSATZ_OPTIONS = ["basic", "strongly_entangling", "hardware_efficient"]

    def __init__(self, n_members: int = 3, config: QuantumConfig = None, base_seed: int = 42):
        if config is None:
            config = QuantumConfig()
        self.n_members = n_members
        self.config = config
        self.base_seed = base_seed
        self.members: List[VQNN] = []
        self.member_weights: np.ndarray = np.ones(n_members) / n_members
        self._build_members()

    def _build_members(self):
        """Instantiate ensemble members with varied seeds and ansatz types."""
        self.members = []
        for i in range(self.n_members):
            # Cycle through ansatz types for diversity
            ansatz = self.ANSATZ_OPTIONS[i % len(self.ANSATZ_OPTIONS)]
            member_config = QuantumConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
                ansatz=ansatz,
                device=self.config.device,
            )
            member = VQNN(config=member_config, seed=self.base_seed + i * 7)
            self.members.append(member)
            logger.info("Ensemble member %d: ansatz='%s', seed=%d", i, ansatz, self.base_seed + i * 7)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        learning_rate: float = 0.01,
        patience: int = 10,
    ) -> List[dict]:
        """Train all ensemble members. Returns list of training histories."""
        histories = []
        val_losses = []

        for i, member in enumerate(self.members):
            logger.info("Training ensemble member %d/%d...", i + 1, self.n_members)
            history = member.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                learning_rate=learning_rate,
                patience=patience,
                verbose=False,
            )
            histories.append(history)
            # Track final val loss for weighting
            if history["val_loss"]:
                val_losses.append(history["val_loss"][-1])
            else:
                val_losses.append(history["train_loss"][-1])

        # Weight members inversely proportional to validation loss
        # (better performers get higher weight)
        val_losses = np.array(val_losses)
        inv_losses = 1.0 / (val_losses + 1e-10)
        self.member_weights = inv_losses / inv_losses.sum()
        logger.info("Ensemble weights: %s", [f"{w:.3f}" for w in self.member_weights])
        return histories

    def predict_signal(self, features: np.ndarray, mode: str = "weighted") -> float:
        """
        Predict a single signal.

        Args:
            features: Feature vector for one time step.
            mode: "weighted" (default) | "uniform" | "vote"

        Returns:
            Signal ∈ [-1, +1]
        """
        raw_signals = np.array([m.predict_signal(features) for m in self.members])

        if mode == "weighted":
            return float(np.dot(self.member_weights, raw_signals))
        elif mode == "uniform":
            return float(raw_signals.mean())
        elif mode == "vote":
            # Majority vote among {-1, 0, +1} labels
            votes = np.sign(raw_signals)
            return float(np.sign(votes.sum()))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def predict_batch(self, X: np.ndarray, mode: str = "weighted") -> np.ndarray:
        """Predict signals for a batch. Returns shape (n_samples,)."""
        return np.array([self.predict_signal(x, mode=mode) for x in X])

    def member_signals(self, features: np.ndarray) -> np.ndarray:
        """Return individual signals from all members (for diagnostics)."""
        return np.array([m.predict_signal(features) for m in self.members])

    def save(self, directory: str):
        """Save all member parameters."""
        os.makedirs(directory, exist_ok=True)
        for i, member in enumerate(self.members):
            member.save(os.path.join(directory, f"member_{i}.npy"))
        np.save(os.path.join(directory, "weights.npy"), self.member_weights)
        logger.info("Ensemble saved to %s/", directory)

    def load(self, directory: str):
        """Load all member parameters."""
        for i, member in enumerate(self.members):
            member.load(os.path.join(directory, f"member_{i}.npy"))
        weights_path = os.path.join(directory, "weights.npy")
        if os.path.exists(weights_path):
            self.member_weights = np.load(weights_path)
        logger.info("Ensemble loaded from %s/", directory)

    def summary(self) -> str:
        lines = [f"\nVQNN Ensemble ({self.n_members} members)"]
        lines.append("─" * 40)
        for i, (member, w) in enumerate(zip(self.members, self.member_weights)):
            lines.append(
                f"  Member {i}: ansatz={member.ansatz}, "
                f"params={member.n_params()}, weight={w:.3f}"
            )
        lines.append("─" * 40)
        total_params = sum(m.n_params() for m in self.members)
        lines.append(f"  Total quantum params: {total_params}")
        return "\n".join(lines)
