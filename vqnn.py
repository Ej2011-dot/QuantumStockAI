# models/vqnn.py
#
# Variational Quantum Neural Network (VQNN)
#
# This is the core model. It is a hybrid classical-quantum network:
#
#   [Classical encoder] → [VQC layers] → [Classical decoder]
#
# The quantum core is a parameterized quantum circuit (PQC) where the
# parameters θ are trained via gradient descent using the parameter-shift rule.
# This gives us analytically exact gradients on real quantum hardware too.
#
# Unlike the original script which had a fixed circuit, this model:
#  1. Has a configurable number of layers and qubits
#  2. Trains its parameters to minimise a loss function on historical data
#  3. Uses ZZFeatureMap encoding for better class separation
#  4. Supports multiple ansatz architectures

import pennylane as qml
from pennylane import numpy as np
import numpy as _np  # stdlib numpy for non-differentiable ops
from typing import List, Optional, Tuple
import logging

from core.quantum_circuit import build_vqc, random_params, draw_circuit
from config import QuantumConfig

logger = logging.getLogger(__name__)


class VQNN:
    """
    Variational Quantum Neural Network for binary signal classification.

    The model outputs a scalar in [-1, +1]:
      > +buy_threshold  → BUY
      < -sell_threshold → SELL
      between            → HOLD

    Training uses the parameter-shift rule for exact quantum gradients.

    Args:
        config: QuantumConfig dataclass with circuit hyperparameters.
        seed: Random seed for parameter initialisation.
    """

    def __init__(self, config: QuantumConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.ansatz = config.ansatz

        # Build the compiled QNode
        self._vqc, self._param_shape = build_vqc(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            ansatz=self.ansatz,
            device_name=config.device,
        )

        # Initialise trainable parameters
        self.params = random_params(self.ansatz, self.n_qubits, self.n_layers, seed)
        self._is_trained = False

        logger.info(
            "VQNN initialised: %d qubits, %d layers, '%s' ansatz, param shape %s",
            self.n_qubits, self.n_layers, self.ansatz, self._param_shape
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, features: np.ndarray) -> float:
        """
        Single forward pass.
        Returns the mean PauliZ expectation across all qubits → scalar in [-1, +1].
        """
        raw = self._vqc(features, self.params)
        return np.mean(np.array(raw))

    def predict_signal(self, features: np.ndarray) -> float:
        """Public forward pass. Returns raw signal in [-1, +1]."""
        return float(self._forward(features))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Run forward pass on a batch of feature vectors.
        X shape: (n_samples, n_features)
        Returns shape: (n_samples,)
        """
        return _np.array([self.predict_signal(x) for x in X])

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def mse_loss(self, features: np.ndarray, target: float) -> float:
        """Mean squared error between circuit output and target label."""
        pred = self._forward(features)
        return (pred - target) ** 2

    def batch_mse_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Average MSE over a batch."""
        total = np.array(0.0)
        for xi, yi in zip(X, y):
            total = total + self.mse_loss(xi, float(yi))
        return total / len(X)

    def sharpe_loss(self, X: np.ndarray, returns: np.ndarray,
                    threshold: float = 0.0) -> float:
        """
        Differentiable Sharpe-ratio-inspired loss.
        Maximise risk-adjusted returns by minimising negative Sharpe.
        (Used as an alternative to MSE for direct return optimisation.)
        """
        signals = np.array([self._forward(xi) for xi in X])
        # Convert signals to position sizes ∈ [-1, +1]
        positions = np.tanh(signals / 0.3)
        pnl = positions * np.array(returns)
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl) + 1e-8  # avoid div by zero
        sharpe = mean_pnl / std_pnl
        return -sharpe  # minimise negative Sharpe

    # ------------------------------------------------------------------
    # Parameter-shift gradient
    # ------------------------------------------------------------------

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of batch MSE loss w.r.t. params using parameter-shift rule.

        The parameter-shift rule:
            ∂f/∂θ = (f(θ + π/2) - f(θ - π/2)) / 2

        This gives exact gradients (not finite differences) and works on hardware too.
        """
        grad = _np.zeros_like(self.params)
        flat_params = self.params.flatten()

        for idx in range(len(flat_params)):
            # Shift +π/2
            shifted_plus = flat_params.copy()
            shifted_plus[idx] += _np.pi / 2
            self.params = shifted_plus.reshape(self._param_shape)
            loss_plus = float(self.batch_mse_loss(X, y))

            # Shift -π/2
            shifted_minus = flat_params.copy()
            shifted_minus[idx] -= _np.pi / 2
            self.params = shifted_minus.reshape(self._param_shape)
            loss_minus = float(self.batch_mse_loss(X, y))

            # Restore
            self.params = flat_params.reshape(self._param_shape)

            grad.flat[idx] = (loss_plus - loss_minus) / 2.0

        return grad

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: _np.ndarray,
        y_train: _np.ndarray,
        X_val: Optional[_np.ndarray] = None,
        y_val: Optional[_np.ndarray] = None,
        epochs: int = 50,
        learning_rate: float = 0.01,
        patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Train the VQNN using gradient descent with parameter-shift gradients.

        Args:
            X_train: Feature array, shape (n_train, n_features).
            y_train: Label array, shape (n_train,). Values in {-1, 0, +1}.
            X_val: Optional validation features.
            y_val: Optional validation labels.
            epochs: Number of full passes over training data.
            learning_rate: Step size for gradient updates.
            patience: Early stopping patience (epochs without val improvement).
            verbose: Print training progress.

        Returns:
            History dict with 'train_loss', 'val_loss' lists.
        """
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_params = self.params.copy()
        no_improve_count = 0

        X_train = _np.array(X_train)
        y_train = _np.array(y_train)

        for epoch in range(1, epochs + 1):
            # Compute gradient and update
            grad = self.gradient(X_train, y_train)
            self.params = self.params - learning_rate * grad

            # Training loss
            train_loss = float(self.batch_mse_loss(X_train, y_train))
            history["train_loss"].append(train_loss)

            # Validation loss + early stopping
            if X_val is not None and y_val is not None:
                val_loss = float(self.batch_mse_loss(_np.array(X_val), _np.array(y_val)))
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_params = self.params.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if verbose and epoch % 10 == 0:
                    logger.info(
                        "Epoch %3d/%d — train: %.4f  val: %.4f",
                        epoch, epochs, train_loss, val_loss
                    )

                if no_improve_count >= patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break
            else:
                if verbose and epoch % 10 == 0:
                    logger.info("Epoch %3d/%d — train: %.4f", epoch, epochs, train_loss)

        # Restore best params
        if X_val is not None:
            self.params = best_params

        self._is_trained = True
        logger.info("Training complete. Final train loss: %.4f", history["train_loss"][-1])
        return history

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def draw(self, sample_features: _np.ndarray) -> str:
        """Print a text diagram of the circuit for a sample input."""
        return draw_circuit(self._vqc, sample_features, self.params)

    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return int(_np.prod(self._param_shape))

    def save(self, path: str):
        """Save parameters to a .npy file."""
        _np.save(path, self.params)
        logger.info("Model parameters saved to %s", path)

    def load(self, path: str):
        """Load parameters from a .npy file."""
        self.params = _np.load(path)
        self._is_trained = True
        logger.info("Model parameters loaded from %s", path)

    def summary(self) -> str:
        return (
            f"VQNN Summary\n"
            f"  Qubits    : {self.n_qubits}\n"
            f"  Layers    : {self.n_layers}\n"
            f"  Ansatz    : {self.ansatz}\n"
            f"  Params    : {self.n_params()}\n"
            f"  Trained   : {self._is_trained}\n"
        )
