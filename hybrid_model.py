# models/hybrid_model.py
#
# Hybrid Classical-Quantum Model
#
# Architecture:
#   Raw OHLCV features
#      │
#      ▼
#   [Classical Pre-net]  — projects raw features → n_qubits dims, normalises to [0, π]
#      │
#      ▼
#   [VQNN Core]          — variational quantum circuit
#      │
#      ▼
#   [Classical Post-net] — maps quantum outputs → buy/hold/sell probability
#      │
#      ▼
#   Signal + Confidence
#
# Having classical layers before and after the quantum core dramatically
# improves expressivity vs a pure VQC, at the cost of having more parameters.

import numpy as np
from typing import Optional, Tuple
import logging
import json
import os

from models.vqnn import VQNN
from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple classical layers (numpy-only, no torch/tensorflow dependency)
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class DenseLayer:
    """Single fully-connected layer with optional activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialisation
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation_name = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.W + self.b
        if self.activation_name == "relu":
            return relu(z)
        elif self.activation_name == "tanh":
            return tanh(z)
        elif self.activation_name == "sigmoid":
            return sigmoid(z)
        elif self.activation_name == "linear":
            return z
        return z

    def to_dict(self) -> dict:
        return {"W": self.W.tolist(), "b": self.b.tolist(), "activation": self.activation_name}

    @classmethod
    def from_dict(cls, d: dict) -> "DenseLayer":
        layer = cls.__new__(cls)
        layer.W = np.array(d["W"])
        layer.b = np.array(d["b"])
        layer.activation_name = d["activation"]
        return layer


# ---------------------------------------------------------------------------
# Hybrid Model
# ---------------------------------------------------------------------------

class HybridQuantumModel:
    """
    Full hybrid model: classical pre-net + VQNN core + classical post-net.

    The quantum circuit sits in the middle, receiving a compressed and
    normalised feature vector from the pre-net, and passing its output
    to the post-net which produces the final trading signal + confidence.
    """

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        qc = config.quantum
        dc = config.data

        n_raw_features = len(dc.features)  # raw input dimension
        n_qubits = qc.n_qubits             # quantum core dimension

        # Classical pre-net: raw features → quantum-compatible features
        # Two layers: compress & normalise
        self.pre_net = [
            DenseLayer(n_raw_features, n_raw_features * 2, activation="relu", seed=config.seed),
            DenseLayer(n_raw_features * 2, n_qubits, activation="tanh", seed=config.seed + 1),
        ]

        # Quantum core
        self.vqnn = VQNN(config=qc, seed=config.seed)

        # Classical post-net: quantum outputs → signal
        # Input: n_qubits PauliZ expectations; output: 1 scalar
        self.post_net = [
            DenseLayer(n_qubits, 16, activation="relu", seed=config.seed + 2),
            DenseLayer(16, 1, activation="tanh", seed=config.seed + 3),
        ]

        self._is_trained = False
        logger.info(
            "HybridQuantumModel ready. Pre-net: %d→%d, VQC: %d qubits × %d layers, Post-net: %d→1",
            n_raw_features, n_qubits, n_qubits, qc.n_layers, n_qubits
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _pre_forward(self, x: np.ndarray) -> np.ndarray:
        """Run input through the classical pre-net."""
        out = x
        for layer in self.pre_net:
            out = layer.forward(out)
        # Scale tanh output to [0, π] for angle encoding compatibility
        return (out + 1.0) * (np.pi / 2.0)

    def _post_forward(self, q_out: np.ndarray) -> float:
        """Run quantum outputs through the classical post-net."""
        out = np.array(q_out)
        for layer in self.post_net:
            out = layer.forward(out)
        return float(out[0])

    def forward(self, raw_features: np.ndarray) -> Tuple[float, str]:
        """
        Full forward pass.

        Args:
            raw_features: 1D array of raw feature values (length = n_features).

        Returns:
            (signal, action):
              signal  ∈ [-1, +1] (negative = bearish, positive = bullish)
              action  ∈ {"BUY", "SELL", "HOLD"}
        """
        # 1. Classical pre-processing
        encoded = self._pre_forward(raw_features)

        # 2. Quantum circuit
        q_out = np.array([self.vqnn._vqc(encoded, self.vqnn.params)])
        q_out = q_out.flatten()

        # 3. Classical post-processing
        signal = self._post_forward(q_out)

        # 4. Threshold to action
        bc = self.config.backtest
        if signal > bc.buy_threshold:
            action = "BUY"
        elif signal < bc.sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        return signal, action

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
        """Batch prediction. Returns (signals, actions)."""
        signals, actions = [], []
        for x in X:
            s, a = self.forward(x)
            signals.append(s)
            actions.append(a)
        return np.array(signals), actions

    # ------------------------------------------------------------------
    # Confidence score
    # ------------------------------------------------------------------

    def confidence(self, signal: float) -> float:
        """
        Return a confidence score ∈ [0, 1].
        Signal magnitude maps to confidence: |signal| closer to 1 = more confident.
        """
        return min(1.0, abs(signal))

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, directory: str):
        """Save the full model (quantum params + classical weights) to a directory."""
        os.makedirs(directory, exist_ok=True)

        # Quantum params
        np.save(os.path.join(directory, "vqnn_params.npy"), self.vqnn.params)

        # Classical layers
        classical_state = {
            "pre_net": [l.to_dict() for l in self.pre_net],
            "post_net": [l.to_dict() for l in self.post_net],
        }
        with open(os.path.join(directory, "classical_weights.json"), "w") as f:
            json.dump(classical_state, f, indent=2)

        logger.info("Model saved to %s/", directory)

    def load(self, directory: str):
        """Load the full model from a directory."""
        self.vqnn.load(os.path.join(directory, "vqnn_params.npy"))

        with open(os.path.join(directory, "classical_weights.json")) as f:
            state = json.load(f)
        self.pre_net = [DenseLayer.from_dict(d) for d in state["pre_net"]]
        self.post_net = [DenseLayer.from_dict(d) for d in state["post_net"]]
        self._is_trained = True
        logger.info("Model loaded from %s/", directory)

    def summary(self) -> str:
        qc = self.config.quantum
        dc = self.config.data
        n_classical = sum(
            l.W.size + l.b.size for l in self.pre_net + self.post_net
        )
        return (
            f"\nHybridQuantumModel\n"
            f"{'─'*40}\n"
            f"  Input features  : {len(dc.features)}\n"
            f"  Pre-net dims    : {len(dc.features)} → {qc.n_qubits}\n"
            f"  Quantum core    : {qc.n_qubits} qubits × {qc.n_layers} layers\n"
            f"  Ansatz          : {qc.ansatz}\n"
            f"  Quantum params  : {self.vqnn.n_params()}\n"
            f"  Classical params: {n_classical}\n"
            f"  Post-net dims   : {qc.n_qubits} → 1\n"
            f"  Total params    : {self.vqnn.n_params() + n_classical}\n"
            f"{'─'*40}\n"
        )
