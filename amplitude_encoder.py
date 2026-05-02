# algorithms/amplitude_encoder.py
#
# Amplitude Encoding — the most data-efficient quantum encoding scheme.
#
# CONCEPT:
# ─────────
# A quantum state with n qubits has 2^n amplitudes.
# Amplitude encoding maps a classical vector of 2^n values into those amplitudes:
#
#   |ψ⟩ = (1/‖x‖) Σ xᵢ |i⟩
#
# For n=6 qubits we can encode 2^6 = 64 values in one state!
# Angle encoding (1 feature per qubit) would need 64 qubits for the same data.
#
# TRADE-OFF:
# Amplitude encoding is exponentially efficient in space but requires a
# state-preparation circuit whose depth grows with the data size.
# We use PennyLane's `qml.AmplitudeEmbedding` which handles this automatically.

import pennylane as qml
from pennylane import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def amplitude_encode(features: np.ndarray, wires: List[int], normalize: bool = True):
    """
    Encode a feature vector using amplitude embedding.

    PennyLane's AmplitudeEmbedding requires:
      - len(features) == 2^len(wires)
      - ‖features‖ == 1 (if normalize=False you must provide a unit vector)

    If len(features) < 2^n, the vector is zero-padded.
    If normalize=True, the vector is L2-normalised automatically.

    Args:
        features: 1D array of classical features.
        wires: List of qubit wire indices.
        normalize: Auto-normalise the input vector.
    """
    n_amplitudes = 2 ** len(wires)
    padded = np.zeros(n_amplitudes)
    n = min(len(features), n_amplitudes)
    padded[:n] = features[:n]

    if normalize:
        norm = np.linalg.norm(padded)
        if norm < 1e-8:
            # Avoid division by zero; default to |0...0⟩
            padded[0] = 1.0
        else:
            padded = padded / norm

    qml.AmplitudeEmbedding(padded, wires=wires, normalize=False)


def angle_amplitude_hybrid(
    angle_features: np.ndarray,
    amplitude_features: np.ndarray,
    angle_wires: List[int],
    amplitude_wires: List[int],
):
    """
    Hybrid encoding:
    - Short-term features (e.g. recent 5 days of returns) → angle encoding (intuitive)
    - Longer feature vectors (e.g. 20-day price history) → amplitude encoding (efficient)

    This splits the circuit into two register blocks and encodes each separately.
    """
    # Angle-encode the short-term features
    for i, wire in enumerate(angle_wires):
        if i < len(angle_features):
            qml.RY(float(angle_features[i]) * np.pi, wires=wire)

    # Amplitude-encode the longer feature vector
    amplitude_encode(amplitude_features, amplitude_wires, normalize=True)


class AmplitudeFeatureMap:
    """
    Wrapper that turns a pandas DataFrame of features into quantum-ready
    amplitude embeddings, handling padding, normalisation, and device setup.

    Usage:
        feature_map = AmplitudeFeatureMap(n_qubits=4)
        circuit = feature_map.build_circuit()
    """

    def __init__(self, n_qubits: int, device_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.n_amplitudes = 2 ** n_qubits
        self.device_name = device_name
        self._dev = qml.device(device_name, wires=n_qubits)

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode features and return the resulting quantum state vector.
        (Useful for debugging and visualising what the encoding actually does.)
        """
        wires = list(range(self.n_qubits))

        @qml.qnode(self._dev)
        def state_circuit(f):
            amplitude_encode(f, wires, normalize=True)
            return qml.state()

        return state_circuit(features)

    def fidelity(self, features_a: np.ndarray, features_b: np.ndarray) -> float:
        """
        Compute the quantum fidelity |⟨ψ_a|ψ_b⟩|² between two encoded states.
        This is the quantum analogue of cosine similarity.
        Useful for measuring how 'similar' two market conditions are.
        """
        state_a = self.encode(features_a)
        state_b = self.encode(features_b)
        overlap = np.abs(np.dot(np.conj(state_a), state_b)) ** 2
        return float(overlap)

    def build_encoding_layer(self) -> callable:
        """
        Return a QNode that applies amplitude encoding.
        Plug this into a larger circuit as the encoding layer.
        """
        wires = list(range(self.n_qubits))

        @qml.qnode(self._dev)
        def encoding_layer(features):
            amplitude_encode(features, wires, normalize=True)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        return encoding_layer
