# algorithms/qpe_volatility.py
#
# Quantum Phase Estimation (QPE) adapted for extracting a volatility signal.
#
# CONCEPT:
# ─────────
# QPE is a fundamental quantum algorithm that extracts the eigenphase θ of a
# unitary operator U (where U|ψ⟩ = e^{2πiθ}|ψ⟩).
#
# HERE WE USE IT AS:
# A phase extraction technique to "read out" how much phase has been
# accumulated by a volatility-encoded unitary — giving us a quantum-native
# estimate of the market's volatility regime.
#
# The volatility U is constructed so that high-vol markets accumulate
# more phase, and low-vol markets accumulate less. QPE then reads this phase.
#
# This is a simplified / educational version — on real hardware you'd use
# the full QPE with classical phase kickback and inverse QFT.

import pennylane as qml
from pennylane import numpy as np
import numpy as _np
from typing import List
import logging

logger = logging.getLogger(__name__)


def volatility_unitary(vol_angle: float, target_wire: int):
    """
    A simple phase-encoding unitary.
    U = RZ(2 * vol_angle) — applies a phase rotation proportional to volatility.
    High volatility → larger rotation → more phase accumulated.
    """
    qml.RZ(2.0 * vol_angle, wires=target_wire)


def inverse_qft(wires: List[int]):
    """
    Inverse Quantum Fourier Transform on the given wires.
    Used in QPE to convert accumulated phase into a readable binary number.
    Decomposed into Hadamard + controlled phase rotations.
    """
    n = len(wires)
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - 1 - i]])
    for i in range(n):
        qml.Hadamard(wires=wires[i])
        for j in range(i + 1, n):
            angle = -_np.pi / (2 ** (j - i))
            qml.ctrl(qml.PhaseShift, control=wires[j])(angle, wires=wires[i])


def build_qpe_circuit(
    n_counting_qubits: int = 4,
    target_wire: int = 4,
    device_name: str = "default.qubit",
):
    """
    Build a QPE circuit that estimates the phase of a volatility unitary.

    Architecture:
      Counting register (n_counting_qubits): accumulates phase via controlled-U^{2^k}
      Target register (1 qubit): the eigenstate of U

    Returns a QNode that takes a normalised volatility value and returns
    the estimated phase as a float in [0, 1].
    """
    n_total = n_counting_qubits + 1
    counting_wires = list(range(n_counting_qubits))
    dev = qml.device(device_name, wires=n_total)

    @qml.qnode(dev)
    def qpe_circuit(vol_angle: float):
        # 1. Prepare counting register in superposition
        for w in counting_wires:
            qml.Hadamard(wires=w)

        # 2. Prepare target qubit in eigenstate of U (|+⟩ for phase unitary)
        qml.Hadamard(wires=target_wire)

        # 3. Controlled-U^{2^k} gates: each counting qubit controls 2^k applications
        for k, control_wire in enumerate(counting_wires):
            n_applications = 2 ** k
            for _ in range(n_applications):
                qml.ctrl(volatility_unitary, control=control_wire)(
                    vol_angle, target_wire=target_wire
                )

        # 4. Inverse QFT on counting register → phase → binary number
        inverse_qft(counting_wires)

        # 5. Measure probabilities over counting register
        return qml.probs(wires=counting_wires)

    return qpe_circuit


class QPEVolatilityDetector:
    """
    Uses Quantum Phase Estimation to classify the current volatility regime.

    Returns:
      - A quantum phase estimate ∈ [0, 1]
      - A regime classification: "LOW", "MEDIUM", or "HIGH"

    Usage:
        detector = QPEVolatilityDetector()
        phase, regime = detector.detect(current_volatility_value)
    """

    REGIMES = [
        (0.0, 0.33, "LOW"),
        (0.33, 0.67, "MEDIUM"),
        (0.67, 1.0, "HIGH"),
    ]

    def __init__(self, n_counting_qubits: int = 4):
        self.n_counting_qubits = n_counting_qubits
        self._circuit = build_qpe_circuit(n_counting_qubits)
        self._n_states = 2 ** n_counting_qubits

    def _vol_to_angle(self, vol: float) -> float:
        """
        Map a normalised volatility value [0, 1] to a rotation angle [0, π].
        High vol → angle near π → large phase accumulation.
        """
        return float(vol) * _np.pi

    def detect(self, normalised_vol: float) -> tuple:
        """
        Run QPE and return (estimated_phase, regime_label).

        Args:
            normalised_vol: Volatility value in [0, 1] (e.g. output of realized_volatility()).

        Returns:
            (phase, regime): float in [0, 1], one of "LOW" / "MEDIUM" / "HIGH"
        """
        angle = self._vol_to_angle(normalised_vol)
        probs = self._circuit(angle)
        probs = _np.array(probs)

        # QPE: the most probable state encodes the phase as a binary fraction
        best_idx = int(_np.argmax(probs))
        estimated_phase = best_idx / self._n_states  # ∈ [0, 1)

        # Map to regime
        regime = "MEDIUM"
        for lo, hi, label in self.REGIMES:
            if lo <= estimated_phase < hi:
                regime = label
                break

        logger.debug(
            "QPE: vol=%.3f → angle=%.3f → phase=%.3f → regime=%s",
            normalised_vol, angle, estimated_phase, regime
        )
        return estimated_phase, regime

    def batch_detect(self, vol_series: _np.ndarray) -> list:
        """Run detection on a full series of volatility values."""
        return [self.detect(v) for v in vol_series]
