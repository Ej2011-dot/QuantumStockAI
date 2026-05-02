# algorithms/grover_optimizer.py
#
# Grover's Algorithm applied to finding the optimal buy/sell signal threshold.
#
# CONCEPT:
# ─────────
# We have N candidate thresholds (e.g. 64 evenly spaced values from -1 to +1).
# Classical grid search evaluates all N. Grover's search finds the best in O(√N).
#
# HOW IT WORKS HERE:
# ─────────────────
# 1. Encode all N candidate thresholds as a uniform superposition |ψ⟩ = 1/√N Σ|x⟩
# 2. Define an oracle that "marks" thresholds which beat a profitability target
#    (the oracle flips the phase of good states: |x⟩ → -|x⟩ if f(x) > target)
# 3. Apply amplitude amplification (Grover diffusion operator) k ≈ π/4·√N times
# 4. Measure — the best threshold collapses to the top with high probability
#
# This gives a quadratic speedup over classical search.
# For N=64 candidates, classical needs 64 evaluations; Grover needs ~6.

import pennylane as qml
from pennylane import numpy as np
from typing import List, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------

def build_profitability_oracle(good_indices: List[int], n_qubits: int) -> Callable:
    """
    Build a phase-flip oracle that marks 'good' threshold indices.

    For each index in good_indices, the oracle flips the phase of that
    basis state: |x⟩ → -|x⟩ if x ∈ good_indices.

    This uses a multi-controlled Z gate approach: for each good index,
    we flip qubits to match that index's binary representation, apply
    a multi-controlled Z, then unflip.

    Args:
        good_indices: List of integer indices representing profitable thresholds.
        n_qubits: Number of qubits (log2 of search space size).
    """
    def oracle(wires):
        for idx in good_indices:
            # Convert index to binary and flip qubits where bit is 0
            binary = format(idx, f'0{n_qubits}b')
            flip_wires = [wires[i] for i, bit in enumerate(binary) if bit == '0']

            # Flip 0-bits so we can use a multi-controlled-Z on all-|1⟩
            for w in flip_wires:
                qml.PauliX(wires=w)

            # Multi-controlled Z (phase flip when all qubits are |1⟩)
            qml.ctrl(qml.PauliZ, control=wires[:-1])(wires=wires[-1])

            # Unflip
            for w in flip_wires:
                qml.PauliX(wires=w)

    return oracle


# ---------------------------------------------------------------------------
# Grover diffusion operator
# ---------------------------------------------------------------------------

def grover_diffusion(wires: List[int]):
    """
    Grover diffusion operator: 2|ψ⟩⟨ψ| - I
    Reflects the state about the uniform superposition.

    Implementation:
      H^⊗n · (2|0⟩⟨0| - I) · H^⊗n
    where (2|0⟩⟨0| - I) is a phase flip on the |0...0⟩ state.
    """
    # H on all qubits
    for w in wires:
        qml.Hadamard(wires=w)
    # Phase flip |0...0⟩ state
    for w in wires:
        qml.PauliX(wires=w)
    qml.ctrl(qml.PauliZ, control=wires[:-1])(wires=wires[-1])
    for w in wires:
        qml.PauliX(wires=w)
    # H on all qubits again
    for w in wires:
        qml.Hadamard(wires=w)


# ---------------------------------------------------------------------------
# Main Grover search circuit
# ---------------------------------------------------------------------------

def grover_search(
    good_indices: List[int],
    n_qubits: int,
    n_iterations: int,
    device_name: str = "default.qubit",
) -> np.ndarray:
    """
    Run Grover's algorithm and return the probability distribution over all states.

    Args:
        good_indices: Indices of 'good' (profitable) threshold candidates.
        n_qubits: Circuit size (search space = 2^n_qubits candidates).
        n_iterations: Number of Grover iterations (optimal ≈ π/4 · √(2^n / |good|)).
        device_name: PennyLane device.

    Returns:
        np.ndarray of shape (2^n_qubits,) — probability of each candidate being selected.
    """
    n_states = 2 ** n_qubits
    wires = list(range(n_qubits))
    dev = qml.device(device_name, wires=n_qubits)
    oracle = build_profitability_oracle(good_indices, n_qubits)

    @qml.qnode(dev)
    def circuit():
        # Step 1: Uniform superposition over all candidates
        for w in wires:
            qml.Hadamard(wires=w)

        # Step 2: Grover iterations
        for _ in range(n_iterations):
            oracle(wires)
            grover_diffusion(wires)

        # Step 3: Return probabilities of all basis states
        return qml.probs(wires=wires)

    return circuit()


# ---------------------------------------------------------------------------
# High-level threshold optimizer
# ---------------------------------------------------------------------------

class GroverThresholdOptimizer:
    """
    Uses Grover's algorithm to find the optimal buy/sell signal threshold
    over a discrete grid of candidates.

    Usage:
        optimizer = GroverThresholdOptimizer(n_qubits=6, threshold_range=(-1, 1))
        best_threshold = optimizer.fit(signals, returns)
        print(f"Optimal threshold: {best_threshold:.4f}")
    """

    def __init__(self, n_qubits: int = 6, threshold_range: Tuple[float, float] = (-1.0, 1.0)):
        self.n_qubits = n_qubits
        self.n_candidates = 2 ** n_qubits
        self.threshold_range = threshold_range
        # Evenly spaced candidate thresholds
        self.candidates = np.linspace(threshold_range[0], threshold_range[1], self.n_candidates)
        self.best_threshold_ = None
        self.profitabilities_ = None

    def _evaluate_thresholds(self, signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Classically compute profitability of each threshold candidate.
        (The oracle needs to know which are 'good' — this is the classical prep step.)

        Profitability metric: average return when signal > threshold (buy signal).
        """
        profitabilities = np.zeros(self.n_candidates)
        for i, thresh in enumerate(self.candidates):
            buy_mask = signals > thresh
            if buy_mask.sum() > 0:
                profitabilities[i] = returns[buy_mask].mean()
        return profitabilities

    def _optimal_iterations(self, n_good: int) -> int:
        """Compute optimal number of Grover iterations: π/4 · √(N/k)."""
        if n_good == 0:
            return 1
        k = max(1, n_good)
        return max(1, int(np.round(np.pi / 4 * np.sqrt(self.n_candidates / k))))

    def fit(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        profitability_percentile: float = 75.0,
    ) -> float:
        """
        Find the optimal threshold using Grover's search.

        Args:
            signals: Model output signals (shape: [n_samples]).
            returns: Actual next-day returns (shape: [n_samples]).
            profitability_percentile: Top-N% of thresholds are marked as 'good'.

        Returns:
            The optimal threshold value as a float.
        """
        logger.info("Evaluating %d threshold candidates classically...", self.n_candidates)
        self.profitabilities_ = self._evaluate_thresholds(signals, returns)

        # Mark the top-percentile thresholds as 'good' for the oracle
        cutoff = np.percentile(self.profitabilities_, profitability_percentile)
        good_indices = [i for i, p in enumerate(self.profitabilities_) if p >= cutoff]

        if not good_indices:
            logger.warning("No good thresholds found. Defaulting to 0.0.")
            self.best_threshold_ = 0.0
            return 0.0

        n_good = len(good_indices)
        n_iter = self._optimal_iterations(n_good)
        logger.info("Grover search: %d good candidates, %d iterations", n_good, n_iter)

        # Run Grover's algorithm
        probs = grover_search(good_indices, self.n_qubits, n_iter)

        # Pick the threshold with the highest post-Grover probability
        best_idx = int(np.argmax(probs))
        self.best_threshold_ = float(self.candidates[best_idx])

        logger.info(
            "Grover search complete. Best threshold: %.4f (prob: %.3f)",
            self.best_threshold_, float(probs[best_idx])
        )
        return self.best_threshold_

    def summary(self) -> dict:
        """Return a summary dict of the search results."""
        if self.profitabilities_ is None:
            return {"status": "not fitted"}
        return {
            "n_candidates": self.n_candidates,
            "n_qubits": self.n_qubits,
            "best_threshold": self.best_threshold_,
            "best_profitability": float(self.profitabilities_[
                np.argmin(np.abs(self.candidates - self.best_threshold_))
            ]) if self.best_threshold_ is not None else None,
            "candidates": self.candidates.tolist(),
            "profitabilities": self.profitabilities_.tolist(),
        }
