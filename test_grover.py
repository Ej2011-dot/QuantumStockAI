# tests/test_grover.py
# Tests for Grover's algorithm implementation.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from algorithms.grover_optimizer import (
    grover_search,
    GroverThresholdOptimizer,
    build_profitability_oracle,
    grover_diffusion,
)


# ── Oracle tests ──────────────────────────────────────────────────────────────

def test_grover_search_single_target():
    """
    With one marked state, Grover's search should amplify that state's probability.
    For N=4 (n_qubits=2), optimal iterations = 1.
    The marked state should have the highest probability after the search.
    """
    n_qubits = 2
    good_indices = [2]  # Mark state |10⟩
    n_iter = 1
    probs = grover_search(good_indices, n_qubits, n_iter)

    assert len(probs) == 2 ** n_qubits
    assert np.argmax(probs) == 2, f"Expected state 2 to win, got {np.argmax(probs)}"
    assert probs[2] > 0.5, f"Marked state probability should be > 0.5, got {probs[2]:.3f}"


def test_grover_search_probability_sums_to_one():
    """Probabilities from Grover search must sum to 1 (valid quantum state)."""
    n_qubits = 3
    good_indices = [1, 5]
    probs = grover_search(good_indices, n_qubits, n_iterations=2)
    assert abs(probs.sum() - 1.0) < 1e-6, f"Probabilities sum to {probs.sum()}, expected 1.0"


def test_grover_search_multiple_targets():
    """Multiple marked states should all have elevated probability."""
    n_qubits = 3
    good_indices = [0, 7]  # Mark |000⟩ and |111⟩
    probs = grover_search(good_indices, n_qubits, n_iterations=1)
    # Both marked states should have higher prob than unmarked ones
    unmarked_avg = np.mean([probs[i] for i in range(8) if i not in good_indices])
    marked_avg = np.mean([probs[i] for i in good_indices])
    assert marked_avg > unmarked_avg, (
        f"Marked states avg prob ({marked_avg:.3f}) should exceed unmarked ({unmarked_avg:.3f})"
    )


def test_grover_search_output_shape():
    n_qubits = 4
    probs = grover_search([3], n_qubits, n_iterations=2)
    assert probs.shape == (2 ** n_qubits,)


# ── GroverThresholdOptimizer tests ────────────────────────────────────────────

@pytest.fixture
def mock_signals_returns():
    rng = np.random.default_rng(42)
    n = 100
    signals = rng.uniform(-1, 1, n)
    # Returns positively correlated with signals (idealised scenario)
    returns = 0.5 * signals + 0.1 * rng.standard_normal(n)
    return signals, returns


def test_optimizer_returns_float(mock_signals_returns):
    signals, returns = mock_signals_returns
    optimizer = GroverThresholdOptimizer(n_qubits=4, threshold_range=(-1, 1))
    result = optimizer.fit(signals, returns)
    assert isinstance(result, float)


def test_optimizer_threshold_in_range(mock_signals_returns):
    signals, returns = mock_signals_returns
    optimizer = GroverThresholdOptimizer(n_qubits=4, threshold_range=(-1, 1))
    result = optimizer.fit(signals, returns)
    assert -1.0 <= result <= 1.0, f"Threshold {result} out of range [-1, 1]"


def test_optimizer_summary_after_fit(mock_signals_returns):
    signals, returns = mock_signals_returns
    optimizer = GroverThresholdOptimizer(n_qubits=4)
    optimizer.fit(signals, returns)
    summary = optimizer.summary()
    assert "best_threshold" in summary
    assert "n_candidates" in summary
    assert summary["n_candidates"] == 2 ** 4


def test_optimizer_empty_good_indices():
    """When no thresholds are profitable, should return 0.0 without error."""
    signals = np.ones(10) * 0.5  # All same signal
    returns = np.ones(10) * -0.01  # All negative returns
    optimizer = GroverThresholdOptimizer(n_qubits=4)
    # Force all profitabilities negative → no 'good' thresholds
    result = optimizer.fit(signals, returns, profitability_percentile=100.0)
    # Should return something (either 0.0 default or a valid threshold)
    assert isinstance(result, float)
