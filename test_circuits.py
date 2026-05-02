# tests/test_circuits.py
# Unit tests for quantum circuit building blocks.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pennylane as qml
import pytest

from core.quantum_circuit import (
    build_vqc,
    random_params,
    params_shape,
    angle_encoding,
    basic_ansatz,
    strongly_entangling_ansatz,
    hardware_efficient_ansatz,
    draw_circuit,
)
from config import QuantumConfig


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return QuantumConfig(n_qubits=4, n_layers=2, ansatz="strongly_entangling")

@pytest.fixture
def sample_features(config):
    rng = np.random.default_rng(0)
    return rng.uniform(0, np.pi, size=config.n_qubits)


# ── param shape tests ─────────────────────────────────────────────────────────

def test_params_shape_basic():
    shape = params_shape("basic", n_qubits=4, n_layers=3)
    assert shape == (3, 4, 2)

def test_params_shape_strongly_entangling():
    shape = params_shape("strongly_entangling", n_qubits=6, n_layers=2)
    assert shape == (2, 6, 3)

def test_params_shape_hardware_efficient():
    shape = params_shape("hardware_efficient", n_qubits=5, n_layers=4)
    assert shape == (4, 5, 2)

def test_params_shape_invalid():
    with pytest.raises(ValueError):
        params_shape("nonexistent_ansatz", 4, 2)


# ── random_params tests ───────────────────────────────────────────────────────

def test_random_params_shape(config):
    params = random_params(config.ansatz, config.n_qubits, config.n_layers, seed=42)
    expected = params_shape(config.ansatz, config.n_qubits, config.n_layers)
    assert params.shape == expected

def test_random_params_small_values(config):
    """Params should be initialised near zero to avoid barren plateaus."""
    params = random_params(config.ansatz, config.n_qubits, config.n_layers, seed=42)
    assert np.abs(params).max() <= 0.5, "Initial params should be small"

def test_random_params_reproducible(config):
    p1 = random_params(config.ansatz, config.n_qubits, config.n_layers, seed=7)
    p2 = random_params(config.ansatz, config.n_qubits, config.n_layers, seed=7)
    np.testing.assert_array_equal(p1, p2)


# ── VQC output tests ──────────────────────────────────────────────────────────

def test_vqc_output_range(config, sample_features):
    """VQC output (PauliZ expectations) should be in [-1, +1]."""
    vqc, shape = build_vqc(config.n_qubits, config.n_layers, config.ansatz)
    params = random_params(config.ansatz, config.n_qubits, config.n_layers)
    output = np.array(vqc(sample_features, params))
    assert output.shape == (config.n_qubits,)
    assert np.all(output >= -1.0 - 1e-6)
    assert np.all(output <= 1.0 + 1e-6)

def test_vqc_different_params_give_different_output(config, sample_features):
    """Two different parameter sets should (almost certainly) give different outputs."""
    vqc, _ = build_vqc(config.n_qubits, config.n_layers, config.ansatz)
    p1 = random_params(config.ansatz, config.n_qubits, config.n_layers, seed=1)
    p2 = random_params(config.ansatz, config.n_qubits, config.n_layers, seed=99)
    out1 = np.array(vqc(sample_features, p1))
    out2 = np.array(vqc(sample_features, p2))
    assert not np.allclose(out1, out2), "Different params should give different outputs"

@pytest.mark.parametrize("ansatz", ["basic", "strongly_entangling", "hardware_efficient"])
def test_all_ansatz_types_run(ansatz, sample_features):
    """All ansatz types should build and run without error."""
    n_qubits, n_layers = 4, 2
    vqc, _ = build_vqc(n_qubits, n_layers, ansatz)
    params = random_params(ansatz, n_qubits, n_layers, seed=0)
    output = np.array(vqc(sample_features, params))
    assert len(output) == n_qubits


# ── circuit draw test ─────────────────────────────────────────────────────────

def test_draw_circuit_returns_string(config, sample_features):
    vqc, _ = build_vqc(config.n_qubits, config.n_layers, config.ansatz)
    params = random_params(config.ansatz, config.n_qubits, config.n_layers)
    diagram = draw_circuit(vqc, sample_features, params)
    assert isinstance(diagram, str)
    assert len(diagram) > 0
    assert "─" in diagram or "0" in diagram  # basic circuit diagram markers
