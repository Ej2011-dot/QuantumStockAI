# core/quantum_circuit.py
# Base quantum circuit layer builder.
# Supports multiple ansatz styles and is the foundation for the VQNN model.

import pennylane as qml
from pennylane import numpy as np
from typing import List, Literal


AnsatzType = Literal["basic", "strongly_entangling", "hardware_efficient"]


# ---------------------------------------------------------------------------
# Encoding layers — classical data → quantum state
# ---------------------------------------------------------------------------

def angle_encoding(features: np.ndarray, wires: List[int]):
    """
    Encode a feature vector as RY rotations on each qubit.
    Each feature is mapped to [0, π] via the preprocessor before this is called.
    This is the simplest encoding — one feature per qubit.
    """
    for i, wire in enumerate(wires):
        qml.RY(features[i], wires=wire)


def zz_feature_map(features: np.ndarray, wires: List[int], reps: int = 2):
    """
    ZZFeatureMap-style encoding (inspired by Qiskit's circuit).
    Applies Hadamard + RZ(2*x_i) + ZZ interactions (RZZ(2*(π-x_i)(π-x_j))).
    More expressive than simple angle encoding for separating feature clusters.
    """
    n = len(wires)
    for _ in range(reps):
        # Hadamard layer
        for wire in wires:
            qml.Hadamard(wires=wire)
        # Single-qubit phase rotations
        for i, wire in enumerate(wires):
            qml.RZ(2.0 * features[i], wires=wire)
        # Two-qubit ZZ interactions (nearest-neighbor)
        for i in range(n - 1):
            qml.IsingZZ(2.0 * (np.pi - features[i]) * (np.pi - features[i + 1]),
                        wires=[wires[i], wires[i + 1]])


# ---------------------------------------------------------------------------
# Variational (trainable) ansatz layers
# ---------------------------------------------------------------------------

def basic_ansatz(params: np.ndarray, wires: List[int]):
    """
    Simple alternating RY rotations + CNOT chain.
    params shape: (n_wires, 2) — one RY before and after the entanglement.

    Circuit structure:
        RY(θ₀) ──●── RY(φ₀)
        RY(θ₁) ──X──●── RY(φ₁)
        RY(θ₂)     ──X──●── RY(φ₂)
        ...
    """
    n = len(wires)
    # Pre-entanglement rotations
    for i, wire in enumerate(wires):
        qml.RY(params[i, 0], wires=wire)
    # CNOT chain (linear entanglement)
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    # Post-entanglement rotations
    for i, wire in enumerate(wires):
        qml.RY(params[i, 1], wires=wire)


def strongly_entangling_ansatz(params: np.ndarray, wires: List[int], layer_idx: int = 0):
    """
    Strongly Entangling Layers (Schuld et al. 2020).
    Applies RZ-RY-RZ rotation block + CNOT with stride to maximise entanglement.
    params shape: (n_wires, 3) — three rotation angles per qubit.

    This is the most expressive single-layer ansatz available.
    """
    n = len(wires)
    # Rotation block: Rz Ry Rz (general SU(2) rotation)
    for i, wire in enumerate(wires):
        qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=wire)
    # Strided CNOT entanglement — stride changes per layer for coverage
    stride = (layer_idx % (n - 1)) + 1
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + stride) % n]])


def hardware_efficient_ansatz(params: np.ndarray, wires: List[int]):
    """
    Hardware-efficient ansatz suitable for real quantum hardware.
    Uses only RY + RZ (cheaper than Rot) and nearest-neighbour CZ gates.
    params shape: (n_wires, 2)
    """
    n = len(wires)
    for i, wire in enumerate(wires):
        qml.RY(params[i, 0], wires=wire)
        qml.RZ(params[i, 1], wires=wire)
    # CZ gates (symmetric, easier to calibrate on hardware)
    for i in range(0, n - 1, 2):
        qml.CZ(wires=[wires[i], wires[i + 1]])
    for i in range(1, n - 1, 2):
        qml.CZ(wires=[wires[i], wires[i + 1]])


# ---------------------------------------------------------------------------
# Parameter shape helpers
# ---------------------------------------------------------------------------

def params_shape(ansatz: AnsatzType, n_qubits: int, n_layers: int) -> tuple:
    """Return the required shape for the trainable parameter array."""
    shapes = {
        "basic": (n_layers, n_qubits, 2),
        "strongly_entangling": (n_layers, n_qubits, 3),
        "hardware_efficient": (n_layers, n_qubits, 2),
    }
    if ansatz not in shapes:
        raise ValueError(f"Unknown ansatz: {ansatz}. Choose from {list(shapes)}")
    return shapes[ansatz]


def random_params(ansatz: AnsatzType, n_qubits: int, n_layers: int,
                  seed: int = 42) -> np.ndarray:
    """Initialise variational parameters with small random values near zero."""
    rng = np.random.default_rng(seed)
    shape = params_shape(ansatz, n_qubits, n_layers)
    # Small init prevents barren plateaus at the start of training
    return rng.uniform(-0.1, 0.1, size=shape)


# ---------------------------------------------------------------------------
# Full VQC circuit factory
# ---------------------------------------------------------------------------

def build_vqc(n_qubits: int, n_layers: int, ansatz: AnsatzType,
              device_name: str = "default.qubit"):
    """
    Factory that returns a compiled QNode for the chosen ansatz.
    The returned function signature is:
        vqc(features, params) → np.ndarray of shape (n_qubits,)

    Returns both the qnode and the expected params shape.
    """
    wires = list(range(n_qubits))
    dev = qml.device(device_name, wires=n_qubits)

    ansatz_fns = {
        "basic": basic_ansatz,
        "strongly_entangling": strongly_entangling_ansatz,
        "hardware_efficient": hardware_efficient_ansatz,
    }
    ansatz_fn = ansatz_fns[ansatz]

    @qml.qnode(dev, diff_method="parameter-shift")
    def vqc(features, params):
        # 1. Encode classical features into the quantum state
        zz_feature_map(features, wires)

        # 2. Apply variational layers
        for layer_idx in range(n_layers):
            if ansatz == "strongly_entangling":
                ansatz_fn(params[layer_idx], wires, layer_idx=layer_idx)
            else:
                ansatz_fn(params[layer_idx], wires)

        # 3. Measure all qubits in the Z basis
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    shape = params_shape(ansatz, n_qubits, n_layers)
    return vqc, shape


def draw_circuit(vqc_fn, features, params) -> str:
    """Return a text diagram of the circuit for a given input."""
    return qml.draw(vqc_fn)(features, params)
