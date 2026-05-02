# config.py — Central configuration for QuantumTrader
# All hyperparameters live here. Pass a Config object around instead of magic numbers.

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QuantumConfig:
    """Quantum circuit configuration."""
    n_qubits: int = 6                    # Number of qubits (= number of features)
    n_layers: int = 3                    # VQC depth (more = more expressive, slower)
    ansatz: str = "strongly_entangling"  # Options: "basic", "strongly_entangling", "hardware_efficient"
    measurement: str = "pauli_z"         # Options: "pauli_z", "pauli_x", "pauli_y"
    device: str = "default.qubit"        # PennyLane device
    shots: Optional[int] = None          # None = exact simulation; int = shot-based


@dataclass
class GroverConfig:
    """Grover's algorithm search configuration."""
    n_threshold_candidates: int = 64     # Search space size (must be power of 2)
    n_grover_iterations: int = 3         # Optimal ≈ π/4 * √N
    threshold_range: tuple = (-1.0, 1.0) # Range for buy/sell signal thresholds
    n_qubits_grover: int = 6            # log2(n_threshold_candidates)


@dataclass
class DataConfig:
    """Data fetching and feature engineering configuration."""
    ticker: str = "AAPL"
    period: str = "2y"                   # yfinance period string
    interval: str = "1d"
    sequence_length: int = 20            # Days of history per sample
    train_split: float = 0.70
    val_split: float = 0.15
    # test_split is implicit: 1 - train - val
    features: List[str] = field(default_factory=lambda: [
        "returns",        # Daily log returns
        "rsi_14",         # RSI (14-day)
        "macd_signal",    # MACD signal line
        "bb_position",    # Position within Bollinger Bands (0–1)
        "volume_zscore",  # Volume z-score vs 20-day avg
        "volatility_20",  # 20-day realized volatility
    ])


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    epochs: int = 100
    learning_rate: float = 0.01
    optimizer: str = "adam"              # Options: "adam", "sgd", "qng" (quantum natural gradient)
    batch_size: int = 32
    early_stopping_patience: int = 15
    gradient_method: str = "parameter_shift"  # Options: "parameter_shift", "adjoint"
    loss: str = "mse"                    # Options: "mse", "bce", "sharpe_loss"
    l2_reg: float = 1e-4


@dataclass
class BacktestConfig:
    """Backtesting engine configuration."""
    initial_capital: float = 10_000.0
    position_sizing: str = "fixed"       # Options: "fixed", "kelly", "confidence_weighted"
    transaction_cost: float = 0.001      # 0.1% per trade (realistic broker fee)
    slippage: float = 0.0005            # 0.05% slippage
    buy_threshold: float = 0.2          # Signal > this → BUY
    sell_threshold: float = -0.2        # Signal < this → SELL
    # Between thresholds → HOLD
    max_position_pct: float = 1.0       # Max fraction of capital in one position
    benchmark: str = "SPY"              # Benchmark for comparison


@dataclass
class Config:
    """Master config — compose all sub-configs."""
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    grover: GroverConfig = field(default_factory=GroverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    results_dir: str = "results/"
    log_level: str = "INFO"
    seed: int = 42

    def validate(self):
        """Sanity-check that config values are consistent."""
        assert self.quantum.n_qubits == len(self.data.features), (
            f"n_qubits ({self.quantum.n_qubits}) must equal number of features "
            f"({len(self.data.features)}). Adjust either quantum.n_qubits or data.features."
        )
        assert self.data.train_split + self.data.val_split < 1.0, (
            "train_split + val_split must be < 1.0 to leave room for test set."
        )
        assert self.backtest.buy_threshold > self.backtest.sell_threshold, (
            "buy_threshold must be greater than sell_threshold."
        )
        return self


# Default config — import this in other modules
DEFAULT_CONFIG = Config().validate()
