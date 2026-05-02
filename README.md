# QuantumTrader Quantum Neural Network Stock Predictor

A full quantum machine learning pipeline combining:
- **Variational Quantum Circuit (VQC)** — the "neural network" with trainable parameters
- **Grover's Algorithm** — quantum search to find optimal circuit parameters faster
- **Classical backtesting** — measures real predictive accuracy vs buy-and-hold
- **Feature engineering** — RSI, MACD, volatility, normalized returns

---

## Project Structure

```
quantum_trader/
├── main.py                  ← Entry point — runs full pipeline
├── config.py                ← All hyperparameters & settings
├── requirements.txt         ← Dependencies
│
├── data/
│   ├── fetcher.py           ← Downloads stock data via yfinance
│   └── features.py          ← Feature engineering (RSI, MACD, volatility)
│
├── core/
│   ├── circuit.py           ← Variational quantum circuit (the QNN)
│   └── encoding.py          ← Encodes classical features → qubits
│
├── grover/
│   ├── search.py            ← Grover's algorithm for parameter search
│   └── oracle.py            ← Oracle marks "good" parameter regions
│
├── models/
│   ├── qnn.py               ← Full quantum neural network class
│   └── trainer.py           ← Training loop with gradient descent
│
├── backtest/
│   ├── engine.py            ← Backtesting engine
│   └── metrics.py           ← Sharpe ratio, accuracy, drawdown, etc.
│
├── utils/
│   ├── logger.py            ← Colored console logging
│   └── plotter.py           ← Charts: equity curve, circuit diagram, signals
│
└── viz/
    └── dashboard.py         ← Terminal dashboard (rich library)
```

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py --ticker AAPL --period 2y --epochs 60
```

### Key flags
| Flag | Default | Description |
|------|---------|-------------|
| --ticker | AAPL | Stock symbol |
| --period | 1y | yfinance period |
| --epochs | 40 | VQC training epochs |
| --layers | 3 | Quantum circuit depth |
| --grover | True | Use Grover search for init |
| --benchmark | True | Compare vs buy-and-hold |

---

## How It Works

### 1. Data & Features
Raw OHLCV data → feature engineering → 5 normalized features per day:
- 5-day normalized return
- RSI (14-day)
- MACD signal line
- 5-day realized volatility
- Volume Z-score

### 2. Quantum Encoding
Each feature is encoded into a qubit rotation angle via angle encoding (RY gates).
5 features → 5 qubits.

### 3. Variational Quantum Circuit (QNN)
Each layer has:
- RY(θ) rotations (trainable)
- RZ(φ) rotations (trainable)
- CNOT entanglement ladder

Output: PauliZ expectation → BUY / SELL signal.

### 4. Grover's Algorithm
Searches a discretized parameter space to find low-loss starting regions.
Gives the VQC a smarter init before gradient descent takes over.

### 5. Training
Adam optimizer via PennyLane. Loss = MSE vs next-day return direction.

### 6. Backtesting
Signal > 0.1 → BUY, Signal < -0.1 → SELL/flat.
Reports accuracy, Sharpe, max drawdown, return vs benchmark.
