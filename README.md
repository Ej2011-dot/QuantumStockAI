# QuantumStockAI
QuantumStockAI is an experimental quantum finance project that encodes recent stock price movements into qubits and generates a trend signal using an entangled quantum circuit built with PennyLane.

## How it Works

1. Fetch recent stock data using Yahoo Finance
2. Normalize the last 5 closing prices
3. Encode prices as rotation angles on qubits
4. Entangle qubits using CNOT gates
5. Measure the final qubit to produce a quantum signal

## Technologies

- Python
- PennyLane
- Quantum circuits
- yfinance

## Example Output

Prices analyzed: [257.46 259.22 260.14 261.01 260.55]  
Quantum signal: -0.34  

Negative  BUY signal  
Positive SELL signal
