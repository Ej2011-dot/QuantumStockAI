import yfinance as yf
import pennylane as qml
from pennylane import numpy as np

# 1. GET THE DATA
print("Downloading Apple Stock data...")
df = yf.download("AAPL", period="1mo", interval="1d", auto_adjust=True)


prices = df['Close'].values[-5:].flatten() 


min_p, max_p = np.min(prices), np.max(prices)
norm_prices = (prices - min_p) / (max_p - min_p) * np.pi


dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def quantum_predictor(data):
    # Step A: turn prices into rotations
    for i in range(5):
        qml.RY(data[i], wires=i)
    
    # Step B: Link the days together (Entanglement)
    for i in range(4):
        qml.CNOT(wires=[i, i+1])
        
    # Step C: Read the final "vibe" of the last qubit
    return qml.expval(qml.PauliZ(4))


signal = quantum_predictor(norm_prices)

# Converting 'signal' to float() fixes the "TypeError" you saw
print(f"\nPrices Analyzed: {prices.round(2)}")
print(f"Quantum Signal: {float(signal):.4f}")
print("If signal is NEGATIVE (-), it's a BUY signal. If POSITIVE (+), it's a SELL signal.")
# 5. DRAW THE CIRCUIT
print("\n--- QUANTUM CIRCUIT DIAGRAM ---")
print(qml.draw(quantum_predictor)(norm_prices))
