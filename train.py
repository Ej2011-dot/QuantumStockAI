#!/usr/bin/env python3
# train.py — Training entrypoint for QuantumTrader
#
# Usage:
#   python train.py --ticker AAPL --period 2y --layers 3 --epochs 80

import argparse
import logging
import os
import sys
import json
from datetime import datetime

import numpy as np

# Make sure local imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

from config import Config, QuantumConfig, DataConfig, TrainingConfig, BacktestConfig
from data.fetcher import StockDataPipeline
from models.hybrid_model import HybridQuantumModel
from algorithms.grover_optimizer import GroverThresholdOptimizer
from utils.backtest import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description="Train QuantumTrader VQNN")
    p.add_argument("--ticker", default="AAPL", help="Stock ticker symbol")
    p.add_argument("--period", default="2y", help="Data period (yfinance format: 1y, 2y, 5y)")
    p.add_argument("--layers", type=int, default=3, help="Number of VQC layers")
    p.add_argument("--epochs", type=int, default=80, help="Training epochs")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--ansatz", default="strongly_entangling",
                   choices=["basic", "strongly_entangling", "hardware_efficient"])
    p.add_argument("--no-grover", action="store_true", help="Skip Grover threshold optimisation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.ticker}_{timestamp}"
    results_dir = os.path.join("results", run_name)
    os.makedirs(results_dir, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────
    config = Config(
        quantum=QuantumConfig(n_layers=args.layers, ansatz=args.ansatz),
        data=DataConfig(ticker=args.ticker, period=args.period),
        training=TrainingConfig(epochs=args.epochs, learning_rate=args.lr),
        seed=args.seed,
        results_dir=results_dir,
    )
    config.validate()

    logger.info("=" * 50)
    logger.info("  QuantumTrader Training Run: %s", run_name)
    logger.info("=" * 50)

    # ── Data ──────────────────────────────────────────────────────────────
    pipeline = StockDataPipeline(config.data)
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run(args.ticker)
    prices = pipeline.raw_df_["Close"].values

    # ── Model ─────────────────────────────────────────────────────────────
    model = HybridQuantumModel(config)
    print(model.summary())

    # ── Training ──────────────────────────────────────────────────────────
    logger.info("Starting training (%d epochs, lr=%.4f)...", args.epochs, args.lr)
    history = model.vqnn.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=config.training.early_stopping_patience,
        verbose=True,
    )

    # ── Save trained model ────────────────────────────────────────────────
    model_dir = os.path.join(results_dir, "model")
    model.save(model_dir)

    # ── Grover threshold optimisation ────────────────────────────────────
    buy_threshold = config.backtest.buy_threshold
    sell_threshold = config.backtest.sell_threshold

    if not args.no_grover:
        logger.info("Running Grover's algorithm to find optimal thresholds...")
        val_signals = model.vqnn.predict_batch(X_val)
        val_returns = y_val  # use label as proxy for return direction

        optimizer = GroverThresholdOptimizer(
            n_qubits=config.grover.n_qubits_grover,
            threshold_range=config.grover.threshold_range,
        )
        best_threshold = optimizer.fit(val_signals, val_returns)
        buy_threshold = best_threshold
        sell_threshold = -best_threshold  # symmetric
        logger.info(
            "Grover optimal thresholds — BUY > %.4f, SELL < %.4f",
            buy_threshold, sell_threshold
        )

        # Save Grover summary
        grover_summary = optimizer.summary()
        with open(os.path.join(results_dir, "grover_results.json"), "w") as f:
            json.dump(grover_summary, f, indent=2)

    # ── Backtest on test set ──────────────────────────────────────────────
    logger.info("Running backtest on held-out test set...")
    test_signals = model.vqnn.predict_batch(X_test)
    discrete_signals = np.where(
        test_signals > buy_threshold, 1,
        np.where(test_signals < sell_threshold, -1, 0)
    )

    # Use the price slice corresponding to the test period
    test_prices = prices[-len(X_test):]
    backtester = Backtester(config.backtest)
    metrics = backtester.run(
        signals=discrete_signals,
        prices=test_prices,
    )
    print(backtester.report())

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "run_name": run_name,
        "ticker": args.ticker,
        "config": {
            "layers": args.layers,
            "ansatz": args.ansatz,
            "epochs": args.epochs,
            "learning_rate": args.lr,
        },
        "training_history": history,
        "backtest_metrics": metrics,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("All results saved to: %s/", results_dir)
    logger.info("To generate predictions: python predict.py --model-dir %s --ticker %s", results_dir, args.ticker)


if __name__ == "__main__":
    main()
