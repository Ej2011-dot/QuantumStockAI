#!/usr/bin/env python3
# predict.py — Generate live signals or run a quick backtest with a saved model.
#
# Usage:
#   python predict.py --ticker AAPL --model-dir results/AAPL_20240101_120000/model
#   python predict.py --ticker TSLA --model-dir results/... --mode backtest

import argparse
import sys
import os
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import Config, DataConfig
from data.fetcher import StockDataPipeline
from models.hybrid_model import HybridQuantumModel
from utils.backtest import Backtester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict")

SIGNAL_EMOJIS = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}


def parse_args():
    p = argparse.ArgumentParser(description="QuantumTrader — live signal or backtest")
    p.add_argument("--ticker", required=True)
    p.add_argument("--model-dir", required=True, help="Path to saved model directory")
    p.add_argument("--mode", default="live", choices=["live", "backtest"])
    p.add_argument("--period", default="6mo")
    return p.parse_args()


def main():
    args = parse_args()

    config = Config(data=DataConfig(ticker=args.ticker, period=args.period))

    # Load model
    model = HybridQuantumModel(config)
    model.load(args.model_dir)
    logger.info("Model loaded from %s", args.model_dir)

    # Fetch latest data
    pipeline = StockDataPipeline(config.data)
    X_all, y_all, dates = pipeline.build_features(pipeline.fetch(args.ticker))
    prices = pipeline.raw_df_["Close"].values

    if args.mode == "live":
        # Use the most recent data point
        latest_features = X_all[-1]
        signal, action = model.forward(latest_features)
        conf = model.confidence(signal)

        print("\n" + "=" * 45)
        print(f"  QuantumTrader — {args.ticker}")
        print("=" * 45)
        print(f"  Signal value  : {signal:+.4f}")
        print(f"  Action        : {SIGNAL_EMOJIS.get(action, action)}")
        print(f"  Confidence    : {conf:.1%}")
        print(f"  Based on      : {len(X_all)} days of data")
        print("=" * 45)
        print("\n  ⚠  This is an experimental quantum model.")
        print("     Do NOT make real trading decisions based on this output.\n")

    elif args.mode == "backtest":
        signals_raw = model.vqnn.predict_batch(X_all)
        threshold = 0.2
        discrete = np.where(signals_raw > threshold, 1, np.where(signals_raw < -threshold, -1, 0))

        backtester = Backtester(config.backtest)
        backtester.run(signals=discrete, prices=prices[-len(X_all):], dates=dates)
        print(backtester.report())


if __name__ == "__main__":
    main()
