# data/fetcher.py
# Pulls OHLCV data from yfinance and runs the feature pipeline.

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

from data.features import build_feature_matrix
from config import DataConfig

logger = logging.getLogger(__name__)


class StockDataPipeline:
    """
    End-to-end data pipeline: fetch → clean → feature engineer → split.

    Usage:
        pipeline = StockDataPipeline(config.data)
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run("AAPL")
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.raw_df_: Optional[pd.DataFrame] = None
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.dates_: Optional[np.ndarray] = None

    def fetch(self, ticker: str) -> pd.DataFrame:
        """Download and clean OHLCV data for a single ticker."""
        logger.info("Downloading %s (%s, %s)...", ticker, self.config.period, self.config.interval)
        df = yf.download(
            ticker,
            period=self.config.period,
            interval=self.config.interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

        # Flatten MultiIndex columns if present (multi-ticker downloads)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Drop rows with any NaN in OHLCV
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        logger.info("Fetched %d rows for %s", len(df), ticker)
        self.raw_df_ = df
        return df

    def build_features(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run feature engineering on a DataFrame."""
        if df is None:
            df = self.raw_df_
        if df is None:
            raise RuntimeError("No data available. Call fetch() first.")

        X, y, dates = build_feature_matrix(df, self.config.features, self.config.sequence_length)
        self.X_ = X
        self.y_ = y
        self.dates_ = dates
        return X, y, dates

    def split(
        self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Chronological train/val/test split (no random shuffling — preserves time order).
        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        if X is None:
            X = self.X_
        if y is None:
            y = self.y_

        n = len(X)
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(
            "Split: train=%d, val=%d, test=%d samples",
            len(X_train), len(X_val), len(X_test)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def run(self, ticker: str) -> Tuple[np.ndarray, ...]:
        """Full pipeline: fetch → features → split. Returns all 6 arrays."""
        df = self.fetch(ticker)
        X, y, _ = self.build_features(df)
        return self.split(X, y)
