# data/features.py
# Feature engineering: turns raw OHLCV data into quantum-ready feature vectors.
# Each feature is normalised to a range suitable for quantum encoding.

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual feature calculators
# ---------------------------------------------------------------------------

def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns. Stationary and better than raw prices."""
    return np.log(close / close.shift(1))


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, normalised to [0, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_raw = 100 - (100 / (1 + rs))
    return rsi_raw / 100.0  # → [0, 1]


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD signal line, normalised by rolling std of close."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    # Normalise by price scale so it's comparable across tickers
    roll_std = close.rolling(slow).std() + 1e-10
    return signal_line / roll_std


def bollinger_position(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Position within Bollinger Bands, ∈ [0, 1].
    0 = at lower band, 0.5 = at middle band, 1 = at upper band.
    Values outside [0,1] indicate breakouts.
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band_width = (upper - lower) + 1e-10
    pos = (close - lower) / band_width
    return pos.clip(0, 1)


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Volume z-score vs rolling mean. Clipped to [-3, 3] then normalised to [0, 1].
    High z-score = unusual volume spike (often precedes big moves).
    """
    roll_mean = volume.rolling(period).mean()
    roll_std = volume.rolling(period).std() + 1e-10
    z = (volume - roll_mean) / roll_std
    return (z.clip(-3, 3) + 3) / 6.0  # → [0, 1]


def realized_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Annualised realised volatility (rolling std of log returns × √252).
    Normalised to [0, 1] by clipping at 200% annual vol.
    """
    ret = log_returns(close)
    vol = ret.rolling(period).std() * np.sqrt(252)
    return vol.clip(0, 2.0) / 2.0  # → [0, 1]


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    feature_names: list,
    sequence_length: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, dates) arrays from a raw OHLCV DataFrame.

    X shape: (n_samples, n_features)
    y shape: (n_samples,) — next-day log return, used as label
    dates shape: (n_samples,) — date index of each sample

    The label is sign-encoded: +1 = up, -1 = down, 0 = flat (<0.05% move).
    Each row of X is the feature vector for one trading day.
    """
    close = df["Close"]
    volume = df["Volume"]

    feature_map = {
        "returns": log_returns(close),
        "rsi_14": rsi(close, 14),
        "macd_signal": macd_signal(close),
        "bb_position": bollinger_position(close),
        "volume_zscore": volume_zscore(volume),
        "volatility_20": realized_volatility(close),
    }

    # Build DataFrame with selected features
    feat_df = pd.DataFrame({name: feature_map[name] for name in feature_names})
    feat_df["next_return"] = log_returns(close).shift(-1)  # forward return = label
    feat_df = feat_df.dropna()

    # Convert to arrays
    X = feat_df[feature_names].values.astype(np.float64)
    y_raw = feat_df["next_return"].values

    # Sign-encode labels: +1, 0, -1
    threshold = 0.0005  # 0.05% — below this is "flat"
    y = np.sign(y_raw)
    y[np.abs(y_raw) < threshold] = 0.0

    dates = feat_df.index.values

    logger.info(
        "Features built: %d samples, %d features. Labels: +1=%.1f%%, 0=%.1f%%, -1=%.1f%%",
        len(X), X.shape[1],
        100 * (y == 1).mean(), 100 * (y == 0).mean(), 100 * (y == -1).mean()
    )
    return X, y, dates
