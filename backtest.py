# utils/backtest.py
# Full backtesting engine with realistic transaction costs and comprehensive metrics.
# This is what turns the model from "interesting demo" to "measurable experiment".

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from config import BacktestConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: str
    exit_date: str
    direction: str       # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    return_pct: float
    pnl: float


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

class Backtester:
    """
    Simulates trading a strategy defined by a series of signals.

    Signals ∈ {-1 (SELL/SHORT), 0 (HOLD), +1 (BUY/LONG)}.
    Handles:
      - Transaction costs (flat % per trade)
      - Slippage (applied at trade entry/exit)
      - Position sizing (fixed or confidence-weighted)
      - Performance metrics vs. buy-and-hold benchmark
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades_: List[Trade] = []
        self.equity_curve_: Optional[np.ndarray] = None
        self.metrics_: Optional[dict] = None

    def run(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run backtest.

        Args:
            signals: Array of {-1, 0, +1} signals, one per bar.
            prices: Close price series (same length as signals).
            dates: Optional date labels for the equity curve.
            confidences: Optional confidence weights ∈ [0, 1] per signal.

        Returns:
            metrics dict with Sharpe, drawdown, CAGR, accuracy, etc.
        """
        cfg = self.config
        capital = cfg.initial_capital
        equity = [capital]
        position = 0          # +1, -1, or 0
        entry_price = None
        entry_date = None
        self.trades_ = []

        cost = cfg.transaction_cost + cfg.slippage  # total round-trip cost per side

        for i in range(1, len(signals)):
            price = float(prices[i])
            signal = int(signals[i])
            conf = float(confidences[i]) if confidences is not None else 1.0

            # Position sizing
            size = min(conf, cfg.max_position_pct)

            # --- Exit existing position if signal flips or goes to HOLD ---
            if position != 0 and signal != position:
                exit_price = price * (1 - np.sign(position) * cost)
                ret = (exit_price - entry_price) / entry_price * position
                pnl = capital * size * ret
                capital += pnl
                self.trades_.append(Trade(
                    entry_date=str(dates[i-1]) if dates is not None else str(i-1),
                    exit_date=str(dates[i]) if dates is not None else str(i),
                    direction="LONG" if position == 1 else "SHORT",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=ret * 100,
                    pnl=pnl,
                ))
                position = 0

            # --- Enter new position ---
            if position == 0 and signal != 0:
                entry_price = price * (1 + np.sign(signal) * cost)
                entry_date = str(dates[i]) if dates is not None else str(i)
                position = signal

            equity.append(capital)

        # Close any open position at the end
        if position != 0 and len(prices) > 0:
            exit_price = float(prices[-1]) * (1 - np.sign(position) * cost)
            ret = (exit_price - entry_price) / entry_price * position
            capital += capital * ret
            equity.append(capital)

        self.equity_curve_ = np.array(equity)

        # Compute benchmark (buy-and-hold)
        bah_return = (float(prices[-1]) - float(prices[0])) / float(prices[0])

        self.metrics_ = self._compute_metrics(
            equity=self.equity_curve_,
            initial_capital=cfg.initial_capital,
            bah_return=bah_return,
        )
        return self.metrics_

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        equity: np.ndarray,
        initial_capital: float,
        bah_return: float,
    ) -> dict:
        """Compute comprehensive performance metrics."""
        returns = np.diff(equity) / equity[:-1]
        n = len(returns)
        n_years = n / 252.0

        # Return metrics
        total_return = (equity[-1] - equity[0]) / equity[0]
        cagr = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1 if n_years > 0 else 0.0

        # Risk metrics
        daily_vol = returns.std() * np.sqrt(252) if n > 1 else 0.0
        sharpe = (returns.mean() * 252) / (daily_vol + 1e-10)

        # Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_drawdown = float(drawdowns.min())
        calmar = cagr / (abs(max_drawdown) + 1e-10)

        # Trade stats
        n_trades = len(self.trades_)
        if n_trades > 0:
            win_trades = [t for t in self.trades_ if t.pnl > 0]
            loss_trades = [t for t in self.trades_ if t.pnl <= 0]
            win_rate = len(win_trades) / n_trades
            avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0.0
            avg_loss = np.mean([t.pnl for t in loss_trades]) if loss_trades else 0.0
            profit_factor = (
                sum(t.pnl for t in win_trades) / (abs(sum(t.pnl for t in loss_trades)) + 1e-10)
            )
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0

        metrics = {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "calmar_ratio": round(float(calmar), 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "daily_volatility_pct": round(daily_vol * 100, 2),
            "n_trades": n_trades,
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(float(profit_factor), 3),
            "avg_win_usd": round(float(avg_win), 2),
            "avg_loss_usd": round(float(avg_loss), 2),
            "final_equity_usd": round(float(equity[-1]), 2),
            "benchmark_return_pct": round(bah_return * 100, 2),
            "alpha_pct": round((total_return - bah_return) * 100, 2),
        }

        logger.info(
            "Backtest complete — Return: %.1f%% | Sharpe: %.2f | MaxDD: %.1f%% | Trades: %d",
            metrics["total_return_pct"], metrics["sharpe_ratio"],
            metrics["max_drawdown_pct"], n_trades,
        )
        return metrics

    def report(self) -> str:
        """Pretty-print the backtest metrics."""
        if self.metrics_ is None:
            return "No backtest results yet. Call run() first."
        m = self.metrics_
        return (
            f"\n{'='*45}\n"
            f"  BACKTEST RESULTS\n"
            f"{'='*45}\n"
            f"  Total Return    : {m['total_return_pct']:>8.2f}%\n"
            f"  CAGR            : {m['cagr_pct']:>8.2f}%\n"
            f"  Benchmark (B&H) : {m['benchmark_return_pct']:>8.2f}%\n"
            f"  Alpha           : {m['alpha_pct']:>8.2f}%\n"
            f"{'─'*45}\n"
            f"  Sharpe Ratio    : {m['sharpe_ratio']:>8.3f}\n"
            f"  Calmar Ratio    : {m['calmar_ratio']:>8.3f}\n"
            f"  Max Drawdown    : {m['max_drawdown_pct']:>8.2f}%\n"
            f"  Daily Vol       : {m['daily_volatility_pct']:>8.2f}%\n"
            f"{'─'*45}\n"
            f"  Total Trades    : {m['n_trades']:>8d}\n"
            f"  Win Rate        : {m['win_rate_pct']:>8.2f}%\n"
            f"  Profit Factor   : {m['profit_factor']:>8.3f}\n"
            f"  Avg Win         : ${m['avg_win_usd']:>7.2f}\n"
            f"  Avg Loss        : ${m['avg_loss_usd']:>7.2f}\n"
            f"{'─'*45}\n"
            f"  Final Equity    : ${m['final_equity_usd']:>,.2f}\n"
            f"{'='*45}\n"
        )
