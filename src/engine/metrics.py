"""Performance metrics calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float
    benchmark_return: float  # Buy and hold

    # Risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: pd.Timedelta
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: pd.Timedelta

    # Exposure
    time_in_market: float  # Percentage
    long_exposure: float
    short_exposure: float

    # Additional
    expectancy: float
    kelly_fraction: float

    # Funding metrics (perps mode)
    total_funding_paid: float = 0.0     # Total funding paid
    total_funding_received: float = 0.0 # Total funding received
    net_funding: float = 0.0            # Net funding (paid - received)
    avg_funding_per_trade: float = 0.0  # Average funding cost per trade
    funding_as_pct_of_pnl: float = 0.0  # Funding cost as % of gross PnL


class Metrics:
    """Calculate comprehensive performance metrics."""

    TRADING_DAYS_PER_YEAR = 365
    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate(
        self,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        initial_capital: float,
        price_series: pd.Series,
        funding_df: pd.DataFrame = None,
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            equity_curve: Time series of portfolio equity
            trades_df: DataFrame of completed trades
            initial_capital: Starting capital
            price_series: Asset price series for benchmark
            funding_df: DataFrame of funding payments (perps mode)

        Returns:
            PerformanceMetrics object with all metrics
        """
        # Returns calculation
        total_return = equity_curve.iloc[-1] - initial_capital
        total_return_pct = total_return / initial_capital

        # Calculate annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365
        annualized_return = (1 + total_return_pct) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0

        # Benchmark (buy and hold)
        benchmark_return = (price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0]

        # Daily/period returns for risk metrics
        returns = equity_curve.pct_change().dropna()

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR * 24 * 60)  # For minute data

        # Drawdown analysis
        max_dd, max_dd_duration = self._calculate_drawdown(equity_curve)

        # Sharpe ratio
        sharpe = self._calculate_sharpe(returns)

        # Sortino ratio
        sortino = self._calculate_sortino(returns)

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Trade statistics
        trade_stats = self._calculate_trade_stats(trades_df)

        # Exposure analysis
        exposure = self._calculate_exposure(trades_df, equity_curve)

        # Expectancy and Kelly
        expectancy = self._calculate_expectancy(trades_df)
        kelly = self._calculate_kelly(trade_stats)

        # Funding stats (perps mode)
        funding_stats = self._calculate_funding_stats(trades_df, funding_df)

        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            volatility=volatility,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            avg_win_pct=trade_stats["avg_win_pct"],
            avg_loss_pct=trade_stats["avg_loss_pct"],
            largest_win=trade_stats["largest_win"],
            largest_loss=trade_stats["largest_loss"],
            avg_trade_duration=trade_stats["avg_trade_duration"],
            time_in_market=exposure["time_in_market"],
            long_exposure=exposure["long_exposure"],
            short_exposure=exposure["short_exposure"],
            expectancy=expectancy,
            kelly_fraction=kelly,
            # Funding metrics
            total_funding_paid=funding_stats["total_funding_paid"],
            total_funding_received=funding_stats["total_funding_received"],
            net_funding=funding_stats["net_funding"],
            avg_funding_per_trade=funding_stats["avg_funding_per_trade"],
            funding_as_pct_of_pnl=funding_stats["funding_as_pct_of_pnl"],
        )

    def _calculate_drawdown(
        self, equity_curve: pd.Series
    ) -> tuple[float, pd.Timedelta]:
        """Calculate maximum drawdown and duration."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max

        max_dd = drawdown.min()

        # Find drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        start_idx = None

        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                drawdown_periods.append((start_idx, i))
                start_idx = None

        if start_idx is not None:
            drawdown_periods.append((start_idx, len(in_drawdown) - 1))

        max_duration = pd.Timedelta(0)
        for start, end in drawdown_periods:
            duration = equity_curve.index[end] - equity_curve.index[start]
            if duration > max_duration:
                max_duration = duration

        return max_dd, max_duration

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        std = returns.std()
        if std < 1e-10:  # Use threshold instead of exact zero check
            return 0.0

        # Convert risk-free rate to per-minute for minute data
        minutes_per_year = 365 * 24 * 60
        rf_per_minute = (1 + self.risk_free_rate) ** (1 / minutes_per_year) - 1

        excess_returns = returns - rf_per_minute
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(minutes_per_year)

        return sharpe

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0.0

        minutes_per_year = 365 * 24 * 60
        rf_per_minute = (1 + self.risk_free_rate) ** (1 / minutes_per_year) - 1

        excess_returns = returns.mean() - rf_per_minute
        downside_std = negative_returns.std()

        sortino = (excess_returns / downside_std) * np.sqrt(minutes_per_year)

        return sortino

    def _calculate_trade_stats(self, trades_df: pd.DataFrame) -> dict:
        """Calculate trade-level statistics."""
        if trades_df.empty:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_trade_duration": pd.Timedelta(0),
            }

        total_trades = len(trades_df)
        winners = trades_df[trades_df["pnl"] > 0]
        losers = trades_df[trades_df["pnl"] <= 0]

        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        gross_profit = winners["pnl"].sum() if not winners.empty else 0
        gross_loss = abs(losers["pnl"].sum()) if not losers.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = winners["pnl"].mean() if not winners.empty else 0
        avg_loss = losers["pnl"].mean() if not losers.empty else 0
        avg_win_pct = winners["pnl_pct"].mean() if not winners.empty else 0
        avg_loss_pct = losers["pnl_pct"].mean() if not losers.empty else 0

        largest_win = trades_df["pnl"].max()
        largest_loss = trades_df["pnl"].min()

        avg_duration = trades_df["duration"].mean()

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "avg_trade_duration": avg_duration,
        }

    def _calculate_exposure(
        self, trades_df: pd.DataFrame, equity_curve: pd.Series
    ) -> dict:
        """Calculate market exposure statistics using interval merging."""
        if trades_df.empty:
            return {
                "time_in_market": 0.0,
                "long_exposure": 0.0,
                "short_exposure": 0.0,
            }

        total_duration = equity_curve.index[-1] - equity_curve.index[0]
        total_minutes = total_duration.total_seconds() / 60

        # Collect intervals by side
        long_intervals = []
        short_intervals = []

        for _, trade in trades_df.iterrows():
            entry = trade["entry_time"]
            exit = trade["exit_time"]
            if trade["side"] == "LONG":
                long_intervals.append((entry, exit))
            else:
                short_intervals.append((entry, exit))

        def merge_intervals(intervals):
            """Merge overlapping intervals and return total duration."""
            if not intervals:
                return 0.0
            # Sort by start time
            sorted_intervals = sorted(intervals, key=lambda x: x[0])
            merged = [sorted_intervals[0]]

            for current in sorted_intervals[1:]:
                prev_start, prev_end = merged[-1]
                curr_start, curr_end = current

                if curr_start <= prev_end:
                    # Overlapping - merge
                    merged[-1] = (prev_start, max(prev_end, curr_end))
                else:
                    # Non-overlapping
                    merged.append(current)

            # Calculate total duration
            total = 0.0
            for start, end in merged:
                total += (end - start).total_seconds() / 60
            return total

        long_minutes = merge_intervals(long_intervals)
        short_minutes = merge_intervals(short_intervals)

        # For total time in market, merge all intervals together
        all_intervals = long_intervals + short_intervals
        total_in_market = merge_intervals(all_intervals)

        return {
            "time_in_market": min(total_in_market / total_minutes, 1.0) if total_minutes > 0 else 0,
            "long_exposure": min(long_minutes / total_minutes, 1.0) if total_minutes > 0 else 0,
            "short_exposure": min(short_minutes / total_minutes, 1.0) if total_minutes > 0 else 0,
        }

    def _calculate_expectancy(self, trades_df: pd.DataFrame) -> float:
        """Calculate trade expectancy (average profit per trade)."""
        if trades_df.empty:
            return 0.0
        return trades_df["pnl"].mean()

    def _calculate_kelly(self, trade_stats: dict) -> float:
        """
        Calculate Kelly fraction for optimal position sizing.

        Kelly = W - (1-W)/R
        Where W = win rate, R = win/loss ratio
        """
        win_rate = trade_stats["win_rate"]
        avg_win = abs(trade_stats["avg_win"])
        avg_loss = abs(trade_stats["avg_loss"])

        if avg_loss == 0 or win_rate == 0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Cap Kelly at reasonable levels
        return max(0, min(kelly, 0.5))

    def _calculate_funding_stats(
        self,
        trades_df: pd.DataFrame,
        funding_df: pd.DataFrame,
    ) -> dict:
        """Calculate funding-related statistics."""
        default_stats = {
            "total_funding_paid": 0.0,
            "total_funding_received": 0.0,
            "net_funding": 0.0,
            "avg_funding_per_trade": 0.0,
            "funding_as_pct_of_pnl": 0.0,
        }

        if funding_df is None or funding_df.empty:
            return default_stats

        # Aggregate from funding log
        if "amount" in funding_df.columns:
            funding_amounts = funding_df["amount"]
            total_paid = funding_amounts[funding_amounts > 0].sum()
            total_received = abs(funding_amounts[funding_amounts < 0].sum())
        else:
            total_paid = 0.0
            total_received = 0.0

        net_funding = total_paid - total_received

        # Per-trade average (from trades_df if available)
        if not trades_df.empty and "funding_paid" in trades_df.columns:
            avg_per_trade = trades_df["funding_paid"].mean()
        else:
            avg_per_trade = net_funding / len(trades_df) if len(trades_df) > 0 else 0.0

        # Funding as percentage of gross PnL
        if not trades_df.empty and "gross_pnl" in trades_df.columns:
            gross_pnl = trades_df["gross_pnl"].sum()
        elif not trades_df.empty:
            # Fallback: gross = net + fees + funding
            gross_pnl = trades_df["pnl"].sum() + trades_df["fees"].sum() + net_funding
        else:
            gross_pnl = 0.0

        funding_pct = (net_funding / abs(gross_pnl)) * 100 if gross_pnl != 0 else 0.0

        return {
            "total_funding_paid": total_paid,
            "total_funding_received": total_received,
            "net_funding": net_funding,
            "avg_funding_per_trade": avg_per_trade,
            "funding_as_pct_of_pnl": funding_pct,
        }

    def summary_string(self, metrics: PerformanceMetrics) -> str:
        """Generate a formatted summary string."""
        lines = [
            "=" * 50,
            "PERFORMANCE SUMMARY",
            "=" * 50,
            "",
            "RETURNS",
            f"  Total Return:      ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2%})",
            f"  Annualized Return: {metrics.annualized_return:.2%}",
            f"  Benchmark (B&H):   {metrics.benchmark_return:.2%}",
            "",
            "RISK METRICS",
            f"  Volatility (Ann.): {metrics.volatility:.2%}",
            f"  Max Drawdown:      {metrics.max_drawdown:.2%}",
            f"  Sharpe Ratio:      {metrics.sharpe_ratio:.2f}",
            f"  Sortino Ratio:     {metrics.sortino_ratio:.2f}",
            f"  Calmar Ratio:      {metrics.calmar_ratio:.2f}",
            "",
            "TRADE STATISTICS",
            f"  Total Trades:      {metrics.total_trades}",
            f"  Win Rate:          {metrics.win_rate:.2%}",
            f"  Profit Factor:     {metrics.profit_factor:.2f}",
            f"  Avg Win:           ${metrics.avg_win:,.2f} ({metrics.avg_win_pct:.2%})",
            f"  Avg Loss:          ${metrics.avg_loss:,.2f} ({metrics.avg_loss_pct:.2%})",
            f"  Largest Win:       ${metrics.largest_win:,.2f}",
            f"  Largest Loss:      ${metrics.largest_loss:,.2f}",
            f"  Avg Trade Duration: {metrics.avg_trade_duration}",
            "",
            "EXPOSURE",
            f"  Time in Market:    {metrics.time_in_market:.2%}",
            f"  Long Exposure:     {metrics.long_exposure:.2%}",
            f"  Short Exposure:    {metrics.short_exposure:.2%}",
            "",
            "POSITION SIZING",
            f"  Expectancy:        ${metrics.expectancy:,.2f}",
            f"  Kelly Fraction:    {metrics.kelly_fraction:.2%}",
        ]

        # Add funding section if applicable
        if metrics.net_funding != 0 or metrics.total_funding_paid != 0:
            lines.extend([
                "",
                "FUNDING COSTS (Perps)",
                f"  Total Paid:        ${metrics.total_funding_paid:,.2f}",
                f"  Total Received:    ${metrics.total_funding_received:,.2f}",
                f"  Net Funding:       ${metrics.net_funding:,.2f}",
                f"  Avg Per Trade:     ${metrics.avg_funding_per_trade:,.2f}",
                f"  % of Gross PnL:    {metrics.funding_as_pct_of_pnl:.1f}%",
            ])

        lines.append("=" * 50)

        return "\n".join(lines)
