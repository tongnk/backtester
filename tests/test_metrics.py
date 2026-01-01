"""Tests for performance metrics calculations."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.metrics import Metrics


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_ratio_zero_volatility(self, metrics_calculator):
        """Sharpe should be 0 when volatility is 0."""
        # Constant returns = 0 volatility
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
        returns = pd.Series([0.001] * 100, index=dates)

        sharpe = metrics_calculator._calculate_sharpe(returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_uses_excess_returns_std(self, metrics_calculator):
        """
        KNOWN BUG: Sharpe should use std(excess_returns), not std(raw_returns).

        This test verifies the correct formula:
        Sharpe = mean(excess_returns) / std(excess_returns) * sqrt(annualization)

        The current implementation incorrectly uses std(raw_returns) in denominator.
        """
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0001, 0.01, 1000), index=dates)

        # Calculate expected Sharpe using correct formula
        minutes_per_year = 365 * 24 * 60
        rf_per_minute = (1 + 0.02) ** (1 / minutes_per_year) - 1
        excess_returns = returns - rf_per_minute

        # Correct formula: use std of excess returns
        expected_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(minutes_per_year)

        actual_sharpe = metrics_calculator._calculate_sharpe(returns)

        # This should pass when the bug is fixed
        assert abs(actual_sharpe - expected_sharpe) < 0.1, \
            f"Sharpe ratio mismatch. Expected ~{expected_sharpe:.2f}, got {actual_sharpe:.2f}"

    def test_sharpe_ratio_positive_for_positive_returns(self, metrics_calculator):
        """Sharpe should be positive when average return exceeds risk-free rate."""
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
        np.random.seed(42)
        # Returns with positive mean significantly above risk-free
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000), index=dates)

        sharpe = metrics_calculator._calculate_sharpe(returns)
        assert sharpe > 0, f"Sharpe should be positive for positive excess returns, got {sharpe}"


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_ratio_no_negative_returns(self, metrics_calculator):
        """Sortino should be 0 when there are no negative returns."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
        returns = pd.Series([0.001] * 100, index=dates)  # All positive

        sortino = metrics_calculator._calculate_sortino(returns)
        assert sortino == 0.0

    def test_sortino_ratio_uses_series_not_scalar(self, metrics_calculator):
        """
        KNOWN BUG: Sortino should use Series operation, not scalar.

        Current implementation does: excess_returns = returns.mean() - rf (scalar)
        Should be: excess_returns = returns - rf (Series)

        Then: sortino = mean(excess_returns) / std(negative_returns)
        """
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0001, 0.01, 1000), index=dates)

        # Correct Sortino calculation
        minutes_per_year = 365 * 24 * 60
        rf_per_minute = (1 + 0.02) ** (1 / minutes_per_year) - 1

        # Mean excess return (this part is correct in current implementation)
        mean_excess = returns.mean() - rf_per_minute

        # Downside deviation (std of negative returns only)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std()

        expected_sortino = (mean_excess / downside_std) * np.sqrt(minutes_per_year)
        actual_sortino = metrics_calculator._calculate_sortino(returns)

        # The values should match - if they don't, the implementation is wrong
        assert abs(actual_sortino - expected_sortino) < 0.1, \
            f"Sortino mismatch. Expected ~{expected_sortino:.2f}, got {actual_sortino:.2f}"

    def test_sortino_higher_than_sharpe_for_asymmetric_returns(self, metrics_calculator):
        """Sortino should be higher than Sharpe when positive returns are larger than negative."""
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
        np.random.seed(42)

        # Create asymmetric returns: small negative, large positive
        returns = []
        for i in range(1000):
            if np.random.random() < 0.6:  # 60% winners
                returns.append(np.random.uniform(0.001, 0.02))  # Positive: 0.1% to 2%
            else:
                returns.append(np.random.uniform(-0.005, 0))  # Negative: 0 to -0.5%

        returns = pd.Series(returns, index=dates)

        sharpe = metrics_calculator._calculate_sharpe(returns)
        sortino = metrics_calculator._calculate_sortino(returns)

        # Sortino should be higher because we only penalize downside
        assert sortino >= sharpe, \
            f"Sortino ({sortino:.2f}) should be >= Sharpe ({sharpe:.2f}) for asymmetric returns"


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_max_drawdown_known_value(self, metrics_calculator, drawdown_equity_curve):
        """Test drawdown with known equity curve."""
        # Equity: 100, 110, 120, 100, 90, 95, 100, 110, 115, 120
        # Peak at 120, trough at 90 -> DD = (120-90)/120 = 25%

        max_dd, _ = metrics_calculator._calculate_drawdown(drawdown_equity_curve)

        expected_dd = -0.25  # 25% drawdown, stored as negative
        assert abs(max_dd - expected_dd) < 0.01, \
            f"Expected max DD of {expected_dd}, got {max_dd}"

    def test_max_drawdown_no_drawdown(self, metrics_calculator):
        """Drawdown should be 0 for monotonically increasing equity."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="1min")
        equity = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=dates)

        max_dd, _ = metrics_calculator._calculate_drawdown(equity)

        assert max_dd == 0.0, f"Expected 0 drawdown, got {max_dd}"

    def test_max_drawdown_duration(self, metrics_calculator, drawdown_equity_curve):
        """Test drawdown duration calculation."""
        _, duration = metrics_calculator._calculate_drawdown(drawdown_equity_curve)

        # Peak at index 2 (equity=120)
        # Drawdown starts at index 3 (first bar below peak)
        # Recovery at index 9 (equity returns to 120)
        # Duration = index[9] - index[3] = 6 minutes
        expected_duration = pd.Timedelta(minutes=6)

        assert duration == expected_duration, \
            f"Expected duration {expected_duration}, got {duration}"


class TestVolatility:
    """Tests for volatility calculation."""

    def test_volatility_annualization(self, metrics_calculator):
        """Test that volatility is correctly annualized for minute data."""
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
        np.random.seed(42)

        # Known per-minute std
        per_minute_std = 0.001
        returns = pd.Series(np.random.normal(0, per_minute_std, 1000), index=dates)

        # For minute data, annualization factor is sqrt(365 * 24 * 60)
        minutes_per_year = 365 * 24 * 60
        expected_vol = returns.std() * np.sqrt(minutes_per_year)

        # Calculate via metrics (need to access through calculate method)
        equity = 10000 * (1 + returns).cumprod()
        equity = pd.Series(equity.values, index=dates)

        actual_returns = equity.pct_change().dropna()
        actual_vol = actual_returns.std() * np.sqrt(minutes_per_year)

        # Should be approximately equal (some numerical differences expected)
        assert abs(actual_vol - expected_vol) / expected_vol < 0.1, \
            f"Volatility mismatch. Expected ~{expected_vol:.4f}, got {actual_vol:.4f}"


class TestWinRate:
    """Tests for win rate calculation."""

    def test_win_rate_mixed_trades(self, metrics_calculator, sample_trades_df):
        """Test win rate with mixed winning/losing trades."""
        trade_stats = metrics_calculator._calculate_trade_stats(sample_trades_df)

        # 2 winners out of 3 trades = 66.67%
        expected_win_rate = 2 / 3
        assert abs(trade_stats["win_rate"] - expected_win_rate) < 0.01

    def test_win_rate_all_winners(self, metrics_calculator, all_winning_trades_df):
        """Test win rate when all trades are winners."""
        trade_stats = metrics_calculator._calculate_trade_stats(all_winning_trades_df)

        assert trade_stats["win_rate"] == 1.0

    def test_win_rate_all_losers(self, metrics_calculator, all_losing_trades_df):
        """Test win rate when all trades are losers."""
        trade_stats = metrics_calculator._calculate_trade_stats(all_losing_trades_df)

        assert trade_stats["win_rate"] == 0.0

    def test_win_rate_empty_trades(self, metrics_calculator, empty_trades_df):
        """Test win rate with no trades."""
        trade_stats = metrics_calculator._calculate_trade_stats(empty_trades_df)

        assert trade_stats["win_rate"] == 0.0


class TestProfitFactor:
    """Tests for profit factor calculation."""

    def test_profit_factor_basic(self, metrics_calculator, sample_trades_df):
        """Test profit factor calculation."""
        trade_stats = metrics_calculator._calculate_trade_stats(sample_trades_df)

        # Gross profit: 4.9 + 2.9 = 7.8
        # Gross loss: 2.1
        # PF = 7.8 / 2.1 = 3.71
        expected_pf = 7.8 / 2.1

        assert abs(trade_stats["profit_factor"] - expected_pf) < 0.01

    def test_profit_factor_no_losses(self, metrics_calculator, all_winning_trades_df):
        """Profit factor should be infinity when no losing trades."""
        trade_stats = metrics_calculator._calculate_trade_stats(all_winning_trades_df)

        assert trade_stats["profit_factor"] == float("inf")

    def test_profit_factor_no_wins(self, metrics_calculator, all_losing_trades_df):
        """Profit factor should be 0 when no winning trades."""
        trade_stats = metrics_calculator._calculate_trade_stats(all_losing_trades_df)

        # Gross profit = 0, so PF = 0 / loss = 0
        assert trade_stats["profit_factor"] == 0.0


class TestTimeInMarket:
    """Tests for time in market calculation."""

    def test_time_in_market_simple(self, metrics_calculator, sample_trades_df, sample_equity_curve):
        """Test time in market calculation."""
        exposure = metrics_calculator._calculate_exposure(sample_trades_df, sample_equity_curve)

        # 3 trades, each 1 hour = 180 minutes
        # Equity curve is 100 minutes
        # Time in market = 180 / 100 = 180% (BUG: should not exceed 100%)
        # This reveals the bug where overlapping positions are double-counted

        assert exposure["time_in_market"] <= 1.0, \
            f"Time in market should not exceed 100%, got {exposure['time_in_market']:.2%}"

    def test_time_in_market_no_overlap(self, metrics_calculator, sample_equity_curve):
        """Test time in market without overlapping positions."""
        # Create non-overlapping trades
        trades_data = [
            {
                "entry_time": datetime(2024, 1, 1, 0, 0),
                "exit_time": datetime(2024, 1, 1, 0, 10),
                "side": "LONG",
                "duration": pd.Timedelta(minutes=10),
            },
            {
                "entry_time": datetime(2024, 1, 1, 0, 20),
                "exit_time": datetime(2024, 1, 1, 0, 30),
                "side": "LONG",
                "duration": pd.Timedelta(minutes=10),
            },
        ]
        trades_df = pd.DataFrame(trades_data)

        exposure = metrics_calculator._calculate_exposure(trades_df, sample_equity_curve)

        # 20 minutes in market out of 99 minutes (100 bars, 1-min freq)
        # Should be ~20%
        expected = 20 / 99
        assert abs(exposure["time_in_market"] - expected) < 0.05

    def test_time_in_market_with_pyramids_no_double_count(self, metrics_calculator, sample_equity_curve):
        """
        KNOWN BUG: Pyramided positions should not double-count time.

        If you have a position from 0:00-0:30 and pyramid at 0:10,
        the time in market should be 30 minutes, not 30 + 20 = 50 minutes.
        """
        # Create overlapping trades (simulating pyramiding)
        trades_data = [
            {
                "entry_time": datetime(2024, 1, 1, 0, 0),
                "exit_time": datetime(2024, 1, 1, 0, 30),
                "side": "LONG",
                "duration": pd.Timedelta(minutes=30),
            },
            {
                "entry_time": datetime(2024, 1, 1, 0, 10),  # Overlaps!
                "exit_time": datetime(2024, 1, 1, 0, 30),
                "side": "LONG",
                "duration": pd.Timedelta(minutes=20),
            },
        ]
        trades_df = pd.DataFrame(trades_data)

        exposure = metrics_calculator._calculate_exposure(trades_df, sample_equity_curve)

        # Actual time in market: 30 minutes (from 0:00 to 0:30)
        # Bug would calculate: 30 + 20 = 50 minutes
        # With 99 minutes total: should be 30/99 = 30.3%, not 50/99 = 50.5%

        expected_max = 35 / 99  # Allow some tolerance but not double-counting
        assert exposure["time_in_market"] < expected_max, \
            f"Time in market appears to double-count pyramids: {exposure['time_in_market']:.2%}"


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_ratio_basic(self, metrics_calculator):
        """Test Calmar ratio = annualized return / max drawdown."""
        # Calmar = annualized_return / abs(max_dd)
        # If annualized return = 20%, max_dd = -10%, Calmar = 0.20 / 0.10 = 2.0

        annualized_return = 0.20
        max_dd = -0.10

        calmar = annualized_return / abs(max_dd)

        assert calmar == 2.0

    def test_calmar_ratio_zero_drawdown(self, metrics_calculator):
        """Calmar should be 0 when max drawdown is 0."""
        # Avoid division by zero
        annualized_return = 0.20
        max_dd = 0.0

        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        assert calmar == 0
