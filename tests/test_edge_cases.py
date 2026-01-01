"""Tests for edge cases and boundary conditions."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.metrics import Metrics, PerformanceMetrics
from src.engine.position import Position, PositionSide, Fill, PositionManager


class TestZeroTrades:
    """Tests for handling zero trades."""

    def test_metrics_with_no_trades(self, metrics_calculator, sample_equity_curve, sample_price_series, empty_trades_df):
        """All trade-related metrics should handle zero trades gracefully."""
        metrics = metrics_calculator.calculate(
            equity_curve=sample_equity_curve,
            trades_df=empty_trades_df,
            initial_capital=10000,
            price_series=sample_price_series,
        )

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == 0.0
        assert metrics.time_in_market == 0.0

    def test_expectancy_with_no_trades(self, metrics_calculator, empty_trades_df):
        """Expectancy should be 0 with no trades."""
        expectancy = metrics_calculator._calculate_expectancy(empty_trades_df)
        assert expectancy == 0.0


class TestSingleTrade:
    """Tests for handling single trade scenarios."""

    def test_metrics_with_single_winning_trade(self, metrics_calculator, sample_equity_curve, sample_price_series):
        """Metrics should work correctly with just one winning trade."""
        single_trade = pd.DataFrame([{
            "entry_time": datetime(2024, 1, 1, 10, 0),
            "exit_time": datetime(2024, 1, 1, 11, 0),
            "side": "LONG",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "quantity": 1.0,
            "pnl": 10.0,
            "pnl_pct": 0.10,
            "fees": 0.0,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": True,
        }])

        trade_stats = metrics_calculator._calculate_trade_stats(single_trade)

        assert trade_stats["total_trades"] == 1
        assert trade_stats["win_rate"] == 1.0
        assert trade_stats["winning_trades"] == 1
        assert trade_stats["losing_trades"] == 0
        assert trade_stats["avg_win"] == 10.0

    def test_metrics_with_single_losing_trade(self, metrics_calculator):
        """Metrics should work correctly with just one losing trade."""
        single_trade = pd.DataFrame([{
            "entry_time": datetime(2024, 1, 1, 10, 0),
            "exit_time": datetime(2024, 1, 1, 11, 0),
            "side": "LONG",
            "entry_price": 100.0,
            "exit_price": 90.0,
            "quantity": 1.0,
            "pnl": -10.0,
            "pnl_pct": -0.10,
            "fees": 0.0,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": False,
        }])

        trade_stats = metrics_calculator._calculate_trade_stats(single_trade)

        assert trade_stats["total_trades"] == 1
        assert trade_stats["win_rate"] == 0.0
        assert trade_stats["winning_trades"] == 0
        assert trade_stats["losing_trades"] == 1
        assert trade_stats["avg_loss"] == -10.0


class TestAllWinningTrades:
    """Tests for 100% win rate scenarios."""

    def test_profit_factor_infinity(self, metrics_calculator, all_winning_trades_df):
        """Profit factor should be infinity when no losses."""
        trade_stats = metrics_calculator._calculate_trade_stats(all_winning_trades_df)

        assert trade_stats["profit_factor"] == float("inf")
        assert trade_stats["win_rate"] == 1.0

    def test_avg_loss_zero(self, metrics_calculator, all_winning_trades_df):
        """Average loss should be 0 when no losing trades."""
        trade_stats = metrics_calculator._calculate_trade_stats(all_winning_trades_df)

        assert trade_stats["avg_loss"] == 0.0


class TestAllLosingTrades:
    """Tests for 0% win rate scenarios."""

    def test_profit_factor_zero(self, metrics_calculator, all_losing_trades_df):
        """Profit factor should be 0 when no wins."""
        trade_stats = metrics_calculator._calculate_trade_stats(all_losing_trades_df)

        assert trade_stats["profit_factor"] == 0.0
        assert trade_stats["win_rate"] == 0.0

    def test_avg_win_zero(self, metrics_calculator, all_losing_trades_df):
        """Average win should be 0 when no winning trades."""
        trade_stats = metrics_calculator._calculate_trade_stats(all_losing_trades_df)

        assert trade_stats["avg_win"] == 0.0


class TestBreakEvenTrades:
    """Tests for break-even trade handling."""

    def test_break_even_classified_as_loss(self, metrics_calculator):
        """Break-even trades (pnl=0) should be classified as losses."""
        break_even_trade = pd.DataFrame([{
            "entry_time": datetime(2024, 1, 1, 10, 0),
            "exit_time": datetime(2024, 1, 1, 11, 0),
            "side": "LONG",
            "entry_price": 100.0,
            "exit_price": 100.0,
            "quantity": 1.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "fees": 0.0,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": False,
        }])

        trade_stats = metrics_calculator._calculate_trade_stats(break_even_trade)

        # Current implementation: pnl <= 0 is a loser
        assert trade_stats["winning_trades"] == 0
        assert trade_stats["losing_trades"] == 1
        assert trade_stats["win_rate"] == 0.0


class TestVeryShortBacktest:
    """Tests for very short backtest periods."""

    def test_metrics_with_few_bars(self, metrics_calculator):
        """Metrics should handle very short equity curves."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1min")
        equity = pd.Series([10000, 10010, 10020, 10015, 10025], index=dates)
        prices = pd.Series([100, 101, 102, 101.5, 102.5], index=dates)
        empty_trades = pd.DataFrame(columns=[
            "entry_time", "exit_time", "side", "entry_price", "exit_price",
            "quantity", "pnl", "pnl_pct", "fees", "pyramid_levels", "duration", "is_winner"
        ])

        # Should not raise any errors
        metrics = metrics_calculator.calculate(
            equity_curve=equity,
            trades_df=empty_trades,
            initial_capital=10000,
            price_series=prices,
        )

        assert metrics.total_return == 25.0
        assert abs(metrics.total_return_pct - 0.0025) < 0.0001

    def test_drawdown_with_two_bars(self, metrics_calculator):
        """Drawdown calculation should work with minimal data."""
        dates = pd.date_range(start="2024-01-01", periods=2, freq="1min")
        equity = pd.Series([100, 90], index=dates)  # 10% drawdown

        max_dd, _ = metrics_calculator._calculate_drawdown(equity)

        assert abs(max_dd - (-0.10)) < 0.01


class TestLargeNumbers:
    """Tests for handling large numbers."""

    def test_large_equity_values(self, metrics_calculator):
        """Metrics should handle large equity values correctly."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
        equity = pd.Series(np.linspace(1e9, 1.1e9, 100), index=dates)  # 1 billion to 1.1 billion
        prices = pd.Series(np.linspace(50000, 55000, 100), index=dates)
        empty_trades = pd.DataFrame()

        # Should not overflow
        returns = equity.pct_change().dropna()
        sharpe = metrics_calculator._calculate_sharpe(returns)

        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)


class TestNegativeEquity:
    """Tests for handling negative equity (margin call scenarios)."""

    def test_negative_equity_handling(self, metrics_calculator):
        """Metrics should handle equity going negative."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1min")
        equity = pd.Series([10000, 5000, 0, -1000, -2000], index=dates)

        # Drawdown calculation with negative equity
        max_dd, _ = metrics_calculator._calculate_drawdown(equity)

        # Max drawdown should be 100%+ (went from 10000 to -2000)
        # This is an edge case that may not be handled correctly
        assert max_dd <= -1.0, f"Expected drawdown >= 100%, got {max_dd}"


class TestPositionEdgeCases:
    """Tests for position management edge cases."""

    def test_zero_quantity_position(self):
        """Position with zero quantity after partial close."""
        pos = Position(side=PositionSide.LONG)

        fill = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_fill(fill)

        exit_fill = Fill(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=110.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_exit(exit_fill)

        assert pos.quantity == 0.0
        assert not pos.is_open

    def test_partial_close(self):
        """Partial close should leave remaining position."""
        pos = Position(side=PositionSide.LONG)

        fill = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=2.0,
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_fill(fill)

        exit_fill = Fill(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=110.0,
            quantity=1.0,  # Only close half
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_exit(exit_fill)

        assert pos.quantity == 1.0
        assert pos.is_open

    def test_position_manager_max_pyramids(self):
        """Cannot pyramid beyond max_positions limit."""
        pm = PositionManager(max_positions=2, fee_rate=0.0, slippage=0.0)

        pm.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        # First pyramid (2nd position) - should succeed
        fill1 = pm.add_to_position(
            timestamp=datetime(2024, 1, 1, 10, 10),
            price=105.0,
            quantity=1.0,
        )
        assert fill1 is not None

        # Second pyramid (3rd position) - should fail
        fill2 = pm.add_to_position(
            timestamp=datetime(2024, 1, 1, 10, 20),
            price=110.0,
            quantity=1.0,
        )
        assert fill2 is None
        assert pm.current_position.pyramid_count == 2


class TestKellyFraction:
    """Tests for Kelly fraction calculation."""

    def test_kelly_capped_at_50_percent(self, metrics_calculator):
        """Kelly fraction should be capped at 50%."""
        # Create stats that would give Kelly > 0.5
        trade_stats = {
            "win_rate": 0.9,  # 90% win rate
            "avg_win": 100.0,
            "avg_loss": 10.0,  # 10:1 win/loss ratio
        }

        kelly = metrics_calculator._calculate_kelly(trade_stats)

        assert kelly <= 0.5

    def test_kelly_non_negative(self, metrics_calculator):
        """Kelly fraction should not be negative."""
        trade_stats = {
            "win_rate": 0.3,  # 30% win rate
            "avg_win": 10.0,
            "avg_loss": 50.0,  # 1:5 win/loss ratio - bad strategy
        }

        kelly = metrics_calculator._calculate_kelly(trade_stats)

        assert kelly >= 0.0

    def test_kelly_zero_when_no_losses(self, metrics_calculator):
        """Kelly should be 0 when avg_loss is 0."""
        trade_stats = {
            "win_rate": 1.0,
            "avg_win": 100.0,
            "avg_loss": 0.0,
        }

        kelly = metrics_calculator._calculate_kelly(trade_stats)

        assert kelly == 0.0
