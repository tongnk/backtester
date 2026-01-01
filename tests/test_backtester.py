"""Tests for backtester execution, especially TP/SL behavior."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.backtester import Backtester, run_backtest
from src.strategy.base import Strategy, Signal, StrategyResult
from src.config import Config


class SimpleTPSLStrategy(Strategy):
    """
    Simple strategy for testing TP/SL execution.

    - Enters LONG on first bar
    - Closes when close price hits TP or SL threshold
    """

    @property
    def name(self) -> str:
        return "SimpleTPSL"

    def __init__(self, params=None):
        super().__init__(params)
        self.tp_pct = self.get_param("tp_pct", 0.05)  # 5% take profit
        self.sl_pct = self.get_param("sl_pct", 0.05)  # 5% stop loss

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        signals = pd.Series(index=df.index, dtype=object)
        signals[:] = Signal.HOLD

        entry_price = None
        in_position = False

        for i in range(len(df)):
            close = df.iloc[i]["close"]

            if not in_position:
                # Enter on first bar
                signals.iloc[i] = Signal.LONG
                entry_price = close
                in_position = True
            else:
                # Check TP/SL
                pnl_pct = (close - entry_price) / entry_price

                if pnl_pct >= self.tp_pct:
                    signals.iloc[i] = Signal.CLOSE
                    in_position = False
                elif pnl_pct <= -self.sl_pct:
                    signals.iloc[i] = Signal.CLOSE
                    in_position = False

        return StrategyResult(signals=signals, indicators={})

    def get_required_columns(self):
        return ["open", "high", "low", "close", "volume"]


def create_ohlc_data(prices, start_date="2024-01-01"):
    """Create OHLC DataFrame from a list of close prices."""
    dates = pd.date_range(start=start_date, periods=len(prices), freq="1min")
    data = {
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000] * len(prices),
    }
    return pd.DataFrame(data, index=dates)


class TestSignalExecution:
    """Tests for basic signal execution."""

    def test_signal_executes_on_next_bar_open(self):
        """Signals should execute at the next bar's open price."""
        # Price pattern: enter at bar 0 (100), close at bar 3 (110)
        # Execution should happen at bar 4 open
        prices = [100, 102, 104, 110, 108]  # TP at bar 3 (10% > 5%)
        data = create_ohlc_data(prices)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.05, "sl_pct": 0.05})
        result = run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.0,
        )

        # Should have 1 completed trade
        assert len(result.trades_df) == 1

        trade = result.trades_df.iloc[0]

        # Entry at bar 1 open (signal at bar 0, execute at bar 1)
        assert abs(trade["entry_price"] - 102) < 0.01  # Bar 1 open

        # Exit at bar 4 open (signal at bar 3, execute at bar 4)
        assert abs(trade["exit_price"] - 108) < 0.01  # Bar 4 open

    def test_no_look_ahead_bias(self):
        """Entry should not use future prices."""
        prices = [100, 150, 200, 250, 300]  # Dramatic price increase
        data = create_ohlc_data(prices)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.50, "sl_pct": 0.05})  # 50% TP
        result = run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.0,
        )

        # If there was look-ahead bias, entry would be at bar 0 (100)
        # But correct execution is at bar 1 open (150)
        trade = result.trades_df.iloc[0]
        assert trade["entry_price"] == 150  # Bar 1 open, not bar 0 close


class TestTakeProfitExecution:
    """Tests for take-profit execution."""

    def test_tp_triggers_when_price_rises_above_threshold(self):
        """TP should trigger when close >= entry * (1 + tp_pct)."""
        # Entry at 100 (bar 0 signal, bar 1 open execution)
        # Bar 1 open = 100, so entry = 100
        # TP = 5%, target = 105
        # Bar 2 close = 103 (no trigger)
        # Bar 3 close = 106 (triggers TP)
        prices = [100, 100, 103, 106, 105]

        # We need to set bar opens explicitly
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({
            "open": [100, 100, 102, 105, 105],  # Bar 1 open = 100 (entry)
            "high": [101, 104, 105, 107, 106],
            "low": [99, 99, 101, 104, 104],
            "close": [100, 103, 104, 106, 105],  # Bar 3 close=106 >= 105 triggers
            "volume": [1000] * 5,
        }, index=dates)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.05, "sl_pct": 0.10})
        result = run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.0,
        )

        trade = result.trades_df.iloc[0]

        # Entry at bar 1 open = 100
        assert trade["entry_price"] == 100

        # TP triggered at bar 3 (close=106), exit at bar 4 open = 105
        assert trade["exit_price"] == 105

        # PnL should be (105 - 100) * qty = 5% of position
        assert trade["pnl"] > 0  # Profitable trade

    def test_tp_execution_gap(self):
        """
        Document the execution gap: TP triggers on close but executes on next open.

        This means actual exit price may differ from the trigger price.
        """
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({
            "open": [100, 100, 102, 108, 103],  # Bar 4 opens at 103 (gap down!)
            "high": [101, 104, 107, 110, 105],
            "low": [99, 99, 101, 105, 102],
            "close": [100, 103, 106, 109, 104],  # Bar 2 close=106 triggers 5% TP
            "volume": [1000] * 5,
        }, index=dates)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.05, "sl_pct": 0.10})
        result = run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.0,
        )

        trade = result.trades_df.iloc[0]

        # Entry at bar 1 open = 100
        assert trade["entry_price"] == 100

        # TP triggered when close >= 105 (bar 2 close=106)
        # But exit executes at bar 3 open = 108 (better than expected!)
        assert trade["exit_price"] == 108

        # This demonstrates the execution gap - exit can be better or worse
        # than the trigger price depending on the next bar's open


class TestStopLossExecution:
    """Tests for stop-loss execution."""

    def test_sl_triggers_when_price_drops_below_threshold(self):
        """SL should trigger when close <= entry * (1 - sl_pct)."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({
            "open": [100, 100, 98, 94, 95],  # Entry at bar 1 open = 100
            "high": [101, 101, 99, 96, 96],
            "low": [99, 97, 93, 93, 94],
            "close": [100, 98, 94, 95, 95],  # Bar 2 close=94 triggers 5% SL
            "volume": [1000] * 5,
        }, index=dates)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.10, "sl_pct": 0.05})
        result = run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.0,
        )

        trade = result.trades_df.iloc[0]

        # Entry at bar 1 open = 100
        assert trade["entry_price"] == 100

        # SL triggered at bar 2 (close=94 <= 95), exit at bar 3 open = 94
        assert trade["exit_price"] == 94

        # Should be a losing trade
        assert trade["pnl"] < 0

    def test_sl_execution_gap_can_increase_loss(self):
        """
        Document that SL execution at next bar open can increase the loss.
        """
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({
            "open": [100, 100, 98, 90, 92],  # Bar 3 gaps down to 90!
            "high": [101, 101, 99, 92, 93],
            "low": [99, 97, 94, 89, 91],
            "close": [100, 98, 95, 91, 92],  # Bar 2 close=95 triggers 5% SL
            "volume": [1000] * 5,
        }, index=dates)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.10, "sl_pct": 0.05})
        result = run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.0,
        )

        trade = result.trades_df.iloc[0]

        # Entry at bar 1 open = 100, SL at 95
        # SL triggered at bar 2 (close=95), but exit at bar 3 open = 90
        # Actual loss is 10%, not the expected 5%
        assert trade["exit_price"] == 90

        # Loss is worse than SL percentage
        actual_loss_pct = (trade["exit_price"] - trade["entry_price"]) / trade["entry_price"]
        assert actual_loss_pct < -0.05  # Loss exceeds the 5% SL


class TestSlippageEffect:
    """Tests for slippage effects on TP/SL."""

    def test_slippage_reduces_pnl(self):
        """Slippage should reduce realized PnL."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({
            "open": [100, 100, 105, 110, 110],
            "high": [101, 106, 111, 111, 111],
            "low": [99, 99, 104, 109, 109],
            "close": [100, 105, 110, 110, 110],  # 5% gain at bar 1
            "volume": [1000] * 5,
        }, index=dates)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.05, "sl_pct": 0.05})

        # Run without slippage
        result_no_slip = run_backtest(
            strategy=strategy,
            data=data.copy(),
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.0,
        )

        # Run with slippage
        result_with_slip = run_backtest(
            strategy=strategy,
            data=data.copy(),
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.001,  # 0.1% slippage
        )

        pnl_no_slip = result_no_slip.trades_df.iloc[0]["pnl"]
        pnl_with_slip = result_with_slip.trades_df.iloc[0]["pnl"]

        # Slippage should reduce PnL
        assert pnl_with_slip < pnl_no_slip

    def test_slippage_applied_to_entry_and_exit(self):
        """Slippage should be applied to both entry and exit."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({
            "open": [100, 100, 105, 110, 110],
            "high": [101, 106, 111, 111, 111],
            "low": [99, 99, 104, 109, 109],
            "close": [100, 105, 110, 110, 110],
            "volume": [1000] * 5,
        }, index=dates)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.05, "sl_pct": 0.05})
        result = run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.01,  # 1% slippage for visibility
        )

        trade = result.trades_df.iloc[0]

        # Entry at bar 1 open = 100, with 1% slippage for LONG = 101
        assert abs(trade["entry_price"] - 101) < 0.01

        # Exit at bar 2 open = 105, with 1% slippage for sell = 103.95
        expected_exit = 105 * 0.99
        assert abs(trade["exit_price"] - expected_exit) < 0.01


class TestFinalPositionClose:
    """Tests for closing positions at end of backtest."""

    def test_open_position_closed_at_end(self):
        """Open positions should be closed at the last bar's close."""
        dates = pd.date_range(start="2024-01-01", periods=3, freq="1min")
        data = pd.DataFrame({
            "open": [100, 100, 102],
            "high": [101, 103, 103],
            "low": [99, 99, 101],
            "close": [100, 102, 103],  # Never hits TP or SL
            "volume": [1000] * 3,
        }, index=dates)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.20, "sl_pct": 0.20})  # Wide TP/SL
        result = run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.0,
        )

        # Should have 1 trade closed at end
        assert len(result.trades_df) == 1

        trade = result.trades_df.iloc[0]

        # Should be closed at final bar's close = 103
        assert trade["exit_price"] == 103


class TestCommissions:
    """Tests for commission/fee handling."""

    def test_fees_deducted_from_pnl(self):
        """Fees should be deducted from realized PnL."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({
            "open": [100, 100, 105, 110, 110],
            "high": [101, 106, 111, 111, 111],
            "low": [99, 99, 104, 109, 109],
            "close": [100, 105, 110, 110, 110],
            "volume": [1000] * 5,
        }, index=dates)

        strategy = SimpleTPSLStrategy({"tp_pct": 0.05, "sl_pct": 0.05})

        # Run without fees
        result_no_fee = run_backtest(
            strategy=strategy,
            data=data.copy(),
            initial_capital=10000,
            fee_rate=0.0,
            slippage=0.0,
        )

        # Run with fees
        result_with_fee = run_backtest(
            strategy=strategy,
            data=data.copy(),
            initial_capital=10000,
            fee_rate=0.001,  # 0.1% fee
            slippage=0.0,
        )

        pnl_no_fee = result_no_fee.trades_df.iloc[0]["pnl"]
        pnl_with_fee = result_with_fee.trades_df.iloc[0]["pnl"]

        # Fees should reduce PnL
        assert pnl_with_fee < pnl_no_fee

        # Fees should be recorded
        assert result_with_fee.trades_df.iloc[0]["fees"] > 0
