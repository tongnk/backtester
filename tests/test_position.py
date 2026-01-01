"""Tests for position management and PnL calculations."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.position import Position, PositionSide, Fill, PositionManager


class TestPositionUnrealizedPnL:
    """Tests for unrealized PnL calculation."""

    def test_long_unrealized_pnl_profit(self):
        """Long position unrealized PnL when price increases."""
        pos = Position(side=PositionSide.LONG)
        fill = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_fill(fill)

        # Price went up to 110
        unrealized = pos.unrealized_pnl(current_price=110.0)

        # Expected: (110 - 100) * 1 = 10
        assert unrealized == 10.0

    def test_long_unrealized_pnl_loss(self):
        """Long position unrealized PnL when price decreases."""
        pos = Position(side=PositionSide.LONG)
        fill = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_fill(fill)

        # Price went down to 90
        unrealized = pos.unrealized_pnl(current_price=90.0)

        # Expected: (90 - 100) * 1 = -10
        assert unrealized == -10.0

    def test_short_unrealized_pnl_profit(self):
        """Short position unrealized PnL when price decreases."""
        pos = Position(side=PositionSide.SHORT)
        fill = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.SHORT,
            fee=0.0,
        )
        pos.add_fill(fill)

        # Price went down to 90 (good for short)
        unrealized = pos.unrealized_pnl(current_price=90.0)

        # Expected: (100 - 90) * 1 = 10
        assert unrealized == 10.0

    def test_short_unrealized_pnl_loss(self):
        """Short position unrealized PnL when price increases."""
        pos = Position(side=PositionSide.SHORT)
        fill = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.SHORT,
            fee=0.0,
        )
        pos.add_fill(fill)

        # Price went up to 110 (bad for short)
        unrealized = pos.unrealized_pnl(current_price=110.0)

        # Expected: (100 - 110) * 1 = -10
        assert unrealized == -10.0


class TestPositionRealizedPnL:
    """Tests for realized PnL calculation."""

    def test_realized_pnl_simple_long(self):
        """Simple long position realized PnL."""
        pos = Position(side=PositionSide.LONG)

        # Entry
        entry_fill = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.1,
        )
        pos.add_fill(entry_fill)

        # Exit
        exit_fill = Fill(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=110.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.11,
        )
        pos.add_exit(exit_fill)

        realized = pos.realized_pnl()

        # Expected: (110 - 100) * 1 - (0.1 + 0.11) = 10 - 0.21 = 9.79
        expected = 10.0 - 0.21
        assert abs(realized - expected) < 0.01, f"Expected {expected}, got {realized}"

    def test_realized_pnl_simple_short(self):
        """Simple short position realized PnL."""
        pos = Position(side=PositionSide.SHORT)

        # Entry
        entry_fill = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.SHORT,
            fee=0.1,
        )
        pos.add_fill(entry_fill)

        # Exit (price went down = profit)
        exit_fill = Fill(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=90.0,
            quantity=1.0,
            side=PositionSide.SHORT,
            fee=0.09,
        )
        pos.add_exit(exit_fill)

        realized = pos.realized_pnl()

        # Expected: (100 - 90) * 1 - (0.1 + 0.09) = 10 - 0.19 = 9.81
        expected = 10.0 - 0.19
        assert abs(realized - expected) < 0.01, f"Expected {expected}, got {realized}"

    def test_realized_pnl_with_fees(self):
        """Verify fees are properly deducted from realized PnL."""
        pos = Position(side=PositionSide.LONG)

        entry_fill = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=1.0,  # $1 fee
        )
        pos.add_fill(entry_fill)

        exit_fill = Fill(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=105.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=1.05,  # $1.05 fee
        )
        pos.add_exit(exit_fill)

        realized = pos.realized_pnl()

        # Gross PnL: (105 - 100) * 1 = 5
        # Total fees: 1 + 1.05 = 2.05
        # Net PnL: 5 - 2.05 = 2.95
        expected = 2.95
        assert abs(realized - expected) < 0.01


class TestPyramidedPositions:
    """Tests for pyramided position handling."""

    def test_pyramided_avg_entry_price(self):
        """Test average entry price calculation for pyramided position."""
        pos = Position(side=PositionSide.LONG)

        # First entry: 1 BTC at $100
        fill1 = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_fill(fill1)

        # Second entry: 1 BTC at $110
        fill2 = Fill(
            timestamp=datetime(2024, 1, 1, 10, 30),
            price=110.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_fill(fill2)

        # Average entry: (100 + 110) / 2 = 105
        assert pos.avg_entry_price == 105.0
        assert pos.quantity == 2.0
        assert pos.pyramid_count == 2

    def test_pyramided_weighted_avg_entry_price(self):
        """Test weighted average entry price for unequal quantities."""
        pos = Position(side=PositionSide.LONG)

        # First entry: 2 BTC at $100
        fill1 = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=2.0,
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_fill(fill1)

        # Second entry: 1 BTC at $130
        fill2 = Fill(
            timestamp=datetime(2024, 1, 1, 10, 30),
            price=130.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.0,
        )
        pos.add_fill(fill2)

        # Weighted avg: (2*100 + 1*130) / 3 = 330/3 = 110
        expected_avg = (2 * 100 + 1 * 130) / 3
        assert abs(pos.avg_entry_price - expected_avg) < 0.01

    def test_pyramided_position_full_close_pnl(self):
        """Test PnL for pyramided position closed all at once."""
        pos = Position(side=PositionSide.LONG)

        # Entry 1: 1 BTC at $100
        fill1 = Fill(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.1,
        )
        pos.add_fill(fill1)

        # Entry 2: 1 BTC at $110
        fill2 = Fill(
            timestamp=datetime(2024, 1, 1, 10, 30),
            price=110.0,
            quantity=1.0,
            side=PositionSide.LONG,
            fee=0.11,
        )
        pos.add_fill(fill2)

        # Exit all at $120
        exit_fill = Fill(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=120.0,
            quantity=2.0,
            side=PositionSide.LONG,
            fee=0.24,
        )
        pos.add_exit(exit_fill)

        realized = pos.realized_pnl()

        # Avg entry: 105, Exit: 120, Qty: 2
        # Gross PnL: (120 - 105) * 2 = 30
        # Total fees: 0.1 + 0.11 + 0.24 = 0.45
        # Net PnL: 30 - 0.45 = 29.55
        expected = 29.55
        assert abs(realized - expected) < 0.01, f"Expected {expected}, got {realized}"


class TestSlippage:
    """Tests for slippage application."""

    def test_slippage_long_entry(self, position_manager):
        """Slippage should make long entries worse (higher price)."""
        pm = PositionManager(max_positions=1, fee_rate=0.0, slippage=0.001)  # 0.1% slippage

        fill = pm.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        # Long entry should be slipped up: 100 * 1.001 = 100.1
        expected_price = 100.0 * 1.001
        assert abs(fill.price - expected_price) < 0.001

    def test_slippage_short_entry(self, position_manager):
        """Slippage should make short entries worse (lower price)."""
        pm = PositionManager(max_positions=1, fee_rate=0.0, slippage=0.001)

        fill = pm.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.SHORT,
        )

        # Short entry should be slipped down: 100 * 0.999 = 99.9
        expected_price = 100.0 * 0.999
        assert abs(fill.price - expected_price) < 0.001

    def test_slippage_long_exit(self):
        """Slippage should make long exits worse (lower price)."""
        pm = PositionManager(max_positions=1, fee_rate=0.0, slippage=0.001)

        # Open position
        pm.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        # Close position
        exit_fill = pm.close_position(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=110.0,
        )

        # Long exit should be slipped down: 110 * 0.999 = 109.89
        expected_price = 110.0 * 0.999
        assert abs(exit_fill.price - expected_price) < 0.001

    def test_slippage_short_exit(self):
        """Slippage should make short exits worse (higher price)."""
        pm = PositionManager(max_positions=1, fee_rate=0.0, slippage=0.001)

        # Open short position
        pm.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.SHORT,
        )

        # Close position
        exit_fill = pm.close_position(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=90.0,
        )

        # Short exit should be slipped up: 90 * 1.001 = 90.09
        expected_price = 90.0 * 1.001
        assert abs(exit_fill.price - expected_price) < 0.001


class TestPositionManager:
    """Tests for PositionManager functionality."""

    def test_open_position_when_flat(self, position_manager):
        """Should be able to open position when flat."""
        fill = position_manager.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        assert fill is not None
        assert not position_manager.is_flat
        assert position_manager.position_side == PositionSide.LONG

    def test_cannot_open_position_when_not_flat(self, position_manager):
        """Should not be able to open new position when one exists."""
        position_manager.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        # Try to open another position
        fill = position_manager.open_position(
            timestamp=datetime(2024, 1, 1, 10, 30),
            price=105.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        assert fill is None

    def test_add_to_position_pyramiding(self, position_manager):
        """Should be able to add to existing position."""
        position_manager.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        fill = position_manager.add_to_position(
            timestamp=datetime(2024, 1, 1, 10, 30),
            price=105.0,
            quantity=1.0,
        )

        assert fill is not None
        assert position_manager.current_position.pyramid_count == 2

    def test_close_position_records_trade(self, position_manager):
        """Closing position should record a trade."""
        position_manager.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        position_manager.close_position(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=110.0,
        )

        assert len(position_manager.trades) == 1
        assert position_manager.is_flat

    def test_flip_position(self, position_manager):
        """Should be able to flip from long to short."""
        position_manager.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        close_fill, open_fill = position_manager.flip_position(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=110.0,
            quantity=1.0,
            new_side=PositionSide.SHORT,
        )

        assert close_fill is not None
        assert open_fill is not None
        assert position_manager.position_side == PositionSide.SHORT
        assert len(position_manager.trades) == 1  # First trade recorded


class TestTradeRecording:
    """Tests for trade recording accuracy."""

    def test_trade_pnl_percentage(self, position_manager):
        """Test PnL percentage calculation in recorded trade."""
        position_manager.open_position(
            timestamp=datetime(2024, 1, 1, 10, 0),
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        position_manager.close_position(
            timestamp=datetime(2024, 1, 1, 11, 0),
            price=110.0,
        )

        trade = position_manager.trades[0]

        # Entry notional: ~100 (with slippage)
        # PnL: (110 - 100) * 1 - fees
        # PnL%: pnl / entry_notional

        # With 0.1% fee rate:
        # Entry: 100 * 1.001 = 100.1 (slippage) -> fee = 0.1001
        # Exit: 110 * 0.999 = 109.89 -> fee = 0.10989
        # Wait, slippage is 0 in fixture, just fees

        # Actually position_manager fixture has slippage=0
        # Entry price = 100, fee = 100 * 0.001 = 0.1
        # Exit price = 110, fee = 110 * 0.001 = 0.11
        # PnL = (110 - 100) * 1 - 0.21 = 9.79
        # Entry notional = 100 * 1 = 100
        # PnL% = 9.79 / 100 = 9.79%

        expected_pnl_pct = 0.0979
        assert abs(trade.pnl_pct - expected_pnl_pct) < 0.01, \
            f"Expected pnl_pct ~{expected_pnl_pct}, got {trade.pnl_pct}"

    def test_trade_duration(self, position_manager):
        """Test trade duration calculation."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        exit_time = datetime(2024, 1, 1, 11, 30)

        position_manager.open_position(
            timestamp=entry_time,
            price=100.0,
            quantity=1.0,
            side=PositionSide.LONG,
        )

        position_manager.close_position(
            timestamp=exit_time,
            price=110.0,
        )

        trade = position_manager.trades[0]
        expected_duration = pd.Timedelta(hours=1, minutes=30)

        assert trade.duration == expected_duration
