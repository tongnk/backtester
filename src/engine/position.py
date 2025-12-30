"""Position and trade tracking with pyramiding support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


class PositionSide(Enum):
    """Position side."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Fill:
    """Represents a single fill/execution."""
    timestamp: datetime
    price: float
    quantity: float
    side: PositionSide
    fee: float = 0.0

    @property
    def notional(self) -> float:
        """Total notional value of the fill."""
        return self.price * self.quantity

    @property
    def cost(self) -> float:
        """Total cost including fees."""
        return self.notional + self.fee


@dataclass
class Position:
    """
    Represents an open position with multiple fills (pyramiding support).

    Tracks:
    - Multiple entries at different prices
    - Weighted average entry price
    - Current unrealized PnL
    - Partial exits
    """
    side: PositionSide
    fills: list[Fill] = field(default_factory=list)
    exit_fills: list[Fill] = field(default_factory=list)
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None

    @property
    def quantity(self) -> float:
        """Current position quantity."""
        entry_qty = sum(f.quantity for f in self.fills)
        exit_qty = sum(f.quantity for f in self.exit_fills)
        return entry_qty - exit_qty

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.quantity > 0

    @property
    def avg_entry_price(self) -> float:
        """Weighted average entry price."""
        if not self.fills:
            return 0.0
        total_cost = sum(f.notional for f in self.fills)
        total_qty = sum(f.quantity for f in self.fills)
        return total_cost / total_qty if total_qty > 0 else 0.0

    @property
    def avg_exit_price(self) -> float:
        """Weighted average exit price."""
        if not self.exit_fills:
            return 0.0
        total_value = sum(f.notional for f in self.exit_fills)
        total_qty = sum(f.quantity for f in self.exit_fills)
        return total_value / total_qty if total_qty > 0 else 0.0

    @property
    def total_fees(self) -> float:
        """Total fees paid for all fills."""
        entry_fees = sum(f.fee for f in self.fills)
        exit_fees = sum(f.fee for f in self.exit_fills)
        return entry_fees + exit_fees

    @property
    def entry_qty(self) -> float:
        """Total entry quantity."""
        return sum(f.quantity for f in self.fills)

    @property
    def exit_qty(self) -> float:
        """Total exit quantity."""
        return sum(f.quantity for f in self.exit_fills)

    def add_fill(self, fill: Fill) -> None:
        """Add an entry fill to the position."""
        if not self.fills:
            self.entry_time = fill.timestamp
        self.fills.append(fill)

    def add_exit(self, fill: Fill) -> None:
        """Add an exit fill to the position."""
        self.exit_fills.append(fill)
        if self.quantity <= 0:
            self.exit_time = fill.timestamp

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if not self.is_open:
            return 0.0

        if self.side == PositionSide.LONG:
            return (current_price - self.avg_entry_price) * self.quantity
        else:  # SHORT
            return (self.avg_entry_price - current_price) * self.quantity

    def realized_pnl(self) -> float:
        """Calculate realized PnL from closed portion."""
        if not self.exit_fills:
            return 0.0

        exit_qty = self.exit_qty
        if self.side == PositionSide.LONG:
            pnl = (self.avg_exit_price - self.avg_entry_price) * exit_qty
        else:  # SHORT
            pnl = (self.avg_entry_price - self.avg_exit_price) * exit_qty

        return pnl - self.total_fees

    @property
    def pyramid_count(self) -> int:
        """Number of pyramid levels (entries)."""
        return len(self.fills)

    @property
    def duration(self) -> Optional[pd.Timedelta]:
        """Position duration."""
        if not self.entry_time:
            return None
        end_time = self.exit_time or datetime.now()
        return pd.Timedelta(end_time - self.entry_time)


@dataclass
class Trade:
    """Represents a completed trade (closed position)."""
    entry_time: datetime
    exit_time: datetime
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    pyramid_levels: int

    @property
    def duration(self) -> pd.Timedelta:
        return pd.Timedelta(self.exit_time - self.entry_time)

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


class PositionManager:
    """
    Manages positions and trade history.

    Supports:
    - Multiple concurrent positions (pyramiding)
    - Position sizing modes
    - Trade logging
    """

    def __init__(
        self,
        max_positions: int = 1,
        fee_rate: float = 0.001,
        slippage: float = 0.0,
    ):
        """
        Initialize the position manager.

        Args:
            max_positions: Maximum concurrent positions
            fee_rate: Trading fee rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0001 = 0.01%)
        """
        self.max_positions = max_positions
        self.fee_rate = fee_rate
        self.slippage = slippage

        self.current_position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.trade_log: list[dict] = []

    @property
    def is_flat(self) -> bool:
        """Check if no position is open."""
        return self.current_position is None or not self.current_position.is_open

    @property
    def position_side(self) -> PositionSide:
        """Current position side."""
        if self.is_flat:
            return PositionSide.FLAT
        return self.current_position.side

    def open_position(
        self,
        timestamp: datetime,
        price: float,
        quantity: float,
        side: PositionSide,
    ) -> Optional[Fill]:
        """
        Open a new position.

        Args:
            timestamp: Entry timestamp
            price: Entry price
            quantity: Position size
            side: Position side (LONG or SHORT)

        Returns:
            Fill object if successful, None otherwise
        """
        if not self.is_flat:
            return None

        # Apply slippage
        if side == PositionSide.LONG:
            fill_price = price * (1 + self.slippage)
        else:
            fill_price = price * (1 - self.slippage)

        # Calculate fee
        notional = fill_price * quantity
        fee = notional * self.fee_rate

        fill = Fill(
            timestamp=timestamp,
            price=fill_price,
            quantity=quantity,
            side=side,
            fee=fee,
        )

        self.current_position = Position(side=side)
        self.current_position.add_fill(fill)

        self._log_action("OPEN", timestamp, fill_price, quantity, side, fee)

        return fill

    def add_to_position(
        self,
        timestamp: datetime,
        price: float,
        quantity: float,
    ) -> Optional[Fill]:
        """
        Add to existing position (pyramid).

        Args:
            timestamp: Entry timestamp
            price: Entry price
            quantity: Additional size

        Returns:
            Fill object if successful, None otherwise
        """
        if self.is_flat:
            return None

        if self.current_position.pyramid_count >= self.max_positions:
            return None

        side = self.current_position.side

        # Apply slippage
        if side == PositionSide.LONG:
            fill_price = price * (1 + self.slippage)
        else:
            fill_price = price * (1 - self.slippage)

        notional = fill_price * quantity
        fee = notional * self.fee_rate

        fill = Fill(
            timestamp=timestamp,
            price=fill_price,
            quantity=quantity,
            side=side,
            fee=fee,
        )

        self.current_position.add_fill(fill)

        self._log_action("ADD", timestamp, fill_price, quantity, side, fee)

        return fill

    def close_position(
        self,
        timestamp: datetime,
        price: float,
        quantity: Optional[float] = None,
    ) -> Optional[Fill]:
        """
        Close position (fully or partially).

        Args:
            timestamp: Exit timestamp
            price: Exit price
            quantity: Quantity to close (None = close all)

        Returns:
            Fill object if successful, None otherwise
        """
        if self.is_flat:
            return None

        close_qty = quantity or self.current_position.quantity
        close_qty = min(close_qty, self.current_position.quantity)

        side = self.current_position.side

        # Apply slippage (opposite direction)
        if side == PositionSide.LONG:
            fill_price = price * (1 - self.slippage)
        else:
            fill_price = price * (1 + self.slippage)

        notional = fill_price * close_qty
        fee = notional * self.fee_rate

        fill = Fill(
            timestamp=timestamp,
            price=fill_price,
            quantity=close_qty,
            side=side,
            fee=fee,
        )

        self.current_position.add_exit(fill)

        self._log_action("CLOSE", timestamp, fill_price, close_qty, side, fee)

        # If fully closed, create trade record
        if not self.current_position.is_open:
            self._record_trade()

        return fill

    def flip_position(
        self,
        timestamp: datetime,
        price: float,
        quantity: float,
        new_side: PositionSide,
    ) -> tuple[Optional[Fill], Optional[Fill]]:
        """
        Close current position and open opposite.

        Args:
            timestamp: Timestamp
            price: Price
            quantity: New position size
            new_side: New position side

        Returns:
            Tuple of (close_fill, open_fill)
        """
        close_fill = None
        if not self.is_flat:
            close_fill = self.close_position(timestamp, price)

        open_fill = self.open_position(timestamp, price, quantity, new_side)

        return close_fill, open_fill

    def _record_trade(self) -> None:
        """Record completed trade."""
        pos = self.current_position

        pnl = pos.realized_pnl()
        entry_notional = pos.avg_entry_price * pos.entry_qty
        pnl_pct = pnl / entry_notional if entry_notional > 0 else 0.0

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=pos.exit_time,
            side=pos.side,
            entry_price=pos.avg_entry_price,
            exit_price=pos.avg_exit_price,
            quantity=pos.entry_qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=pos.total_fees,
            pyramid_levels=pos.pyramid_count,
        )

        self.trades.append(trade)
        self.current_position = None

    def _log_action(
        self,
        action: str,
        timestamp: datetime,
        price: float,
        quantity: float,
        side: PositionSide,
        fee: float,
    ) -> None:
        """Log a trading action."""
        self.trade_log.append({
            "action": action,
            "timestamp": timestamp,
            "price": price,
            "quantity": quantity,
            "side": side.name,
            "fee": fee,
        })

    def get_trade_log_df(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        return pd.DataFrame(self.trade_log)

    def get_trades_df(self) -> pd.DataFrame:
        """Get completed trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        data = []
        for t in self.trades:
            data.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side.name,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "fees": t.fees,
                "pyramid_levels": t.pyramid_levels,
                "duration": t.duration,
                "is_winner": t.is_winner,
            })

        return pd.DataFrame(data)

    def reset(self) -> None:
        """Reset all positions and trade history."""
        self.current_position = None
        self.trades = []
        self.trade_log = []
