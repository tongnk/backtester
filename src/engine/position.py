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
class FundingPayment:
    """
    Represents a single funding payment for perpetual futures.

    Funding mechanics:
    - Long + positive rate = pay funding (amount > 0)
    - Long + negative rate = receive funding (amount < 0)
    - Short + positive rate = receive funding (amount < 0)
    - Short + negative rate = pay funding (amount > 0)
    """
    timestamp: datetime
    rate: float           # Funding rate (can be positive or negative)
    position_value: float # Position notional at funding time
    amount: float         # Actual payment (positive = paid, negative = received)
    side: PositionSide    # Position side at funding time


@dataclass
class Position:
    """
    Represents an open position with multiple fills (pyramiding support).

    Tracks:
    - Multiple entries at different prices
    - Weighted average entry price
    - Current unrealized PnL
    - Partial exits
    - Funding payments (for perpetual futures)
    """
    side: PositionSide
    fills: list[Fill] = field(default_factory=list)
    exit_fills: list[Fill] = field(default_factory=list)
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    # Funding tracking for perpetual futures
    funding_payments: list[FundingPayment] = field(default_factory=list)

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
    def total_funding(self) -> float:
        """Total funding paid (positive) or received (negative)."""
        return sum(fp.amount for fp in self.funding_payments)

    def add_funding_payment(self, payment: FundingPayment) -> None:
        """Record a funding payment."""
        self.funding_payments.append(payment)

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
        """Calculate realized PnL from closed portion (includes fees and funding)."""
        if not self.exit_fills:
            return 0.0

        exit_qty = self.exit_qty
        if self.side == PositionSide.LONG:
            pnl = (self.avg_exit_price - self.avg_entry_price) * exit_qty
        else:  # SHORT
            pnl = (self.avg_entry_price - self.avg_exit_price) * exit_qty

        # Deduct fees and funding costs
        return pnl - self.total_fees - self.total_funding

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
    # Funding tracking for perpetual futures
    funding_paid: float = 0.0      # Total funding paid (positive = cost)
    funding_count: int = 0         # Number of funding events during trade

    @property
    def duration(self) -> pd.Timedelta:
        return pd.Timedelta(self.exit_time - self.entry_time)

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def gross_pnl(self) -> float:
        """PnL before fees and funding."""
        return self.pnl + self.fees + self.funding_paid


class PositionManager:
    """
    Manages positions and trade history.

    Supports:
    - Multiple concurrent positions (pyramiding)
    - Position sizing modes
    - Trade logging
    - Funding rate tracking (perpetual futures)
    """

    def __init__(
        self,
        max_positions: int = 1,
        fee_rate: float = 0.001,
        slippage: float = 0.0,
        perps_mode: bool = False,
    ):
        """
        Initialize the position manager.

        Args:
            max_positions: Maximum concurrent positions
            fee_rate: Trading fee rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0001 = 0.01%)
            perps_mode: Whether using perpetual futures (enables funding)
        """
        self.max_positions = max_positions
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.perps_mode = perps_mode

        self.current_position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.trade_log: list[dict] = []
        self.funding_log: list[dict] = []  # Track funding events

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

    def apply_funding(
        self,
        timestamp: datetime,
        funding_rate: float,
        mark_price: float,
    ) -> Optional[FundingPayment]:
        """
        Apply funding rate to current position.

        Called at funding times (00:00, 08:00, 16:00 UTC) when position is open.

        Funding mechanics:
        - Long position + positive rate = pay funding
        - Long position + negative rate = receive funding
        - Short position + positive rate = receive funding
        - Short position + negative rate = pay funding

        Args:
            timestamp: Funding timestamp
            funding_rate: Funding rate (e.g., 0.0001 for 0.01%)
            mark_price: Mark price at funding time

        Returns:
            FundingPayment if position was open, None otherwise
        """
        if self.is_flat:
            return None

        position = self.current_position
        position_value = position.quantity * mark_price

        # Calculate payment direction based on position side
        if position.side == PositionSide.LONG:
            # Longs pay positive funding, receive negative
            payment_amount = funding_rate * position_value
        else:  # SHORT
            # Shorts receive positive funding, pay negative
            payment_amount = -funding_rate * position_value

        funding_payment = FundingPayment(
            timestamp=timestamp,
            rate=funding_rate,
            position_value=position_value,
            amount=payment_amount,
            side=position.side,
        )

        position.add_funding_payment(funding_payment)

        # Log the funding event
        self.funding_log.append({
            "timestamp": timestamp,
            "rate": funding_rate,
            "position_value": position_value,
            "amount": payment_amount,
            "side": position.side.name,
        })

        return funding_payment

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
            funding_paid=pos.total_funding,
            funding_count=len(pos.funding_payments),
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
                # Funding columns (for perps)
                "funding_paid": t.funding_paid,
                "funding_count": t.funding_count,
                "gross_pnl": t.gross_pnl,
            })

        return pd.DataFrame(data)

    def get_funding_log_df(self) -> pd.DataFrame:
        """Get funding log as DataFrame."""
        return pd.DataFrame(self.funding_log)

    def reset(self) -> None:
        """Reset all positions and trade history."""
        self.current_position = None
        self.trades = []
        self.trade_log = []
        self.funding_log = []
