"""Core backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config, SizingMode, PositionConfig
from ..strategy.base import Strategy, Signal, StrategyResult
from .position import PositionManager, PositionSide
from .metrics import Metrics, PerformanceMetrics


@dataclass
class BacktestResult:
    """Container for backtest results."""
    # Core data
    equity_curve: pd.Series
    trades_df: pd.DataFrame
    trade_log_df: pd.DataFrame

    # Strategy data
    signals: pd.Series
    indicators: dict[str, pd.Series]

    # Performance
    metrics: PerformanceMetrics

    # Metadata
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float


class Backtester:
    """
    Core backtesting engine with support for:
    - Multiple position sizing modes
    - Pyramiding
    - Fee and slippage modeling
    - No look-ahead bias
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the backtester.

        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or Config()
        self.position_manager: Optional[PositionManager] = None
        self.metrics_calculator = Metrics()

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        initial_capital: Optional[float] = None,
    ) -> BacktestResult:
        """
        Run backtest for a strategy.

        Args:
            strategy: Strategy instance to backtest
            data: OHLC DataFrame with timestamp index
            initial_capital: Starting capital (uses config default if not provided)

        Returns:
            BacktestResult with all backtest data
        """
        initial_capital = initial_capital or self.config.backtest.initial_capital

        # Validate data
        strategy.validate_data(data)

        # Initialize position manager
        self.position_manager = PositionManager(
            max_positions=self.config.position.max_pyramid_levels,
            fee_rate=self.config.backtest.fee_rate,
            slippage=self.config.backtest.slippage,
        )

        # Run strategy to get signals
        print(f"Running strategy: {strategy.name}")
        strategy_result = strategy.run(data)

        # Skip warmup period
        warmup = strategy.warmup_period()
        data = data.iloc[warmup:].copy()
        signals = strategy_result.signals.iloc[warmup:]

        # Initialize tracking
        capital = initial_capital
        equity = []
        equity_timestamps = []

        # Process each bar
        print(f"Processing {len(data)} bars...")
        for i in range(len(data)):
            timestamp = data.index[i]
            current_bar = data.iloc[i]

            # Get signal for this bar
            signal = signals.iloc[i]

            # Calculate current equity
            current_price = current_bar["close"]
            unrealized_pnl = 0.0
            if not self.position_manager.is_flat:
                unrealized_pnl = self.position_manager.current_position.unrealized_pnl(current_price)

            current_equity = capital + unrealized_pnl
            equity.append(current_equity)
            equity_timestamps.append(timestamp)

            # Process signal (use next bar's open for execution to avoid look-ahead bias)
            if i < len(data) - 1:
                next_bar = data.iloc[i + 1]
                execution_price = next_bar["open"]
                execution_time = data.index[i + 1]

                capital = self._process_signal(
                    signal=signal,
                    current_price=current_price,
                    execution_price=execution_price,
                    execution_time=execution_time,
                    capital=capital,
                    current_equity=current_equity,
                )

        # Close any remaining position at the end
        if not self.position_manager.is_flat:
            final_price = data.iloc[-1]["close"]
            final_time = data.index[-1]
            self.position_manager.close_position(final_time, final_price)
            realized_pnl = self.position_manager.trades[-1].pnl
            capital += realized_pnl

        # Create equity curve
        equity_curve = pd.Series(equity, index=equity_timestamps)

        # Get trade data
        trades_df = self.position_manager.get_trades_df()
        trade_log_df = self.position_manager.get_trade_log_df()

        # Calculate metrics
        metrics = self.metrics_calculator.calculate(
            equity_curve=equity_curve,
            trades_df=trades_df,
            initial_capital=initial_capital,
            price_series=data["close"],
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades_df=trades_df,
            trade_log_df=trade_log_df,
            signals=signals,
            indicators=strategy_result.indicators,
            metrics=metrics,
            strategy_name=strategy.name,
            symbol=self.config.data.symbol,
            start_date=data.index[0].to_pydatetime(),
            end_date=data.index[-1].to_pydatetime(),
            initial_capital=initial_capital,
            final_capital=capital,
        )

    def _process_signal(
        self,
        signal: Signal,
        current_price: float,
        execution_price: float,
        execution_time: datetime,
        capital: float,
        current_equity: float,
    ) -> float:
        """
        Process a trading signal.

        Returns updated capital after any trades.
        """
        if signal is None or signal == Signal.HOLD:
            return capital

        current_side = self.position_manager.position_side

        # Calculate position size
        position_size = self._calculate_position_size(
            equity=current_equity,
            price=execution_price,
        )

        if signal == Signal.LONG:
            if current_side == PositionSide.SHORT:
                # Close short and open long
                fill = self.position_manager.close_position(execution_time, execution_price)
                if fill:
                    capital += self.position_manager.trades[-1].pnl
                self.position_manager.open_position(
                    execution_time, execution_price, position_size, PositionSide.LONG
                )
            elif current_side == PositionSide.FLAT:
                # Open long
                self.position_manager.open_position(
                    execution_time, execution_price, position_size, PositionSide.LONG
                )

        elif signal == Signal.SHORT:
            if current_side == PositionSide.LONG:
                # Close long and open short
                fill = self.position_manager.close_position(execution_time, execution_price)
                if fill:
                    capital += self.position_manager.trades[-1].pnl
                self.position_manager.open_position(
                    execution_time, execution_price, position_size, PositionSide.SHORT
                )
            elif current_side == PositionSide.FLAT:
                # Open short
                self.position_manager.open_position(
                    execution_time, execution_price, position_size, PositionSide.SHORT
                )

        elif signal == Signal.CLOSE:
            if current_side != PositionSide.FLAT:
                fill = self.position_manager.close_position(execution_time, execution_price)
                if fill and self.position_manager.trades:
                    capital += self.position_manager.trades[-1].pnl

        elif signal == Signal.ADD:
            # Add to existing position (pyramid)
            if not self.position_manager.is_flat:
                # Check if position is profitable enough to add
                if self._should_pyramid(current_price):
                    add_size = self._calculate_pyramid_size(current_equity, execution_price)
                    self.position_manager.add_to_position(
                        execution_time, execution_price, add_size
                    )

        return capital

    def _calculate_position_size(
        self,
        equity: float,
        price: float,
    ) -> float:
        """Calculate position size based on sizing mode."""
        config = self.config.position

        if config.sizing_mode == SizingMode.FIXED_PERCENT:
            # size_value is percentage of equity (e.g., 1.0 = 100%)
            notional = equity * config.size_value
            return notional / price

        elif config.sizing_mode == SizingMode.FIXED_AMOUNT:
            # size_value is fixed BTC amount
            return config.size_value

        elif config.sizing_mode == SizingMode.KELLY:
            # Use Kelly fraction from previous trades
            # For simplicity, use configured size_value as base fraction
            kelly_fraction = min(config.size_value, 0.25)  # Cap at 25%
            notional = equity * kelly_fraction
            return notional / price

        return equity / price  # Default to 100%

    def _should_pyramid(self, current_price: float) -> bool:
        """Check if position is profitable enough to pyramid."""
        if self.position_manager.is_flat:
            return False

        position = self.position_manager.current_position
        threshold = self.config.position.pyramid_threshold

        unrealized_pnl_pct = position.unrealized_pnl(current_price) / (
            position.avg_entry_price * position.quantity
        )

        return unrealized_pnl_pct >= threshold

    def _calculate_pyramid_size(
        self,
        equity: float,
        price: float,
    ) -> float:
        """Calculate size for pyramid add."""
        # Use half the normal position size for pyramids
        base_size = self._calculate_position_size(equity, price)
        return base_size * 0.5

    def print_summary(self, result: BacktestResult) -> None:
        """Print backtest summary to console."""
        print(self.metrics_calculator.summary_string(result.metrics))
        print(f"\nStrategy: {result.strategy_name}")
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        print(f"Final Capital: ${result.final_capital:,.2f}")


def run_backtest(
    strategy: Strategy,
    data: pd.DataFrame,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0001,
    sizing_mode: str = "fixed_percent",
    size_value: float = 1.0,
    max_pyramid_levels: int = 1,
    pyramid_threshold: float = 0.02,
) -> BacktestResult:
    """
    Convenience function to run a backtest with custom parameters.

    Args:
        strategy: Strategy instance
        data: OHLC DataFrame
        initial_capital: Starting capital
        fee_rate: Fee rate per trade
        slippage: Slippage rate
        sizing_mode: 'fixed_percent', 'fixed_amount', or 'kelly'
        size_value: Size value (interpretation depends on mode)
        max_pyramid_levels: Maximum pyramid entries
        pyramid_threshold: Profit threshold to pyramid

    Returns:
        BacktestResult
    """
    config = Config()
    config.backtest.initial_capital = initial_capital
    config.backtest.fee_rate = fee_rate
    config.backtest.slippage = slippage
    config.position.sizing_mode = SizingMode(sizing_mode)
    config.position.size_value = size_value
    config.position.max_pyramid_levels = max_pyramid_levels
    config.position.pyramid_threshold = pyramid_threshold

    backtester = Backtester(config)
    return backtester.run(strategy, data, initial_capital)
