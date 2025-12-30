"""Moving Average Crossover Strategy - A classic trend-following strategy."""

from __future__ import annotations

import pandas as pd

from ..base import Strategy, StrategyResult, Signal
from ..indicators import Indicators


class MACrossoverStrategy(Strategy):
    """
    Moving Average Crossover Strategy.

    Goes long when fast MA crosses above slow MA.
    Goes short when fast MA crosses below slow MA.

    Parameters:
        fast_period: Period for fast moving average (default: 10)
        slow_period: Period for slow moving average (default: 50)
        ma_type: Type of moving average - 'sma' or 'ema' (default: 'ema')
        use_close_signal: If True, close position on opposite crossover (default: True)
    """

    @property
    def name(self) -> str:
        return "MA Crossover"

    @property
    def description(self) -> str:
        fast = self.params.get("fast_period", 10)
        slow = self.params.get("slow_period", 50)
        ma_type = self.params.get("ma_type", "ema").upper()
        return f"{ma_type} {fast}/{slow} Crossover Strategy"

    def _validate_params(self) -> None:
        fast = self.params.get("fast_period", 10)
        slow = self.params.get("slow_period", 50)

        if fast >= slow:
            raise ValueError(f"fast_period ({fast}) must be less than slow_period ({slow})")

        if fast < 1 or slow < 1:
            raise ValueError("Periods must be positive integers")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fast and slow moving averages."""
        fast_period = self.params.get("fast_period", 10)
        slow_period = self.params.get("slow_period", 50)
        ma_type = self.params.get("ma_type", "ema").lower()

        close = df["close"]

        if ma_type == "sma":
            df["fast_ma"] = Indicators.sma(close, fast_period)
            df["slow_ma"] = Indicators.sma(close, slow_period)
        else:  # ema
            df["fast_ma"] = Indicators.ema(close, fast_period)
            df["slow_ma"] = Indicators.ema(close, slow_period)

        # Calculate crossover signals
        df["ma_diff"] = df["fast_ma"] - df["slow_ma"]
        df["ma_diff_prev"] = df["ma_diff"].shift(1)

        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """
        Generate signals based on MA crossovers.

        Long signal: Fast MA crosses above Slow MA
        Short signal: Fast MA crosses below Slow MA
        """
        signals = pd.Series(index=df.index, dtype=object)
        signals[:] = Signal.HOLD

        use_close_signal = self.params.get("use_close_signal", True)

        # Bullish crossover: fast crosses above slow
        bullish_cross = (df["ma_diff"] > 0) & (df["ma_diff_prev"] <= 0)

        # Bearish crossover: fast crosses below slow
        bearish_cross = (df["ma_diff"] < 0) & (df["ma_diff_prev"] >= 0)

        if use_close_signal:
            # Crossover acts as both entry and exit
            signals[bullish_cross] = Signal.LONG
            signals[bearish_cross] = Signal.SHORT
        else:
            # Only entry signals, no automatic exit
            signals[bullish_cross] = Signal.LONG
            signals[bearish_cross] = Signal.SHORT

        # Store indicators for visualization
        indicators = {
            "fast_ma": df["fast_ma"],
            "slow_ma": df["slow_ma"],
        }

        metadata = {
            "bullish_crosses": bullish_cross.sum(),
            "bearish_crosses": bearish_cross.sum(),
        }

        return StrategyResult(signals=signals, indicators=indicators, metadata=metadata)

    def warmup_period(self) -> int:
        """Return warmup period based on slow MA period."""
        return self.params.get("slow_period", 50) + 1

    def get_required_columns(self) -> list[str]:
        """This strategy only needs basic OHLC data."""
        return ["close"]


class TripleMAStrategy(Strategy):
    """
    Triple Moving Average Strategy.

    Uses three MAs to confirm trend direction and filter signals.

    Long: Fast > Medium > Slow (all aligned upward)
    Short: Fast < Medium < Slow (all aligned downward)

    Parameters:
        fast_period: Period for fast MA (default: 10)
        medium_period: Period for medium MA (default: 25)
        slow_period: Period for slow MA (default: 50)
        ma_type: Type of moving average - 'sma' or 'ema' (default: 'ema')
    """

    @property
    def name(self) -> str:
        return "Triple MA"

    @property
    def description(self) -> str:
        fast = self.params.get("fast_period", 10)
        medium = self.params.get("medium_period", 25)
        slow = self.params.get("slow_period", 50)
        return f"Triple MA {fast}/{medium}/{slow} Strategy"

    def _validate_params(self) -> None:
        fast = self.params.get("fast_period", 10)
        medium = self.params.get("medium_period", 25)
        slow = self.params.get("slow_period", 50)

        if not (fast < medium < slow):
            raise ValueError(f"Periods must be ordered: fast ({fast}) < medium ({medium}) < slow ({slow})")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate three moving averages."""
        fast_period = self.params.get("fast_period", 10)
        medium_period = self.params.get("medium_period", 25)
        slow_period = self.params.get("slow_period", 50)
        ma_type = self.params.get("ma_type", "ema").lower()

        close = df["close"]

        ma_func = Indicators.ema if ma_type == "ema" else Indicators.sma

        df["fast_ma"] = ma_func(close, fast_period)
        df["medium_ma"] = ma_func(close, medium_period)
        df["slow_ma"] = ma_func(close, slow_period)

        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """Generate signals based on triple MA alignment."""
        signals = pd.Series(index=df.index, dtype=object)
        signals[:] = Signal.HOLD

        # Bullish alignment: fast > medium > slow
        bullish = (df["fast_ma"] > df["medium_ma"]) & (df["medium_ma"] > df["slow_ma"])
        bullish_prev = bullish.shift(1).fillna(False)

        # Bearish alignment: fast < medium < slow
        bearish = (df["fast_ma"] < df["medium_ma"]) & (df["medium_ma"] < df["slow_ma"])
        bearish_prev = bearish.shift(1).fillna(False)

        # Signal on alignment change
        signals[bullish & ~bullish_prev] = Signal.LONG
        signals[bearish & ~bearish_prev] = Signal.SHORT

        # Close when alignment breaks
        neutral = ~bullish & ~bearish
        neutral_prev = neutral.shift(1).fillna(True)
        signals[neutral & ~neutral_prev] = Signal.CLOSE

        indicators = {
            "fast_ma": df["fast_ma"],
            "medium_ma": df["medium_ma"],
            "slow_ma": df["slow_ma"],
        }

        return StrategyResult(signals=signals, indicators=indicators)

    def warmup_period(self) -> int:
        return self.params.get("slow_period", 50) + 1
