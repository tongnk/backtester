"""Implied Volatility Based Strategies - Using Deribit DVOL data."""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..base import Strategy, StrategyResult, Signal
from ..indicators import Indicators


class IVMeanReversionStrategy(Strategy):
    """
    IV Mean Reversion Strategy.

    Trades based on implied volatility deviating from its mean.

    When IV is high (above upper threshold):
    - Expects volatility to decrease, prices often consolidate after spikes
    - Goes short on IV spikes (contrarian)

    When IV is low (below lower threshold):
    - Expects volatility increase, prices may move sharply
    - Goes long in anticipation of breakout

    Parameters:
        iv_lookback: Lookback period for IV mean/std calculation (default: 168 = 1 week in hours)
        upper_threshold: Standard deviations above mean to trigger short (default: 2.0)
        lower_threshold: Standard deviations below mean to trigger long (default: -1.5)
        exit_threshold: Standard deviations from mean to exit (default: 0.5)
        use_price_confirmation: Require price trend confirmation (default: True)
        price_ma_period: Period for price trend MA (default: 20)
    """

    @property
    def name(self) -> str:
        return "IV Mean Reversion"

    @property
    def description(self) -> str:
        return "Trade based on implied volatility mean reversion"

    def _validate_params(self) -> None:
        upper = self.params.get("upper_threshold", 2.0)
        lower = self.params.get("lower_threshold", -1.5)

        if upper <= lower:
            raise ValueError(f"upper_threshold ({upper}) must be greater than lower_threshold ({lower})")

    def get_required_columns(self) -> list[str]:
        """This strategy requires IV data."""
        return ["open", "high", "low", "close", "volume", "iv"]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate IV z-score and price trend indicators."""
        iv_lookback = self.params.get("iv_lookback", 168)  # 1 week in hours
        price_ma_period = self.params.get("price_ma_period", 20)

        # Calculate IV z-score
        iv = df["iv"]
        iv_mean = iv.rolling(window=iv_lookback, min_periods=iv_lookback // 2).mean()
        iv_std = iv.rolling(window=iv_lookback, min_periods=iv_lookback // 2).std()

        df["iv_zscore"] = (iv - iv_mean) / iv_std
        df["iv_mean"] = iv_mean
        df["iv_std"] = iv_std

        # Price trend for confirmation
        df["price_ma"] = Indicators.ema(df["close"], price_ma_period)
        df["price_above_ma"] = df["close"] > df["price_ma"]

        # IV momentum
        df["iv_change"] = iv.pct_change(periods=24)  # 24-hour change

        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """Generate signals based on IV z-score thresholds."""
        signals = pd.Series(index=df.index, dtype=object)
        signals[:] = Signal.HOLD

        upper_threshold = self.params.get("upper_threshold", 2.0)
        lower_threshold = self.params.get("lower_threshold", -1.5)
        exit_threshold = self.params.get("exit_threshold", 0.5)
        use_price_confirmation = self.params.get("use_price_confirmation", True)

        iv_zscore = df["iv_zscore"]
        price_above_ma = df["price_above_ma"]

        # High IV: expect reversion, short
        high_iv = iv_zscore > upper_threshold
        high_iv_prev = high_iv.shift(1).fillna(False)

        # Low IV: expect expansion, long
        low_iv = iv_zscore < lower_threshold
        low_iv_prev = low_iv.shift(1).fillna(False)

        # Mean reversion zone
        neutral_iv = (iv_zscore > -exit_threshold) & (iv_zscore < exit_threshold)

        if use_price_confirmation:
            # Short on high IV only if price is above MA (contrarian at tops)
            signals[high_iv & ~high_iv_prev & price_above_ma] = Signal.SHORT
            # Long on low IV only if price is below MA (contrarian at bottoms)
            signals[low_iv & ~low_iv_prev & ~price_above_ma] = Signal.LONG
        else:
            signals[high_iv & ~high_iv_prev] = Signal.SHORT
            signals[low_iv & ~low_iv_prev] = Signal.LONG

        # Exit when IV returns to mean
        neutral_prev = neutral_iv.shift(1).fillna(False)
        signals[neutral_iv & ~neutral_prev] = Signal.CLOSE

        indicators = {
            "iv": df["iv"],
            "iv_zscore": df["iv_zscore"],
            "iv_mean": df["iv_mean"],
            "price_ma": df["price_ma"],
        }

        metadata = {
            "high_iv_signals": high_iv.sum(),
            "low_iv_signals": low_iv.sum(),
        }

        return StrategyResult(signals=signals, indicators=indicators, metadata=metadata)

    def warmup_period(self) -> int:
        return self.params.get("iv_lookback", 168)


class IVBreakoutStrategy(Strategy):
    """
    IV Breakout Strategy.

    Trades in the direction of price movement when IV spikes,
    assuming high IV indicates a real market move.

    When IV increases significantly:
    - If price is rising: Go long (momentum)
    - If price is falling: Go short (momentum)

    Parameters:
        iv_spike_threshold: Minimum IV increase % to trigger signal (default: 0.20 = 20%)
        iv_lookback: Lookback period for IV change calculation (default: 24)
        price_lookback: Lookback for price direction (default: 5)
        exit_iv_decrease: IV decrease % to exit (default: 0.10 = 10%)
        min_price_move: Minimum price change % to confirm (default: 0.01 = 1%)
    """

    @property
    def name(self) -> str:
        return "IV Breakout"

    @property
    def description(self) -> str:
        return "Trade momentum when IV spikes confirm price moves"

    def get_required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "iv"]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate IV change and price momentum."""
        iv_lookback = self.params.get("iv_lookback", 24)
        price_lookback = self.params.get("price_lookback", 5)

        # IV change
        df["iv_pct_change"] = df["iv"].pct_change(periods=iv_lookback)

        # IV moving average for context
        df["iv_ma"] = Indicators.sma(df["iv"], iv_lookback)

        # Price momentum
        df["price_change"] = df["close"].pct_change(periods=price_lookback)
        df["price_direction"] = np.sign(df["price_change"])

        # ATR for position sizing context
        df["atr"] = Indicators.atr(df["high"], df["low"], df["close"], 14)

        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """Generate signals on IV spikes with price confirmation."""
        signals = pd.Series(index=df.index, dtype=object)
        signals[:] = Signal.HOLD

        spike_threshold = self.params.get("iv_spike_threshold", 0.20)
        exit_decrease = self.params.get("exit_iv_decrease", 0.10)
        min_price_move = self.params.get("min_price_move", 0.01)

        iv_change = df["iv_pct_change"]
        price_change = df["price_change"]
        price_direction = df["price_direction"]

        # IV spike detection
        iv_spike = iv_change > spike_threshold
        iv_spike_prev = iv_spike.shift(1).fillna(False)

        # Price confirmation
        bullish_price = (price_change > min_price_move) & (price_direction > 0)
        bearish_price = (price_change < -min_price_move) & (price_direction < 0)

        # Entry signals
        signals[(iv_spike & ~iv_spike_prev) & bullish_price] = Signal.LONG
        signals[(iv_spike & ~iv_spike_prev) & bearish_price] = Signal.SHORT

        # Exit on IV decrease
        iv_decrease = iv_change < -exit_decrease
        iv_decrease_prev = iv_decrease.shift(1).fillna(False)
        signals[iv_decrease & ~iv_decrease_prev] = Signal.CLOSE

        indicators = {
            "iv": df["iv"],
            "iv_pct_change": df["iv_pct_change"],
            "iv_ma": df["iv_ma"],
            "price_change": df["price_change"],
        }

        return StrategyResult(signals=signals, indicators=indicators)

    def warmup_period(self) -> int:
        return max(
            self.params.get("iv_lookback", 24),
            self.params.get("price_lookback", 5),
        ) + 1


class IVRegimeStrategy(Strategy):
    """
    IV Regime Strategy.

    Uses IV levels to determine market regime and trade accordingly:
    - Low IV regime: Trade breakouts (momentum)
    - High IV regime: Trade mean reversion (contrarian)

    Parameters:
        iv_percentile_lookback: Lookback for percentile calculation (default: 720 = 30 days hourly)
        low_iv_percentile: Percentile threshold for low IV regime (default: 25)
        high_iv_percentile: Percentile threshold for high IV regime (default: 75)
        breakout_period: Period for breakout detection (default: 20)
        mean_reversion_period: Period for mean reversion signals (default: 14)
    """

    @property
    def name(self) -> str:
        return "IV Regime"

    @property
    def description(self) -> str:
        return "Adaptive strategy based on IV regime detection"

    def get_required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "iv"]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate IV regime and regime-specific indicators."""
        lookback = self.params.get("iv_percentile_lookback", 720)
        low_pct = self.params.get("low_iv_percentile", 25)
        high_pct = self.params.get("high_iv_percentile", 75)
        breakout_period = self.params.get("breakout_period", 20)
        mr_period = self.params.get("mean_reversion_period", 14)

        iv = df["iv"]

        # Calculate IV percentile rank
        df["iv_percentile"] = iv.rolling(window=lookback, min_periods=lookback // 2).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )

        # Regime classification
        df["low_iv_regime"] = df["iv_percentile"] < low_pct
        df["high_iv_regime"] = df["iv_percentile"] > high_pct
        df["normal_regime"] = ~df["low_iv_regime"] & ~df["high_iv_regime"]

        # Breakout indicators (for low IV regime)
        upper, middle, lower = Indicators.donchian_channels(df["high"], df["low"], breakout_period)
        df["donchian_upper"] = upper
        df["donchian_lower"] = lower
        df["donchian_middle"] = middle

        # Mean reversion indicators (for high IV regime)
        df["rsi"] = Indicators.rsi(df["close"], mr_period)
        bb_middle, bb_upper, bb_lower = Indicators.bollinger_bands(df["close"], mr_period, 2.0)
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower

        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """Generate regime-adaptive signals."""
        signals = pd.Series(index=df.index, dtype=object)
        signals[:] = Signal.HOLD

        close = df["close"]

        # Low IV regime: Breakout strategy
        low_iv = df["low_iv_regime"]
        breakout_long = (close > df["donchian_upper"].shift(1)) & low_iv
        breakout_short = (close < df["donchian_lower"].shift(1)) & low_iv

        signals[breakout_long] = Signal.LONG
        signals[breakout_short] = Signal.SHORT

        # High IV regime: Mean reversion strategy
        high_iv = df["high_iv_regime"]
        rsi = df["rsi"]

        mr_long = (rsi < 30) & (close < df["bb_lower"]) & high_iv
        mr_short = (rsi > 70) & (close > df["bb_upper"]) & high_iv

        signals[mr_long] = Signal.LONG
        signals[mr_short] = Signal.SHORT

        # Exit on regime change
        regime_change = (
            (df["low_iv_regime"] != df["low_iv_regime"].shift(1)) |
            (df["high_iv_regime"] != df["high_iv_regime"].shift(1))
        )
        signals[regime_change] = Signal.CLOSE

        indicators = {
            "iv": df["iv"],
            "iv_percentile": df["iv_percentile"],
            "donchian_upper": df["donchian_upper"],
            "donchian_lower": df["donchian_lower"],
            "rsi": df["rsi"],
            "bb_upper": df["bb_upper"],
            "bb_lower": df["bb_lower"],
        }

        metadata = {
            "low_iv_periods": df["low_iv_regime"].sum(),
            "high_iv_periods": df["high_iv_regime"].sum(),
        }

        return StrategyResult(signals=signals, indicators=indicators, metadata=metadata)

    def warmup_period(self) -> int:
        return self.params.get("iv_percentile_lookback", 720)
