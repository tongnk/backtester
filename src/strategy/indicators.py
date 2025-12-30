"""Technical indicators library - all vectorized for performance."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Indicators:
    """Collection of technical indicators for strategy development."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.

        Args:
            series: Price series
            period: Lookback period

        Returns:
            SMA series
        """
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.

        Args:
            series: Price series
            period: Lookback period

        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            series: Price series
            period: RSI period (default 14)

        Returns:
            RSI series (0-100)
        """
        delta = series.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def bollinger_bands(
        series: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Args:
            series: Price series
            period: SMA period (default 20)
            std_dev: Standard deviation multiplier (default 2)

        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        middle = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return middle, upper, lower

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Average True Range.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 14)

        Returns:
            ATR series
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()

        return atr

    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.

        Args:
            series: Price series
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line period (default 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period (default 14)
            d_period: %D period (default 3)

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period, min_periods=d_period).mean()

        return k, d

    @staticmethod
    def vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Volume Weighted Average Price (cumulative).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series

        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ADX period (default 14)

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = low.diff()

        plus_dm = np.where(
            (high_diff > low_diff.abs()) & (high_diff > 0), high_diff, 0.0
        )
        minus_dm = np.where(
            (low_diff.abs() > high_diff) & (low_diff < 0), low_diff.abs(), 0.0
        )

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Calculate ATR
        atr = Indicators.atr(high, low, close, period)

        # Smooth DM values
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum indicator.

        Args:
            series: Price series
            period: Lookback period

        Returns:
            Momentum series
        """
        return series.diff(period)

    @staticmethod
    def roc(series: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change (percentage).

        Args:
            series: Price series
            period: Lookback period

        Returns:
            ROC series (percentage)
        """
        return 100 * series.pct_change(period)

    @staticmethod
    def williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Williams %R.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period (default 14)

        Returns:
            Williams %R series (-100 to 0)
        """
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()

        return -100 * (highest_high - close) / (highest_high - lowest_low)

    @staticmethod
    def cci(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> pd.Series:
        """
        Commodity Channel Index.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: CCI period (default 20)

        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period, min_periods=period).mean()
        mad = typical_price.rolling(window=period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        return (typical_price - sma) / (0.015 * mad)

    @staticmethod
    def donchian_channels(
        high: pd.Series, low: pd.Series, period: int = 20
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels.

        Args:
            high: High price series
            low: Low price series
            period: Channel period (default 20)

        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        upper = high.rolling(window=period, min_periods=period).max()
        lower = low.rolling(window=period, min_periods=period).min()
        middle = (upper + lower) / 2

        return upper, middle, lower

    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_multiplier: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: EMA period (default 20)
            atr_multiplier: ATR multiplier (default 2)

        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        middle = close.ewm(span=period, adjust=False).mean()
        atr = Indicators.atr(high, low, close, period)

        upper = middle + (atr_multiplier * atr)
        lower = middle - (atr_multiplier * atr)

        return upper, middle, lower

    @staticmethod
    def supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Supertrend indicator.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 10)
            multiplier: ATR multiplier (default 3)

        Returns:
            Tuple of (supertrend_line, direction) where direction is 1 for uptrend, -1 for downtrend
        """
        atr = Indicators.atr(high, low, close, period)
        hl2 = (high + low) / 2

        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1

        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1

            # Adjust bands based on previous values
            if direction.iloc[i] == 1 and lower_band.iloc[i] < supertrend.iloc[i - 1]:
                supertrend.iloc[i] = supertrend.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = supertrend.iloc[i - 1]

        return supertrend, direction
