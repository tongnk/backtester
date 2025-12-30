"""IV-Momentum Strategy - Combines volatility normality with price momentum breakouts."""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..base import Strategy, StrategyResult, Signal


class IVMomentumStrategy(Strategy):
    """
    Volatility-Momentum Strategy.

    Entry logic:
    - Calculate realized volatility z-score over a rolling window (default: 7 days)
    - Calculate 24-hour rolling price momentum (% change)
    - Map momentum values into z-score over 7-day lookback

    Entry conditions:
    - Volatility is "normal" (z-score within ±1 SD)
    - Price momentum is extreme (z-score > 2 SD for long, < -2 SD for short)

    Exit logic:
    - Stop loss at configurable % (default: 5%)
    - Take profit at configurable % (default: 5%)

    Parameters:
        vol_lookback: Lookback periods for volatility z-score calculation (default: 168 = 7 days hourly)
        vol_window: Window for realized volatility calculation (default: 24 = 24 hours)
        momentum_period: Period for 24h momentum calculation (default: 24 hours)
        momentum_lookback: Lookback for momentum z-score (default: 168 = 7 days)
        vol_threshold: Volatility must be within this many SDs to enter (default: 1.0)
        momentum_threshold: Momentum z-score threshold for entry (default: 2.0)
        stop_loss_pct: Stop loss percentage (default: 0.05 = 5%)
        take_profit_pct: Take profit percentage (default: 0.05 = 5%)
        use_iv: If True and IV data available, use IV instead of realized vol (default: False)
    """

    @property
    def name(self) -> str:
        return "Volatility Momentum"

    @property
    def description(self) -> str:
        return "Enter on strong momentum when volatility is within normal range, with 5% SL/TP"

    def _validate_params(self) -> None:
        vol_threshold = self.params.get("vol_threshold", 1.0)
        momentum_threshold = self.params.get("momentum_threshold", 2.0)

        if vol_threshold <= 0:
            raise ValueError(f"vol_threshold ({vol_threshold}) must be positive")
        if momentum_threshold <= 0:
            raise ValueError(f"momentum_threshold ({momentum_threshold}) must be positive")

    def get_required_columns(self) -> list[str]:
        """Only requires OHLCV - IV is optional."""
        return ["open", "high", "low", "close", "volume"]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility z-score and momentum z-score."""
        vol_lookback = self.params.get("vol_lookback", 168)  # 7 days in hours
        vol_window = self.params.get("vol_window", 24)  # 24 hours for realized vol
        momentum_period = self.params.get("momentum_period", 24)  # 24 hours
        momentum_lookback = self.params.get("momentum_lookback", 168)  # 7 days
        use_iv = self.params.get("use_iv", False)

        # Calculate volatility - use IV if available and requested, otherwise realized vol
        if use_iv and "iv" in df.columns and df["iv"].notna().sum() > len(df) * 0.5:
            vol = df["iv"]
        else:
            # Calculate realized volatility from returns (annualized)
            returns = df["close"].pct_change()
            vol = returns.rolling(window=vol_window, min_periods=vol_window // 2).std() * np.sqrt(525600)  # Annualize for 1-min data

        df["volatility"] = vol

        # Calculate volatility z-score
        vol_mean = vol.rolling(window=vol_lookback, min_periods=vol_lookback // 2).mean()
        vol_std = vol.rolling(window=vol_lookback, min_periods=vol_lookback // 2).std()
        df["vol_zscore"] = (vol - vol_mean) / vol_std
        df["vol_mean"] = vol_mean
        df["vol_std"] = vol_std

        # Calculate 24-hour momentum (percentage change)
        df["momentum_24h"] = df["close"].pct_change(periods=momentum_period) * 100

        # Calculate z-score of momentum over 7-day lookback
        momentum = df["momentum_24h"]
        momentum_mean = momentum.rolling(window=momentum_lookback, min_periods=momentum_lookback // 2).mean()
        momentum_std = momentum.rolling(window=momentum_lookback, min_periods=momentum_lookback // 2).std()
        df["momentum_zscore"] = (momentum - momentum_mean) / momentum_std
        df["momentum_mean"] = momentum_mean
        df["momentum_std"] = momentum_std

        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """Generate signals based on volatility normality and momentum extremes."""
        signals = pd.Series(index=df.index, dtype=object)
        signals[:] = Signal.HOLD

        vol_threshold = self.params.get("vol_threshold", 1.0)
        momentum_threshold = self.params.get("momentum_threshold", 2.0)
        stop_loss_pct = self.params.get("stop_loss_pct", 0.05)
        take_profit_pct = self.params.get("take_profit_pct", 0.05)

        vol_zscore = df["vol_zscore"]
        momentum_zscore = df["momentum_zscore"]
        close = df["close"]

        # Volatility is within normal range (within ±threshold SDs)
        vol_normal = (vol_zscore >= -vol_threshold) & (vol_zscore <= vol_threshold)

        # Momentum breakout conditions
        momentum_long = momentum_zscore > momentum_threshold
        momentum_short = momentum_zscore < -momentum_threshold

        # Track position state for SL/TP
        position = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0.0

        for i in range(len(df)):
            current_price = close.iloc[i]
            current_vol_normal = vol_normal.iloc[i] if not pd.isna(vol_normal.iloc[i]) else False
            current_momentum_long = momentum_long.iloc[i] if not pd.isna(momentum_long.iloc[i]) else False
            current_momentum_short = momentum_short.iloc[i] if not pd.isna(momentum_short.iloc[i]) else False

            if position == 0:
                # Check for entry
                if current_vol_normal and current_momentum_long:
                    signals.iloc[i] = Signal.LONG
                    position = 1
                    entry_price = current_price
                elif current_vol_normal and current_momentum_short:
                    signals.iloc[i] = Signal.SHORT
                    position = -1
                    entry_price = current_price
            else:
                # Check for SL/TP exit
                if position == 1:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                        signals.iloc[i] = Signal.CLOSE
                        position = 0
                        entry_price = 0.0
                elif position == -1:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                    if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                        signals.iloc[i] = Signal.CLOSE
                        position = 0
                        entry_price = 0.0

        indicators = {
            "volatility": df["volatility"],
            "vol_zscore": df["vol_zscore"],
            "vol_mean": df["vol_mean"],
            "momentum_24h": df["momentum_24h"],
            "momentum_zscore": df["momentum_zscore"],
            "momentum_mean": df["momentum_mean"],
        }

        # Count signals for metadata
        long_entries = (signals == Signal.LONG).sum()
        short_entries = (signals == Signal.SHORT).sum()
        closes = (signals == Signal.CLOSE).sum()

        metadata = {
            "long_entries": long_entries,
            "short_entries": short_entries,
            "total_closes": closes,
            "vol_normal_periods": vol_normal.sum(),
            "momentum_long_periods": momentum_long.sum(),
            "momentum_short_periods": momentum_short.sum(),
        }

        return StrategyResult(signals=signals, indicators=indicators, metadata=metadata)

    def warmup_period(self) -> int:
        vol_lookback = self.params.get("vol_lookback", 168)
        momentum_lookback = self.params.get("momentum_lookback", 168)
        momentum_period = self.params.get("momentum_period", 24)
        return max(vol_lookback, momentum_lookback + momentum_period)
