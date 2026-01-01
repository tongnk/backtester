"""IV-Momentum Strategy - Combines volatility normality with price momentum breakouts."""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..base import Strategy, StrategyResult, Signal


class IVMomentumStrategy(Strategy):
    """
    Volatility-Momentum Strategy with configurable direction and exit modes.

    Entry logic:
    - Calculate implied volatility (Deribit DVOL) z-score over a rolling window (default: 7 days)
    - Calculate 24-hour rolling price momentum (% change)
    - Map momentum values into z-score over 7-day lookback

    Entry conditions:
    - Volatility is "normal" (z-score within ±1 SD)
    - Price momentum is extreme (z-score > 2 SD for long, < -2 SD for short)

    Exit modes (in priority order):
    1. Time-based exit: Exit after hold_hours (if set)
    2. Stop loss: Fixed % or ATR-based
    3. Take profit: Fixed % or vol-based

    Parameters:
        # Direction filtering
        direction: Trade direction - "both", "long_only", or "short_only" (default: "both")

        # Entry parameters
        vol_lookback: Lookback periods for IV z-score calculation (default: 168 = 7 days hourly)
        momentum_period: Period for momentum calculation in hours (default: 24)
        momentum_lookback: Lookback for momentum z-score (default: 168 = 7 days)
        vol_threshold: Volatility must be within this many SDs to enter (default: 1.0)
        momentum_threshold: Momentum z-score threshold for entry (default: 2.0)

        # Exit parameters - Simple mode
        hold_hours: Fixed hold period in hours, None to disable (default: None)
        stop_loss_pct: Stop loss percentage (default: 0.05 = 5%)
        take_profit_pct: Take profit percentage (default: 0.05 = 5%)

        # Exit parameters - Advanced mode
        use_atr_stops: Use ATR-based stops - 'sl_only', 'both', or False (default: False)
        atr_period: ATR calculation period on hourly data (default: 14 hours)
        sl_atr_mult: Stop loss multiplier for ATR (default: 3.0)
        tp_sd_fraction: TP as fraction of 1 SD expected move, None to use fixed (default: None)
        sl_sd_fraction: SL as fraction of 1 SD expected move, None to use fixed (default: None)

        # Pyramiding (disabled by default)
        pyramid_enabled: Enable pyramiding on winning trades (default: False)
        pyramid_threshold: Profit % required to pyramid (default: 0.025 = 2.5%)
        trail_to_breakeven: Move SL to breakeven after pyramid (default: True)
    """

    @property
    def name(self) -> str:
        return "Volatility Momentum"

    @property
    def description(self) -> str:
        direction = self.params.get("direction", "both")
        hold_hours = self.params.get("hold_hours", None)

        dir_str = {"both": "long/short", "long_only": "long-only", "short_only": "short-only"}[direction]

        if hold_hours:
            return f"IV-momentum {dir_str}, {hold_hours}h time exit"
        return f"IV-momentum {dir_str} with SL/TP"

    def _validate_params(self) -> None:
        direction = self.params.get("direction", "both")
        vol_threshold = self.params.get("vol_threshold", 1.0)
        momentum_threshold = self.params.get("momentum_threshold", 2.0)
        hold_hours = self.params.get("hold_hours", None)

        if direction not in ("both", "long_only", "short_only"):
            raise ValueError(f"direction must be 'both', 'long_only', or 'short_only', got '{direction}'")
        if vol_threshold <= 0:
            raise ValueError(f"vol_threshold ({vol_threshold}) must be positive")
        if momentum_threshold <= 0:
            raise ValueError(f"momentum_threshold ({momentum_threshold}) must be positive")
        if hold_hours is not None and hold_hours <= 0:
            raise ValueError(f"hold_hours ({hold_hours}) must be positive")

    def get_required_columns(self) -> list[str]:
        """Requires OHLCV plus IV data."""
        return ["open", "high", "low", "close", "volume", "iv"]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility z-score, momentum z-score, and ATR."""
        vol_lookback = self.params.get("vol_lookback", 168)  # 7 days in hours
        momentum_period = self.params.get("momentum_period", 24)  # 24 hours
        momentum_lookback = self.params.get("momentum_lookback", 168)  # 7 days
        atr_period = self.params.get("atr_period", 14)

        if "iv" not in df.columns or df["iv"].isna().all():
            raise ValueError("IV data is required for IVMomentumStrategy but is missing or empty.")

        # Use Deribit DVOL implied volatility (convert percent to decimal if needed).
        vol = df["iv"].copy()
        if vol.max() > 3:
            vol = vol / 100.0

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

        # Calculate ATR (Average True Range) on hourly-resampled data
        # This gives meaningful ATR values for stop calculation on minute data
        hourly = df.resample("1h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna()

        high_h = hourly["high"]
        low_h = hourly["low"]
        close_h = hourly["close"]

        tr1 = high_h - low_h  # Current high - current low
        tr2 = abs(high_h - close_h.shift(1))  # Current high - previous close
        tr3 = abs(low_h - close_h.shift(1))  # Current low - previous close
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        hourly_atr = tr.rolling(window=atr_period, min_periods=atr_period // 2).mean()

        # Map hourly ATR back to minute bars using forward fill
        df["atr"] = hourly_atr.reindex(df.index, method="ffill")

        # Calculate vol-based TP and SL (fraction of 1 SD expected move)
        tp_sd_fraction = self.params.get("tp_sd_fraction", 0.5)
        sl_sd_fraction = self.params.get("sl_sd_fraction", 0.5)
        take_profit_pct = self.params.get("take_profit_pct", 0.05)
        stop_loss_pct = self.params.get("stop_loss_pct", 0.05)

        hours_per_year = 8760
        # Convert annualized vol to expected move over momentum_period (24h)
        # 1 SD move = annualized_vol * sqrt(hours / hours_per_year)
        expected_move_1sd = vol * np.sqrt(momentum_period / hours_per_year)

        # Vol-based TP
        if tp_sd_fraction is not None and tp_sd_fraction > 0:
            df["vol_tp"] = expected_move_1sd * tp_sd_fraction
            df["vol_tp"] = df["vol_tp"].fillna(take_profit_pct)
        else:
            df["vol_tp"] = take_profit_pct

        # Vol-based SL
        if sl_sd_fraction is not None and sl_sd_fraction > 0:
            df["vol_sl"] = expected_move_1sd * sl_sd_fraction
            df["vol_sl"] = df["vol_sl"].fillna(stop_loss_pct)
        else:
            df["vol_sl"] = stop_loss_pct

        # Apply floor to vol-based TP/SL if enabled
        vol_floor_enabled = self.params.get("vol_floor_enabled", False)
        if vol_floor_enabled:
            tp_floor_pct = self.params.get("tp_floor_pct", 0.025)
            sl_floor_pct = self.params.get("sl_floor_pct", 0.015)
            df["vol_tp"] = df["vol_tp"].clip(lower=tp_floor_pct)
            df["vol_sl"] = df["vol_sl"].clip(lower=sl_floor_pct)

        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """Generate signals based on volatility normality and momentum extremes."""
        signals = pd.Series(index=df.index, dtype=object)
        signals[:] = Signal.HOLD

        # Direction filtering
        direction = self.params.get("direction", "both")
        allow_long = direction in ("both", "long_only")
        allow_short = direction in ("both", "short_only")

        # Entry parameters
        vol_threshold = self.params.get("vol_threshold", 1.0)
        momentum_threshold = self.params.get("momentum_threshold", 2.0)

        # Exit parameters - simple mode
        hold_hours = self.params.get("hold_hours", None)  # Time-based exit
        stop_loss_pct = self.params.get("stop_loss_pct", 0.05)
        take_profit_pct = self.params.get("take_profit_pct", 0.05)

        # Exit parameters - advanced mode
        use_atr_stops = self.params.get("use_atr_stops", False)  # Default to fixed %
        sl_atr_mult = self.params.get("sl_atr_mult", 3.0)
        trail_atr_mult = self.params.get("trail_atr_mult", 6.0)

        # Pyramiding (disabled by default)
        pyramid_enabled = self.params.get("pyramid_enabled", False)
        pyramid_threshold = self.params.get("pyramid_threshold", 0.025)
        trail_to_breakeven = self.params.get("trail_to_breakeven", True)

        vol_zscore = df["vol_zscore"]
        momentum_zscore = df["momentum_zscore"]
        close = df["close"]
        atr = df["atr"]
        vol_tp = df["vol_tp"]
        vol_sl = df["vol_sl"]

        # Volatility is within normal range (within ±threshold SDs)
        vol_normal = (vol_zscore >= -vol_threshold) & (vol_zscore <= vol_threshold)

        # Momentum breakout conditions
        momentum_long = momentum_zscore > momentum_threshold
        momentum_short = momentum_zscore < -momentum_threshold

        # Track position state for SL/TP and pyramiding
        position = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0.0
        pyramided = False
        avg_entry_price = 0.0  # Weighted average after pyramid
        pyramid_add_count = 0

        # ATR-based tracking
        entry_atr = 0.0  # ATR at entry for initial stop
        stop_level = 0.0  # Absolute stop price level
        highest_since_entry = 0.0  # For trailing (long)
        lowest_since_entry = 0.0  # For trailing (short)

        # Dynamic exit state
        entry_bar_index = 0  # For time-based exit
        breakeven_moved = False  # For breakeven after profit (without pyramid)

        for i in range(len(df)):
            current_price = close.iloc[i]
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.0
            current_vol_normal = vol_normal.iloc[i] if not pd.isna(vol_normal.iloc[i]) else False
            current_momentum_long = momentum_long.iloc[i] if not pd.isna(momentum_long.iloc[i]) else False
            current_momentum_short = momentum_short.iloc[i] if not pd.isna(momentum_short.iloc[i]) else False
            current_momentum_zscore = momentum_zscore.iloc[i] if not pd.isna(momentum_zscore.iloc[i]) else 0.0

            # Determine if we use ATR for stops
            use_atr_sl = use_atr_stops in ("sl_only", "both", True) and current_atr > 0
            use_atr_trail = use_atr_stops == "both" and current_atr > 0

            if position == 0:
                # Check for entry (with direction filtering)
                if allow_long and current_vol_normal and current_momentum_long:
                    signals.iloc[i] = Signal.LONG
                    position = 1
                    entry_price = current_price
                    avg_entry_price = current_price
                    pyramided = False
                    entry_atr = current_atr
                    highest_since_entry = current_price
                    lowest_since_entry = current_price
                    entry_bar_index = i  # For time-based exit
                    # Set initial stop level
                    current_vol_sl = vol_sl.iloc[i] if not pd.isna(vol_sl.iloc[i]) else stop_loss_pct
                    if use_atr_sl:
                        stop_level = entry_price - (sl_atr_mult * current_atr)
                    else:
                        stop_level = entry_price * (1 - current_vol_sl)
                elif allow_short and current_vol_normal and current_momentum_short:
                    signals.iloc[i] = Signal.SHORT
                    position = -1
                    entry_price = current_price
                    avg_entry_price = current_price
                    pyramided = False
                    entry_atr = current_atr
                    highest_since_entry = current_price
                    lowest_since_entry = current_price
                    entry_bar_index = i  # For time-based exit
                    # Set initial stop level
                    current_vol_sl = vol_sl.iloc[i] if not pd.isna(vol_sl.iloc[i]) else stop_loss_pct
                    if use_atr_sl:
                        stop_level = entry_price + (sl_atr_mult * current_atr)
                    else:
                        stop_level = entry_price * (1 + current_vol_sl)
            else:
                # Update highest/lowest since entry
                highest_since_entry = max(highest_since_entry, current_price)
                lowest_since_entry = min(lowest_since_entry, current_price)

                # Calculate current PnL
                if position == 1:  # Long position
                    pnl_pct = (current_price - avg_entry_price) / avg_entry_price
                    pnl_from_entry = (current_price - entry_price) / entry_price
                else:  # Short position
                    pnl_pct = (avg_entry_price - current_price) / avg_entry_price
                    pnl_from_entry = (entry_price - current_price) / entry_price

                # Calculate effective stop level
                if use_atr_trail:
                    if position == 1:
                        trail_stop = highest_since_entry - (trail_atr_mult * current_atr)
                        effective_stop = max(stop_level, trail_stop)
                    else:
                        trail_stop = lowest_since_entry + (trail_atr_mult * current_atr)
                        effective_stop = min(stop_level, trail_stop)
                    if pyramided and trail_to_breakeven:
                        if position == 1:
                            effective_stop = max(effective_stop, avg_entry_price)
                        else:
                            effective_stop = min(effective_stop, avg_entry_price)
                else:
                    if pyramided and trail_to_breakeven:
                        if position == 1:
                            effective_stop = max(stop_level, avg_entry_price)
                        else:
                            effective_stop = min(stop_level, avg_entry_price)
                    else:
                        effective_stop = stop_level

                # Helper to reset position state
                def reset_position():
                    nonlocal position, entry_price, avg_entry_price, pyramided
                    nonlocal stop_level, highest_since_entry, lowest_since_entry, entry_bar_index
                    position = 0
                    entry_price = 0.0
                    avg_entry_price = 0.0
                    pyramided = False
                    stop_level = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    entry_bar_index = 0

                # EXIT PRIORITY ORDER:
                # 1. Time-based exit (if hold_hours is set)
                # 2. Stop loss hit
                # 3. Take profit hit
                # 4. Pyramid (if enabled)

                exited = False

                # 1. Time-based exit
                if hold_hours is not None:
                    bars_held = i - entry_bar_index
                    hours_held = bars_held / 60
                    if hours_held >= hold_hours:
                        signals.iloc[i] = Signal.CLOSE
                        reset_position()
                        exited = True

                # 2. Stop loss check
                if not exited:
                    if position == 1 and current_price <= effective_stop:
                        signals.iloc[i] = Signal.CLOSE
                        reset_position()
                        exited = True
                    elif position == -1 and current_price >= effective_stop:
                        signals.iloc[i] = Signal.CLOSE
                        reset_position()
                        exited = True

                # 3. Take profit check (only if no time exit and not using ATR trailing)
                if not exited and hold_hours is None and not use_atr_trail:
                    tp_target = vol_tp.iloc[i] if not pd.isna(vol_tp.iloc[i]) else take_profit_pct
                    if pnl_pct >= tp_target:
                        signals.iloc[i] = Signal.CLOSE
                        reset_position()
                        exited = True

                # 4. Pyramid check
                if not exited and pyramid_enabled and not pyramided and pnl_from_entry >= pyramid_threshold:
                    signals.iloc[i] = Signal.ADD
                    pyramided = True
                    pyramid_add_count += 1
                    avg_entry_price = (entry_price + current_price) / 2
                    if trail_to_breakeven:
                        if position == 1:
                            stop_level = max(stop_level, avg_entry_price)
                        else:
                            stop_level = min(stop_level, avg_entry_price)

        indicators = {
            "volatility": df["volatility"],
            "vol_zscore": df["vol_zscore"],
            "vol_mean": df["vol_mean"],
            "momentum_24h": df["momentum_24h"],
            "momentum_zscore": df["momentum_zscore"],
            "momentum_mean": df["momentum_mean"],
            "atr": df["atr"],
            "vol_tp": df["vol_tp"],
            "vol_sl": df["vol_sl"],
        }

        # Count signals for metadata
        long_entries = (signals == Signal.LONG).sum()
        short_entries = (signals == Signal.SHORT).sum()
        adds = (signals == Signal.ADD).sum()
        closes = (signals == Signal.CLOSE).sum()

        metadata = {
            "direction": direction,
            "hold_hours": hold_hours,
            "long_entries": long_entries,
            "short_entries": short_entries,
            "pyramid_adds": adds,
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
        atr_period = self.params.get("atr_period", 14)
        return max(vol_lookback, momentum_lookback + momentum_period, atr_period)
