# Strategy Variables Reference

## Quick Reference Table

| Variable | Default | Description | Section |
|----------|---------|-------------|---------|
| [vol_lookback](#vol_lookback) | 168 | Hours for volatility z-score calculation (7 days) | Entry |
| [vol_window](#vol_window) | 24 | Hours for realized vol calculation | Entry |
| [vol_threshold](#vol_threshold) | 1.0 | Max vol z-score to allow entry (±1 SD) | Entry |
| [momentum_period](#momentum_period) | 24 | Hours for momentum calculation | Entry |
| [momentum_lookback](#momentum_lookback) | 168 | Hours for momentum z-score (7 days) | Entry |
| [momentum_threshold](#momentum_threshold) | 2.5 | Min momentum z-score for entry | Entry |
| [stop_loss_pct](#stop_loss_pct) | 0.05 | Fixed stop loss percentage (5%) | Exit |
| [take_profit_pct](#take_profit_pct) | 0.05 | Fixed take profit percentage (5%) | Exit |
| [tp_sd_fraction](#tp_sd_fraction) | 0.5 | TP as fraction of 1 SD expected move | Exit |
| [pyramid_enabled](#pyramid_enabled) | True | Enable pyramiding on winners | Position |
| [pyramid_threshold](#pyramid_threshold) | 0.025 | Profit % to trigger pyramid (2.5%) | Position |
| [trail_to_breakeven](#trail_to_breakeven) | True | Move SL to breakeven after pyramid | Position |
| [use_atr_stops](#use_atr_stops) | False | Use ATR-based stops | Exit |
| [atr_period](#atr_period) | 14 | ATR calculation period (hours) | Exit |
| [sl_atr_mult](#sl_atr_mult) | 3.0 | SL multiplier for ATR | Exit |
| [trail_atr_mult](#trail_atr_mult) | 6.0 | Trailing stop multiplier for ATR | Exit |
| [use_iv](#use_iv) | False | Use IV instead of realized vol | Entry |

---

## Entry Conditions

### vol_lookback

Lookback period (in hours) for calculating the volatility z-score.

- **Default**: 168 (7 days)
- **Impact**: Longer = smoother z-score, fewer signals. Shorter = more reactive.
- **Used for**: Determining if current volatility is "normal" (within ±vol_threshold SDs)

### vol_window

Window (in hours) for calculating realized volatility from returns.

- **Default**: 24 (1 day)
- **Calculation**: `std(returns) * sqrt(525600)` to annualize from minute data
- **Impact**: Shorter = more reactive to recent vol spikes

### vol_threshold

Maximum allowed volatility z-score for entry. Only enter when vol is within this range.

- **Default**: 1.0 (within ±1 SD of mean)
- **Impact**:
  - Lower (0.5): Only enter in very calm markets
  - Higher (1.5): Allow entry in wider range of vol conditions
- **Tested**: 0.5 and 1.5 both underperformed default 1.0

### momentum_period

Period (in hours) for calculating price momentum (% change).

- **Default**: 24 (24-hour momentum)
- **Calculation**: `(price_now - price_24h_ago) / price_24h_ago`
- **Impact**: Shorter = more signals, captures smaller moves. Longer = fewer signals.

### momentum_lookback

Lookback period (in hours) for calculating momentum z-score.

- **Default**: 168 (7 days)
- **Used for**: Mapping raw momentum into z-score to identify "extreme" moves

### momentum_threshold

Minimum momentum z-score required for entry.

- **Default**: 2.5 (>2.5 SD for long, <-2.5 SD for short)
- **Impact**:
  - Lower (2.0): More signals, lower win rate
  - Higher (3.0): Fewer signals, not enough data
- **Best performer**: 2.5 SD showed +1.58% return vs -0.20% for 2.0 SD

---

## Exit Conditions

### stop_loss_pct

Fixed stop loss percentage from entry price.

- **Default**: 0.05 (5%)
- **Calculation**: Long SL = entry × (1 - 0.05), Short SL = entry × (1 + 0.05)
- **Note**: Overridden by ATR-based SL if `use_atr_stops` is enabled

### take_profit_pct

Fixed take profit percentage from entry/average price.

- **Default**: 0.05 (5%)
- **Note**: Overridden by `tp_sd_fraction` if set > 0

### tp_sd_fraction

Take profit as a fraction of the 1 SD expected 24h move.

- **Default**: 0.5 (0.5 × 1 SD move)
- **Calculation**:
  ```
  1 SD move = annualized_vol × sqrt(24h / 8760h)
  TP target = 1 SD move × tp_sd_fraction
  ```
- **Expected hit rates**:
  | Fraction | Expected Hit Rate |
  |----------|-------------------|
  | 0.25 | ~80% |
  | 0.50 | ~62% |
  | 0.75 | ~45% |
  | 1.00 | ~32% |

### use_atr_stops

Enable ATR-based stops instead of fixed percentages.

- **Default**: False
- **Options**:
  - `False`: Use fixed % stops
  - `"sl_only"`: ATR for SL, fixed for TP
  - `"both"`: ATR for both SL and trailing TP
- **Note**: ATR trailing tested and underperformed fixed % (see learnings.md)

### atr_period

Period for ATR calculation (on hourly-resampled data).

- **Default**: 14 hours
- **Note**: ATR calculated on hourly data, then forward-filled to minute bars

### sl_atr_mult

Stop loss distance as multiple of ATR.

- **Default**: 3.0 (SL = entry ± 3×ATR)
- **Tested**: 2x, 3x, 4x - all underperformed fixed 5%

### trail_atr_mult

Trailing stop distance as multiple of ATR (when `use_atr_stops="both"`).

- **Default**: 6.0 (trail = high/low ± 6×ATR)
- **Tested**: 4x, 6x, 8x, 10x - all underperformed fixed TP

---

## Position Management

### pyramid_enabled

Enable adding to winning positions.

- **Default**: True
- **Behavior**: When profit reaches `pyramid_threshold`, add equal size to position

### pyramid_threshold

Profit percentage required to trigger pyramid.

- **Default**: 0.025 (2.5%)
- **Behavior**:
  1. Position reaches 2.5% profit from entry
  2. Add equal size at current price
  3. Average entry = (original + current) / 2
  4. If `trail_to_breakeven`, SL moves to avg entry

### trail_to_breakeven

Move stop loss to breakeven (average entry) after pyramiding.

- **Default**: True
- **Benefit**: Protects against giving back the initial profit after adding

---

## Data Source

### use_iv

Use implied volatility from Deribit instead of realized volatility.

- **Default**: False
- **Note**: IV data has limited historical coverage (~0.2%). Currently using realized vol as proxy.
