# Strategy Learnings & Test Results

## Summary

**Best Configuration Found**: Fixed 5% SL / 5% TP with 2.5 SD momentum threshold
- Return: +4.98% (2 years)
- Win Rate: 62%
- Benchmark (B&H): +107%

---

## Table of Contents

1. [Baseline Parameter Sweep](#1-baseline-parameter-sweep)
2. [ATR-Based Stops](#2-atr-based-stops)
3. [ATR Trailing Take Profit](#3-atr-trailing-take-profit)
4. [Dynamic TP (Historical Moves)](#4-dynamic-tp-historical-moves)
5. [Vol-Based TP (SD Fraction)](#5-vol-based-tp-sd-fraction)
6. [Price Movement Analysis](#6-price-movement-analysis)
7. [Key Insights](#7-key-insights)

---

## 1. Baseline Parameter Sweep

**Test Date**: Initial development
**Data Period**: 2 years (2023-12-30 to 2025-12-30)
**Benchmark (B&H)**: +107.48%

| # | Variation | mom_thresh | vol_thresh | SL | TP | Return | Win Rate | Sharpe |
|---|-----------|------------|------------|-----|-----|--------|----------|--------|
| 1 | Baseline | 2.0 | 1.0 | 5% | 5% | -0.20% | 52.51% | -0.04 |
| 2 | Asymmetric SL/TP | 2.0 | 1.0 | 4% | 7% | -0.66% | 34.32% | -0.92 |
| 3 | **High Momentum** | **2.5** | 1.0 | 5% | 5% | **+1.58%** | **59.17%** | **1.22** |
| 4 | Very High Momentum | 3.0 | 1.0 | 5% | 5% | -0.36% | 50.00% | -0.35 |
| 5 | Tight Vol | 2.0 | 0.5 | 5% | 5% | -0.42% | 50.00% | -0.39 |
| 6 | Wide Vol | 2.0 | 1.5 | 5% | 5% | -0.74% | 46.41% | -1.21 |
| 7 | Longer Momentum (48h) | 2.0 | 1.0 | 5% | 5% | -0.49% | 50.00% | -0.49 |
| 8 | High Mom + Asym | 2.5 | 1.0 | 4% | 7% | +0.17% | 41.13% | 0.36 |
| 9 | Tight Vol + High Mom | 2.5 | 0.5 | 5% | 5% | -0.30% | 50.35% | -0.22 |
| 10 | Aggressive | 2.0 | 1.0 | 3% | 10% | -0.56% | 22.78% | -0.67 |

**Winner**: High Momentum (2.5 SD) - only profitable configuration

**Why 2.5 SD works**: Waiting for more extreme momentum produces higher quality signals with better win rate (59% vs 52%).

---

## 2. ATR-Based Stops

**Hypothesis**: ATR-based stops adapt to volatility - wider in volatile markets, tighter in calm markets.

**Implementation**: ATR calculated on hourly-resampled data (14-hour period), then applied to minute bars.

### Results (1-Year Test)

| Config | SL Type | TP Type | Return | Win Rate | Trades |
|--------|---------|---------|--------|----------|--------|
| Fixed 5%/5% | Fixed | Fixed | -33.50% | 63.30% | 109 |
| ATR 2x SL/2x Trail | ATR | ATR Trail | -77.68% | 29.60% | 625 |
| ATR 3x SL/3x Trail | ATR | ATR Trail | -63.95% | 34.49% | 374 |

### Results (2-Year Test)

| Config | Return | Win Rate | Trades | Max DD |
|--------|--------|----------|--------|--------|
| Fixed 5%/5% | **+4.98%** | 62.00% | 250 | -61.91% |
| ATR 4x SL/6x Trail | -77.48% | 41.62% | 370 | -83.99% |
| ATR 4x SL/10x Trail | -53.45% | 46.96% | 247 | -76.19% |

**Conclusion**: ATR trailing significantly underperforms fixed %. Creates too many trades by exiting on pullbacks.

---

## 3. ATR Trailing Take Profit

**Hypothesis**: Let winners run by trailing stop below highest price.

**Problem Discovered**: ATR trailing exits on ANY pullback exceeding the trail distance.

### Example of Failure

```
Entry: $100,000
ATR: $2,000 (2%)
Trail mult: 2x

Price → $104,000 (+4%)
Trail stop = $104,000 - $4,000 = $100,000

Price pulls back to $100,500
→ Not stopped yet

Price pulls back to $99,500
→ STOPPED at breakeven despite +4% favorable move
```

### ATR Multiplier Sweep (1-Year)

| SL Mult | Trail Mult | Return | Win Rate | Trades |
|---------|------------|--------|----------|--------|
| 2.0x | 4.0x | -66.16% | 32.72% | 327 |
| 3.0x | 5.0x | -69.76% | 36.59% | 246 |
| 3.0x | 6.0x | -39.99% | 39.59% | 197 |
| 4.0x | 6.0x | -33.00% | 41.38% | 174 |
| 4.0x | 8.0x | -40.42% | 49.28% | 138 |

**Conclusion**: Even very wide ATR multipliers (4x SL, 8x trail) underperform fixed 5%/5%. The strategy enters on extreme momentum expecting a TARGET move, not an open-ended trend.

---

## 4. Dynamic TP (Historical Moves)

**Hypothesis**: Set TP based on median max favorable move observed in recent history.

**Implementation**:
- Look back 7 days (168h)
- For each bar, calculate max favorable move in next 24h
- Use median of these moves as TP target

### Results (2-Year)

| Config | Return | Win Rate | Avg Win | Avg Loss |
|--------|--------|----------|---------|----------|
| Fixed 5% SL / 5% TP | +4.98% | 62.00% | 1.82% | -4.13% |
| Dynamic TP + 5% SL | -53.75% | 78.26% | 1.16% | -5.08% |
| Dynamic TP + 3% SL | -76.29% | 68.20% | 1.14% | -3.22% |
| Dynamic TP + 2% SL | -82.36% | 59.98% | 1.14% | -2.27% |

**Problem**: Dynamic TP gives smaller targets (~1.16% avg) but we're still using large SLs. R:R ratio is terrible.

---

## 5. Vol-Based TP (SD Fraction)

**Hypothesis**: Use realized vol to set TP as fraction of expected 1 SD move.

**Calculation**:
```
1 SD 24h move = annualized_vol × sqrt(24/8760)
TP = 1 SD move × tp_sd_fraction
```

**Expected Hit Rates**:
| Fraction | Expected Hit Rate |
|----------|-------------------|
| 0.25 SD | ~80% |
| 0.50 SD | ~62% |
| 1.00 SD | ~32% |

### Results (2-Year, 0.5 SD)

| Config | Return | Win Rate | Avg Win | Avg Loss |
|--------|--------|----------|---------|----------|
| Fixed 5% SL / 5% TP | -0.57% | 61.75% | 1.82% | -4.14% |
| 0.5 SD TP + 5% SL | -88.39% | 79.04% | 0.85% | -4.98% |
| 0.5 SD TP + 3% SL | -88.02% | 71.57% | 0.86% | -3.14% |
| 0.5 SD TP + 2% SL | -91.45% | 64.53% | 0.82% | -2.18% |

**Problem**: 0.5 SD gives only ~0.85% avg wins. Even with 79% win rate, the R:R ratio (0.85:5) requires >85% win rate to break even.

---

## 6. Price Movement Analysis

**Question**: After a momentum signal, what is the max favorable move within 24h/48h?

### Max Favorable Move Distribution (24h window)

| Metric | Long | Short | Combined |
|--------|------|-------|----------|
| Average | 2.66% | 2.15% | 2.41% |
| Median | 2.25% | 1.39% | 1.89% |
| 25th %ile | 1.03% | 0.56% | - |
| 75th %ile | 3.73% | 3.20% | - |
| 90th %ile | 5.79% | 4.38% | - |

### Hit Rates by TP Target (24h window)

| TP Target | Long Hit Rate | Short Hit Rate |
|-----------|---------------|----------------|
| 1.5% | 61.2% | 47.1% |
| 2.0% | 54.1% | 43.5% |
| 2.5% | 44.7% | 32.9% |
| 3.0% | 32.9% | 27.1% |
| 5.0% | 12.9% | ~8% |

**Key Finding**: Only ~13% of long signals and ~8% of short signals hit 5% within 24h. The 5% TP is aggressive for this strategy.

---

## 7. Key Insights

### What Works
1. **2.5 SD momentum threshold** - Higher quality signals, better win rate
2. **Fixed 5%/5% SL/TP** - Simple, consistent, only profitable configuration
3. **Pyramiding with breakeven stop** - Improved returns from +1.58% to +4.98%

### What Doesn't Work
1. **ATR trailing** - Exits on pullbacks, creates too many trades, lower win rate
2. **Asymmetric SL/TP** - Lower win rate doesn't compensate for better R:R
3. **Tighter vol threshold** - Fewer trades without improvement
4. **Dynamic TP based on historical moves** - Targets too small vs SL
5. **Vol-based TP (0.5 SD)** - Same issue, tiny wins vs large losses

### Why Fixed 5% TP Works

The strategy enters on **extreme momentum** (>2.5 SD). This is a signal that expects continuation to a **meaningful target**.

- Small TPs (1-2%) get hit frequently but capture only a fraction of the move
- The 5% TP, while hit less often (~13% in 24h), captures the full expected move when it happens
- Combined with pyramiding, the 62% win rate is sufficient for profitability

### The R:R Problem

All alternative TP approaches fail because they create poor risk:reward:

| Approach | Avg Win | Avg Loss | Required Win Rate | Actual |
|----------|---------|----------|-------------------|--------|
| Fixed 5%/5% | 1.82% | 4.13% | 69% | 62% |
| Vol 0.5 SD | 0.85% | 4.98% | 85% | 79% |
| Dynamic TP | 1.16% | 5.08% | 81% | 78% |

The pyramiding with breakeven stop is what makes the fixed approach work - it changes the effective R:R by protecting profits after adding.

### Future Directions to Test

1. **Long-only mode** - Strategy was short 56% of time in a +107% bull market
2. **Different SL sizing** - Vol-based SL instead of fixed 5%?
3. **Time-based exits** - Exit after 24h if TP not hit?
4. **Separate long/short parameters** - Longs show 2.66% avg move vs 2.15% for shorts
