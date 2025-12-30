# Backtest Results - Volatility Momentum Strategy

## Overview

This document contains the results of a parameter sweep for the Volatility Momentum strategy, tested over 2 years of BTC/USDT 1-minute data (2023-12-30 to 2025-12-30).

**Benchmark (Buy & Hold):** +107.48%

---

## Parameter Variations Tested

| # | Name | Description |
|---|------|-------------|
| 1 | Baseline | Current default parameters - 5% SL/TP, 2.0 SD momentum threshold |
| 2 | Asymmetric SL/TP (4%/7%) | Tighter stop loss (4%) with wider take profit (7%) to account for execution slippage and let winners run |
| 3 | High Momentum (2.5 SD) | Increased momentum threshold to 2.5 SD - only enter on more extreme price moves |
| 4 | Very High Momentum (3.0 SD) | Further increased momentum threshold to 3.0 SD - only enter on very rare, extreme moves |
| 5 | Tight Vol (0.5 SD) | Restricted volatility threshold to 0.5 SD - only enter when market is very calm |
| 6 | Wide Vol (1.5 SD) | Relaxed volatility threshold to 1.5 SD - allow entry in wider range of volatility conditions |
| 7 | Longer Momentum (48h) | Extended momentum measurement period from 24h to 48h - capture larger moves |
| 8 | High Mom + Asym SL/TP | Combined high momentum (2.5 SD) with asymmetric SL/TP (4%/7%) |
| 9 | Tight Vol + High Mom | Combined tight volatility (0.5 SD) with high momentum (2.5 SD) |
| 10 | Aggressive (3%/10%) | Very tight stop loss (3%) with wide take profit (10%) - cut losses fast, let winners run big |

---

## Full Results Table

| # | Variation | vol_lookback | vol_window | mom_period | mom_lookback | vol_thresh | mom_thresh | SL | TP | Return | Max DD | Sharpe | Trades | Win Rate | Avg Win | Avg Loss | PF |
|---|-----------|--------------|------------|------------|--------------|------------|------------|-----|-----|--------|--------|--------|--------|----------|---------|----------|------|
| 1 | Baseline | 168 | 24 | 24 | 168 | 1.0 | 2.0 | 5% | 5% | -0.20% | -0.70% | -0.04 | 179 | 52.51% | 4.86% | -5.37% | 0.94 |
| 2 | Asymmetric SL/TP | 168 | 24 | 24 | 168 | 1.0 | 2.0 | 4% | 7% | -0.66% | -0.75% | -0.92 | 169 | 34.32% | 6.86% | -4.34% | 0.81 |
| 3 | **High Momentum** | 168 | 24 | 24 | 168 | 1.0 | **2.5** | 5% | 5% | **+1.58%** | -0.44% | **1.22** | 169 | **59.17%** | 4.86% | -5.36% | **1.34** |
| 4 | Very High Momentum | 168 | 24 | 24 | 168 | 1.0 | **3.0** | 5% | 5% | -0.36% | -0.56% | -0.35 | 150 | 50.00% | 4.95% | -5.29% | 0.88 |
| 5 | Tight Vol | 168 | 24 | 24 | 168 | **0.5** | 2.0 | 5% | 5% | -0.42% | -0.64% | -0.39 | 168 | 50.00% | 4.91% | -5.30% | 0.87 |
| 6 | Wide Vol | 168 | 24 | 24 | 168 | **1.5** | 2.0 | 5% | 5% | -0.74% | -0.83% | -1.21 | 181 | 46.41% | 4.88% | -5.38% | 0.79 |
| 7 | Longer Momentum | 168 | 24 | **48** | 168 | 1.0 | 2.0 | 5% | 5% | -0.49% | -0.64% | -0.49 | 180 | 50.00% | 4.88% | -5.36% | 0.87 |
| 8 | High Mom + Asym | 168 | 24 | 24 | 168 | 1.0 | **2.5** | **4%** | **7%** | +0.17% | -0.67% | 0.36 | 141 | 41.13% | 6.84% | -4.34% | 1.06 |
| 9 | Tight Vol + High Mom | 168 | 24 | 24 | 168 | **0.5** | **2.5** | 5% | 5% | -0.30% | -0.67% | -0.22 | 143 | 50.35% | 4.93% | -5.25% | 0.89 |
| 10 | Aggressive | 168 | 24 | 24 | 168 | 1.0 | 2.0 | **3%** | **10%** | -0.56% | -0.74% | -0.67 | 158 | 22.78% | 9.70% | -3.35% | 0.79 |

**Bold** = changed from baseline or best performer

---

## Key Findings

### Best Performer: High Momentum (2.5 SD)

The only profitable variation was **#3 - High Momentum (2.5 SD)** with:
- **+1.58% return** (vs -0.20% baseline)
- **59.17% win rate** (vs 52.51% baseline)
- **1.34 profit factor** (vs 0.94 baseline)
- **1.22 Sharpe ratio** (only positive Sharpe)

### Why It Works

Increasing the momentum threshold from 2.0 to 2.5 SD:
1. **Fewer but better trades** - 169 trades vs 179 (6% fewer)
2. **Higher win rate** - 59.17% vs 52.51% (+6.66 percentage points)
3. **Same avg win/loss** - Both ~4.86% / -5.36%

The improvement comes entirely from **better signal quality** - waiting for more extreme momentum produces more reliable signals.

### What Didn't Work

| Approach | Why It Failed |
|----------|---------------|
| Asymmetric SL/TP (4%/7%) | Lower win rate (34%) didn't compensate for better R:R ratio |
| Very High Momentum (3.0 SD) | Too selective - missed good opportunities, win rate dropped back to 50% |
| Tight Vol (0.5 SD) | Filtering on calm markets didn't improve signal quality |
| Wide Vol (1.5 SD) | More trades in volatile conditions = worse signals |
| Longer Momentum (48h) | Larger moves didn't produce better entry timing |
| Aggressive (3%/10%) | 22.78% win rate too low - stopped out too frequently |

---

## Execution Analysis

All variations showed consistent slippage between target and actual exits:

| Target | Actual Avg |
|--------|------------|
| 5% TP | ~4.86-4.95% win |
| 5% SL | ~5.25-5.38% loss |
| 4% SL | ~4.34% loss |
| 7% TP | ~6.84-6.86% win |
| 3% SL | ~3.35% loss |
| 10% TP | ~9.70% win |

**Conclusion:** Expect ~0.1-0.4% slippage on exits due to execution at next bar's open.

---

## Recommendations

1. **Use 2.5 SD momentum threshold** - Only improvement that produced positive returns
2. **Keep symmetric 5%/5% SL/TP** - Asymmetric didn't help in this strategy
3. **Keep vol_threshold at 1.0** - Tighter/wider didn't improve results
4. **Consider long-only mode** - Strategy was short 56% of time in a +107% bull market

---

## Files

- Chart: `output/parameter_sweep_YYYYMMDD_HHMMSS.html`
- Detailed CSV: `output/parameter_sweep_detailed.csv`
- Variable documentation: `docs/variables/`
