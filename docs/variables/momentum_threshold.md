# momentum_threshold

## Description

The minimum absolute z-score of momentum required for entry. The strategy only enters positions when momentum is statistically extreme - i.e., when the momentum z-score exceeds `momentum_threshold` standard deviations.

## How It Works

1. The strategy calculates the current momentum z-score
2. **Long entry**: momentum z-score > +`momentum_threshold`
3. **Short entry**: momentum z-score < -`momentum_threshold`
4. This ensures trades are only taken on statistically significant price moves

## Default Value

- **2.0** (momentum must exceed ±2 standard deviations)

## Impact

- **Higher values** (e.g., 2.5, 3.0): More selective, only enters on extreme moves, fewer trades, potentially higher win rate
- **Lower values** (e.g., 1.5, 1.0): More signals, catches smaller moves, more trades, potentially more false signals

## Statistical Context

| Threshold | % of observations beyond | Interpretation |
|-----------|--------------------------|----------------|
| 1.0 SD | ~32% | Common occurrence |
| 1.5 SD | ~13% | Somewhat unusual |
| 2.0 SD | ~5% | Unusual |
| 2.5 SD | ~1.2% | Rare |
| 3.0 SD | ~0.3% | Very rare |

## Considerations

- 2.0 SD is a common starting point (captures ~5% most extreme moves)
- Higher thresholds reduce noise but may miss valid opportunities
- In backtesting, 2.5 SD showed the best results for this strategy
- The optimal value depends on market conditions and desired trade frequency
