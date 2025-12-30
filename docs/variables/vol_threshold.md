# vol_threshold

## Description

The maximum absolute z-score of volatility allowed for entry. The strategy only enters positions when volatility is "normal" - i.e., when the volatility z-score is within ±`vol_threshold` standard deviations of its mean.

## How It Works

1. The strategy calculates the current volatility z-score
2. Entry is only allowed if: `-vol_threshold <= vol_zscore <= +vol_threshold`
3. This ensures trades are taken when the market is in a "normal" volatility regime

## Default Value

- **1.0** (volatility must be within ±1 standard deviation)

## Impact

- **Higher values** (e.g., 1.5, 2.0): More permissive, allows entry in wider range of volatility conditions, more trades
- **Lower values** (e.g., 0.5, 0.3): More restrictive, only enters in very calm markets, fewer trades

## Rationale

The hypothesis is that momentum signals are more reliable when volatility is normal:
- High volatility periods may see erratic price action and false breakouts
- Low volatility periods may precede explosive moves
- Normal volatility suggests orderly price discovery where momentum is meaningful

## Considerations

- Setting this too tight (e.g., 0.3) may result in very few trades
- Setting this too loose (e.g., 2.0) essentially disables the volatility filter
- The optimal value depends on the asset's volatility characteristics
