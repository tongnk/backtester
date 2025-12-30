# vol_window

## Description

The number of periods (in hours) used to calculate the realized volatility itself. This is the window over which price returns are measured to compute the standard deviation (volatility).

## How It Works

1. Price returns are calculated as percentage change between bars
2. The standard deviation of returns is computed over the `vol_window` period
3. This value is annualized by multiplying by `sqrt(525600)` (minutes per year for 1-min data)
4. The result represents the current realized volatility

## Default Value

- **24** (equivalent to 24 hours)

## Impact

- **Higher values**: Smoother volatility estimate, less reactive to short-term spikes
- **Lower values**: More responsive to recent price action, captures short-term volatility changes faster

## Considerations

- 24 hours is a common choice as it captures a full trading day cycle
- Shorter windows (e.g., 12 hours) may be better for capturing intraday volatility regimes
- Longer windows (e.g., 48-72 hours) provide more stable estimates but lag behind actual conditions
