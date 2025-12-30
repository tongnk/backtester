# vol_lookback

## Description

The number of periods (in hours) used to calculate the z-score of realized volatility. This parameter determines how far back the strategy looks to establish what "normal" volatility looks like.

## How It Works

1. The strategy calculates realized volatility at each bar
2. It then computes the rolling mean and standard deviation of volatility over the `vol_lookback` period
3. The z-score is calculated as: `(current_vol - mean) / std`
4. This z-score is compared against `vol_threshold` to determine if volatility is "normal"

## Default Value

- **168** (equivalent to 7 days of hourly data)

## Impact

- **Higher values**: More stable baseline, slower to adapt to regime changes, fewer false signals
- **Lower values**: More responsive to recent conditions, may produce more signals, potentially noisier

## Considerations

- Should be long enough to capture a meaningful distribution of volatility states
- Too short may cause the strategy to constantly adapt, losing the concept of "normal"
- Too long may miss important regime shifts in the market
