# momentum_lookback

## Description

The number of periods (in hours) used to calculate the z-score of momentum. This parameter determines how far back the strategy looks to establish what "normal" momentum looks like, allowing it to identify when momentum is statistically extreme.

## How It Works

1. The strategy calculates raw momentum (price change %) at each bar
2. It computes the rolling mean and standard deviation of momentum over `momentum_lookback` periods
3. The z-score is calculated as: `(current_momentum - mean) / std`
4. This z-score is compared against `momentum_threshold` to identify extreme moves

## Default Value

- **168** (equivalent to 7 days of hourly data)

## Impact

- **Higher values**: More conservative baseline, only truly exceptional moves register as extreme
- **Lower values**: More adaptive to recent conditions, may flag moves as "extreme" more often

## Considerations

- Should be long enough to establish a meaningful distribution of momentum values
- 7 days (168 hours) captures roughly a week of market behavior
- Too short a lookback may cause the strategy to constantly recalibrate, reducing signal quality
- Should generally match or exceed `vol_lookback` for consistency
