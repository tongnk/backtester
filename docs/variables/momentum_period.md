# momentum_period

## Description

The number of periods (in hours) over which price momentum is measured. This determines the lookback for calculating the percentage price change that defines "momentum."

## How It Works

1. Momentum is calculated as the percentage change in price over `momentum_period` bars
2. Formula: `(current_price - price_N_periods_ago) / price_N_periods_ago * 100`
3. This raw momentum value is then converted to a z-score using `momentum_lookback`

## Default Value

- **24** (equivalent to 24 hours of price change)

## Impact

- **Higher values** (e.g., 48, 72): Captures larger, more sustained moves; filters out noise; fewer signals
- **Lower values** (e.g., 12, 6): More responsive to short-term moves; more signals; potentially more false positives

## Considerations

- 24 hours captures a full daily cycle, common in crypto due to 24/7 trading
- Shorter periods may catch quick reversals but also more noise
- Longer periods identify stronger trends but may enter late
- Should align with your intended holding period (shorter momentum = shorter trades)
