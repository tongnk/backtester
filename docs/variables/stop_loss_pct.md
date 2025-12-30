# stop_loss_pct

## Description

The percentage loss at which a position is automatically closed to limit downside. Expressed as a decimal (e.g., 0.05 = 5%).

## How It Works

1. When a position is opened, the entry price is recorded
2. At each bar, the unrealized P&L percentage is calculated:
   - **Long**: `(current_price - entry_price) / entry_price`
   - **Short**: `(entry_price - current_price) / entry_price`
3. If unrealized P&L <= -`stop_loss_pct`, the position is closed

## Default Value

- **0.05** (5% stop loss)

## Impact

- **Tighter stops** (e.g., 3%, 2%): Limits losses per trade, but may get stopped out on normal volatility, more trades
- **Wider stops** (e.g., 7%, 10%): Gives trades room to breathe, but larger losses when wrong

## Execution Consideration

In backtesting, the stop is checked on close prices but executed at the next bar's open. This means:
- Actual losses often exceed the stop loss percentage due to slippage
- In the backtest, a 5% stop loss resulted in average losses of ~5.3-5.5%

## Considerations

- Should account for the asset's typical volatility
- BTC can easily move 2-3% in an hour, so very tight stops may not work well
- The relationship between stop loss and take profit affects overall profitability
- Consider setting stops tighter than your actual target to account for execution slippage
