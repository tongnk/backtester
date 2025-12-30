# take_profit_pct

## Description

The percentage profit at which a position is automatically closed to lock in gains. Expressed as a decimal (e.g., 0.05 = 5%).

## How It Works

1. When a position is opened, the entry price is recorded
2. At each bar, the unrealized P&L percentage is calculated:
   - **Long**: `(current_price - entry_price) / entry_price`
   - **Short**: `(entry_price - current_price) / entry_price`
3. If unrealized P&L >= `take_profit_pct`, the position is closed

## Default Value

- **0.05** (5% take profit)

## Impact

- **Tighter targets** (e.g., 3%, 2%): Locks in profits quickly, higher win rate, but may miss larger moves
- **Wider targets** (e.g., 7%, 10%): Captures bigger moves, but lower win rate, more time at risk

## Execution Consideration

In backtesting, the take profit is checked on close prices but executed at the next bar's open. This means:
- Actual profits may be slightly less than the target due to price movement
- In the backtest, a 5% take profit resulted in average wins of ~4.8-4.9%

## Risk-Reward Ratio

The ratio of `take_profit_pct` to `stop_loss_pct` affects strategy dynamics:

| SL | TP | Ratio | Implication |
|----|-----|-------|-------------|
| 5% | 5% | 1:1 | Need >50% win rate to profit |
| 4% | 7% | 1:1.75 | Need >36% win rate to profit |
| 3% | 10% | 1:3.3 | Need >23% win rate to profit |

## Considerations

- Asymmetric SL/TP (wider TP than SL) can be profitable even with lower win rates
- The optimal ratio depends on the strategy's natural win rate
- Consider the asset's typical move sizes when setting targets
