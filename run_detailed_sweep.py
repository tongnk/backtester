#!/usr/bin/env python3
"""Get detailed metrics for all parameter variations."""

import sys
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.config import Config
from src.data import DataManager
from src.strategy.examples.iv_momentum import IVMomentumStrategy
from src.engine import Backtester


def _run_single_backtest(args):
    """Worker function for parallel execution."""
    name, params, data = args
    config = Config()  # Create fresh config in worker
    strategy = IVMomentumStrategy(params=params)
    backtester = Backtester(config)
    result = backtester.run(strategy, data)

    m = result.metrics
    return {
        "name": name,
        "vol_lookback": params["vol_lookback"],
        "momentum_period": params["momentum_period"],
        "momentum_lookback": params["momentum_lookback"],
        "vol_threshold": params["vol_threshold"],
        "momentum_threshold": params["momentum_threshold"],
        "stop_loss_pct": params["stop_loss_pct"],
        "take_profit_pct": params["take_profit_pct"],
        "total_return_pct": m.total_return_pct,
        "max_drawdown": m.max_drawdown,
        "sharpe_ratio": m.sharpe_ratio,
        "total_trades": m.total_trades,
        "win_rate": m.win_rate,
        "avg_win_pct": m.avg_win_pct,
        "avg_loss_pct": m.avg_loss_pct,
        "profit_factor": m.profit_factor,
    }

VARIATIONS = {
    "1. Baseline": {
        "vol_lookback": 168,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "2. Asymmetric SL/TP (4%/7%)": {
        "vol_lookback": 168,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.07,
    },
    "3. High Momentum (2.5 SD)": {
        "vol_lookback": 168,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.5,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "4. Very High Momentum (3.0 SD)": {
        "vol_lookback": 168,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 3.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "5. Tight Vol (0.5 SD)": {
        "vol_lookback": 168,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 0.5,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "6. Wide Vol (1.5 SD)": {
        "vol_lookback": 168,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.5,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "7. Longer Momentum (48h)": {
        "vol_lookback": 168,
        "momentum_period": 48,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "8. High Mom + Asym SL/TP": {
        "vol_lookback": 168,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.5,
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.07,
    },
    "9. Tight Vol + High Mom": {
        "vol_lookback": 168,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 0.5,
        "momentum_threshold": 2.5,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "10. Aggressive (3%/10%)": {
        "vol_lookback": 168,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.10,
    },
}


def main():
    config = Config()
    start_date = datetime(2023, 12, 30)
    end_date = datetime(2025, 12, 30)

    print("Loading data...")
    data_manager = DataManager(
        ohlc_dir=config.data.ohlc_dir,
        external_dir=config.data.external_dir,
    )

    data = data_manager.get_combined_data(
        symbol=config.data.symbol,
        start_date=start_date,
        end_date=end_date,
        include_iv=True,
        update=False,
    )
    print(f"Data loaded: {len(data):,} rows\n")

    # Run all variations in parallel
    n_workers = min(cpu_count(), len(VARIATIONS))
    print(f"Running {len(VARIATIONS)} backtests in parallel ({n_workers} workers)...")

    # Prepare arguments for parallel execution
    args_list = [(name, params, data.copy()) for name, params in VARIATIONS.items()]

    # Execute in parallel
    with Pool(n_workers) as pool:
        results = pool.map(_run_single_backtest, args_list)

    print("All backtests completed.")

    # Print as CSV-like format for easy parsing
    print("\n" + "=" * 120)
    print("DETAILED RESULTS")
    print("=" * 120)

    for r in results:
        print(f"\n{r['name']}")
        print(f"  Parameters: vol_thresh={r['vol_threshold']}, mom_thresh={r['momentum_threshold']}, "
              f"SL={r['stop_loss_pct']*100:.0f}%, TP={r['take_profit_pct']*100:.0f}%, "
              f"mom_period={r['momentum_period']}")
        print(f"  Return: {r['total_return_pct']:.2f}% | MaxDD: {r['max_drawdown']:.2f}% | "
              f"Sharpe: {r['sharpe_ratio']:.2f}")
        print(f"  Trades: {r['total_trades']} | Win Rate: {r['win_rate']:.2f}% | "
              f"Avg Win: {r['avg_win_pct']:.2f}% | Avg Loss: {r['avg_loss_pct']:.2f}% | "
              f"PF: {r['profit_factor']:.2f}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("output/parameter_sweep_detailed.csv", index=False)
    print(f"\nDetailed results saved to: output/parameter_sweep_detailed.csv")


if __name__ == "__main__":
    main()
