#!/usr/bin/env python3
"""
BTC Backtesting Tool - Main Entry Point

Usage:
    python run_backtest.py [--config CONFIG] [--strategy STRATEGY] [--update-data]

Examples:
    # Run with default settings
    python run_backtest.py

    # Run with custom config
    python run_backtest.py --config my_config.yaml

    # Update data before running
    python run_backtest.py --update-data

    # Run specific strategy
    python run_backtest.py --strategy iv_mean_reversion
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data import DataManager
from src.strategy.examples.ma_crossover import MACrossoverStrategy, TripleMAStrategy
from src.strategy.examples.iv_strategy import (
    IVMeanReversionStrategy,
    IVBreakoutStrategy,
    IVRegimeStrategy,
)
from src.strategy.examples.iv_momentum import IVMomentumStrategy
from src.engine import Backtester
from src.visualization.charts import ChartGenerator, generate_report


# Available strategies
STRATEGIES = {
    "ma_crossover": {
        "class": MACrossoverStrategy,
        "params": {"fast_period": 10, "slow_period": 50, "ma_type": "ema"},
        "requires_iv": False,
    },
    "triple_ma": {
        "class": TripleMAStrategy,
        "params": {"fast_period": 10, "medium_period": 25, "slow_period": 50},
        "requires_iv": False,
    },
    "iv_mean_reversion": {
        "class": IVMeanReversionStrategy,
        "params": {
            "iv_lookback": 168,
            "upper_threshold": 2.0,
            "lower_threshold": -1.5,
            "use_price_confirmation": True,
        },
        "requires_iv": True,
    },
    "iv_breakout": {
        "class": IVBreakoutStrategy,
        "params": {
            "iv_spike_threshold": 0.20,
            "iv_lookback": 24,
            "min_price_move": 0.01,
        },
        "requires_iv": True,
    },
    "iv_regime": {
        "class": IVRegimeStrategy,
        "params": {
            "iv_percentile_lookback": 720,
            "low_iv_percentile": 25,
            "high_iv_percentile": 75,
        },
        "requires_iv": True,
    },
    "iv_momentum": {
        "class": IVMomentumStrategy,
        "params": {
            "vol_lookback": 168,           # 7 days in hours
            "vol_window": 24,              # 24-hour window for realized vol
            "momentum_period": 24,         # 24-hour momentum
            "momentum_lookback": 168,      # 7 days for momentum z-score
            "vol_threshold": 1.0,          # Volatility within 1 SD
            "momentum_threshold": 2.5,     # Momentum > 2.5 SD (best from sweep)
            "stop_loss_pct": 0.05,         # 5% stop loss (baseline)
            "take_profit_pct": 0.05,       # 5% take profit (baseline)
            "use_iv": False,               # Use realized vol instead of IV
            "pyramid_enabled": True,       # Enable pyramiding
            "pyramid_threshold": 0.025,    # Add at 2.5% profit
            "trail_to_breakeven": True,    # Move SL to breakeven after add
            "use_atr_stops": False,        # Disable ATR stops
            "atr_period": 14,              # ATR period (calculated on hourly data)
            "sl_atr_mult": 3.0,            # Stop loss = entry - 3×ATR
            "trail_atr_mult": 6.0,         # Trailing stop = highest - 6×ATR (when 'both')
            "tp_sd_fraction": None,        # Disable vol-based TP for baseline
            "sl_sd_fraction": None,        # Disable vol-based SL for baseline

            # Dynamic exit approaches (all disabled by default for baseline)
            "signal_invalidation_enabled": False,  # Exit when momentum fades
            "momentum_exit_threshold": 1.0,        # Exit long when zscore < 1.0

            "time_exit_enabled": False,            # Exit after max_hold_hours
            "max_hold_hours": 24,                  # Max hours to hold

            "asymmetric_tp_enabled": False,        # Use separate TP for long/short
            "tp_long_pct": 0.03,                   # 3% TP for longs
            "tp_short_pct": 0.02,                  # 2% TP for shorts

            "breakeven_after_profit_enabled": False,  # Move SL to BE after profit
            "breakeven_trigger_pct": 0.015,           # Trigger at 1.5% profit

            "vol_floor_enabled": False,            # Enforce min TP/SL with vol-based
            "tp_floor_pct": 0.025,                 # Minimum 2.5% TP
            "sl_floor_pct": 0.015,                 # Minimum 1.5% SL
        },
        "requires_iv": False,
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BTC Backtesting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="ma_crossover",
        choices=list(STRATEGIES.keys()),
        help="Strategy to backtest",
    )

    parser.add_argument(
        "--update-data",
        action="store_true",
        help="Fetch/update data before running backtest",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for backtest (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for backtest (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital",
    )

    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip chart generation",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("BTC BACKTESTING TOOL")
    print("=" * 60)

    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = Config()
        print("Using default configuration")

    # Override config with command line args
    if args.capital:
        config.backtest.initial_capital = args.capital

    config.output_dir = Path(args.output_dir)

    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = datetime.now() - timedelta(days=365)  # 1 year default

    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    print(f"\nBacktest Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${config.backtest.initial_capital:,.2f}")
    print(f"Strategy: {args.strategy}")

    # Initialize data manager
    data_manager = DataManager(
        ohlc_dir=config.data.ohlc_dir,
        external_dir=config.data.external_dir,
    )

    # Get strategy configuration
    strategy_config = STRATEGIES[args.strategy]
    requires_iv = strategy_config["requires_iv"]

    # Load or update data
    print("\n" + "-" * 40)
    print("DATA LOADING")
    print("-" * 40)

    if args.update_data:
        print("Fetching/updating market data...")

        def progress_callback(current, total):
            if total:
                pct = current / total * 100
                print(f"\rProgress: {pct:.1f}% ({current}/{total})", end="", flush=True)
            else:
                print(f"\rFetched {current} batches...", end="", flush=True)

        data = data_manager.update_ohlc_data(
            symbol=config.data.symbol,
            interval=config.data.interval,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback,
        )
        print()  # New line after progress

        if requires_iv:
            print("Fetching IV data...")
            data_manager.update_iv_data(
                currency="BTC",
                start_date=start_date,
                end_date=end_date,
            )
    else:
        print("Loading cached data...")

    # Get combined data
    data = data_manager.get_combined_data(
        symbol=config.data.symbol,
        start_date=start_date,
        end_date=end_date,
        include_iv=requires_iv,
        update=False,
    )

    if data.empty:
        print("\nERROR: No data available. Run with --update-data to fetch data.")
        print("\nExample:")
        print(f"  python run_backtest.py --strategy {args.strategy} --update-data")
        sys.exit(1)

    # Show data info
    data_info = data_manager.get_data_info(config.data.symbol)
    print(f"\nData loaded: {len(data):,} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    if requires_iv:
        if "iv" not in data.columns or data["iv"].isna().all():
            print("\nWARNING: IV data not available. Strategy may not work correctly.")
            print("Run with --update-data to fetch IV data.")
        else:
            iv_coverage = data["iv"].notna().sum() / len(data) * 100
            print(f"IV data coverage: {iv_coverage:.1f}%")

    # Initialize strategy
    print("\n" + "-" * 40)
    print("RUNNING BACKTEST")
    print("-" * 40)

    strategy_class = strategy_config["class"]
    strategy_params = strategy_config["params"]
    strategy = strategy_class(params=strategy_params)

    print(f"Strategy: {strategy.name}")
    print(f"Description: {strategy.description}")
    print(f"Parameters: {strategy_params}")

    # Run backtest
    backtester = Backtester(config)
    result = backtester.run(strategy, data)

    # Print results
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)
    backtester.print_summary(result)

    # Generate charts
    if not args.no_chart:
        print("\n" + "-" * 40)
        print("GENERATING CHARTS")
        print("-" * 40)

        output_path = generate_report(
            result=result,
            ohlc_data=data,
            output_dir=config.output_dir,
            filename=f"backtest_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        print(f"\nResults saved to: {output_path}")
        print(f"Open the HTML file in your browser to view interactive charts.")

    # Save trade log
    trades_path = config.output_dir / f"trades_{args.strategy}.csv"
    result.trades_df.to_csv(trades_path, index=False)
    print(f"Trade log saved to: {trades_path}")

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
