#!/usr/bin/env python3
"""
Parameter Sweep for Volatility Momentum Strategy

Hypotheses:
1. Baseline - Current parameters
2. Asymmetric SL/TP - Tighter SL (4%), wider TP (7%) to account for execution slippage
3. Higher momentum threshold (2.5 SD) - Fewer but higher conviction trades
4. Even higher momentum (3.0 SD) - Only extreme momentum
5. Tighter vol threshold (0.5 SD) - Only enter in very calm markets
6. Wider vol threshold (1.5 SD) - More lenient on volatility
7. Longer momentum period (48h) - Capture bigger moves
8. Combined: High momentum + Asymmetric SL/TP
9. Combined: Tight vol + High momentum
10. Aggressive: 3% SL, 10% TP - Cut losses fast, let winners run big
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import Config
from src.data import DataManager
from src.strategy.examples.iv_momentum import IVMomentumStrategy
from src.engine import Backtester


# Parameter variations to test
VARIATIONS = {
    "1. Baseline": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "2. Asymmetric SL/TP (4%/7%)": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.07,
    },
    "3. High Momentum (2.5 SD)": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.5,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "4. Very High Momentum (3.0 SD)": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 3.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "5. Tight Vol (0.5 SD)": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 0.5,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "6. Wide Vol (1.5 SD)": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.5,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "7. Longer Momentum (48h)": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 48,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "8. High Mom + Asym SL/TP": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.5,
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.07,
    },
    "9. Tight Vol + High Mom": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 0.5,
        "momentum_threshold": 2.5,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.05,
    },
    "10. Aggressive (3%/10%)": {
        "vol_lookback": 168,
        "vol_window": 24,
        "momentum_period": 24,
        "momentum_lookback": 168,
        "vol_threshold": 1.0,
        "momentum_threshold": 2.0,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.10,
    },
}


def run_backtest(params: dict, data: pd.DataFrame, config: Config) -> dict:
    """Run a single backtest and return results."""
    strategy = IVMomentumStrategy(params=params)
    backtester = Backtester(config)
    result = backtester.run(strategy, data)

    return {
        "equity_curve": result.equity_curve,
        "total_return": result.metrics.total_return,
        "total_return_pct": result.metrics.total_return_pct,
        "max_drawdown": result.metrics.max_drawdown,
        "sharpe_ratio": result.metrics.sharpe_ratio,
        "win_rate": result.metrics.win_rate,
        "total_trades": result.metrics.total_trades,
        "profit_factor": result.metrics.profit_factor,
    }


def main():
    print("=" * 60)
    print("PARAMETER SWEEP - Volatility Momentum Strategy")
    print("=" * 60)

    # Setup
    config = Config()
    start_date = datetime(2023, 12, 30)
    end_date = datetime(2025, 12, 30)

    # Load data
    print("\nLoading data...")
    data_manager = DataManager(
        ohlc_dir=config.data.ohlc_dir,
        external_dir=config.data.external_dir,
    )

    data = data_manager.get_combined_data(
        symbol=config.data.symbol,
        start_date=start_date,
        end_date=end_date,
        include_iv=False,
        update=False,
    )
    print(f"Data loaded: {len(data):,} rows")

    # Run all variations
    results = {}
    for name, params in VARIATIONS.items():
        print(f"\nRunning: {name}")
        print(f"  Params: vol_thresh={params['vol_threshold']}, mom_thresh={params['momentum_threshold']}, "
              f"SL={params['stop_loss_pct']*100:.0f}%, TP={params['take_profit_pct']*100:.0f}%")

        result = run_backtest(params, data.copy(), config)
        results[name] = result

        print(f"  Return: {result['total_return_pct']:.2f}% | "
              f"MaxDD: {result['max_drawdown']:.2f}% | "
              f"Trades: {result['total_trades']} | "
              f"Win Rate: {result['win_rate']:.1f}%")

    # Calculate buy & hold for comparison
    print("\nCalculating Buy & Hold benchmark...")
    initial_price = data.iloc[0]["close"]
    final_price = data.iloc[-1]["close"]
    bh_return_pct = (final_price - initial_price) / initial_price * 100
    bh_equity = 10000 * (1 + data["close"].pct_change().fillna(0)).cumprod()
    bh_equity.index = data.index

    # Create comparison chart
    print("\nGenerating comparison chart...")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curves Comparison", "Drawdown")
    )

    # Color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Add equity curves
    for i, (name, result) in enumerate(results.items()):
        equity = result["equity_curve"]
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                name=f"{name} ({result['total_return_pct']:.1f}%)",
                line=dict(color=colors[i], width=1.5),
                hovertemplate=f"{name}<br>%{{x}}<br>${{y:,.0f}}<extra></extra>"
            ),
            row=1, col=1
        )

    # Add buy & hold
    fig.add_trace(
        go.Scatter(
            x=bh_equity.index,
            y=bh_equity.values,
            mode="lines",
            name=f"Buy & Hold ({bh_return_pct:.1f}%)",
            line=dict(color="black", width=2, dash="dash"),
        ),
        row=1, col=1
    )

    # Add drawdown for each strategy
    for i, (name, result) in enumerate(results.items()):
        equity = result["equity_curve"]
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name=name,
                line=dict(color=colors[i], width=1),
                showlegend=False,
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title="Volatility Momentum Strategy - Parameter Sweep Results",
        height=800,
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # Save chart
    output_path = Path("output") / f"parameter_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    fig.write_html(str(output_path))
    print(f"\nChart saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Variation':<35} {'Return':>10} {'MaxDD':>10} {'Sharpe':>10} {'WinRate':>10} {'Trades':>10} {'PF':>10}")
    print("-" * 100)

    # Sort by return
    sorted_results = sorted(results.items(), key=lambda x: x[1]["total_return_pct"], reverse=True)

    for name, result in sorted_results:
        print(f"{name:<35} {result['total_return_pct']:>9.2f}% {result['max_drawdown']:>9.2f}% "
              f"{result['sharpe_ratio']:>10.2f} {result['win_rate']:>9.1f}% "
              f"{result['total_trades']:>10} {result['profit_factor']:>10.2f}")

    print("-" * 100)
    print(f"{'Buy & Hold':<35} {bh_return_pct:>9.2f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
