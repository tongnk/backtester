"""Interactive chart generation using Plotly."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..engine.backtester import BacktestResult


def _resample_ohlc(ohlc_data: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Resample OHLC data to a lower frequency for faster chart rendering.

    Args:
        ohlc_data: DataFrame with OHLC columns and datetime index
        freq: Target frequency (e.g., "1h" for hourly)

    Returns:
        Resampled DataFrame
    """
    return ohlc_data.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()


class ChartGenerator:
    """Generate interactive HTML charts for backtest results."""

    # Color scheme
    COLORS = {
        "price_up": "#26a69a",
        "price_down": "#ef5350",
        "long_entry": "#00c853",
        "long_exit": "#69f0ae",
        "short_entry": "#ff1744",
        "short_exit": "#ff8a80",
        "equity": "#2196f3",
        "drawdown": "#f44336",
        "benchmark": "#9e9e9e",
        "indicator1": "#ff9800",
        "indicator2": "#9c27b0",
        "indicator3": "#00bcd4",
    }

    def __init__(self, theme: str = "plotly_dark"):
        """
        Initialize chart generator.

        Args:
            theme: Plotly template theme
        """
        self.theme = theme

    def create_backtest_chart(
        self,
        result: BacktestResult,
        ohlc_data: pd.DataFrame,
        show_indicators: bool = True,
        show_volume: bool = True,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create comprehensive backtest visualization.

        Args:
            result: BacktestResult from backtester
            ohlc_data: OHLC DataFrame with same index as result
            show_indicators: Whether to show strategy indicators
            show_volume: Whether to show volume subplot
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Determine subplot layout
        rows = 3  # Price, Equity, PnL distribution
        if show_volume:
            rows += 1
        if show_indicators and result.indicators:
            rows += 1

        row_heights = self._calculate_row_heights(rows, show_volume, show_indicators)

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=self._get_subplot_titles(rows, show_volume, show_indicators),
        )

        current_row = 1

        # 1. Price chart with trade markers
        self._add_price_chart(fig, ohlc_data, result, current_row)
        current_row += 1

        # 2. Volume (optional)
        if show_volume:
            self._add_volume_chart(fig, ohlc_data, current_row)
            current_row += 1

        # 3. Indicators (optional)
        if show_indicators and result.indicators:
            self._add_indicator_chart(fig, result.indicators, current_row)
            current_row += 1

        # 4. Equity curve
        self._add_equity_chart(fig, result, current_row)
        current_row += 1

        # Update layout
        chart_title = title or f"{result.strategy_name} Backtest Results"
        self._update_layout(fig, chart_title, result)

        return fig

    def _calculate_row_heights(
        self, rows: int, show_volume: bool, show_indicators: bool
    ) -> list[float]:
        """Calculate row heights for subplots."""
        if rows == 3:
            return [0.5, 0.25, 0.25]
        elif rows == 4:
            if show_volume:
                return [0.45, 0.15, 0.2, 0.2]
            else:
                return [0.4, 0.2, 0.2, 0.2]
        else:  # 5 rows
            return [0.35, 0.1, 0.15, 0.2, 0.2]

    def _get_subplot_titles(
        self, rows: int, show_volume: bool, show_indicators: bool
    ) -> list[str]:
        """Get subplot titles."""
        titles = ["Price & Trades"]
        if show_volume:
            titles.append("Volume")
        if show_indicators:
            titles.append("Indicators")
        titles.extend(["Equity Curve", ""])
        return titles[:rows]

    def _add_price_chart(
        self,
        fig: go.Figure,
        ohlc_data: pd.DataFrame,
        result: BacktestResult,
        row: int,
    ) -> None:
        """Add candlestick price chart with trade markers."""
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=ohlc_data.index,
                open=ohlc_data["open"],
                high=ohlc_data["high"],
                low=ohlc_data["low"],
                close=ohlc_data["close"],
                name="Price",
                increasing_line_color=self.COLORS["price_up"],
                decreasing_line_color=self.COLORS["price_down"],
            ),
            row=row,
            col=1,
        )

        # Add trade markers
        if not result.trades_df.empty:
            trades = result.trades_df

            # Long entries
            long_entries = trades[trades["side"] == "LONG"]
            if not long_entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=long_entries["entry_time"],
                        y=long_entries["entry_price"],
                        mode="markers",
                        name="Long Entry",
                        marker=dict(
                            symbol="triangle-up",
                            size=12,
                            color=self.COLORS["long_entry"],
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate="Long Entry<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )

            # Long exits
            if not long_entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=long_entries["exit_time"],
                        y=long_entries["exit_price"],
                        mode="markers",
                        name="Long Exit",
                        marker=dict(
                            symbol="triangle-down",
                            size=10,
                            color=self.COLORS["long_exit"],
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate="Long Exit<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )

            # Short entries
            short_entries = trades[trades["side"] == "SHORT"]
            if not short_entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=short_entries["entry_time"],
                        y=short_entries["entry_price"],
                        mode="markers",
                        name="Short Entry",
                        marker=dict(
                            symbol="triangle-down",
                            size=12,
                            color=self.COLORS["short_entry"],
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate="Short Entry<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )

            # Short exits
            if not short_entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=short_entries["exit_time"],
                        y=short_entries["exit_price"],
                        mode="markers",
                        name="Short Exit",
                        marker=dict(
                            symbol="triangle-up",
                            size=10,
                            color=self.COLORS["short_exit"],
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate="Short Exit<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )

    def _add_volume_chart(
        self, fig: go.Figure, ohlc_data: pd.DataFrame, row: int
    ) -> None:
        """Add volume bar chart."""
        colors = [
            self.COLORS["price_up"] if c >= o else self.COLORS["price_down"]
            for o, c in zip(ohlc_data["open"], ohlc_data["close"])
        ]

        fig.add_trace(
            go.Bar(
                x=ohlc_data.index,
                y=ohlc_data["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=row,
            col=1,
        )

    def _add_indicator_chart(
        self, fig: go.Figure, indicators: dict[str, pd.Series], row: int
    ) -> None:
        """Add indicator lines."""
        color_cycle = [
            self.COLORS["indicator1"],
            self.COLORS["indicator2"],
            self.COLORS["indicator3"],
        ]

        for i, (name, series) in enumerate(indicators.items()):
            color = color_cycle[i % len(color_cycle)]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1),
                ),
                row=row,
                col=1,
            )

    def _add_equity_chart(
        self, fig: go.Figure, result: BacktestResult, row: int
    ) -> None:
        """Add equity curve with drawdown fill."""
        equity = result.equity_curve

        # Calculate drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max

        # Equity line
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity,
                mode="lines",
                name="Equity",
                line=dict(color=self.COLORS["equity"], width=2),
                hovertemplate="Equity: $%{y:,.2f}<br>Time: %{x}<extra></extra>",
            ),
            row=row,
            col=1,
        )

        # Benchmark (starting equity * price change)
        if not result.trades_df.empty:
            initial_equity = result.initial_capital
            benchmark = initial_equity * (
                1 + (equity.index.to_series().apply(lambda x: 0))
            )  # Placeholder

        # Drawdown fill (on secondary y-axis would be better, but simplified here)
        # We'll add a small drawdown indicator below
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=rolling_max,
                mode="lines",
                name="High Water Mark",
                line=dict(color=self.COLORS["benchmark"], width=1, dash="dash"),
                opacity=0.5,
            ),
            row=row,
            col=1,
        )

    def _update_layout(
        self, fig: go.Figure, title: str, result: BacktestResult
    ) -> None:
        """Update figure layout with styling and annotations."""
        metrics = result.metrics

        # Create expanded summary table as HTML
        metrics_text = (
            f"<b>PERFORMANCE SUMMARY</b><br>"
            f"<br>"
            f"<b>Returns</b><br>"
            f"Total Return: ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2%})<br>"
            f"Annualized: {metrics.annualized_return:.2%}<br>"
            f"<br>"
            f"<b>Risk</b><br>"
            f"Sharpe: {metrics.sharpe_ratio:.2f} | Sortino: {metrics.sortino_ratio:.2f}<br>"
            f"Max DD: {metrics.max_drawdown:.2%}<br>"
            f"<br>"
            f"<b>Trades</b><br>"
            f"Total: {metrics.total_trades} | Win Rate: {metrics.win_rate:.1%}<br>"
            f"Profit Factor: {metrics.profit_factor:.2f}<br>"
            f"Avg Win: ${metrics.avg_win:,.2f} | Avg Loss: ${metrics.avg_loss:,.2f}"
        )

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(size=20),
            ),
            template=self.theme,
            height=1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
            ),
            xaxis_rangeslider_visible=False,
            annotations=[
                dict(
                    text=metrics_text,
                    xref="paper",
                    yref="paper",
                    x=0.99,
                    y=0.99,
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=12, family="monospace"),
                    bgcolor="rgba(0,0,0,0.8)",
                    bordercolor="rgba(255,255,255,0.3)",
                    borderwidth=1,
                    borderpad=12,
                )
            ],
            hovermode="x unified",
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)

    def save_html(
        self,
        fig: go.Figure,
        filepath: str | Path,
        include_plotlyjs: bool = True,
    ) -> None:
        """
        Save chart to HTML file.

        Args:
            fig: Plotly Figure
            filepath: Output file path
            include_plotlyjs: Include Plotly.js in the file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(
            str(filepath),
            include_plotlyjs=include_plotlyjs,
            full_html=True,
        )

        print(f"Chart saved to: {filepath}")

    def show(self, fig: go.Figure) -> None:
        """Display chart in browser."""
        fig.show()

    def create_trade_analysis_chart(self, result: BacktestResult) -> go.Figure:
        """
        Create trade analysis charts.

        Shows:
        - PnL distribution histogram
        - Win/loss by trade
        - Cumulative PnL
        """
        trades = result.trades_df

        if trades.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades to analyze",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20),
            )
            return fig

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "PnL Distribution",
                "Trade Returns",
                "Cumulative PnL",
                "Win/Loss Streaks",
            ],
        )

        # 1. PnL Distribution
        fig.add_trace(
            go.Histogram(
                x=trades["pnl"],
                name="PnL Distribution",
                nbinsx=30,
                marker_color=self.COLORS["equity"],
            ),
            row=1,
            col=1,
        )

        # 2. Trade Returns (bar chart)
        colors = [
            self.COLORS["long_entry"] if pnl > 0 else self.COLORS["short_entry"]
            for pnl in trades["pnl"]
        ]
        fig.add_trace(
            go.Bar(
                x=list(range(len(trades))),
                y=trades["pnl"],
                name="Trade PnL",
                marker_color=colors,
            ),
            row=1,
            col=2,
        )

        # 3. Cumulative PnL
        cumulative_pnl = trades["pnl"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(trades))),
                y=cumulative_pnl,
                mode="lines+markers",
                name="Cumulative PnL",
                line=dict(color=self.COLORS["equity"]),
            ),
            row=2,
            col=1,
        )

        # 4. Win/Loss streak analysis
        streaks = self._calculate_streaks(trades["pnl"])
        streak_colors = [
            self.COLORS["long_entry"] if s > 0 else self.COLORS["short_entry"]
            for s in streaks
        ]
        fig.add_trace(
            go.Bar(
                x=list(range(len(streaks))),
                y=streaks,
                name="Streaks",
                marker_color=streak_colors,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Trade Analysis",
            template=self.theme,
            height=800,
            showlegend=False,
        )

        return fig

    def _calculate_streaks(self, pnl_series: pd.Series) -> list[int]:
        """Calculate win/loss streaks."""
        streaks = []
        current_streak = 0
        prev_win = None

        for pnl in pnl_series:
            is_win = pnl > 0

            if prev_win is None:
                current_streak = 1 if is_win else -1
            elif is_win == prev_win:
                current_streak += 1 if is_win else -1
            else:
                streaks.append(current_streak)
                current_streak = 1 if is_win else -1

            prev_win = is_win

        if current_streak != 0:
            streaks.append(current_streak)

        return streaks


def generate_report(
    result: BacktestResult,
    ohlc_data: pd.DataFrame,
    output_dir: str | Path = "output",
    filename: str = "backtest_report",
    resample_freq: str = "1h",
) -> Path:
    """
    Generate a complete backtest report.

    Args:
        result: BacktestResult from backtester
        ohlc_data: OHLC DataFrame
        output_dir: Output directory
        filename: Base filename (without extension)
        resample_freq: Frequency to resample OHLC data (e.g., "1h" for hourly)

    Returns:
        Path to the generated HTML file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ChartGenerator()

    # Resample OHLC data for faster rendering
    resampled_ohlc = _resample_ohlc(ohlc_data, freq=resample_freq)
    print(f"Resampled OHLC from {len(ohlc_data):,} to {len(resampled_ohlc):,} rows ({resample_freq})")

    # Create main chart with resampled data
    main_chart = generator.create_backtest_chart(
        result=result,
        ohlc_data=resampled_ohlc,
        show_indicators=True,
        show_volume=True,
    )

    # Save main chart
    main_path = output_dir / f"{filename}.html"
    generator.save_html(main_chart, main_path)

    # Create and save trade analysis
    trade_chart = generator.create_trade_analysis_chart(result)
    trade_path = output_dir / f"{filename}_trades.html"
    generator.save_html(trade_chart, trade_path)

    return main_path
