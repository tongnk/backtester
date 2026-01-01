"""Shared test fixtures for backtester tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.metrics import Metrics, PerformanceMetrics
from src.engine.position import Position, PositionSide, Fill, PositionManager, Trade


@pytest.fixture
def sample_equity_curve():
    """Create a simple equity curve for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
    # Start at 10000, grow to ~11000 with some volatility
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.001, 100)
    equity = 10000 * (1 + returns).cumprod()
    return pd.Series(equity, index=dates)


@pytest.fixture
def sample_price_series():
    """Create a simple price series for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.002, 100)
    prices = 100 * (1 + returns).cumprod()
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_trades_df():
    """Create sample trades DataFrame."""
    data = [
        {
            "entry_time": datetime(2024, 1, 1, 10, 0),
            "exit_time": datetime(2024, 1, 1, 11, 0),
            "side": "LONG",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "quantity": 1.0,
            "pnl": 4.9,  # 5.0 - 0.1 fees
            "pnl_pct": 0.049,
            "fees": 0.1,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": True,
        },
        {
            "entry_time": datetime(2024, 1, 1, 12, 0),
            "exit_time": datetime(2024, 1, 1, 13, 0),
            "side": "LONG",
            "entry_price": 105.0,
            "exit_price": 103.0,
            "quantity": 1.0,
            "pnl": -2.1,  # -2.0 - 0.1 fees
            "pnl_pct": -0.02,
            "fees": 0.1,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": False,
        },
        {
            "entry_time": datetime(2024, 1, 1, 14, 0),
            "exit_time": datetime(2024, 1, 1, 15, 0),
            "side": "SHORT",
            "entry_price": 103.0,
            "exit_price": 100.0,
            "quantity": 1.0,
            "pnl": 2.9,  # 3.0 - 0.1 fees
            "pnl_pct": 0.029,
            "fees": 0.1,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": True,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def empty_trades_df():
    """Create empty trades DataFrame."""
    return pd.DataFrame(columns=[
        "entry_time", "exit_time", "side", "entry_price", "exit_price",
        "quantity", "pnl", "pnl_pct", "fees", "pyramid_levels", "duration", "is_winner"
    ])


@pytest.fixture
def metrics_calculator():
    """Create metrics calculator instance."""
    return Metrics(risk_free_rate=0.02)


@pytest.fixture
def position_manager():
    """Create position manager instance."""
    return PositionManager(max_positions=3, fee_rate=0.001, slippage=0.0)


@pytest.fixture
def known_returns_series():
    """Create a returns series with known statistics for testing Sharpe/Sortino."""
    # Create returns with known mean and std
    # Mean: 0.001 (0.1% per period)
    # Std: 0.02 (2% per period)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
    np.random.seed(123)
    returns = np.random.normal(0.001, 0.02, 1000)
    return pd.Series(returns, index=dates)


@pytest.fixture
def drawdown_equity_curve():
    """Create equity curve with known drawdown."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="1min")
    # Equity: 100, 110, 120, 100, 90, 95, 100, 110, 115, 120
    # Max DD: (120 - 90) / 120 = 25%
    equity = pd.Series([100, 110, 120, 100, 90, 95, 100, 110, 115, 120], index=dates)
    return equity


@pytest.fixture
def all_winning_trades_df():
    """Create trades where all are winners."""
    data = [
        {
            "entry_time": datetime(2024, 1, 1, 10, 0),
            "exit_time": datetime(2024, 1, 1, 11, 0),
            "side": "LONG",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "quantity": 1.0,
            "pnl": 5.0,
            "pnl_pct": 0.05,
            "fees": 0.0,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": True,
        },
        {
            "entry_time": datetime(2024, 1, 1, 12, 0),
            "exit_time": datetime(2024, 1, 1, 13, 0),
            "side": "LONG",
            "entry_price": 105.0,
            "exit_price": 110.0,
            "quantity": 1.0,
            "pnl": 5.0,
            "pnl_pct": 0.048,
            "fees": 0.0,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": True,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def all_losing_trades_df():
    """Create trades where all are losers."""
    data = [
        {
            "entry_time": datetime(2024, 1, 1, 10, 0),
            "exit_time": datetime(2024, 1, 1, 11, 0),
            "side": "LONG",
            "entry_price": 100.0,
            "exit_price": 95.0,
            "quantity": 1.0,
            "pnl": -5.0,
            "pnl_pct": -0.05,
            "fees": 0.0,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": False,
        },
        {
            "entry_time": datetime(2024, 1, 1, 12, 0),
            "exit_time": datetime(2024, 1, 1, 13, 0),
            "side": "LONG",
            "entry_price": 95.0,
            "exit_price": 90.0,
            "quantity": 1.0,
            "pnl": -5.0,
            "pnl_pct": -0.053,
            "fees": 0.0,
            "pyramid_levels": 1,
            "duration": pd.Timedelta(hours=1),
            "is_winner": False,
        },
    ]
    return pd.DataFrame(data)
