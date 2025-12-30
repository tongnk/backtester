"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class Signal(Enum):
    """Trading signal types."""
    LONG = 1       # Open long position
    SHORT = -1     # Open short position
    CLOSE = 0      # Close current position
    HOLD = None    # No action / hold current position
    ADD = 2        # Add to existing position (pyramid)


@dataclass
class StrategyParams:
    """Base class for strategy parameters."""
    pass


@dataclass
class StrategyResult:
    """Container for strategy calculation results."""
    signals: pd.Series  # Signal at each timestamp
    indicators: dict[str, pd.Series] = field(default_factory=dict)  # Calculated indicators
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement:
    - name: Strategy identifier
    - calculate_indicators: Compute technical indicators
    - generate_signals: Generate trading signals

    Strategies receive OHLC data and optional external data,
    and produce signals for the backtesting engine.
    """

    def __init__(self, params: Optional[dict] = None):
        """
        Initialize the strategy.

        Args:
            params: Strategy-specific parameters
        """
        self.params = params or {}
        self._validate_params()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass

    @property
    def description(self) -> str:
        """Return a description of the strategy."""
        return "No description provided."

    def _validate_params(self) -> None:
        """
        Validate strategy parameters.

        Override this method to add custom validation.
        Raises ValueError if parameters are invalid.
        """
        pass

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators required by the strategy.

        Args:
            df: DataFrame with OHLC data (columns: open, high, low, close, volume)
                May also contain external data (e.g., 'iv' for implied volatility)

        Returns:
            DataFrame with added indicator columns
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """
        Generate trading signals based on indicators.

        IMPORTANT: Signals must only use data available at each timestamp.
        Do not use future data (look-ahead bias).

        Args:
            df: DataFrame with OHLC data and calculated indicators

        Returns:
            StrategyResult containing:
            - signals: Series of Signal values (LONG=1, SHORT=-1, CLOSE=0, HOLD=None, ADD=2)
            - indicators: Dict of indicator series for visualization
            - metadata: Any additional information
        """
        pass

    def run(self, df: pd.DataFrame) -> StrategyResult:
        """
        Run the complete strategy pipeline.

        Args:
            df: DataFrame with OHLC and optional external data

        Returns:
            StrategyResult with signals and indicators
        """
        # Calculate indicators
        df_with_indicators = self.calculate_indicators(df.copy())

        # Generate signals
        result = self.generate_signals(df_with_indicators)

        return result

    def get_required_columns(self) -> list[str]:
        """
        Return list of required DataFrame columns.

        Override to specify additional required columns (e.g., 'iv' for IV data).
        """
        return ["open", "high", "low", "close", "volume"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has all required columns.

        Args:
            df: Input DataFrame

        Returns:
            True if valid, raises ValueError otherwise
        """
        required = self.get_required_columns()
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return True

    def warmup_period(self) -> int:
        """
        Return the number of periods needed before strategy can generate signals.

        Override this to specify the warmup period based on indicator lookback.
        """
        return 0

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value with optional default.

        Args:
            key: Parameter name
            default: Default value if not found

        Returns:
            Parameter value
        """
        return self.params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """
        Set a parameter value.

        Args:
            key: Parameter name
            value: Parameter value
        """
        self.params[key] = value
        self._validate_params()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"
