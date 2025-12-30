"""Configuration management for the backtesting tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import yaml


class SizingMode(Enum):
    """Position sizing modes."""
    FIXED_PERCENT = "fixed_percent"
    FIXED_AMOUNT = "fixed_amount"
    KELLY = "kelly"


@dataclass
class DataConfig:
    """Data-related configuration."""
    ohlc_dir: Path = field(default_factory=lambda: Path("data/ohlc"))
    external_dir: Path = field(default_factory=lambda: Path("data/external"))
    symbol: str = "BTCUSDT"
    interval: str = "1m"

    def __post_init__(self):
        self.ohlc_dir = Path(self.ohlc_dir)
        self.external_dir = Path(self.external_dir)


@dataclass
class PositionConfig:
    """Position sizing and management configuration."""
    sizing_mode: SizingMode = SizingMode.FIXED_PERCENT
    size_value: float = 1.0  # 100% of capital for fixed_percent, or BTC amount for fixed_amount
    max_positions: int = 1  # Max concurrent positions (>1 enables pyramiding)
    pyramid_threshold: float = 0.02  # Add to position when profit exceeds this %
    max_pyramid_levels: int = 3  # Maximum number of adds to a position

    def __post_init__(self):
        if isinstance(self.sizing_mode, str):
            self.sizing_mode = SizingMode(self.sizing_mode)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 10000.0
    fee_rate: float = 0.001  # 0.1% per trade
    slippage: float = 0.0001  # 0.01% slippage

    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    output_dir: Path = field(default_factory=lambda: Path("output"))

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            data=DataConfig(**data.get("data", {})),
            position=PositionConfig(**data.get("position", {})),
            backtest=BacktestConfig(**data.get("backtest", {})),
            output_dir=Path(data.get("output_dir", "output"))
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            "data": {
                "ohlc_dir": str(self.data.ohlc_dir),
                "external_dir": str(self.data.external_dir),
                "symbol": self.data.symbol,
                "interval": self.data.interval,
            },
            "position": {
                "sizing_mode": self.position.sizing_mode.value,
                "size_value": self.position.size_value,
                "max_positions": self.position.max_positions,
                "pyramid_threshold": self.position.pyramid_threshold,
                "max_pyramid_levels": self.position.max_pyramid_levels,
            },
            "backtest": {
                "initial_capital": self.backtest.initial_capital,
                "fee_rate": self.backtest.fee_rate,
                "slippage": self.backtest.slippage,
                "start_date": self.backtest.start_date,
                "end_date": self.backtest.end_date,
            },
            "output_dir": str(self.output_dir),
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
