"""
Probability Threshold Optimizer.

Optimizes long/short thresholds.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from backend.core.walkforward_engine import WalkForwardEngine
from backend.core.metrics import MetricsCalculator


@dataclass
class ThresholdConfiguration:
    """Threshold configuration."""
    long_threshold: float
    short_threshold: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    win_rate: float
    test_sharpe_ratio: float = 0.0
    test_max_drawdown: float = 1.0
    test_total_trades: int = 0


class ThresholdOptimizer:
    """Optimize probability thresholds."""

    def __init__(
        self,
        walkforward_engine: WalkForwardEngine,
        long_threshold_range: Tuple[float, float] = (0.55, 0.65),
        short_threshold_range: Tuple[float, float] = (0.35, 0.45),
        step: float = 0.01
    ):
        self.walkforward_engine = walkforward_engine
        self.long_threshold_range = long_threshold_range
        self.short_threshold_range = short_threshold_range
        self.step = step
        self.metrics_calculator = MetricsCalculator()
        self.configurations: List[ThresholdConfiguration] = []
        self.current_long_threshold = 0.55
        self.current_short_threshold = 0.45

    def generate_thresholds(self) -> List[Tuple[float, float]]:
        long_start, long_end = self.long_threshold_range
        short_start, short_end = self.short_threshold_range

        long_thresholds = np.arange(long_start, long_end + self.step, self.step)
        short_thresholds = np.arange(short_start, short_end + self.step, self.step)

        return [
            (long_t, short_t)
            for long_t in long_thresholds
            for short_t in short_thresholds
            if long_t > short_t
        ]

    def evaluate_thresholds(
        self,
        predictions: pd.Series,
        data: pd.DataFrame,
        price_col: str = 'close',
        date_column: str = 'timestamp'
    ) -> Dict[str, float]:
        signals = pd.Series(0, index=predictions.index, dtype=int)
        signals[predictions > self.current_long_threshold] = 1
        signals[predictions < self.current_short_threshold] = -1

        if price_col not in data.columns:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}

        price_returns = data[price_col].pct_change().fillna(0.0)
        strategy_returns = price_returns * signals.reindex(price_returns.index, fill_value=0)

        if len(strategy_returns) == 0 or strategy_returns.std() == 0:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0, 'profit_factor': 0.0, 'total_trades': 0, 'win_rate': 0.0}

        equity = (1 + strategy_returns).cumprod()
        metrics = self.metrics_calculator.calculate_all_metrics(strategy_returns, equity)

        return {
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'profit_factor': metrics.profit_factor,
            'total_trades': metrics.total_trades,
            'win_rate': metrics.win_rate
        }

    def optimize(
        self,
        predictions: pd.Series,
        data: pd.DataFrame,
        price_col: str = 'close',
        date_column: str = 'timestamp',
        validation_split: float = 0.7
    ) -> ThresholdConfiguration:
        """
        Optimize thresholds using 60/20/20 split.

        - 60% train section is ignored for threshold search
        - 20% middle section is used for threshold optimization
        - 20% final section is held out for final report
        """
        n_total = len(data)
        train_end = int(n_total * 0.6)
        val_end = int(n_total * 0.8)

        threshold_data = data.iloc[train_end:val_end].copy()
        threshold_predictions = predictions.reindex(threshold_data.index)

        final_test_data = data.iloc[val_end:].copy()
        final_test_predictions = predictions.reindex(final_test_data.index)

        threshold_combinations = self.generate_thresholds()

        best_config = None
        best_sharpe = -np.inf

        for long_t, short_t in threshold_combinations:
            self.current_long_threshold = long_t
            self.current_short_threshold = short_t

            metrics = self.evaluate_thresholds(
                threshold_predictions,
                threshold_data,
                price_col,
                date_column
            )

            config = ThresholdConfiguration(
                long_threshold=long_t,
                short_threshold=short_t,
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                max_drawdown=metrics.get('max_drawdown', 1.0),
                profit_factor=metrics.get('profit_factor', 0.0),
                total_trades=metrics.get('total_trades', 0),
                win_rate=metrics.get('win_rate', 0.0)
            )
            self.configurations.append(config)

            if config.sharpe_ratio > best_sharpe:
                best_sharpe = config.sharpe_ratio
                best_config = config

        if best_config is None:
            best_config = ThresholdConfiguration(
                long_threshold=0.55,
                short_threshold=0.45,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                profit_factor=0.0,
                total_trades=0,
                win_rate=0.0
            )

        if len(final_test_data) > 0:
            self.current_long_threshold = best_config.long_threshold
            self.current_short_threshold = best_config.short_threshold
            test_metrics = self.evaluate_thresholds(
                final_test_predictions,
                final_test_data,
                price_col,
                date_column
            )
            best_config.test_sharpe_ratio = test_metrics.get('sharpe_ratio', 0.0)
            best_config.test_max_drawdown = test_metrics.get('max_drawdown', 1.0)
            best_config.test_total_trades = test_metrics.get('total_trades', 0)

        return best_config

    def get_top_configurations(self, top_n: int = 5) -> List[ThresholdConfiguration]:
        sorted_configs = sorted(self.configurations, key=lambda x: x.sharpe_ratio, reverse=True)
        return sorted_configs[:top_n]
