"""
标签优化器

优化标签配置（预测时间范围和阈值）：
- 使用 Triple Barrier Method（Lopez de Prado）
- 测试不同的 horizon / sl / tp 组合
- 使用 walk-forward 验证评估每个配置
- 存储每个配置的性能指标
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import warnings
import pandas as pd
import numpy as np

from backend.config import LabelConfig
from backend.core.walkforward_engine import WalkForwardEngine
from backend.core.metrics import MetricsCalculator


@dataclass
class LabelConfiguration:
    """标签配置"""
    horizon: int
    threshold: float  # 兼容旧字段：映射为 min_ret
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    win_rate: float
    sl_multiplier: float = 1.5
    tp_multiplier: float = 2.0
    min_ret: float = 0.0
    use_asymmetric: bool = False  # 兼容历史字段
    long_threshold: Optional[float] = None
    short_threshold: Optional[float] = None


class LabelOptimizer:
    """标签优化器（Triple Barrier）"""

    def __init__(
        self,
        walkforward_engine: WalkForwardEngine,
        config: LabelConfig
    ):
        self.walkforward_engine = walkforward_engine
        self.config = config
        self.configurations: List[LabelConfiguration] = []
        self.metrics_calculator = MetricsCalculator()

    def _calculate_atr(self, data: pd.DataFrame, atr_period: int = 14) -> pd.Series:
        """计算 ATR（只依赖历史数据，无前视）。"""
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Triple barrier labeling requires 'high', 'low', and 'close' columns")

        high = data['high']
        low = data['low']
        prev_close = data['close'].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(window=atr_period, min_periods=1).mean()

    def create_triple_barrier_labels(
        self,
        data: pd.DataFrame,
        price_col: str,
        horizon: int,
        sl_multiplier: float,
        tp_multiplier: float,
        atr_period: int = 14,
        min_ret: float = 0.0
    ) -> pd.Series:
        """
        Labels:
         1  = Take profit hit first (long signal)
        -1  = Stop loss hit first (short signal)
         0  = Vertical barrier hit (no clear direction)

        Barrier check uses data['high'] and data['low'] (path dependent), never close-only.
        """
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if sl_multiplier <= 0 or tp_multiplier <= 0:
            raise ValueError("sl_multiplier and tp_multiplier must be > 0")
        if tp_multiplier <= sl_multiplier:
            raise ValueError("tp_multiplier must be > sl_multiplier for positive EV")
        if price_col not in data.columns:
            raise ValueError(f"price_col '{price_col}' not found in data")
        if not all(col in data.columns for col in ['high', 'low']):
            raise ValueError("Triple barrier labeling requires data['high'] and data['low']")

        prices = data[price_col].values
        highs = data['high'].values
        lows = data['low'].values
        atr = self._calculate_atr(data, atr_period=atr_period).values

        labels = np.zeros(len(data), dtype=int)

        for i in range(len(data) - horizon):
            entry = prices[i]
            if np.isnan(entry) or entry <= 0:
                labels[i] = 0
                continue

            atr_i = atr[i]
            if np.isnan(atr_i) or atr_i <= 0:
                labels[i] = 0
                continue

            sl_price = entry - sl_multiplier * atr_i
            tp_price = entry + tp_multiplier * atr_i

            label = 0
            for j in range(1, horizon + 1):
                idx = i + j

                # Must use LOW/HIGH for touch detection
                if lows[idx] <= sl_price:
                    label = -1
                    break

                if highs[idx] >= tp_price:
                    # only keep positive class if move is above min_ret
                    realized_ret = (tp_price - entry) / entry
                    label = 1 if realized_ret >= min_ret else 0
                    break

            labels[i] = label

        return pd.Series(labels, index=data.index)

    def label_distribution_check(self, labels: pd.Series) -> None:
        """Check class distribution and print warnings."""
        if labels is None or len(labels) == 0:
            print("Distribution: {-1: 0.00%, 0: 0.00%, 1: 0.00%}")
            return

        labels = labels.dropna()
        if len(labels) == 0:
            print("Distribution: {-1: 0.00%, 0: 0.00%, 1: 0.00%}")
            return

        dist = labels.value_counts(normalize=True)
        pct_m1 = float(dist.get(-1, 0.0) * 100)
        pct_0 = float(dist.get(0, 0.0) * 100)
        pct_1 = float(dist.get(1, 0.0) * 100)

        print(f"Distribution: {{-1: {pct_m1:.2f}%, 0: {pct_0:.2f}%, 1: {pct_1:.2f}%}}")

        if pct_1 < 20.0 or pct_1 > 50.0:
            print("Warning: Class 1 outside [20%, 50%] (imbalanced labels).")
        if pct_0 > 60.0:
            print("Warning: Class 0 > 60% (too many neutrals, consider wider barriers).")

    def create_labels(
        self,
        data: pd.DataFrame,
        price_col: str,
        horizon: int,
        threshold: float,
        use_dynamic_threshold: bool = False,
        atr_values: Optional[pd.Series] = None,
        atr_multiplier: float = 0.5,
        use_asymmetric: bool = False,
        long_threshold: Optional[float] = None,
        short_threshold: Optional[float] = None
    ) -> pd.Series:
        """Deprecated backward-compatible interface."""
        warnings.warn(
            "create_labels() is deprecated; use create_triple_barrier_labels() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        sl_mult = float(short_threshold) if (use_asymmetric and short_threshold is not None) else float(atr_multiplier)
        tp_mult = float(long_threshold) if (use_asymmetric and long_threshold is not None) else max(sl_mult + 0.5, sl_mult * 1.5)

        return self.create_triple_barrier_labels(
            data=data,
            price_col=price_col,
            horizon=horizon,
            sl_multiplier=sl_mult,
            tp_multiplier=tp_mult,
            atr_period=14,
            min_ret=float(threshold)
        )

    def evaluate_label_config(
        self,
        data: pd.DataFrame,
        features: List[str],
        price_col: str,
        horizon: int,
        threshold: float,
        model_train_fn,
        model_predict_fn,
        use_dynamic_threshold: bool = False,
        atr_multiplier: float = 1.0,
        sl_multiplier: float = 1.5,
        tp_multiplier: float = 2.0,
        min_ret: float = 0.0
    ) -> Dict[str, float]:
        """评估标签配置性能（使用 Triple Barrier 标签）。"""
        labels = self.create_triple_barrier_labels(
            data=data,
            price_col=price_col,
            horizon=horizon,
            sl_multiplier=sl_multiplier,
            tp_multiplier=tp_multiplier,
            atr_period=14,
            min_ret=min_ret
        )

        self.label_distribution_check(labels.iloc[:-horizon] if horizon > 0 else labels)

        valid_data = data.iloc[:-horizon].copy() if horizon > 0 else data.copy()
        valid_labels = labels.iloc[:-horizon] if horizon > 0 else labels
        valid_data['label'] = valid_labels

        def train_wrapper(train_data, **kwargs):
            X_train = train_data[features]
            y_train = train_data['label']
            return model_train_fn(X_train, y_train, **kwargs)

        def predict_wrapper(model, test_data):
            X_test = test_data[features]
            return model_predict_fn(model, X_test)

        def metrics_wrapper(predictions, test_data):
            returns = self._predictions_to_returns(
                predictions,
                test_data,
                price_col,
                horizon,
                threshold
            )

            if len(returns) == 0 or returns.sum() == 0:
                return {
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'profit_factor': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0
                }

            equity = (1 + returns).cumprod()
            metrics = self.metrics_calculator.calculate_all_metrics(returns, equity)

            return {
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate
            }

        try:
            results = self.walkforward_engine.run_validation(
                data=valid_data,
                model_train_fn=train_wrapper,
                model_predict_fn=predict_wrapper,
                metrics_fn=metrics_wrapper
            )
            return results.aggregated_metrics
        except Exception as e:
            print(
                f"评估配置 (horizon={horizon}, sl={sl_multiplier}, tp={tp_multiplier}, min_ret={min_ret}) 时出错: {e}"
            )
            return {
                'mean_sharpe_ratio': 0.0,
                'max_max_drawdown': 1.0,
                'mean_profit_factor': 0.0,
                'total_total_trades': 0,
                'overall_win_rate': 0.0
            }

    def _predictions_to_returns(
        self,
        predictions: np.ndarray,
        test_data: pd.DataFrame,
        price_col: str,
        horizon: int,
        threshold: float
    ) -> pd.Series:
        """将预测转换为收益率。"""
        if price_col not in test_data.columns:
            return pd.Series(0.0, index=test_data.index)

        prices = test_data[price_col].values
        returns = np.zeros(len(test_data))

        for i in range(len(test_data) - horizon):
            if predictions[i] > 0.5:
                entry_price = prices[i]
                if i + horizon < len(prices):
                    exit_price = prices[i + horizon]
                    returns[i] = (exit_price - entry_price) / entry_price

        return pd.Series(returns, index=test_data.index)

    def optimize_labels(
        self,
        data: pd.DataFrame,
        features: List[str],
        price_col: str,
        model_train_fn,
        model_predict_fn
    ) -> LabelConfiguration:
        """网格搜索 Triple Barrier 参数。"""
        best_config = None
        best_sharpe = -np.inf

        horizons = [6, 8, 10, 12]
        sl_grid = [1.0, 1.5, 2.0]
        tp_grid = [1.5, 2.0, 3.0]
        min_ret = 0.0

        for horizon in horizons:
            for sl_multiplier in sl_grid:
                for tp_multiplier in tp_grid:
                    if tp_multiplier <= sl_multiplier:
                        continue

                    print(
                        f"评估配置: horizon={horizon}, sl_multiplier={sl_multiplier:.2f}, "
                        f"tp_multiplier={tp_multiplier:.2f}, min_ret={min_ret:.4f}"
                    )

                    metrics = self.evaluate_label_config(
                        data=data,
                        features=features,
                        price_col=price_col,
                        horizon=horizon,
                        threshold=min_ret,
                        model_train_fn=model_train_fn,
                        model_predict_fn=model_predict_fn,
                        use_dynamic_threshold=False,
                        atr_multiplier=1.0,
                        sl_multiplier=sl_multiplier,
                        tp_multiplier=tp_multiplier,
                        min_ret=min_ret
                    )

                    sharpe = metrics.get('mean_sharpe_ratio', 0.0)
                    max_dd = metrics.get('max_max_drawdown', 1.0)
                    profit_factor = metrics.get('mean_profit_factor', 0.0)
                    total_trades = metrics.get('total_total_trades', 0)
                    win_rate = metrics.get('overall_win_rate', 0.0)

                    config = LabelConfiguration(
                        horizon=horizon,
                        threshold=min_ret,
                        sharpe_ratio=sharpe,
                        max_drawdown=max_dd,
                        profit_factor=profit_factor,
                        total_trades=total_trades,
                        win_rate=win_rate,
                        sl_multiplier=sl_multiplier,
                        tp_multiplier=tp_multiplier,
                        min_ret=min_ret
                    )

                    self.configurations.append(config)

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_config = config

        return best_config if best_config else LabelConfiguration(
            horizon=6,
            threshold=0.0,
            sharpe_ratio=0.0,
            max_drawdown=1.0,
            profit_factor=0.0,
            total_trades=0,
            win_rate=0.0,
            sl_multiplier=1.5,
            tp_multiplier=2.0,
            min_ret=0.0
        )

    def get_configuration_results(self) -> pd.DataFrame:
        """获取所有配置评估结果。"""
        if not self.configurations:
            return pd.DataFrame()

        results_data = []
        for config in self.configurations:
            results_data.append({
                'horizon': config.horizon,
                'threshold': config.threshold,
                'sl_multiplier': config.sl_multiplier,
                'tp_multiplier': config.tp_multiplier,
                'min_ret': config.min_ret,
                'sharpe_ratio': config.sharpe_ratio,
                'max_drawdown': config.max_drawdown,
                'profit_factor': config.profit_factor,
                'total_trades': config.total_trades,
                'win_rate': config.win_rate
            })

        return pd.DataFrame(results_data).sort_values('sharpe_ratio', ascending=False)
