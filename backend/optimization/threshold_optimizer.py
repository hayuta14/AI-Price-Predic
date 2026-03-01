"""
Probability Threshold Optimizer

Optimize long/short thresholds để tối ưu performance:
- long_threshold ∈ [0.55, 0.65]
- short_threshold ∈ [0.35, 0.45]
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import product

from backend.core.walkforward_engine import WalkForwardEngine
from backend.core.metrics import MetricsCalculator


@dataclass
class ThresholdConfiguration:
    """Cấu hình threshold"""
    long_threshold: float
    short_threshold: float
    sharpe_ratio: float  # Validation Sharpe
    max_drawdown: float  # Validation Max DD
    profit_factor: float
    total_trades: int
    win_rate: float
    test_sharpe_ratio: float = 0.0  # Test Sharpe (out-of-sample)
    test_max_drawdown: float = 1.0  # Test Max DD
    test_total_trades: int = 0  # Test Trades


class ThresholdOptimizer:
    """Optimize probability thresholds"""
    
    def __init__(
        self,
        walkforward_engine: WalkForwardEngine,
        long_threshold_range: Tuple[float, float] = (0.55, 0.65),
        short_threshold_range: Tuple[float, float] = (0.35, 0.45),
        step: float = 0.01
    ):
        """
        Khởi tạo Threshold Optimizer
        
        Args:
            walkforward_engine: Walk-forward engine
            long_threshold_range: Range cho long threshold
            short_threshold_range: Range cho short threshold
            step: Bước nhảy khi search
        """
        self.walkforward_engine = walkforward_engine
        self.long_threshold_range = long_threshold_range
        self.short_threshold_range = short_threshold_range
        self.step = step
        self.metrics_calculator = MetricsCalculator()
        self.configurations: List[ThresholdConfiguration] = []
    
    def generate_thresholds(self) -> List[Tuple[float, float]]:
        """
        Generate tất cả combinations của thresholds
        
        Returns:
            List of (long_threshold, short_threshold) tuples
        """
        long_start, long_end = self.long_threshold_range
        short_start, short_end = self.short_threshold_range
        
        long_thresholds = np.arange(long_start, long_end + self.step, self.step)
        short_thresholds = np.arange(short_start, short_end + self.step, self.step)
        
        # Đảm bảo long > short
        combinations = [
            (long_t, short_t)
            for long_t in long_thresholds
            for short_t in short_thresholds
            if long_t > short_t
        ]
        
        return combinations
    
    def evaluate_thresholds(
        self,
        predictions: pd.Series,
        data: pd.DataFrame,
        price_col: str = 'close',
        date_column: str = 'timestamp'
    ) -> Dict[str, float]:
        """
        Đánh giá performance của threshold configuration
        
        Args:
            predictions: Model predictions
            data: Price data
            price_col: Tên cột giá
            date_column: Tên cột date
            
        Returns:
            Metrics dictionary
        """
        # Tạo signals từ predictions
        signals = pd.Series(0, index=predictions.index, dtype=int)
        signals[predictions > self.current_long_threshold] = 1
        signals[predictions < self.current_short_threshold] = -1
        
        # Tính returns
        if price_col not in data.columns:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
        
        price_returns = data[price_col].pct_change().fillna(0.0)
        strategy_returns = price_returns * signals.reindex(price_returns.index, fill_value=0)
        
        if len(strategy_returns) == 0 or strategy_returns.std() == 0:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
        
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
        Optimize thresholds trên VALIDATION SET (không dùng test set)
        
        Args:
            predictions: Model predictions
            data: Price data
            price_col: Tên cột giá
            date_column: Tên cột date
            validation_split: Tỷ lệ validation (0.7 = 70% đầu làm validation, 30% cuối làm test)
            
        Returns:
            Best threshold configuration
        """
        # Tách data thành validation và test
        # Validation: 70% đầu để optimize threshold
        # Test: 30% cuối để đánh giá cuối cùng (KHÔNG được dùng để optimize)
        n_total = len(data)
        n_validation = int(n_total * validation_split)
        
        validation_data = data.iloc[:n_validation].copy()
        validation_predictions = predictions.iloc[:n_validation] if len(predictions) >= n_validation else predictions.iloc[:len(validation_data)]
        
        test_data = data.iloc[n_validation:].copy()
        test_predictions = predictions.iloc[n_validation:] if len(predictions) > n_validation else pd.Series()
        
        print(f"\n🔍 Đang optimize thresholds trên VALIDATION SET:")
        print(f"   • Validation: {len(validation_data)} samples ({validation_split*100:.0f}%)")
        print(f"   • Test: {len(test_data)} samples ({(1-validation_split)*100:.0f}%) - KHÔNG dùng để optimize")
        
        threshold_combinations = self.generate_thresholds()
        
        best_config = None
        best_sharpe = -np.inf
        
        print(f"\n🔍 Đang optimize {len(threshold_combinations)} threshold combinations trên validation set...")
        
        for i, (long_t, short_t) in enumerate(threshold_combinations):
            self.current_long_threshold = long_t
            self.current_short_threshold = short_t
            
            try:
                # CHỈ evaluate trên validation set
                metrics = self.evaluate_thresholds(
                    validation_predictions, validation_data, price_col, date_column
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
                
                sharpe = config.sharpe_ratio
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_config = config
                
                if (i + 1) % 10 == 0:
                    print(f"   Đã test {i+1}/{len(threshold_combinations)} combinations trên validation set...")
            
            except Exception as e:
                print(f"   Error evaluating thresholds ({long_t}, {short_t}): {e}")
                continue
        
        # Đánh giá best config trên TEST SET (out-of-sample)
        if best_config is not None and len(test_data) > 0:
            print(f"\n📊 Đánh giá best threshold trên TEST SET (out-of-sample)...")
            self.current_long_threshold = best_config.long_threshold
            self.current_short_threshold = best_config.short_threshold
            
            try:
                test_metrics = self.evaluate_thresholds(
                    test_predictions, test_data, price_col, date_column
                )
                print(f"   • Test Sharpe: {test_metrics.get('sharpe_ratio', 0.0):.4f}")
                print(f"   • Test Max DD: {test_metrics.get('max_drawdown', 1.0)*100:.2f}%")
                print(f"   • Test Trades: {test_metrics.get('total_trades', 0)}")
                
                # Lưu test metrics vào config
                best_config.test_sharpe_ratio = test_metrics.get('sharpe_ratio', 0.0)
                best_config.test_max_drawdown = test_metrics.get('max_drawdown', 1.0)
                best_config.test_total_trades = test_metrics.get('total_trades', 0)
            except Exception as e:
                print(f"   Warning: Không thể đánh giá trên test set: {e}")
        
        if best_config is None:
            # Fallback: sử dụng default
            best_config = ThresholdConfiguration(
                long_threshold=0.55,
                short_threshold=0.45,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                profit_factor=0.0,
                total_trades=0,
                win_rate=0.0
            )
        
        return best_config
    
    def get_top_configurations(self, top_n: int = 5) -> List[ThresholdConfiguration]:
        """
        Lấy top N configurations
        
        Args:
            top_n: Số lượng configurations
            
        Returns:
            List top configurations
        """
        sorted_configs = sorted(
            self.configurations,
            key=lambda x: x.sharpe_ratio,
            reverse=True
        )
        return sorted_configs[:top_n]
