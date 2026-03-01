"""
标签优化器

优化标签配置（预测时间范围和阈值）：
- 测试不同的horizon和threshold组合
- 使用walk-forward验证评估每个配置
- 存储每个配置的性能指标
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backend.config import LabelConfig
from backend.core.walkforward_engine import WalkForwardEngine
from backend.core.metrics import MetricsCalculator


@dataclass
class LabelConfiguration:
    """标签配置"""
    horizon: int              # 预测时间范围（根数）
    threshold: float          # 收益阈值（百分比）hoặc dict cho asymmetric
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    win_rate: float
    use_asymmetric: bool = False  # Có sử dụng asymmetric labels không
    long_threshold: Optional[float] = None  # Threshold cho long (nếu asymmetric)
    short_threshold: Optional[float] = None  # Threshold cho short (nếu asymmetric)


class LabelOptimizer:
    """
    标签优化器
    
    通过网格搜索优化标签参数：
    - horizon: 预测时间范围
    - threshold: 收益阈值
    """
    
    def __init__(
        self,
        walkforward_engine: WalkForwardEngine,
        config: LabelConfig
    ):
        """
        初始化标签优化器
        
        Args:
            walkforward_engine: Walk-forward验证引擎
            config: 标签配置
        """
        self.walkforward_engine = walkforward_engine
        self.config = config
        self.configurations: List[LabelConfiguration] = []
        self.metrics_calculator = MetricsCalculator()
    
    def create_labels(
        self,
        data: pd.DataFrame,
        price_col: str,
        horizon: int,
        threshold: float,
        use_dynamic_threshold: bool = False,
        atr_values: Optional[pd.Series] = None,
        atr_multiplier: float = 0.5,  # Mặc định 0.5 * ATR như yêu cầu
        use_asymmetric: bool = False,
        long_threshold: Optional[float] = None,
        short_threshold: Optional[float] = None
    ) -> pd.Series:
        """
        创建标签
        
        标签定义：
        - Asymmetric: 1 = long (上涨超过long_threshold), -1 = short (下跌超过short_threshold), 0 = 否则
        - Symmetric: 1 = 上涨超过threshold, 0 = 否则
        
        Args:
            data: 价格数据
            price_col: 价格列名
            horizon: 预测时间范围
            threshold: 收益阈值（百分比）或固定值
            use_dynamic_threshold: 是否使用动态阈值（基于ATR）
            atr_values: ATR值序列（如果使用动态阈值）
            atr_multiplier: ATR乘数（如果使用动态阈值）
            use_asymmetric: 是否使用asymmetric labels
            long_threshold: Long threshold (nếu asymmetric)
            short_threshold: Short threshold (nếu asymmetric)
            
        Returns:
            标签序列 (1/-1/0 nếu asymmetric, 1/0 nếu symmetric)
        """
        prices = data[price_col].values
        labels = np.zeros(len(data))
        
        # 计算动态阈值
        if use_dynamic_threshold and atr_values is not None:
            # Dynamic threshold = ATR * multiplier / price
            dynamic_thresholds = (atr_values * atr_multiplier / prices).fillna(threshold)
        else:
            dynamic_thresholds = pd.Series(threshold, index=data.index)
        
        # Kiểm tra asymmetric labels
        if use_asymmetric and long_threshold is not None and short_threshold is not None:
            # Asymmetric labels: 1 = long, -1 = short, 0 = neutral
            labels = np.zeros(len(data))
            
            # Tính thresholds
            if use_dynamic_threshold and atr_values is not None:
                dynamic_long_thresholds = (atr_values * atr_multiplier / prices).fillna(long_threshold)
                dynamic_short_thresholds = (atr_values * atr_multiplier / prices).fillna(short_threshold)
            else:
                dynamic_long_thresholds = pd.Series(long_threshold, index=data.index)
                dynamic_short_thresholds = pd.Series(short_threshold, index=data.index)
            
            for i in range(len(data) - horizon):
                current_price = prices[i]
                future_prices = prices[i+1:i+horizon+1]
                
                # Tính max return và min return
                max_return = (future_prices.max() - current_price) / current_price
                min_return = (future_prices.min() - current_price) / current_price
                
                # Lấy thresholds
                current_long_threshold = dynamic_long_thresholds.iloc[i] if use_dynamic_threshold else long_threshold
                current_short_threshold = dynamic_short_thresholds.iloc[i] if use_dynamic_threshold else short_threshold
                
                # Long label: max return >= long_threshold
                if max_return >= current_long_threshold:
                    labels[i] = 1
                # Short label: min_return <= -short_threshold (giá giảm)
                elif abs(min_return) >= current_short_threshold and min_return < 0:
                    labels[i] = -1
                # Neutral: không đạt threshold nào
                else:
                    labels[i] = 0
            
            return pd.Series(labels, index=data.index)
        
        else:
            # Symmetric labels: 1 = up, 0 = otherwise
            for i in range(len(data) - horizon):
                current_price = prices[i]
                future_prices = prices[i+1:i+horizon+1]
                
                # 计算未来最高收益
                max_return = (future_prices.max() - current_price) / current_price
                
                # 获取阈值（固定或动态）
                current_threshold = dynamic_thresholds.iloc[i] if use_dynamic_threshold else threshold
                
                # 如果最大收益超过阈值，标记为1
                if max_return >= current_threshold:
                    labels[i] = 1
            
            return pd.Series(labels, index=data.index)
    
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
        atr_multiplier: float = 1.0
    ) -> Dict[str, float]:
        """
        评估标签配置的性能
        
        Args:
            data: 完整数据集
            features: 特征列表
            price_col: 价格列名
            horizon: 预测时间范围
            threshold: 收益阈值
            model_train_fn: 模型训练函数
            model_predict_fn: 模型预测函数
            use_dynamic_threshold: 是否使用动态阈值
            atr_multiplier: ATR乘数（如果使用动态阈值）
            
        Returns:
            性能指标字典
        """
        # 计算ATR（如果需要动态阈值）
        atr_values = None
        if use_dynamic_threshold:
            if all(col in data.columns for col in ['high', 'low', 'close']):
                from backend.data.feature_engineering import calculate_atr
                atr_values = calculate_atr(data, 14)
        
        # 创建标签
        labels = self.create_labels(
            data, price_col, horizon, threshold,
            use_dynamic_threshold=use_dynamic_threshold,
            atr_values=atr_values,
            atr_multiplier=atr_multiplier
        )
        
        # 准备数据（移除最后horizon个样本，因为无法计算标签）
        valid_data = data.iloc[:-horizon].copy() if horizon > 0 else data.copy()
        valid_labels = labels.iloc[:-horizon] if horizon > 0 else labels
        
        # 添加标签列
        valid_data['label'] = valid_labels
        
        # 运行walk-forward验证
        def train_wrapper(train_data, **kwargs):
            X_train = train_data[features]
            y_train = train_data['label']
            return model_train_fn(X_train, y_train, **kwargs)
        
        def predict_wrapper(model, test_data):
            X_test = test_data[features]
            return model_predict_fn(model, X_test)
        
        def metrics_wrapper(predictions, test_data):
            # 将预测转换为交易信号和收益率
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
            print(f"评估配置 (horizon={horizon}, threshold={threshold}) 时出错: {e}")
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
        """
        将预测转换为收益率
        
        Args:
            predictions: 模型预测值（上涨概率）
            test_data: 测试数据
            price_col: 价格列名
            horizon: 预测时间范围
            threshold: 收益阈值
            
        Returns:
            收益率序列
        """
        if price_col not in test_data.columns:
            return pd.Series(0.0, index=test_data.index)
        
        prices = test_data[price_col].values
        returns = np.zeros(len(test_data))
        
        for i in range(len(test_data) - horizon):
            if predictions[i] > 0.5:  # 预测会上涨
                # 做多
                entry_price = prices[i]
                # 在horizon根K线后平仓
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
        """
        优化标签配置
        
        网格搜索所有horizon和threshold组合，包括动态阈值
        
        Args:
            data: 完整数据集
            features: 特征列表
            price_col: 价格列名
            model_train_fn: 模型训练函数
            model_predict_fn: 模型预测函数
            
        Returns:
            最优标签配置
        """
        best_config = None
        best_sharpe = -np.inf
        
        # 网格搜索：固定阈值
        for horizon in self.config.horizons:
            for threshold in self.config.thresholds:
                print(f"评估配置: horizon={horizon}, threshold={threshold:.3f} (固定)")
                
                metrics = self.evaluate_label_config(
                    data=data,
                    features=features,
                    price_col=price_col,
                    horizon=horizon,
                    threshold=threshold,
                    model_train_fn=model_train_fn,
                    model_predict_fn=model_predict_fn,
                    use_dynamic_threshold=False
                )
                
                sharpe = metrics.get('mean_sharpe_ratio', 0.0)
                max_dd = metrics.get('max_max_drawdown', 1.0)
                profit_factor = metrics.get('mean_profit_factor', 0.0)
                total_trades = metrics.get('total_total_trades', 0)
                win_rate = metrics.get('overall_win_rate', 0.0)
                
                config = LabelConfiguration(
                    horizon=horizon,
                    threshold=threshold,
                    sharpe_ratio=sharpe,
                    max_drawdown=max_dd,
                    profit_factor=profit_factor,
                    total_trades=total_trades,
                    win_rate=win_rate
                )
                
                self.configurations.append(config)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_config = config
        
        # 网格搜索：动态阈值（基于ATR）
        if self.config.use_dynamic_threshold:
            for horizon in self.config.horizons:
                for atr_multiplier in self.config.atr_multiplier_range:
                    # 使用一个基准threshold，但实际会乘以ATR
                    threshold = 0.0  # 占位符，实际使用ATR * multiplier
                    print(f"评估配置: horizon={horizon}, ATR_multiplier={atr_multiplier:.2f} (动态)")
                    
                    metrics = self.evaluate_label_config(
                        data=data,
                        features=features,
                        price_col=price_col,
                        horizon=horizon,
                        threshold=threshold,
                        model_train_fn=model_train_fn,
                        model_predict_fn=model_predict_fn,
                        use_dynamic_threshold=True,
                        atr_multiplier=atr_multiplier
                    )
                    
                    sharpe = metrics.get('mean_sharpe_ratio', 0.0)
                    max_dd = metrics.get('max_max_drawdown', 1.0)
                    profit_factor = metrics.get('mean_profit_factor', 0.0)
                    total_trades = metrics.get('total_total_trades', 0)
                    win_rate = metrics.get('overall_win_rate', 0.0)
                    
                    # 对于动态阈值，存储atr_multiplier作为threshold的负值（用于区分）
                    config = LabelConfiguration(
                        horizon=horizon,
                        threshold=-atr_multiplier,  # 负值表示动态阈值
                        sharpe_ratio=sharpe,
                        max_drawdown=max_dd,
                        profit_factor=profit_factor,
                        total_trades=total_trades,
                        win_rate=win_rate
                    )
                    
                    self.configurations.append(config)
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_config = config
        
        return best_config if best_config else LabelConfiguration(
            horizon=self.config.horizons[0],
            threshold=self.config.thresholds[0],
            sharpe_ratio=0.0,
            max_drawdown=1.0,
            profit_factor=0.0,
            total_trades=0,
            win_rate=0.0
        )
    
    def get_configuration_results(self) -> pd.DataFrame:
        """
        获取所有配置的评估结果
        
        Returns:
            DataFrame包含所有配置的性能指标
        """
        if not self.configurations:
            return pd.DataFrame()
        
        results_data = []
        for config in self.configurations:
            results_data.append({
                'horizon': config.horizon,
                'threshold': config.threshold,
                'sharpe_ratio': config.sharpe_ratio,
                'max_drawdown': config.max_drawdown,
                'profit_factor': config.profit_factor,
                'total_trades': config.total_trades,
                'win_rate': config.win_rate
            })
        
        return pd.DataFrame(results_data).sort_values('sharpe_ratio', ascending=False)
