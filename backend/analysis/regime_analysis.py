"""
市场状态（Regime）分析模块

将数据集按市场状态分段：
- 波动率百分位
- 趋势强度
- 报告每个状态的性能指标
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backend.core.metrics import MetricsCalculator


@dataclass
class RegimeMetrics:
    """市场状态的性能指标"""
    regime_name: str
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    win_rate: float
    volatility: float
    trend_strength: float


class RegimeAnalyzer:
    """
    市场状态分析器
    
    将市场分为不同状态（regime），分析策略在不同状态下的表现
    """
    
    def __init__(self):
        """初始化市场状态分析器"""
        self.metrics_calculator = MetricsCalculator()
    
    def calculate_volatility_percentile(
        self,
        returns: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        计算波动率百分位
        
        Args:
            returns: 收益率序列
            lookback: 回看期数
            
        Returns:
            波动率百分位序列（0-100）
        """
        # 计算滚动波动率
        rolling_vol = returns.rolling(window=lookback).std()
        
        # 计算历史波动率分布
        vol_percentiles = rolling_vol.rolling(window=lookback * 10, min_periods=lookback).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50,
            raw=False
        )
        
        return vol_percentiles.fillna(50.0)
    
    def calculate_trend_strength(
        self,
        prices: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        计算趋势强度
        
        使用线性回归的R²作为趋势强度指标
        
        Args:
            prices: 价格序列
            lookback: 回看期数
            
        Returns:
            趋势强度序列（0-1）
        """
        trend_strength = pd.Series(index=prices.index, dtype=float)
        
        for i in range(lookback, len(prices)):
            window_prices = prices.iloc[i-lookback:i+1]
            window_indices = np.arange(len(window_prices))
            
            # 线性回归
            if len(window_prices) > 1:
                coeffs = np.polyfit(window_indices, window_prices.values, 1)
                y_pred = np.polyval(coeffs, window_indices)
                
                # 计算R²
                ss_res = np.sum((window_prices.values - y_pred) ** 2)
                ss_tot = np.sum((window_prices.values - window_prices.mean()) ** 2)
                
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                trend_strength.iloc[i] = max(0.0, min(1.0, r_squared))
            else:
                trend_strength.iloc[i] = 0.0
        
        return trend_strength.fillna(0.0)
    
    def segment_by_volatility(
        self,
        returns: pd.Series,
        n_regimes: int = 3
    ) -> pd.Series:
        """
        按波动率分段
        
        Args:
            returns: 收益率序列
            n_regimes: 状态数量
            
        Returns:
            状态标签序列
        """
        vol_percentiles = self.calculate_volatility_percentile(returns)
        
        # 分段
        labels = pd.Series(index=returns.index, dtype=str)
        
        for i in range(n_regimes):
            lower = i * 100 / n_regimes
            upper = (i + 1) * 100 / n_regimes
            
            if i == n_regimes - 1:
                mask = vol_percentiles >= lower
            else:
                mask = (vol_percentiles >= lower) & (vol_percentiles < upper)
            
            labels[mask] = f"波动率_{i+1}"
        
        return labels.fillna("波动率_1")
    
    def segment_by_trend(
        self,
        prices: pd.Series,
        n_regimes: int = 3
    ) -> pd.Series:
        """
        按趋势强度分段
        
        Args:
            prices: 价格序列
            n_regimes: 状态数量
            
        Returns:
            状态标签序列
        """
        trend_strength = self.calculate_trend_strength(prices)
        
        # 分段
        labels = pd.Series(index=prices.index, dtype=str)
        
        for i in range(n_regimes):
            lower = i / n_regimes
            upper = (i + 1) / n_regimes
            
            if i == n_regimes - 1:
                mask = trend_strength >= lower
            else:
                mask = (trend_strength >= lower) & (trend_strength < upper)
            
            labels[mask] = f"趋势_{i+1}"
        
        return labels.fillna("趋势_1")
    
    def segment_by_combined(
        self,
        returns: pd.Series,
        prices: pd.Series,
        vol_regimes: int = 2,
        trend_regimes: int = 2
    ) -> pd.Series:
        """
        按波动率和趋势组合分段
        
        Args:
            returns: 收益率序列
            prices: 价格序列
            vol_regimes: 波动率状态数
            trend_regimes: 趋势状态数
            
        Returns:
            组合状态标签序列
        """
        vol_labels = self.segment_by_volatility(returns, vol_regimes)
        trend_labels = self.segment_by_trend(prices, trend_regimes)
        
        # 组合标签
        combined_labels = vol_labels + "_" + trend_labels
        
        return combined_labels
    
    def analyze_regimes(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        regime_labels: pd.Series
    ) -> Dict[str, RegimeMetrics]:
        """
        分析各市场状态的性能
        
        Args:
            returns: 策略收益率序列
            equity_curve: 权益曲线
            regime_labels: 状态标签序列
            
        Returns:
            每个状态的性能指标字典
        """
        regime_metrics_dict = {}
        
        unique_regimes = regime_labels.unique()
        
        for regime in unique_regimes:
            if pd.isna(regime):
                continue
            
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) == 0:
                continue
            
            # 计算该状态的权益曲线
            regime_equity = equity_curve[regime_mask]
            
            # 计算指标
            metrics = self.metrics_calculator.calculate_all_metrics(
                regime_returns,
                regime_equity
            )
            
            # 计算该状态的波动率和趋势强度
            regime_vol = regime_returns.std()
            
            # 创建RegimeMetrics
            regime_metrics = RegimeMetrics(
                regime_name=str(regime),
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown,
                profit_factor=metrics.profit_factor,
                total_trades=metrics.total_trades,
                win_rate=metrics.win_rate,
                volatility=regime_vol,
                trend_strength=0.0  # 可以进一步计算
            )
            
            regime_metrics_dict[str(regime)] = regime_metrics
        
        return regime_metrics_dict
    
    def generate_regime_report(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        regime_labels: pd.Series
    ) -> pd.DataFrame:
        """
        生成市场状态分析报告
        
        Args:
            returns: 策略收益率序列
            equity_curve: 权益曲线
            regime_labels: 状态标签序列
            
        Returns:
            包含各状态性能指标的DataFrame
        """
        regime_metrics = self.analyze_regimes(returns, equity_curve, regime_labels)
        
        report_data = []
        for regime_name, metrics in regime_metrics.items():
            report_data.append({
                '市场状态': regime_name,
                'Sharpe比率': metrics.sharpe_ratio,
                '最大回撤': metrics.max_drawdown,
                '盈亏比': metrics.profit_factor,
                '交易次数': metrics.total_trades,
                '胜率': metrics.win_rate,
                '波动率': metrics.volatility
            })
        
        df = pd.DataFrame(report_data)
        
        # 计算跨状态的稳定性（Sharpe标准差）
        if len(df) > 1:
            sharpe_std = df['Sharpe比率'].std()
            df.loc[len(df)] = {
                '市场状态': '跨状态稳定性',
                'Sharpe比率': sharpe_std,
                '最大回撤': 0.0,
                '盈亏比': 0.0,
                '交易次数': 0,
                '胜率': 0.0,
                '波动率': 0.0
            }
        
        return df
