"""
风险参数优化器

优化风险管理参数：
- 每笔交易风险百分比
- ATR倍数（止损距离）
- 风险收益比（止盈/止损）
- 使用walk-forward验证评估
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backend.core.walkforward_engine import WalkForwardEngine
from backend.core.backtest_engine import BacktestEngine
from backend.core.risk_engine import RiskEngine
from backend.core.metrics import MetricsCalculator
from backend.config import RiskConfig, TradingConfig


@dataclass
class RiskConfiguration:
    """风险配置"""
    risk_per_trade: float
    atr_multiplier: float
    reward_risk_ratio: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int


class RiskOptimizer:
    """
    风险参数优化器
    
    通过网格搜索优化风险参数组合
    """
    
    def __init__(
        self,
        walkforward_engine: WalkForwardEngine,
        initial_equity: float = 100000.0
    ):
        """
        初始化风险优化器
        
        Args:
            walkforward_engine: Walk-forward验证引擎
            initial_equity: 初始权益
        """
        self.walkforward_engine = walkforward_engine
        self.initial_equity = initial_equity
        self.configurations: List[RiskConfiguration] = []
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_risk_config(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        atr_values: pd.Series,
        risk_per_trade: float,
        atr_multiplier: float,
        reward_risk_ratio: float,
        price_col: str = 'close',
        timestamp_col: str = 'timestamp',
        signals: pd.Series = None
    ) -> Dict[str, float]:
        """
        评估风险配置的性能
        
        Args:
            data: 价格数据
            predictions: 模型预测序列
            atr_values: ATR值序列
            risk_per_trade: 每笔交易风险百分比
            atr_multiplier: ATR倍数
            reward_risk_ratio: 风险收益比
            price_col: 价格列名
            timestamp_col: 时间戳列名
            
        Returns:
            性能指标字典
        """
        # 创建风险配置
        risk_config = RiskConfig(
            risk_per_trade=risk_per_trade,
            max_leverage=10.0,
            max_daily_loss=0.05,
            max_drawdown_stop=0.20,
            volatility_scaling=True
        )
        
        trading_config = TradingConfig()
        
        # 创建风险引擎和回测引擎
        risk_engine = RiskEngine(risk_config)
        risk_engine.reset(self.initial_equity)
        
        backtest_engine = BacktestEngine(
            risk_engine=risk_engine,
            trading_config=trading_config,
            initial_equity=self.initial_equity
        )
        
        # 生成交易信号（若未提供）
        if signals is None:
            signals = pd.Series(0, index=data.index)
            for i in range(len(data)):
                if i < len(predictions):
                    if predictions.iloc[i] > 0.6:
                        signals.iloc[i] = 1  # 做多
                    elif predictions.iloc[i] < 0.4:
                        signals.iloc[i] = -1  # 做空
        
        # 运行回测
        try:
            results = backtest_engine.run_backtest(
                data=data,
                predictions=predictions,
                signals=signals,
                atr_values=atr_values,
                atr_multiplier=atr_multiplier,
                reward_risk_ratio=reward_risk_ratio,
                timestamp_col=timestamp_col,
                price_col=price_col
            )
            
            return results.metrics
        except Exception as e:
            print(f"评估风险配置时出错: {e}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }
    
    def optimize_risk_parameters(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        atr_values: pd.Series,
        risk_per_trade_range: Tuple[float, float, float] = (0.01, 0.03, 0.005),  # (min, max, step)
        atr_multiplier_range: Tuple[float, float, float] = (1.5, 3.0, 0.5),
        reward_risk_ratio_range: Tuple[float, float, float] = (1.5, 3.0, 0.5),
        price_col: str = 'close',
        timestamp_col: str = 'timestamp',
        signals: pd.Series = None
    ) -> RiskConfiguration:
        """
        优化风险参数
        
        Args:
            data: 价格数据
            predictions: 模型预测序列
            atr_values: ATR值序列
            risk_per_trade_range: 风险百分比范围 (min, max, step)
            atr_multiplier_range: ATR倍数范围
            reward_risk_ratio_range: 风险收益比范围
            price_col: 价格列名
            timestamp_col: 时间戳列名
            
        Returns:
            最优风险配置
        """
        best_config = None
        best_sharpe = -np.inf
        
        # 生成参数网格
        risk_values = np.arange(
            risk_per_trade_range[0],
            risk_per_trade_range[1] + risk_per_trade_range[2],
            risk_per_trade_range[2]
        )
        
        atr_values_range = np.arange(
            atr_multiplier_range[0],
            atr_multiplier_range[1] + atr_multiplier_range[2],
            atr_multiplier_range[2]
        )
        
        rr_values = np.arange(
            reward_risk_ratio_range[0],
            reward_risk_ratio_range[1] + reward_risk_ratio_range[2],
            reward_risk_ratio_range[2]
        )
        
        total_combinations = len(risk_values) * len(atr_values_range) * len(rr_values)
        current = 0
        
        # 网格搜索
        for risk_pct in risk_values:
            for atr_mult in atr_values_range:
                for rr_ratio in rr_values:
                    current += 1
                    print(f"评估风险配置 ({current}/{total_combinations}): "
                          f"risk={risk_pct:.3f}, atr_mult={atr_mult:.1f}, rr={rr_ratio:.1f}")
                    
                    metrics = self.evaluate_risk_config(
                        data=data,
                        predictions=predictions,
                        atr_values=atr_values,
                        risk_per_trade=risk_pct,
                        atr_multiplier=atr_mult,
                        reward_risk_ratio=rr_ratio,
                        price_col=price_col,
                        timestamp_col=timestamp_col
                    )
                    
                    sharpe = metrics.get('sharpe_ratio', 0.0)
                    max_dd = metrics.get('max_drawdown', 1.0)
                    profit_factor = metrics.get('profit_factor', 0.0)
                    total_trades = metrics.get('total_trades', 0)
                    
                    config = RiskConfiguration(
                        risk_per_trade=risk_pct,
                        atr_multiplier=atr_mult,
                        reward_risk_ratio=rr_ratio,
                        sharpe_ratio=sharpe,
                        max_drawdown=max_dd,
                        profit_factor=profit_factor,
                        total_trades=total_trades
                    )
                    
                    self.configurations.append(config)
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_config = config
        
        return best_config if best_config else RiskConfiguration(
            risk_per_trade=0.02,
            atr_multiplier=2.0,
            reward_risk_ratio=2.0,
            sharpe_ratio=0.0,
            max_drawdown=1.0,
            profit_factor=0.0,
            total_trades=0
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
                'risk_per_trade': config.risk_per_trade,
                'atr_multiplier': config.atr_multiplier,
                'reward_risk_ratio': config.reward_risk_ratio,
                'sharpe_ratio': config.sharpe_ratio,
                'max_drawdown': config.max_drawdown,
                'profit_factor': config.profit_factor,
                'total_trades': config.total_trades
            })
        
        return pd.DataFrame(results_data).sort_values('sharpe_ratio', ascending=False)
