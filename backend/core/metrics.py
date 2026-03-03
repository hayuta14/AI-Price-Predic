"""
性能指标计算模块

实现hedge-fund级别的性能评估指标：
- Sharpe Ratio（风险调整收益）
- Maximum Drawdown（最大回撤）
- Profit Factor（盈亏比）
- Calmar Ratio（收益回撤比）
- 稳定性指标

所有指标计算都考虑：
- 时间序列特性
- 风险调整
- 稳健性
"""
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    calmar_ratio: float
    total_return: float
    annualized_return: float
    volatility: float
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_dd_duration: int  # 最大回撤持续时间（天数）
    recovery_factor: float  # 恢复因子 = 总收益 / 最大回撤


class MetricsCalculator:
    """
    性能指标计算器
    
    提供全面的交易策略性能评估指标，符合hedge-fund标准
    """
    
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 365 * 24 * 4):
        """
        初始化指标计算器
        
        Args:
            risk_free_rate: 无风险利率（年化）
            periods_per_year: 每年交易周期数（15分钟K线：365*24*4=35040）
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def calculate_sharpe_ratio(
        self, 
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        计算Sharpe比率
        
        Sharpe = (平均收益率 - 无风险利率) / 收益率标准差
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率（如果为None，使用实例默认值）
            
        Returns:
            Sharpe比率（年化）
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            return 0.0
        
        # 移除NaN
        returns = returns.dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # 计算平均收益率和标准差
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # 年化
        sharpe = (mean_return - risk_free_rate / self.periods_per_year) / std_return
        sharpe *= np.sqrt(self.periods_per_year)
        
        return sharpe
    
    def calculate_max_drawdown(
        self, 
        equity_curve: Union[pd.Series, np.ndarray]
    ) -> tuple[float, int, pd.Series]:
        """
        计算最大回撤
        
        Args:
            equity_curve: 权益曲线（累计净值）
            
        Returns:
            (最大回撤, 最大回撤持续时间, 回撤序列)
        """
        equity = pd.Series(equity_curve) if not isinstance(equity_curve, pd.Series) else equity_curve
        equity = equity.dropna()
        
        if len(equity) == 0:
            return 0.0, 0, pd.Series()
        
        # 计算累计最大值（峰值）
        cumulative_max = equity.expanding().max()
        
        # 计算回撤
        drawdown = (equity - cumulative_max) / cumulative_max
        
        # 最大回撤
        max_dd = abs(drawdown.min())
        
        # 计算最大回撤持续时间
        max_dd_duration = self._calculate_dd_duration(drawdown)
        
        return max_dd, max_dd_duration, drawdown
    
    def _calculate_dd_duration(self, drawdown: pd.Series) -> int:
        """
        计算最大回撤持续时间
        
        Args:
            drawdown: 回撤序列
            
        Returns:
            最大回撤持续时间（周期数）
        """
        if len(drawdown) == 0:
            return 0
        
        # 找到最大回撤点
        max_dd_idx = drawdown.idxmin()
        
        # 找到回撤开始点（峰值点）
        before_max_dd = drawdown[:max_dd_idx]
        if len(before_max_dd) == 0:
            peak_idx = drawdown.index[0]
        else:
            peak_idx = before_max_dd.idxmax()
            if pd.isna(peak_idx):
                peak_idx = drawdown.index[0]
        
        # 找到恢复点（回到峰值）
        after_max_dd = drawdown[max_dd_idx:]
        if len(after_max_dd) == 0:
            recovery_idx = drawdown.index[-1]
        else:
            recovery_idx = after_max_dd.idxmax()
            if pd.isna(recovery_idx):
                # 如果还未恢复，返回到当前的时间
                recovery_idx = drawdown.index[-1]
        
        duration = recovery_idx - peak_idx if not pd.isna(peak_idx) else len(drawdown)
        
        # Convert to int - handle both numeric and datetime indices
        if isinstance(duration, pd.Timedelta):
            return int(duration.total_seconds() / 86400)  # Convert to days
        elif isinstance(duration, (int, float, np.integer, np.floating)):
            return int(duration)
        else:
            return len(drawdown)
    
    def calculate_profit_factor(
        self, 
        returns: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        计算盈亏比（Profit Factor）
        
        Profit Factor = 总盈利 / 总亏损（绝对值）
        
        Args:
            returns: 收益率序列
            
        Returns:
            盈亏比
        """
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        returns = returns.dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # 分离盈利和亏损
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float('inf') if profits > 0 else 0.0
        
        return profits / losses
    
    def calculate_calmar_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        equity_curve: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> float:
        """
        计算Calmar比率
        
        Calmar = 年化收益率 / 最大回撤
        
        Args:
            returns: 收益率序列
            equity_curve: 权益曲线（可选，如果提供则用于计算回撤）
            
        Returns:
            Calmar比率
        """
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        returns = returns.dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # 计算年化收益率
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (self.periods_per_year / n_periods) - 1
        
        # 计算最大回撤
        if equity_curve is not None:
            max_dd, _, _ = self.calculate_max_drawdown(equity_curve)
        else:
            # 从收益率计算权益曲线
            equity = (1 + returns).cumprod()
            max_dd, _, _ = self.calculate_max_drawdown(equity)
        
        if max_dd == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_dd
    
    def calculate_trade_statistics(
        self,
        returns: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        计算交易统计信息
        
        Args:
            returns: 收益率序列
            
        Returns:
            包含交易统计的字典
        """
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # 只统计非零收益（实际交易）
        trades = returns[returns != 0]
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]
        
        total_trades = len(trades)
        n_wins = len(winning_trades)
        n_losses = len(losing_trades)
        
        win_rate = n_wins / total_trades if total_trades > 0 else 0.0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0
        largest_win = winning_trades.max() if len(winning_trades) > 0 else 0.0
        largest_loss = losing_trades.min() if len(losing_trades) > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': n_wins,
            'losing_trades': n_losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def calculate_all_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        equity_curve: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> PerformanceMetrics:
        """
        计算所有性能指标
        
        Args:
            returns: 收益率序列
            equity_curve: 权益曲线（可选）
            
        Returns:
            PerformanceMetrics对象
        """
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        returns = returns.dropna()
        
        if len(returns) == 0:
            # 返回零值指标
            return PerformanceMetrics(
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                calmar_ratio=0.0,
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                max_dd_duration=0,
                recovery_factor=0.0
            )
        
        # 计算权益曲线（如果未提供）
        if equity_curve is None:
            equity_curve = (1 + returns).cumprod()
        else:
            equity_curve = pd.Series(equity_curve) if not isinstance(equity_curve, pd.Series) else equity_curve
        
        # 计算各项指标
        sharpe = self.calculate_sharpe_ratio(returns)
        max_dd, max_dd_duration, _ = self.calculate_max_drawdown(equity_curve)
        profit_factor = self.calculate_profit_factor(returns)
        calmar = self.calculate_calmar_ratio(returns, equity_curve)
        
        # 收益率统计
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (self.periods_per_year / n_periods) - 1
        volatility = returns.std() * np.sqrt(self.periods_per_year)
        
        # 交易统计
        trade_stats = self.calculate_trade_statistics(returns)
        
        # 恢复因子
        recovery_factor = total_return / max_dd if max_dd > 0 else 0.0
        
        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            max_dd_duration=max_dd_duration,
            recovery_factor=recovery_factor
        )
    
    def calculate_regime_metrics(
        self,
        returns: pd.Series,
        regime_labels: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        按市场状态（regime）计算指标
        
        Args:
            returns: 收益率序列
            regime_labels: 市场状态标签序列（与returns对齐）
            
        Returns:
            每个regime的指标字典
        """
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        regime_labels = pd.Series(regime_labels) if not isinstance(regime_labels, pd.Series) else regime_labels
        
        if len(returns) != len(regime_labels):
            raise ValueError("收益率序列和状态标签长度不匹配")
        
        regime_metrics = {}
        unique_regimes = regime_labels.unique()
        
        for regime in unique_regimes:
            regime_returns = returns[regime_labels == regime]
            
            if len(regime_returns) == 0:
                continue
            
            # 计算该regime的权益曲线
            regime_equity = (1 + regime_returns).cumprod()
            
            # 计算所有指标
            metrics = self.calculate_all_metrics(regime_returns, regime_equity)
            
            # 转换为字典
            regime_metrics[str(regime)] = {
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'total_return': metrics.total_return,
                'volatility': metrics.volatility
            }
        
        return regime_metrics
