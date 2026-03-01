"""
蒙特卡洛模拟模块

通过随机化交易顺序和滑点来评估策略的稳健性：
- 随机化交易执行顺序
- 随机化滑点
- 1000次模拟
- 输出Sharpe、最大回撤、最终权益的分布
- 返回最坏5%情况的回撤指标
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backend.core.backtest_engine import BacktestResults
from backend.config import MonteCarloConfig


@dataclass
class MonteCarloResults:
    """蒙特卡洛模拟结果"""
    sharpe_distribution: np.ndarray
    max_drawdown_distribution: np.ndarray
    final_equity_distribution: np.ndarray
    worst_5pct_drawdown: float
    mean_sharpe: float
    mean_max_dd: float
    mean_final_equity: float
    sharpe_std: float
    max_dd_std: float


class MonteCarloSimulator:
    """
    蒙特卡洛模拟器
    
    通过随机化交易执行来评估策略稳健性
    """
    
    def __init__(self, config: MonteCarloConfig):
        """
        初始化蒙特卡洛模拟器
        
        Args:
            config: 蒙特卡洛配置
        """
        self.config = config
    
    def randomize_trade_order(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        随机化交易顺序
        
        Args:
            trades: 交易日志DataFrame
            
        Returns:
            随机排序后的交易日志
        """
        # 保持时间顺序，但随机化执行顺序（在时间窗口内）
        trades = trades.copy()
        
        # 按日期分组，在每天内随机排序
        if 'entry_time' in trades.columns:
            trades['date'] = pd.to_datetime(trades['entry_time']).dt.date
            trades = trades.groupby('date', group_keys=False).apply(
                lambda x: x.sample(frac=1, random_state=None)
            ).reset_index(drop=True)
            trades = trades.drop('date', axis=1)
        else:
            # 如果没有时间列，完全随机排序
            trades = trades.sample(frac=1, random_state=None).reset_index(drop=True)
        
        return trades
    
    def randomize_slippage(
        self,
        price: float,
        base_slippage_rate: float,
        direction: int
    ) -> float:
        """
        随机化滑点
        
        Args:
            price: 原始价格
            base_slippage_rate: 基础滑点率
            direction: 交易方向（1=做多, -1=做空）
            
        Returns:
            考虑随机滑点后的价格
        """
        # 滑点随机化：基础滑点 ± 50%
        slippage_multiplier = np.random.uniform(0.5, 1.5)
        slippage = price * base_slippage_rate * slippage_multiplier
        
        if direction == 1:  # 做多
            return price + slippage
        else:  # 做空
            return price - slippage
    
    def simulate_from_trades(
        self,
        trades: pd.DataFrame,
        initial_equity: float,
        base_slippage_rate: float = 0.0002
    ) -> Dict[str, float]:
        """
        从交易日志模拟一次运行
        
        Args:
            trades: 交易日志DataFrame
            initial_equity: 初始权益
            base_slippage_rate: 基础滑点率
            
        Returns:
            性能指标字典
        """
        equity = initial_equity
        equity_history = [equity]
        
        # 随机化交易顺序
        randomized_trades = self.randomize_trade_order(trades)
        
        for _, trade in randomized_trades.iterrows():
            # 应用随机滑点
            entry_price = self.randomize_slippage(
                trade.get('entry_price', 0),
                base_slippage_rate,
                1 if trade.get('direction', 'LONG') == 'LONG' else -1
            )
            
            exit_price = self.randomize_slippage(
                trade.get('exit_price', entry_price),
                base_slippage_rate,
                -1 if trade.get('direction', 'LONG') == 'LONG' else 1
            )
            
            # 计算盈亏（简化：假设仓位大小不变）
            position_size = trade.get('position_size', 0)
            
            if trade.get('direction', 'LONG') == 'LONG':
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            
            # 扣除费用
            fees = trade.get('fees', 0)
            pnl -= fees
            
            equity += pnl
            equity_history.append(equity)
        
        # 计算指标
        equity_series = pd.Series(equity_history)
        returns = equity_series.pct_change().fillna(0.0)
        
        # Sharpe比率
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)  # 年化
        else:
            sharpe = 0.0
        
        # 最大回撤
        cumulative_max = equity_series.expanding().max()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # 最终权益
        final_equity = equity
        
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_equity': final_equity
        }
    
    def run_simulation(
        self,
        backtest_results: BacktestResults,
        initial_equity: float,
        base_slippage_rate: float = 0.0002
    ) -> MonteCarloResults:
        """
        运行蒙特卡洛模拟
        
        Args:
            backtest_results: 原始回测结果
            initial_equity: 初始权益
            base_slippage_rate: 基础滑点率
            
        Returns:
            蒙特卡洛模拟结果
        """
        if backtest_results.trade_log.empty:
            # 如果没有交易，返回零值结果
            return MonteCarloResults(
                sharpe_distribution=np.array([0.0] * self.config.n_simulations),
                max_drawdown_distribution=np.array([0.0] * self.config.n_simulations),
                final_equity_distribution=np.array([initial_equity] * self.config.n_simulations),
                worst_5pct_drawdown=0.0,
                mean_sharpe=0.0,
                mean_max_dd=0.0,
                mean_final_equity=initial_equity,
                sharpe_std=0.0,
                max_dd_std=0.0
            )
        
        sharpe_list = []
        max_dd_list = []
        final_equity_list = []
        
        print(f"运行蒙特卡洛模拟，共 {self.config.n_simulations} 次...")
        
        for i in range(self.config.n_simulations):
            if (i + 1) % 100 == 0:
                print(f"完成 {i + 1}/{self.config.n_simulations} 次模拟")
            
            results = self.simulate_from_trades(
                trades=backtest_results.trade_log,
                initial_equity=initial_equity,
                base_slippage_rate=base_slippage_rate
            )
            
            sharpe_list.append(results['sharpe_ratio'])
            max_dd_list.append(results['max_drawdown'])
            final_equity_list.append(results['final_equity'])
        
        sharpe_array = np.array(sharpe_list)
        max_dd_array = np.array(max_dd_list)
        final_equity_array = np.array(final_equity_list)
        
        # 计算最坏5%情况的回撤
        worst_5pct_drawdown = np.percentile(max_dd_array, 100 * (1 - self.config.confidence_level))
        
        return MonteCarloResults(
            sharpe_distribution=sharpe_array,
            max_drawdown_distribution=max_dd_array,
            final_equity_distribution=final_equity_array,
            worst_5pct_drawdown=worst_5pct_drawdown,
            mean_sharpe=sharpe_array.mean(),
            mean_max_dd=max_dd_array.mean(),
            mean_final_equity=final_equity_array.mean(),
            sharpe_std=sharpe_array.std(),
            max_dd_std=max_dd_array.std()
        )
    
    def get_distribution_summary(self, results: MonteCarloResults) -> pd.DataFrame:
        """
        获取分布摘要统计
        
        Args:
            results: 蒙特卡洛模拟结果
            
        Returns:
            包含统计摘要的DataFrame
        """
        summary_data = {
            '指标': ['Sharpe比率', '最大回撤', '最终权益'],
            '均值': [
                results.mean_sharpe,
                results.mean_max_dd,
                results.mean_final_equity
            ],
            '标准差': [
                results.sharpe_std,
                results.max_dd_std,
                results.final_equity_distribution.std()
            ],
            '最小值': [
                results.sharpe_distribution.min(),
                results.max_drawdown_distribution.min(),
                results.final_equity_distribution.min()
            ],
            '25%分位数': [
                np.percentile(results.sharpe_distribution, 25),
                np.percentile(results.max_drawdown_distribution, 25),
                np.percentile(results.final_equity_distribution, 25)
            ],
            '中位数': [
                np.percentile(results.sharpe_distribution, 50),
                np.percentile(results.max_drawdown_distribution, 50),
                np.percentile(results.final_equity_distribution, 50)
            ],
            '75%分位数': [
                np.percentile(results.sharpe_distribution, 75),
                np.percentile(results.max_drawdown_distribution, 75),
                np.percentile(results.final_equity_distribution, 75)
            ],
            '最大值': [
                results.sharpe_distribution.max(),
                results.max_drawdown_distribution.max(),
                results.final_equity_distribution.max()
            ],
            f'最坏{self.config.confidence_level*100}%': [
                np.percentile(results.sharpe_distribution, self.config.confidence_level * 100),
                results.worst_5pct_drawdown,
                np.percentile(results.final_equity_distribution, self.config.confidence_level * 100)
            ]
        }
        
        return pd.DataFrame(summary_data)
