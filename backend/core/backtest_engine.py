"""
回测引擎

实现完整的交易回测模拟，包括：
- 模型信号生成
- 入场/出场逻辑
- ATR-based止损
- 风险/收益比止盈
- 交易成本（手续费、滑点、资金费率）
- 权益曲线跟踪
- 交易日志记录

严格模拟真实交易环境，确保回测结果可靠
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

from backend.config import TradingConfig
from backend.core.risk_engine import RiskEngine, PositionSize
from backend.core.metrics import MetricsCalculator


class TradeDirection(Enum):
    """交易方向"""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Trade:
    """单笔交易记录"""
    trade_id: int
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: TradeDirection
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    funding_cost: float = 0.0
    exit_reason: str = ""
    is_open: bool = True
    # 退出所在K线的OHLC信息 & 是否跳空离场
    exit_bar_open: Optional[float] = None
    exit_bar_high: Optional[float] = None
    exit_bar_low: Optional[float] = None
    exit_bar_close: Optional[float] = None
    gap_exit: bool = False


@dataclass
class BacktestResults:
    """回测结果"""
    equity_curve: pd.Series
    returns: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    trade_log: pd.DataFrame
    daily_returns: pd.Series


class BacktestEngine:
    """
    回测引擎
    
    模拟完整的交易执行流程，包括：
    1. 信号生成（从模型预测）
    2. 仓位计算（从风险引擎）
    3. 订单执行（考虑滑点和费用）
    4. 止损/止盈检查
    5. 权益更新
    """
    
    def __init__(
        self,
        risk_engine: RiskEngine,
        trading_config: TradingConfig,
        initial_equity: float = 100000.0
    ):
        """
        初始化回测引擎
        
        Args:
            risk_engine: 风险管理引擎
            trading_config: 交易配置
            initial_equity: 初始权益
        """
        self.risk_engine = risk_engine
        self.trading_config = trading_config
        self.initial_equity = initial_equity
        
        self.current_equity = initial_equity
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.equity_history: List[float] = [initial_equity]
        self.time_history: List[datetime] = []
        
        # 指标计算器：基于15分钟K线的年化参数
        self.metrics_calculator = MetricsCalculator.for_15min_bars()
        self.trade_counter = 0
        # Realistic execution simulator (fees, slippage, funding)
        self.execution_simulator = RealisticExecutionSimulator(
            {
                "taker_fee": self.trading_config.fee_rate,
                "maker_fee": self.trading_config.fee_rate * 0.4,
                "slippage_k": self.trading_config.slippage_rate,
                "funding_interval": 32,
            }
        )
    
    def reset(self, initial_equity: Optional[float] = None):
        """
        重置回测引擎
        
        Args:
            initial_equity: 初始权益（如果为None，使用实例默认值）
        """
        if initial_equity is not None:
            self.initial_equity = initial_equity
        
        self.current_equity = self.initial_equity
        self.trades = []
        self.open_trades = []
        self.equity_history = [self.initial_equity]
        self.time_history = []
        self.trade_counter = 0
        self.risk_engine.reset(self.initial_equity)
    
    def _apply_slippage(self, price: float, direction: TradeDirection) -> float:
        """
        应用滑点
        
        Args:
            price: 原始价格
            direction: 交易方向
            
        Returns:
            考虑滑点后的价格
        """
        # 使用RealisticExecutionSimulator，根据波动率/成交量估算滑点.
        # 当前实现中如果缺少微观结构特征，则退化为固定比例滑点。
        # 这里使用一个保守的基准波动率和成交量比率占位。
        baseline_vol = 0.01
        volume_ratio = 1.0
        side = 1 if direction == TradeDirection.LONG else -1 if direction == TradeDirection.SHORT else 0
        if side == 0:
            return price

        exec_info = self.execution_simulator.calculate_fill_price(
            order_price=price,
            side=side,
            current_volatility=baseline_vol,
            volume_ratio=volume_ratio,
        )
        return float(exec_info["fill_price"])
    
    def _calculate_fees(self, position_value: float) -> float:
        """
        计算手续费
        
        Args:
            position_value: 仓位价值
            
        Returns:
            手续费金额
        """
        return position_value * self.trading_config.fee_rate
    
    def _calculate_funding_cost(
        self,
        position_value: float,
        direction: TradeDirection,
        hours: float = 8.0,
    ) -> float:
        """
        使用RealisticExecutionSimulator按8小时节奏计算资金费率成本.
        """
        side = 1 if direction == TradeDirection.LONG else -1 if direction == TradeDirection.SHORT else 0
        if side == 0 or position_value <= 0:
            return 0.0

        candles_held = int(hours * 4)  # 15m bars per hour
        return float(
            self.execution_simulator.calculate_funding_cost(
                position_size=1.0,
                position_value=position_value,
                side=side,
                funding_rate=self.trading_config.funding_rate,
                candles_held=candles_held,
            )
        )
    
    def enter_trade(
        self,
        timestamp: datetime,
        price: float,
        direction: TradeDirection,
        prediction: float,  # 模型预测值（用于确定方向）
        atr: float,
        atr_multiplier: float = 2.0,
        reward_risk_ratio: float = 2.0
    ) -> Optional[Trade]:
        """
        开仓
        
        Args:
            timestamp: 时间戳
            price: 当前价格
            direction: 交易方向（如果为None，根据prediction自动判断）
            prediction: 模型预测值
            atr: ATR值
            atr_multiplier: ATR倍数（用于止损）
            reward_risk_ratio: 风险收益比
            
        Returns:
            Trade对象（如果开仓成功），否则None
        """
        # 更新权益
        self.risk_engine.update_equity(self.current_equity)
        
        # 检查是否允许交易
        allowed, reason = self.risk_engine.check_trade_allowed()
        if not allowed:
            return None
        
        # 确定交易方向
        if direction == TradeDirection.FLAT:
            # 根据预测自动判断
            if prediction > 0.5:
                direction = TradeDirection.LONG
            elif prediction < 0.5:
                direction = TradeDirection.SHORT
            else:
                return None
        
        # 计算止损和止盈价格
        stop_distance = atr * atr_multiplier
        
        if direction == TradeDirection.LONG:
            stop_price = price - stop_distance
            take_profit_price = price + stop_distance * reward_risk_ratio
        else:  # SHORT
            stop_price = price + stop_distance
            take_profit_price = price - stop_distance * reward_risk_ratio
        
        # 计算仓位大小
        position_size = self.risk_engine.calculate_position_size(
            entry_price=price,
            stop_price=stop_price,
            atr=atr
        )
        
        if position_size.position_size == 0:
            return None
        
        # 应用滑点
        entry_price = self._apply_slippage(price, direction)
        
        # 计算费用
        fees = self._calculate_fees(position_size.position_value)
        
        # 创建交易
        trade = Trade(
            trade_id=self.trade_counter,
            entry_time=timestamp,
            exit_time=None,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            position_size=position_size.position_size,
            stop_loss=stop_price,
            take_profit=take_profit_price,
            fees=fees,
            slippage=abs(entry_price - price) * position_size.position_size,
            is_open=True
        )
        
        # 更新权益（扣除费用）
        self.current_equity -= fees
        self.risk_engine.update_equity(self.current_equity)
        
        self.trade_counter += 1
        self.open_trades.append(trade)
        self.trades.append(trade)
        
        return trade
    
    def update_trades(
        self,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        atr: Optional[float] = None
    ) -> List[Trade]:
        """
        更新所有开仓交易（检查止损/止盈）
        
        Args:
            timestamp: 当前时间戳
            open_price: 当前K线开盘价
            high_price: 当前K线最高价
            low_price: 当前K线最低价
            close_price: 当前K线收盘价
            atr: ATR值（可选，用于动态止损）
            
        Returns:
            已平仓的交易列表
        """
        closed_trades: List[Trade] = []

        # 使用整根K线的OHLC进行更真实的撮合模拟
        for trade in self.open_trades[:]:  # 使用切片复制，避免迭代时修改
            exit_price: Optional[float] = None
            exit_reason: str = ""
            gap_exit: bool = False

            if trade.direction == TradeDirection.LONG:
                # 1) 跳空低开直接击穿止损 -> 按开盘价离场（保守处理为跳空止损）
                if open_price < trade.stop_loss:
                    exit_price = open_price
                    exit_reason = "止损"
                    gap_exit = True
                else:
                    # 2) 盘中同时触及止损与止盈时，保守假设先触发止损
                    stop_hit = low_price <= trade.stop_loss
                    tp_hit = high_price >= trade.take_profit

                    if stop_hit:
                        # 止损价基础上再施加额外滑点（不利方向）
                        sl_rate = float(self.trading_config.slippage_rate)
                        exit_price = trade.stop_loss * (1.0 - sl_rate * 2.0)
                        exit_reason = "止损"
                    elif tp_hit:
                        exit_price = trade.take_profit
                        exit_reason = "止盈"

            else:  # SHORT
                # 1) 跳空高开直接击穿止损 -> 按开盘价离场（保守处理为跳空止损）
                if open_price > trade.stop_loss:
                    exit_price = open_price
                    exit_reason = "止损"
                    gap_exit = True
                else:
                    # 2) 盘中同时触及止损与止盈时，保守假设先触发止损
                    stop_hit = high_price >= trade.stop_loss
                    tp_hit = low_price <= trade.take_profit

                    if stop_hit:
                        sl_rate = float(self.trading_config.slippage_rate)
                        exit_price = trade.stop_loss * (1.0 + sl_rate * 2.0)
                        exit_reason = "止损"
                    elif tp_hit:
                        exit_price = trade.take_profit
                        exit_reason = "止盈"

            if exit_price is not None:
                # 记录退出时所在K线的OHLC和是否跳空离场
                trade.exit_bar_open = open_price
                trade.exit_bar_high = high_price
                trade.exit_bar_low = low_price
                trade.exit_bar_close = close_price
                trade.gap_exit = gap_exit

                closed_trade = self._close_trade(trade, timestamp, exit_price, exit_reason)
                closed_trades.append(closed_trade)
                self.open_trades.remove(trade)

        return closed_trades
    
    def _close_trade(
        self,
        trade: Trade,
        timestamp: datetime,
        exit_price: float,
        exit_reason: str
    ) -> Trade:
        """
        平仓
        
        Args:
            trade: 交易对象
            timestamp: 平仓时间
            exit_price: 平仓价格
            exit_reason: 平仓原因
            
        Returns:
            更新后的交易对象
        """
        # 应用滑点
        actual_exit_price = self._apply_slippage(exit_price, trade.direction)
        
        # 计算盈亏
        if trade.direction == TradeDirection.LONG:
            pnl = (actual_exit_price - trade.entry_price) * trade.position_size
        else:  # SHORT
            pnl = (trade.entry_price - actual_exit_price) * trade.position_size
        
        # 计算费用
        exit_fees = self._calculate_fees(trade.position_size * actual_exit_price)
        
        # 计算资金费率成本
        hours_held = (timestamp - trade.entry_time).total_seconds() / 3600.0
        funding_cost = self._calculate_funding_cost(
            trade.position_size * trade.entry_price,
            trade.direction,
            hours_held
        )
        
        # 总盈亏（扣除所有成本）
        total_pnl = pnl - exit_fees - funding_cost
        pnl_pct = total_pnl / (trade.position_size * trade.entry_price) if trade.position_size * trade.entry_price > 0 else 0.0
        
        # 更新交易记录
        trade.exit_time = timestamp
        trade.exit_price = actual_exit_price
        trade.pnl = total_pnl
        trade.pnl_pct = pnl_pct
        trade.fees += exit_fees
        trade.funding_cost = funding_cost
        trade.exit_reason = exit_reason
        trade.is_open = False
        
        # 更新权益
        self.current_equity += total_pnl
        self.risk_engine.update_equity(self.current_equity)
        
        # 更新每日盈亏（简化：假设每笔交易在同一天）
        self.risk_engine.update_daily_pnl(total_pnl)
        
        return trade
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        signals: Optional[pd.Series] = None,
        atr_values: Optional[pd.Series] = None,
        atr_multiplier: float = 2.0,
        reward_risk_ratio: float = 2.0,
        timestamp_col: str = 'timestamp',
        price_col: str = 'close'
    ) -> BacktestResults:
        """
        运行完整回测
        
        Args:
            data: 价格数据
            predictions: 模型预测序列（与data对齐）
            signals: 交易信号序列（可选，如果为None，根据predictions自动生成）
            atr_values: ATR值序列（可选）
            atr_multiplier: ATR倍数
            reward_risk_ratio: 风险收益比
            timestamp_col: 时间戳列名
            price_col: 价格列名
            
        Returns:
            BacktestResults对象
        """
        self.reset()
        
        # 确保数据按时间排序
        data = data.sort_values(by=timestamp_col).reset_index(drop=True)
        predictions = predictions.reindex(data.index)
        
        if atr_values is not None:
            atr_values = atr_values.reindex(data.index)
        
        # 遍历每个时间点
        for idx, row in data.iterrows():
            timestamp = row[timestamp_col]
            price = row[price_col]
            open_price = row.get('open', price)
            high_price = row.get('high', price)
            low_price = row.get('low', price)
            prediction = predictions.loc[idx] if idx in predictions.index else 0.5
            
            # 获取ATR值
            atr = atr_values.iloc[idx] if atr_values is not None and idx < len(atr_values) else price * 0.01
            
            # 更新现有交易（使用整根K线的OHLC进行模拟）
            self.update_trades(
                timestamp=timestamp,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=price,
                atr=atr,
            )
            
            # 检查是否有开仓信号
            if signals is not None:
                signal = signals.iloc[idx] if idx < len(signals) else 0
            else:
                # 自动生成信号：预测值偏离0.5足够远时开仓
                signal = 1 if prediction > 0.6 else (-1 if prediction < 0.4 else 0)
            
            # 如果有信号且没有开仓，尝试开仓
            if signal != 0 and len(self.open_trades) == 0:
                direction = TradeDirection.LONG if signal > 0 else TradeDirection.SHORT
                self.enter_trade(
                    timestamp=timestamp,
                    price=price,
                    direction=direction,
                    prediction=prediction,
                    atr=atr,
                    atr_multiplier=atr_multiplier,
                    reward_risk_ratio=reward_risk_ratio
                )
            
            # 记录权益
            self.equity_history.append(self.current_equity)
            self.time_history.append(timestamp)
        
        # 强制平仓所有未平仓交易（使用最后一根K线信息）
        final_row = data.iloc[-1]
        final_price = final_row[price_col]
        final_timestamp = final_row[timestamp_col]
        final_open = final_row.get('open', final_price)
        final_high = final_row.get('high', final_price)
        final_low = final_row.get('low', final_price)
        
        for trade in self.open_trades[:]:
            trade.exit_bar_open = final_open
            trade.exit_bar_high = final_high
            trade.exit_bar_low = final_low
            trade.exit_bar_close = final_price
            trade.gap_exit = False

            self._close_trade(trade, final_timestamp, final_price, "强制平仓")
            self.open_trades.remove(trade)
        
        # 计算权益曲线和收益率（15分钟级别）
        equity_series = pd.Series(self.equity_history[1:], index=self.time_history)
        returns = equity_series.pct_change().fillna(0.0)

        # 直接在15分钟级别上计算所有指标（已通过periods_per_year正确年化）
        metrics = self.metrics_calculator.calculate_all_metrics(returns, equity_series)
        
        # 转换为字典
        metrics_dict = {
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'omega_ratio': metrics.omega_ratio,
            'max_drawdown': metrics.max_drawdown,
            'profit_factor': metrics.profit_factor,
            'calmar_ratio': metrics.calmar_ratio,
            'total_return': metrics.total_return,
            'annualized_return': metrics.annualized_return,
            'volatility': metrics.volatility,
            'win_rate': metrics.win_rate,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'max_consecutive_losses': metrics.max_consecutive_losses,
            'max_consecutive_wins': metrics.max_consecutive_wins,
            'avg_trade_duration': metrics.avg_trade_duration,
            'expectancy': metrics.expectancy,
            'max_dd_duration': metrics.max_dd_duration,
            'recovery_factor': metrics.recovery_factor
        }
        
        # 生成交易日志DataFrame
        trade_log = self._generate_trade_log()
        
        return BacktestResults(
            equity_curve=equity_series,
            returns=returns,
            trades=self.trades,
            metrics=metrics_dict,
            trade_log=trade_log,
            daily_returns=self._calculate_daily_returns(equity_series)
        )

    def _generate_trade_log(self) -> pd.DataFrame:
        """生成交易日志DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        log_data = []
        for trade in self.trades:
            log_data.append({
                'trade_id': trade.trade_id,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'direction': trade.direction.name,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_size': trade.position_size,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'fees': trade.fees,
                'slippage': trade.slippage,
                'funding_cost': trade.funding_cost,
                'exit_reason': trade.exit_reason,
                # 退出所在K线的OHLC信息
                'bar_open': trade.exit_bar_open,
                'bar_high': trade.exit_bar_high,
                'bar_low': trade.exit_bar_low,
                'bar_close': trade.exit_bar_close,
                # 是否因跳空而在开盘价离场
                'gap_exit': trade.gap_exit,
            })

        return pd.DataFrame(log_data)

    def _calculate_daily_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate daily returns."""
        if len(equity_curve) == 0:
            return pd.Series()

        # Group by date
        equity_curve.index = pd.to_datetime(equity_curve.index)
        daily_equity = equity_curve.resample('D').last()
        daily_returns = daily_equity.pct_change().fillna(0.0)

        return daily_returns


class RealisticExecutionSimulator:
    """
    更加贴近期货交易现实的执行模拟:
    - 手续费 (taker/maker)
    - 波动率驱动的滑点
    - 成交量不足时的部分成交
    - 周期性资金费率
    """

    def __init__(self, config: Dict) -> None:
        self.base_fee: float = float(config.get("taker_fee", 0.0005))
        self.maker_fee: float = float(config.get("maker_fee", 0.0002))
        self.slippage_k: float = float(config.get("slippage_k", 0.1))
        # 32 * 15m = 8h
        self.funding_interval: int = int(config.get("funding_interval", 32))

    def calculate_fill_price(
        self,
        order_price: float,
        side: int,
        current_volatility: float,
        volume_ratio: float,
    ) -> Dict[str, float]:
        """
        根据波动率 & 流动性估算实际成交价.

        Args:
            order_price: 下单价格
            side: 1=buy, -1=sell
            current_volatility: 当前波动率（例如15m收益率的rolling std）
            volume_ratio: 当前成交量与均值之比 (>1流动性更好)
        """
        # Slippage 基于波动率
        slippage_pct = self.slippage_k * max(current_volatility, 0.0)

        # 流动性较好 -> 滑点略小；流动性较差 -> 滑点放大
        if volume_ratio > 1.5:
            slippage_pct *= 0.7
        elif volume_ratio < 0.5:
            slippage_pct *= 1.5

        slippage_pct = max(slippage_pct, 0.0)

        # 按交易方向施加不利滑点
        fill_price = order_price * (1.0 + side * slippage_pct)

        return {
            "fill_price": float(fill_price),
            "slippage_cost": float(abs(fill_price - order_price) / max(order_price, 1e-9)),
            "fee": float(self.base_fee),
        }

    def calculate_funding_cost(
        self,
        position_size: float,
        position_value: float,
        side: int,
        funding_rate: float,
        candles_held: int,
    ) -> float:
        """
        资金费率每 funding_interval 根K线收取一次.

        正资金费率 -> 多头支付空头.
        """
        if candles_held <= 0 or position_value <= 0:
            return 0.0

        payments = candles_held // max(self.funding_interval, 1)
        if payments <= 0:
            return 0.0

        # side=1 多头; side=-1 空头; 正 funding_rate 时多头付钱
        cost_per_payment = funding_rate * position_value * side
        # 我们关心的是成本 => 对策略而言是pnl的减少
        return float(cost_per_payment * payments)

    def simulate_partial_fill(
        self,
        order_size: float,
        available_liquidity: float,
    ) -> float:
        """
        模拟在流动性不足时的部分成交比例.
        """
        if order_size <= 0:
            return 0.0
        fill_ratio = min(1.0, max(available_liquidity, 0.0) / (order_size * 1.5))
        return float(order_size * fill_ratio)
    
