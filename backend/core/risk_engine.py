"""
风险管理引擎

实现hedge-fund级别的动态风险管理：
- 基于波动率的动态仓位管理
- ATR-based止损距离计算
- 杠杆控制
- 最大回撤保护
- 单日亏损限制

核心原则：
- 风险优先：每笔交易的风险是固定的（账户权益的百分比）
- 动态调整：根据市场波动率调整仓位大小
- 严格限制：多重风险控制机制
"""
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backend.config import RiskConfig


@dataclass
class PositionSize:
    """仓位大小计算结果"""
    position_size: float      # 仓位大小（合约数量）
    position_value: float     # 仓位价值（USD）
    leverage: float           # 杠杆倍数
    stop_distance: float      # 止损距离（价格）
    risk_amount: float        # 风险金额（USD）


@dataclass
class RiskState:
    """风险状态"""
    equity: float                    # 当前权益
    daily_pnl: float                 # 当日盈亏
    max_equity: float                # 历史最高权益
    current_drawdown: float          # 当前回撤
    is_trading_allowed: bool         # 是否允许交易
    reason: str                      # 状态原因


class RiskEngine:
    """
    风险管理引擎
    
    实现动态风险管理和仓位计算：
    1. 基于ATR的止损距离
    2. 固定风险百分比计算仓位
    3. 波动率缩放
    4. 多重风险限制检查
    """
    
    def __init__(self, config: RiskConfig):
        """
        初始化风险管理引擎
        
        Args:
            config: 风险管理配置
        """
        self.config = config
        self.risk_state = RiskState(
            equity=100000.0,  # 初始权益（默认10万USD）
            daily_pnl=0.0,
            max_equity=100000.0,
            current_drawdown=0.0,
            is_trading_allowed=True,
            reason="正常"
        )
    
    def update_equity(self, equity: float):
        """
        更新账户权益
        
        Args:
            equity: 当前账户权益
        """
        self.risk_state.equity = equity
        
        # 更新历史最高权益
        if equity > self.risk_state.max_equity:
            self.risk_state.max_equity = equity
        
        # 计算当前回撤
        self.risk_state.current_drawdown = (
            (self.risk_state.max_equity - equity) / self.risk_state.max_equity
        )
        
        # 检查最大回撤限制
        if self.risk_state.current_drawdown >= self.config.max_drawdown_stop:
            self.risk_state.is_trading_allowed = False
            self.risk_state.reason = f"达到最大回撤限制: {self.risk_state.current_drawdown:.2%}"
        else:
            self.risk_state.is_trading_allowed = True
            self.risk_state.reason = "正常"
    
    def update_daily_pnl(self, pnl: float):
        """
        更新当日盈亏
        
        Args:
            pnl: 当日盈亏金额
        """
        self.risk_state.daily_pnl = pnl
        
        # 检查单日亏损限制
        daily_loss_pct = abs(pnl) / self.risk_state.equity if self.risk_state.equity > 0 else 0
        
        if pnl < 0 and daily_loss_pct >= self.config.max_daily_loss:
            self.risk_state.is_trading_allowed = False
            self.risk_state.reason = f"达到单日亏损限制: {daily_loss_pct:.2%}"
    
    def reset_daily_pnl(self):
        """重置当日盈亏（新的一天开始时调用）"""
        self.risk_state.daily_pnl = 0.0
        # 重新检查交易权限
        if self.risk_state.current_drawdown < self.config.max_drawdown_stop:
            self.risk_state.is_trading_allowed = True
            self.risk_state.reason = "正常"
    
    def calculate_atr_stop_distance(
        self,
        atr: float,
        atr_multiplier: float = 2.0,
        price: Optional[float] = None
    ) -> float:
        """
        计算基于ATR的止损距离
        
        Args:
            atr: ATR值
            atr_multiplier: ATR倍数
            price: 当前价格（可选，用于计算百分比距离）
            
        Returns:
            止损距离（价格单位）
        """
        stop_distance = atr * atr_multiplier
        
        if price is not None:
            # 也可以返回百分比距离
            return stop_distance / price
        
        return stop_distance
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        atr: Optional[float] = None,
        volatility_scale: Optional[float] = None
    ) -> PositionSize:
        """
        计算仓位大小
        
        核心公式：
        Position Size = (Equity * Risk%) / Stop Distance
        
        Args:
            entry_price: 入场价格
            stop_price: 止损价格
            atr: ATR值（可选，用于波动率缩放）
            volatility_scale: 波动率缩放因子（可选）
            
        Returns:
            PositionSize对象
        """
        if not self.risk_state.is_trading_allowed:
            return PositionSize(
                position_size=0.0,
                position_value=0.0,
                leverage=0.0,
                stop_distance=0.0,
                risk_amount=0.0
            )
        
        # 计算止损距离（价格）
        stop_distance = abs(entry_price - stop_price)
        
        if stop_distance == 0:
            return PositionSize(
                position_size=0.0,
                position_value=0.0,
                leverage=0.0,
                stop_distance=0.0,
                risk_amount=0.0
            )
        
        # 计算风险金额
        risk_amount = self.risk_state.equity * self.config.risk_per_trade
        
        # 波动率缩放
        if self.config.volatility_scaling and volatility_scale is not None:
            risk_amount *= volatility_scale
        elif self.config.volatility_scaling and atr is not None:
            # 使用ATR计算波动率缩放
            # 假设基准ATR为价格的1%
            baseline_atr_pct = 0.01
            current_atr_pct = atr / entry_price if entry_price > 0 else baseline_atr_pct
            volatility_scale = baseline_atr_pct / current_atr_pct
            volatility_scale = np.clip(volatility_scale, 0.5, 2.0)  # 限制缩放范围
            risk_amount *= volatility_scale
        
        # 计算仓位大小（合约数量）
        # 对于BTCUSDT期货，1个合约 = 1 USD
        position_size = risk_amount / stop_distance
        
        # 计算仓位价值
        position_value = position_size * entry_price
        
        # 计算杠杆
        leverage = position_value / self.risk_state.equity if self.risk_state.equity > 0 else 0.0
        
        # 检查最大杠杆限制
        if leverage > self.config.max_leverage:
            # 按最大杠杆调整仓位
            max_position_value = self.risk_state.equity * self.config.max_leverage
            position_size = max_position_value / entry_price
            position_value = max_position_value
            leverage = self.config.max_leverage
        
        return PositionSize(
            position_size=position_size,
            position_value=position_value,
            leverage=leverage,
            stop_distance=stop_distance,
            risk_amount=risk_amount
        )
    
    def calculate_volatility_scale(
        self,
        returns: pd.Series,
        lookback: Optional[int] = None
    ) -> float:
        """
        计算波动率缩放因子
        
        当市场波动率较高时，减小仓位；波动率较低时，增加仓位
        
        Args:
            returns: 历史收益率序列
            lookback: 回看期数（如果为None，使用配置值）
            
        Returns:
            波动率缩放因子（0.5-2.0）
        """
        if lookback is None:
            lookback = self.config.volatility_lookback
        
        if len(returns) < lookback:
            return 1.0
        
        # 计算最近N期的波动率
        recent_vol = returns.tail(lookback).std()
        
        # 计算长期波动率（全样本）
        long_term_vol = returns.std()
        
        if long_term_vol == 0:
            return 1.0
        
        # 缩放因子：波动率越高，仓位越小
        scale = long_term_vol / recent_vol
        
        # 限制在合理范围内
        scale = np.clip(scale, 0.5, 2.0)
        
        return scale
    
    def check_trade_allowed(self) -> Tuple[bool, str]:
        """
        检查是否允许交易
        
        Returns:
            (是否允许, 原因)
        """
        return (self.risk_state.is_trading_allowed, self.risk_state.reason)
    
    def get_risk_state(self) -> RiskState:
        """
        获取当前风险状态
        
        Returns:
            RiskState对象
        """
        return self.risk_state
    
    def reset(self, initial_equity: float = 100000.0):
        """
        重置风险引擎
        
        Args:
            initial_equity: 初始权益
        """
        self.risk_state = RiskState(
            equity=initial_equity,
            daily_pnl=0.0,
            max_equity=initial_equity,
            current_drawdown=0.0,
            is_trading_allowed=True,
            reason="正常"
        )
