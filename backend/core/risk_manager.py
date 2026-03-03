"""
Futures Risk Manager

Advanced risk controls for futures trading with volatility/regime/confidence sizing.
"""
from __future__ import annotations

from typing import Dict
import logging


logger = logging.getLogger(__name__)


class FuturesRiskManager:
    def __init__(self, config: Dict):
        self.max_leverage = config.get('max_leverage', 5)
        self.base_risk_per_trade = config.get('base_risk_pct', 0.01)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_drawdown_kill = config.get('max_drawdown_kill', 0.15)
        self.volatility_lookback = config.get('vol_lookback', 96)

        self.daily_pnl = 0.0
        self.peak_equity = 1.0
        self.is_killed = False

    def calculate_position_size(
        self,
        equity: float,
        signal: int,
        confidence: float,
        current_volatility: float,
        regime: str,
        atr: float,
        price: float,
    ) -> Dict:
        if self.is_killed:
            return {'size': 0.0, 'leverage': 0.0, 'reason': 'kill_switch'}

        if self.daily_pnl <= -self.max_daily_loss:
            return {'size': 0.0, 'leverage': 0.0, 'reason': 'daily_loss_limit'}

        if signal == 0 or price <= 0 or atr <= 0:
            return {'size': 0.0, 'leverage': 0.0, 'reason': 'invalid_inputs'}

        atr_sl_distance = atr * 2.0
        risk_amount = equity * self.base_risk_per_trade
        base_size = risk_amount / max((atr_sl_distance / price), 1e-9)

        vol_scalar = self._volatility_scalar(current_volatility)
        regime_scalar = self._regime_scalar(regime)
        confidence_scalar = self._confidence_scalar(confidence)

        position_size = base_size * vol_scalar * regime_scalar * confidence_scalar

        leverage = (position_size * price) / max(equity, 1e-9)
        leverage = min(leverage, self.max_leverage)
        position_size = (leverage * equity) / price

        return {
            'size': round(position_size, 6),
            'leverage': round(leverage, 2),
            'vol_scalar': vol_scalar,
            'regime_scalar': regime_scalar,
            'confidence_scalar': confidence_scalar,
            'sl_price': price - signal * atr_sl_distance,
            'tp_price': price + signal * atr_sl_distance * 1.5,
        }

    def _volatility_scalar(self, current_vol: float) -> float:
        avg_vol = 0.015
        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        if ratio > 3.0:
            return 0.2
        if ratio > 2.0:
            return 0.4
        if ratio > 1.5:
            return 0.7
        if ratio < 0.5:
            return 0.8
        return 1.0

    def _regime_scalar(self, regime: str) -> float:
        scalars = {
            'trending_up': 1.0,
            'trending_down': 1.0,
            'sideways': 0.25,
            'high_volatility': 0.4,
            'low_volatility': 0.7,
        }
        return scalars.get(regime, 0.5)

    def _confidence_scalar(self, confidence: float) -> float:
        if confidence > 0.75:
            return 1.0
        if confidence > 0.65:
            return 0.75
        if confidence > 0.55:
            return 0.5
        return 0.0

    def update_daily_pnl(self, pnl_delta: float, current_equity: float) -> None:
        self.daily_pnl += pnl_delta
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (current_equity - self.peak_equity) / max(self.peak_equity, 1e-9)
        if drawdown <= -self.max_drawdown_kill:
            self.is_killed = True
            logger.warning(f"KILL SWITCH ACTIVATED: Drawdown {drawdown:.1%}")

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    rm = FuturesRiskManager({})
    size = rm.calculate_position_size(
        equity=100000,
        signal=1,
        confidence=0.72,
        current_volatility=0.012,
        regime='trending_up',
        atr=120,
        price=50000,
    )
    print(size)
