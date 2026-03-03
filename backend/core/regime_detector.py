"""
Regime Detector Module

Detect market regimes for risk sizing and model behavior adaptation.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class RegimeDetector:
    """Detect market regime to adjust risk and model behavior."""

    def __init__(self) -> None:
        self._last_regime: str | None = None

    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if not all(col in out.columns for col in ["high", "low", "close"]):
            out["regime"] = MarketRegime.LOW_VOLATILITY.value
            out["regime_size_mult"] = 0.8
            return out

        out["adx"] = self._calculate_adx(out, 14)
        out["rv_short"] = out["close"].pct_change().rolling(8).std()
        out["rv_long"] = out["close"].pct_change().rolling(48).std()
        out["vol_regime"] = out["rv_short"] / out["rv_long"].replace(0, np.nan)

        out["ema_fast"] = out["close"].ewm(span=20).mean()
        out["ema_slow"] = out["close"].ewm(span=50).mean()
        out["trend_direction"] = np.sign(out["ema_fast"] - out["ema_slow"])

        out["regime"] = out.apply(self._classify_regime, axis=1)

        regime_multiplier: Dict[str, float] = {
            MarketRegime.TRENDING_UP.value: 1.0,
            MarketRegime.TRENDING_DOWN.value: 1.0,
            MarketRegime.SIDEWAYS.value: 0.3,
            MarketRegime.HIGH_VOLATILITY.value: 0.5,
            MarketRegime.LOW_VOLATILITY.value: 0.8,
        }
        out["regime_size_mult"] = out["regime"].map(regime_multiplier).fillna(0.5)

        if len(out) > 0:
            last_regime = str(out["regime"].iloc[-1])
            if last_regime != self._last_regime:
                logger.info(f"Regime changed: {self._last_regime} -> {last_regime}")
                self._last_regime = last_regime

        return out

    def _classify_regime(self, row: pd.Series) -> str:
        adx = row.get("adx", 20)
        vol_ratio = row.get("vol_regime", 1.0)
        trend = row.get("trend_direction", 0)

        if pd.notna(vol_ratio) and vol_ratio > 2.0:
            return MarketRegime.HIGH_VOLATILITY.value
        if pd.notna(adx) and adx < 20:
            return MarketRegime.SIDEWAYS.value
        if pd.notna(adx) and adx > 25 and trend > 0:
            return MarketRegime.TRENDING_UP.value
        if pd.notna(adx) and adx > 25 and trend < 0:
            return MarketRegime.TRENDING_DOWN.value
        return MarketRegime.LOW_VOLATILITY.value

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.abs().rolling(period).mean() / atr.replace(0, np.nan))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        return dx.rolling(period).mean().fillna(20.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    base = np.cumsum(np.random.randn(n) * 10) + 50000
    sample = pd.DataFrame(
        {
            "timestamp": idx,
            "open": base,
            "high": base + np.random.rand(n) * 50,
            "low": base - np.random.rand(n) * 50,
            "close": base + np.random.randn(n) * 5,
            "volume": np.random.randint(100, 2000, n),
        }
    )

    detector = RegimeDetector()
    out = detector.detect_regime(sample)
    print(out[["close", "adx", "vol_regime", "regime", "regime_size_mult"]].tail())
