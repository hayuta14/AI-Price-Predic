"""
Adaptive Position Sizing

Thay đổi risk per trade dựa trên model confidence:
- Risk = base_risk * confidence_multiplier
- Confidence = |probability - 0.5| / 0.5
- Giúp giảm DD và tăng capital efficiency
"""
import pandas as pd
import numpy as np
from typing import Optional


class AdaptivePositionSizing:
    """Adaptive position sizing dựa trên model confidence"""
    
    def __init__(
        self,
        base_risk: float = 0.01,
        use_adaptive: bool = True,
        min_risk: float = 0.005,
        max_risk: float = 0.02,
        confidence_threshold: float = 0.1
    ):
        """
        Khởi tạo Adaptive Position Sizing
        
        Args:
            base_risk: Risk cơ bản mỗi trade (mặc định: 1%)
            use_adaptive: Có sử dụng adaptive sizing không
            min_risk: Risk tối thiểu (0.5%)
            max_risk: Risk tối đa (2%)
            confidence_threshold: Ngưỡng confidence tối thiểu để trade
        """
        self.base_risk = base_risk
        self.use_adaptive = use_adaptive
        self.min_risk = min_risk
        self.max_risk = max_risk
        self.confidence_threshold = confidence_threshold
    
    def calculate_confidence(self, probability: float) -> float:
        """
        Tính confidence từ probability
        
        Args:
            probability: Model probability (0-1)
            
        Returns:
            Confidence score (0-1)
        """
        # Confidence = |prob - 0.5| / 0.5
        # prob = 0.5 → confidence = 0 (không chắc chắn)
        # prob = 0.0 hoặc 1.0 → confidence = 1 (rất chắc chắn)
        confidence = abs(probability - 0.5) / 0.5
        return min(1.0, max(0.0, confidence))
    
    def calculate_risk(
        self,
        probability: float,
        signal: int
    ) -> float:
        """
        Tính risk per trade dựa trên confidence
        
        Args:
            probability: Model probability
            signal: Signal direction (1, -1, 0)
            
        Returns:
            Risk per trade (0-1)
        """
        if not self.use_adaptive or signal == 0:
            return self.base_risk
        
        # Tính confidence
        confidence = self.calculate_confidence(probability)
        
        # Kiểm tra confidence threshold
        if confidence < self.confidence_threshold:
            return 0.0  # Không trade nếu confidence quá thấp
        
        # Tiered risk sizing thay vì linear để tránh risk quá nhỏ khi confidence trung bình
        # confidence: [0, 1]
        # - low:    [threshold, 0.35) -> 0.6x base_risk
        # - medium: [0.35, 0.70)      -> 1.0x base_risk
        # - high:   [0.70, 1.00]      -> 1.4x base_risk
        if confidence >= 0.70:
            risk_multiplier = 1.4
        elif confidence >= 0.35:
            risk_multiplier = 1.0
        else:
            risk_multiplier = 0.6

        risk = self.base_risk * risk_multiplier

        # Giới hạn trong [min_risk, max_risk]
        risk = max(self.min_risk, min(self.max_risk, risk))
        
        return risk
    
    def calculate_position_sizes(
        self,
        predictions: pd.Series,
        signals: pd.Series,
        prices: pd.Series,
        atr_values: Optional[pd.Series] = None,
        equity: float = 100000.0
    ) -> pd.Series:
        """
        Tính position sizes cho tất cả signals
        
        Args:
            predictions: Model predictions (probabilities)
            signals: Trading signals (1, -1, 0)
            prices: Prices
            atr_values: ATR values (optional, for stop loss calculation)
            equity: Current equity
            
        Returns:
            Series với position sizes (số lượng contracts/shares)
        """
        position_sizes = pd.Series(0.0, index=signals.index)
        
        for idx in signals.index:
            if signals.loc[idx] != 0:
                prob = predictions.loc[idx] if idx in predictions.index else 0.5
                signal = signals.loc[idx]
                
                # Tính risk
                risk = self.calculate_risk(prob, signal)
                
                if risk > 0:
                    # Tính position size dựa trên risk
                    if atr_values is not None and idx in atr_values.index:
                        # Sử dụng ATR để tính stop loss
                        atr = atr_values.loc[idx]
                        price = prices.loc[idx] if idx in prices.index else 0
                        
                        if price > 0 and atr > 0:
                            # Stop loss = ATR * multiplier (ví dụ: 2 * ATR)
                            stop_loss_pct = (2.0 * atr) / price
                            
                            # Position size = (equity * risk) / stop_loss_amount
                            risk_amount = equity * risk
                            stop_loss_amount = equity * stop_loss_pct
                            
                            if stop_loss_amount > 0:
                                position_size = risk_amount / stop_loss_amount
                            else:
                                position_size = 0.0
                        else:
                            position_size = 0.0
                    else:
                        # Fallback: sử dụng fixed stop loss (ví dụ: 2%)
                        stop_loss_pct = 0.02
                        risk_amount = equity * risk
                        stop_loss_amount = equity * stop_loss_pct
                        position_size = risk_amount / stop_loss_amount if stop_loss_amount > 0 else 0.0
                    
                    position_sizes.loc[idx] = position_size
        
        return position_sizes
