"""
Volatility Filter

Filter trades dựa trên volatility regime:
- Chỉ trade khi volatility phù hợp
- Sử dụng volatility features có MI cao (volatility_20, atr_pct, hl_range_ma)
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict


class VolatilityFilter:
    """Filter trades dựa trên volatility"""
    
    def __init__(
        self,
        use_volatility_filter: bool = True,
        min_volatility: Optional[float] = None,
        max_volatility: Optional[float] = None,
        use_atr_filter: bool = True,
        atr_percentile_low: float = 0.2,
        atr_percentile_high: float = 0.8
    ):
        """
        Khởi tạo Volatility Filter
        
        Args:
            use_volatility_filter: Có sử dụng volatility filter không
            min_volatility: Volatility tối thiểu (None = không giới hạn)
            max_volatility: Volatility tối đa (None = không giới hạn)
            use_atr_filter: Có filter theo ATR percentile không
            atr_percentile_low: ATR percentile thấp (chỉ trade khi ATR > percentile này)
            atr_percentile_high: ATR percentile cao (chỉ trade khi ATR < percentile này)
        """
        self.use_volatility_filter = use_volatility_filter
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.use_atr_filter = use_atr_filter
        self.atr_percentile_low = atr_percentile_low
        self.atr_percentile_high = atr_percentile_high
    
    def should_trade(
        self,
        data: pd.DataFrame,
        index: int,
        volatility_col: str = 'volatility_20',
        atr_col: str = 'atr_14_pct',
        hl_range_col: str = 'hl_range_ma'
    ) -> bool:
        """
        Kiểm tra xem có nên trade tại index này không
        
        Args:
            data: DataFrame với volatility features
            index: Index hiện tại
            volatility_col: Tên cột volatility
            atr_col: Tên cột ATR percentage
            hl_range_col: Tên cột high-low range MA
            
        Returns:
            True nếu nên trade, False nếu không
        """
        if not self.use_volatility_filter:
            return True
        
        if index >= len(data):
            return False
        
        row = data.iloc[index]
        
        # 1. Filter theo volatility_20
        if volatility_col in data.columns:
            vol_value = row[volatility_col]
            if pd.isna(vol_value):
                return False
            
            # Tính volatility percentile
            if index >= 50:
                vol_history = data[volatility_col].iloc[max(0, index-50):index]
                vol_percentile = (vol_history < vol_value).sum() / len(vol_history)
                
                # Chỉ trade khi volatility không quá cao (tránh whipsaw)
                if vol_percentile > 0.8:  # Volatility quá cao
                    return False
        
        # 2. Filter theo ATR percentage
        if self.use_atr_filter and atr_col in data.columns:
            atr_value = row[atr_col]
            if pd.isna(atr_value):
                return False
            
            # Tính ATR percentile
            if index >= 50:
                atr_history = data[atr_col].iloc[max(0, index-50):index]
                atr_percentile = (atr_history < atr_value).sum() / len(atr_history)
                
                # Chỉ trade khi ATR trong range hợp lý
                if atr_percentile < self.atr_percentile_low or atr_percentile > self.atr_percentile_high:
                    return False
        
        # 3. Filter theo high-low range
        if hl_range_col in data.columns:
            hl_value = row[hl_range_col]
            if pd.isna(hl_value):
                return False
            
            # Tính HL range percentile
            if index >= 20:
                hl_history = data[hl_range_col].iloc[max(0, index-20):index]
                hl_percentile = (hl_history < hl_value).sum() / len(hl_history)
                
                # Chỉ trade khi range không quá nhỏ (tránh low volatility noise)
                if hl_percentile < 0.1:  # Range quá nhỏ
                    return False
        
        return True
    
    def filter_signals(
        self,
        signals: pd.Series,
        data: pd.DataFrame,
        volatility_col: str = 'volatility_20',
        atr_col: str = 'atr_14_pct',
        hl_range_col: str = 'hl_range_ma'
    ) -> pd.Series:
        """
        Filter signals dựa trên volatility
        
        Args:
            signals: Series signals (1, -1, 0)
            data: DataFrame với volatility features
            volatility_col: Tên cột volatility
            atr_col: Tên cột ATR percentage
            hl_range_col: Tên cột high-low range MA
            
        Returns:
            Filtered signals
        """
        if not self.use_volatility_filter:
            return signals
        
        filtered_signals = signals.copy()
        
        for idx in signals.index:
            if signals.loc[idx] != 0:  # Có signal
                data_idx = data.index.get_loc(idx) if idx in data.index else -1
                if data_idx >= 0:
                    if not self.should_trade(data, data_idx, volatility_col, atr_col, hl_range_col):
                        filtered_signals.loc[idx] = 0  # Filter out
        
        return filtered_signals
