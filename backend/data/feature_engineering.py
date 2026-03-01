"""
Feature Engineering Module

Tạo các đặc trưng kỹ thuật phong phú cho mô hình XGBoost:
- RSI multiple periods (7, 14, 21)
- SMA 20/50/100
- EMA cross
- ATR
- Volatility rolling
- Volume features
- Market regime features
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Tính toán chỉ báo RSI (Relative Strength Index)
    
    Args:
        prices: Chuỗi giá
        period: Chu kỳ tính toán
        
    Returns:
        Chuỗi RSI
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Tính toán Simple Moving Average
    
    Args:
        prices: Chuỗi giá
        period: Chu kỳ
        
    Returns:
        Chuỗi SMA
    """
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Tính toán Exponential Moving Average
    
    Args:
        prices: Chuỗi giá
        period: Chu kỳ
        
    Returns:
        Chuỗi EMA
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Tính toán ATR (Average True Range)
    
    Args:
        data: DataFrame chứa high, low, close
        period: Chu kỳ tính toán
        
    Returns:
        Chuỗi ATR
    """
    if not all(col in data.columns for col in ['high', 'low', 'close']):
        return pd.Series(0.0, index=data.index)
    
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.fillna(tr.mean() if len(tr) > 0 else 0.0)


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Tính toán độ biến động (rolling volatility)
    
    Args:
        returns: Chuỗi lợi nhuận
        window: Cửa sổ rolling
        
    Returns:
        Chuỗi volatility
    """
    return returns.rolling(window=window).std()


def calculate_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán các đặc trưng volume
    
    Args:
        data: DataFrame chứa volume
        
    Returns:
        DataFrame với các cột volume features
    """
    result = pd.DataFrame(index=data.index)
    
    if 'volume' not in data.columns:
        return result
    
    volume = data['volume']
    
    # Volume moving averages
    result['volume_sma_10'] = volume.rolling(10).mean()
    result['volume_sma_20'] = volume.rolling(20).mean()
    result['volume_sma_50'] = volume.rolling(50).mean()
    
    # Volume ratio
    result['volume_ratio_10'] = volume / result['volume_sma_10']
    result['volume_ratio_20'] = volume / result['volume_sma_20']
    
    # Volume change
    result['volume_change'] = volume.pct_change()
    result['volume_change_abs'] = abs(result['volume_change'])
    
    # Price-volume relationship
    if 'close' in data.columns:
        price_change = data['close'].pct_change()
        result['price_volume_corr'] = price_change.rolling(20).corr(volume.pct_change())
    
    return result.fillna(0.0)


def calculate_market_regime(data: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Tính toán market regime features
    
    Args:
        data: DataFrame chứa giá
        lookback: Chu kỳ nhìn lại
        
    Returns:
        DataFrame với các cột regime features
    """
    result = pd.DataFrame(index=data.index)
    
    if 'close' not in data.columns:
        return result
    
    prices = data['close']
    returns = prices.pct_change()
    
    # Trend regime (uptrend/downtrend/sideways)
    sma_short = calculate_sma(prices, 20)
    sma_long = calculate_sma(prices, 50)
    result['trend_regime'] = np.where(
        sma_short > sma_long, 1,  # Uptrend
        np.where(sma_short < sma_long, -1, 0)  # Downtrend, else sideways
    )
    
    # Volatility regime
    volatility = calculate_volatility(returns, window=lookback)
    vol_median = volatility.rolling(lookback * 2).median()
    result['volatility_regime'] = np.where(
        volatility > vol_median, 1,  # High volatility
        -1  # Low volatility
    )
    
    # Price position in range
    high_rolling = prices.rolling(lookback).max()
    low_rolling = prices.rolling(lookback).min()
    result['price_position'] = (prices - low_rolling) / (high_rolling - low_rolling)
    result['price_position'] = result['price_position'].fillna(0.5)
    
    # Momentum regime
    momentum = returns.rolling(lookback).sum()
    result['momentum_regime'] = np.where(
        momentum > 0, 1,  # Positive momentum
        -1  # Negative momentum
    )
    
    return result.fillna(0.0)


def create_all_features(data: pd.DataFrame, timestamp_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tạo tất cả các đặc trưng cho mô hình
    
    Args:
        data: DataFrame chứa OHLCV data
        
    Returns:
        Tuple (data_with_features, feature_list)
    """
    df = data.copy()
    features = []
    
    # Đảm bảo dữ liệu được sắp xếp theo thời gian
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Returns
    if 'close' in df.columns:
        df['returns'] = df['close'].pct_change()
        df['returns_abs'] = abs(df['returns'])
        features.extend(['returns', 'returns_abs'])
    
    # 2. RSI multiple periods (multi-timeframe structure)
    if 'close' in df.columns:
        # RSI 7 và RSI 21 cho multi-timeframe analysis
        for period in [7, 14, 21]:
            col_name = f'rsi_{period}'
            df[col_name] = calculate_rsi(df['close'], period)
            features.append(col_name)
        
        # Multi-timeframe RSI ratios
        if all(col in df.columns for col in ['rsi_7', 'rsi_21']):
            df['rsi_7_21_ratio'] = df['rsi_7'] / df['rsi_21']
            df['rsi_7_21_diff'] = df['rsi_7'] - df['rsi_21']
            features.extend(['rsi_7_21_ratio', 'rsi_7_21_diff'])
        
        # Higher timeframe RSI (50, 100) cho trend context
        for period in [50, 100]:
            col_name = f'rsi_{period}'
            df[col_name] = calculate_rsi(df['close'], period)
            features.append(col_name)
        
        # RSI higher timeframe ratios
        if all(col in df.columns for col in ['rsi_14', 'rsi_50']):
            df['rsi_14_50_ratio'] = df['rsi_14'] / df['rsi_50']
            features.append('rsi_14_50_ratio')
    
    # 3. SMA 20/50/100
    if 'close' in df.columns:
        for period in [20, 50, 100]:
            col_name = f'sma_{period}'
            df[col_name] = calculate_sma(df['close'], period)
            features.append(col_name)
        
        # SMA ratios
        if all(col in df.columns for col in ['sma_20', 'sma_50', 'sma_100']):
            df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
            df['sma_50_100_ratio'] = df['sma_50'] / df['sma_100']
            df['price_sma20_ratio'] = df['close'] / df['sma_20']
            df['price_sma50_ratio'] = df['close'] / df['sma_50']
            features.extend(['sma_20_50_ratio', 'sma_50_100_ratio', 
                           'price_sma20_ratio', 'price_sma50_ratio'])
    
    # 4. EMA cross
    if 'close' in df.columns:
        df['ema_12'] = calculate_ema(df['close'], 12)
        df['ema_26'] = calculate_ema(df['close'], 26)
        df['ema_50'] = calculate_ema(df['close'], 50)
        features.extend(['ema_12', 'ema_26', 'ema_50'])
        
        # EMA cross signals
        if all(col in df.columns for col in ['ema_12', 'ema_26']):
            df['ema_cross'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
            df['ema_cross_distance'] = (df['ema_12'] - df['ema_26']) / df['ema_26']
            features.extend(['ema_cross', 'ema_cross_distance'])
    
    # 5. ATR
    if all(col in df.columns for col in ['high', 'low', 'close']):
        df['atr_14'] = calculate_atr(df, 14)
        df['atr_21'] = calculate_atr(df, 21)
        features.extend(['atr_14', 'atr_21'])
        
        # ATR normalized by price (atr_pct - feature quan trọng)
        if 'close' in df.columns:
            df['atr_14_pct'] = df['atr_14'] / df['close']
            df['atr_21_pct'] = df['atr_21'] / df['close']
            # Đảm bảo atr_pct có sẵn (alias cho atr_14_pct)
            df['atr_pct'] = df['atr_14_pct']
            features.extend(['atr_14_pct', 'atr_21_pct', 'atr_pct'])
    
    # 6. Volatility rolling (quan trọng cho reward distribution)
    if 'returns' in df.columns:
        # rolling_vol_20 là feature quan trọng
        for window in [10, 20, 50]:
            col_name = f'volatility_{window}'
            df[col_name] = calculate_volatility(df['returns'], window)
            features.append(col_name)
        
        # Đảm bảo rolling_vol_20 có sẵn (alias cho volatility_20)
        if 'volatility_20' in df.columns:
            df['rolling_vol_20'] = df['volatility_20']
            if 'rolling_vol_20' not in features:
                features.append('rolling_vol_20')
    
    # 7. Volume features
    volume_features = calculate_volume_features(df)
    if not volume_features.empty:
        df = pd.concat([df, volume_features], axis=1)
        features.extend(volume_features.columns.tolist())
    
    # 8. Market regime features
    regime_features = calculate_market_regime(df, lookback=50)
    if not regime_features.empty:
        df = pd.concat([df, regime_features], axis=1)
        features.extend(regime_features.columns.tolist())
    
    # 8.1. Higher timeframe trend regime
    if 'close' in df.columns:
        # Trend regime trên timeframe dài hơn (100 periods)
        sma_short_htf = calculate_sma(df['close'], 50)
        sma_long_htf = calculate_sma(df['close'], 100)
        df['trend_regime_htf'] = np.where(
            sma_short_htf > sma_long_htf, 1,  # Uptrend higher timeframe
            np.where(sma_short_htf < sma_long_htf, -1, 0)  # Downtrend, else sideways
        )
        features.append('trend_regime_htf')
        
        # Trend regime alignment (lower vs higher timeframe)
        if 'trend_regime' in df.columns:
            df['trend_regime_alignment'] = (df['trend_regime'] == df['trend_regime_htf']).astype(int)
            features.append('trend_regime_alignment')
    
    # 9. Additional price features
    if all(col in df.columns for col in ['high', 'low', 'close']):
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_ma'] = df['hl_range'].rolling(20).mean()
        features.extend(['hl_range', 'hl_range_ma'])
        
        # Body size (close - open)
        if 'open' in df.columns:
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['body_size_ma'] = df['body_size'].rolling(20).mean()
            features.extend(['body_size', 'body_size_ma'])
    
    # 10. Directional Features (quan trọng cho XGBoost interaction)
    if 'close' in df.columns:
        # Momentum features
        for period in [3, 5, 10, 20]:
            col_name = f'momentum_{period}'
            df[col_name] = df['close'].pct_change(period)
            features.append(col_name)
        
        # Return features (short-term)
        for period in [1, 3, 6]:
            col_name = f'return_{period}'
            df[col_name] = df['close'].pct_change(period)
            features.append(col_name)
        
        # Trend slope (linear regression slope)
        for window in [10, 20, 50]:
            col_name = f'trend_slope_{window}'
            slopes = []
            for i in range(len(df)):
                if i < window:
                    slopes.append(0.0)
                else:
                    y = df['close'].iloc[i-window+1:i+1].values
                    x = np.arange(len(y))
                    if len(y) > 1 and y.std() > 0:
                        slope = np.polyfit(x, y, 1)[0] / y.mean()  # Normalized
                    else:
                        slope = 0.0
                    slopes.append(slope)
            df[col_name] = slopes
            features.append(col_name)
        
        # Price position in recent range (directional indicator)
        for window in [10, 20, 50]:
            col_name = f'price_position_{window}'
            high_rolling = df['close'].rolling(window).max()
            low_rolling = df['close'].rolling(window).min()
            df[col_name] = (df['close'] - low_rolling) / (high_rolling - low_rolling)
            df[col_name] = df[col_name].fillna(0.5)
            features.append(col_name)
    
    # 11. EMA Cross Features (directional)
    if 'close' in df.columns:
        # EMA cross signals
        if all(col in df.columns for col in ['ema_12', 'ema_26']):
            # EMA cross direction
            df['ema_cross_direction'] = np.where(
                df['ema_12'] > df['ema_26'], 1, -1
            )
            # EMA cross strength (normalized distance)
            df['ema_cross_strength'] = (df['ema_12'] - df['ema_26']) / df['ema_26']
            features.extend(['ema_cross_direction', 'ema_cross_strength'])
        
        # EMA slope (trend direction)
        if 'ema_12' in df.columns:
            df['ema_12_slope'] = df['ema_12'].pct_change(3)
            features.append('ema_12_slope')
        if 'ema_26' in df.columns:
            df['ema_26_slope'] = df['ema_26'].pct_change(3)
            features.append('ema_26_slope')
    
    # 12. Volatility Regime Flag (directional indicator)
    if 'returns' in df.columns:
        # Volatility percentile
        vol_20 = calculate_volatility(df['returns'], 20)
        vol_50 = calculate_volatility(df['returns'], 50)
        df['volatility_regime_flag'] = np.where(
            vol_20 > vol_50.rolling(50).quantile(0.7), 1,  # High volatility
            np.where(vol_20 < vol_50.rolling(50).quantile(0.3), -1, 0)  # Low volatility
        )
        features.append('volatility_regime_flag')
        
        # Volatility trend
        df['volatility_trend'] = vol_20.pct_change(5)
        features.append('volatility_trend')
    
    # 13. RSI Continuous Features (thay binary bằng continuous)
    if 'close' in df.columns:
        for period in [7, 14, 21]:
            rsi_col = f'rsi_{period}'
            if rsi_col in df.columns:
                # RSI momentum (change)
                df[f'{rsi_col}_momentum'] = df[rsi_col].diff(3)
                features.append(f'{rsi_col}_momentum')
                
                # RSI distance from 50 (neutral point)
                df[f'{rsi_col}_distance_from_50'] = (df[rsi_col] - 50) / 50
                features.append(f'{rsi_col}_distance_from_50')
                
                # RSI slope (rate of change)
                df[f'{rsi_col}_slope'] = df[rsi_col].diff(3) / 3  # Average change per period
                features.append(f'{rsi_col}_slope')
                
                # RSI z-score (normalized by rolling mean/std)
                rsi_mean = df[rsi_col].rolling(50).mean()
                rsi_std = df[rsi_col].rolling(50).std()
                df[f'{rsi_col}_zscore'] = (df[rsi_col] - rsi_mean) / rsi_std
                df[f'{rsi_col}_zscore'] = df[f'{rsi_col}_zscore'].fillna(0.0)
                features.append(f'{rsi_col}_zscore')
                
                # RSI percentile (position in historical range)
                rsi_min = df[rsi_col].rolling(50).min()
                rsi_max = df[rsi_col].rolling(50).max()
                df[f'{rsi_col}_percentile'] = (df[rsi_col] - rsi_min) / (rsi_max - rsi_min)
                df[f'{rsi_col}_percentile'] = df[f'{rsi_col}_percentile'].fillna(0.5)
                features.append(f'{rsi_col}_percentile')
    
    # 14. Context Features - RSI chỉ có nghĩa khi có context
    if 'close' in df.columns:
        for period in [7, 14, 21]:
            rsi_col = f'rsi_{period}'
            if rsi_col in df.columns:
                # RSI * Trend Context (RSI overbought chỉ có nghĩa khi uptrend)
                if 'trend_slope_20' in df.columns:
                    df[f'{rsi_col}_trend_context'] = df[rsi_col] * df['trend_slope_20']
                    features.append(f'{rsi_col}_trend_context')
                
                if 'trend_regime' in df.columns:
                    df[f'{rsi_col}_trend_regime'] = df[rsi_col] * df['trend_regime']
                    features.append(f'{rsi_col}_trend_regime')
                
                # RSI * Volatility Context (RSI chỉ có nghĩa khi volatility regime thấp)
                if 'volatility_regime_flag' in df.columns:
                    df[f'{rsi_col}_vol_context'] = df[rsi_col] * (df['volatility_regime_flag'] == -1).astype(int)
                    features.append(f'{rsi_col}_vol_context')
                
                if 'volatility_20' in df.columns:
                    # Normalize volatility để interaction có nghĩa
                    vol_norm = df['volatility_20'] / df['volatility_20'].rolling(50).mean()
                    df[f'{rsi_col}_vol_norm_context'] = df[rsi_col] / (1 + vol_norm)
                    features.append(f'{rsi_col}_vol_norm_context')
                
                # RSI * Volume Context (Volume confirmation)
                if 'volume' in df.columns:
                    volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
                    df[f'{rsi_col}_volume_context'] = df[rsi_col] * volume_ratio
                    features.append(f'{rsi_col}_volume_context')
                
                # Combined Context: RSI * Trend * Volatility * Volume
                if all(col in df.columns for col in ['trend_slope_20', 'volatility_20', 'volume']):
                    trend_norm = df['trend_slope_20'] / (abs(df['trend_slope_20']).rolling(20).mean() + 1e-6)
                    vol_norm = df['volatility_20'] / (df['volatility_20'].rolling(50).mean() + 1e-6)
                    vol_ratio = df['volume'] / (df['volume'].rolling(20).mean() + 1e-6)
                    df[f'{rsi_col}_full_context'] = df[rsi_col] * trend_norm * (1 / (1 + vol_norm)) * vol_ratio
                    features.append(f'{rsi_col}_full_context')
    
    # 15. Additional Interaction Features (XGBoost có thể tự học nhưng thêm explicit giúp)
    if 'close' in df.columns and 'returns' in df.columns:
        # Price * Volume interaction
        if 'volume' in df.columns:
            df['price_volume_interaction'] = df['returns'] * df['volume'].pct_change()
            features.append('price_volume_interaction')
        
        # Momentum * Volatility interaction
        if 'momentum_5' in df.columns and 'volatility_20' in df.columns:
            df['momentum_vol_interaction'] = df['momentum_5'] * df['volatility_20']
            features.append('momentum_vol_interaction')
        
        # RSI * Trend interaction
        if 'rsi_14' in df.columns and 'trend_slope_20' in df.columns:
            df['rsi_trend_interaction'] = (df['rsi_14'] - 50) / 50 * df['trend_slope_20']
            features.append('rsi_trend_interaction')
    
    # Fill NaN values
    df[features] = df[features].bfill().fillna(0.0)
    
    # 10. Volatility Regime (ATR percentile)
    if all(col in df.columns for col in ['high', 'low', 'close']):
        if 'atr_14' not in df.columns:
            df['atr_14'] = calculate_atr(df, 14)
        if 'atr_14' in df.columns:
            # ATR percentile (volatility regime)
            df['atr_percentile_20'] = df['atr_14'].rolling(20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            )
            df['atr_percentile_50'] = df['atr_14'].rolling(50).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            )
            df['atr_regime'] = pd.cut(
                df['atr_percentile_50'],
                bins=[0, 0.33, 0.67, 1.0],
                labels=[0, 1, 2]  # Low, Medium, High volatility
            ).astype(float)
            features.extend(['atr_percentile_20', 'atr_percentile_50', 'atr_regime'])
    
    # 11. Market Structure (HH/HL logic)
    if 'close' in df.columns:
        # Higher Highs / Higher Lows / Lower Highs / Lower Lows
        df['hh'] = ((df['close'] > df['close'].shift(1)) & 
                    (df['close'].shift(1) > df['close'].shift(2))).astype(int)
        df['hl'] = ((df['close'] > df['close'].shift(1)) & 
                    (df['close'].shift(1) <= df['close'].shift(2))).astype(int)
        df['lh'] = ((df['close'] <= df['close'].shift(1)) & 
                    (df['close'].shift(1) > df['close'].shift(2))).astype(int)
        df['ll'] = ((df['close'] <= df['close'].shift(1)) & 
                    (df['close'].shift(1) <= df['close'].shift(2))).astype(int)
        
        # Market structure score (trending up = positive, trending down = negative)
        df['market_structure_score'] = (df['hh'] * 1 + df['hl'] * 0.5 - df['lh'] * 0.5 - df['ll'] * 1)
        df['market_structure_ma'] = df['market_structure_score'].rolling(10).mean()
        features.extend(['hh', 'hl', 'lh', 'll', 'market_structure_score', 'market_structure_ma'])
    
    # 12. Multi-timeframe Alignment
    if 'close' in df.columns:
        # SMA alignment across timeframes
        sma_short = calculate_sma(df['close'], 20)
        sma_medium = calculate_sma(df['close'], 50)
        sma_long = calculate_sma(df['close'], 100)
        
        # Alignment score: all aligned = 1, mixed = 0, all opposite = -1
        df['mtf_alignment'] = (
            ((sma_short > sma_medium).astype(int) + (sma_medium > sma_long).astype(int) - 1) +
            ((sma_short < sma_medium).astype(int) + (sma_medium < sma_long).astype(int) - 1) * -1
        )
        features.append('mtf_alignment')
    
    # 13. Distance from VWAP
    if 'close' in df.columns and 'volume' in df.columns:
        # VWAP calculation
        typical_price = (df['high'] + df['low'] + df['close']) / 3 if all(c in df.columns for c in ['high', 'low']) else df['close']
        df['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Distance from VWAP
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        df['vwap_distance_ma'] = df['vwap_distance'].rolling(10).mean()
        features.extend(['vwap', 'vwap_distance', 'vwap_distance_ma'])
    elif 'close' in df.columns:
        # Fallback: use SMA as VWAP proxy
        sma_20 = calculate_sma(df['close'], 20)
        df['vwap_distance'] = (df['close'] - sma_20) / sma_20
        features.append('vwap_distance')
    
    # 14. Liquidity Proxy
    if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
        # Bid-ask spread proxy: (high - low) / close
        df['spread_pct'] = (df['high'] - df['low']) / df['close']
        
        # Volume-weighted spread
        df['volume_weighted_spread'] = df['spread_pct'] * df['volume']
        df['liquidity_score'] = 1.0 / (df['volume_weighted_spread'] + 1e-6)
        df['liquidity_score_ma'] = df['liquidity_score'].rolling(20).mean()
        features.extend(['spread_pct', 'volume_weighted_spread', 'liquidity_score', 'liquidity_score_ma'])
    
    # 15. Time-of-day features
    if timestamp_col and timestamp_col in df.columns:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df['hour'] = df[timestamp_col].dt.hour
            df['day_of_week'] = df[timestamp_col].dt.dayofweek
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            features.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])
        except:
            pass
    
    return df, features
