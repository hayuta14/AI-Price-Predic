"""
Binance Data Fetcher

Lấy dữ liệu giá từ Binance API để train mô hình
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import time
import logging

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """
    Lấy dữ liệu từ Binance API
    
    Hỗ trợ cả python-binance và requests để lấy dữ liệu
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Khởi tạo Binance Data Fetcher
        
        Args:
            api_key: Binance API key (không bắt buộc cho public endpoints)
            api_secret: Binance API secret (không bắt buộc cho public endpoints)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        
        if BINANCE_AVAILABLE:
            try:
                if api_key and api_secret:
                    self.client = Client(api_key=api_key, api_secret=api_secret)
                else:
                    self.client = Client()  # Public client không cần API key
                logger.info("Đã khởi tạo Binance client thành công")
            except Exception as e:
                logger.warning(f"Không thể khởi tạo Binance client: {e}")
                self.client = None
    
    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Lấy dữ liệu kline (candlestick) từ Binance
        
        Args:
            symbol: Cặp giao dịch (mặc định: BTCUSDT)
            interval: Khung thời gian (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_date: Ngày bắt đầu (nếu None, lấy từ limit nến gần nhất)
            end_date: Ngày kết thúc (nếu None, lấy đến hiện tại)
            limit: Số lượng nến tối đa mỗi lần request (tối đa 1000)
            
        Returns:
            DataFrame chứa dữ liệu OHLCV với columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Bắt đầu lấy dữ liệu {symbol} {interval} từ Binance...")
        
        if self.client:
            return self._fetch_with_binance_client(symbol, interval, start_date, end_date, limit)
        elif REQUESTS_AVAILABLE:
            return self._fetch_with_requests(symbol, interval, start_date, end_date, limit)
        else:
            raise ImportError(
                "Cần cài đặt 'python-binance' hoặc 'requests' để lấy dữ liệu từ Binance.\n"
                "Chạy: pip install python-binance hoặc pip install requests"
            )
    
    def _fetch_with_binance_client(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> pd.DataFrame:
        """Lấy dữ liệu sử dụng python-binance client"""
        all_klines = []
        
        # Chuyển đổi interval sang format của Binance
        interval_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "3m": Client.KLINE_INTERVAL_3MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "2h": Client.KLINE_INTERVAL_2HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "6h": Client.KLINE_INTERVAL_6HOUR,
            "8h": Client.KLINE_INTERVAL_8HOUR,
            "12h": Client.KLINE_INTERVAL_12HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
            "3d": Client.KLINE_INTERVAL_3DAY,
            "1w": Client.KLINE_INTERVAL_1WEEK,
            "1M": Client.KLINE_INTERVAL_1MONTH,
        }
        
        binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_15MINUTE)
        
        # Xác định thời gian bắt đầu
        if start_date is None:
            # Nếu không có start_date, tính toán từ end_date hoặc hiện tại
            if end_date is None:
                end_date = datetime.now()
            # Lấy khoảng 1 năm dữ liệu mặc định
            start_date = end_date - timedelta(days=365)
        
        if end_date is None:
            end_date = datetime.now()
        
        current_start = start_date
        
        logger.info(f"Lấy dữ liệu từ {current_start.strftime('%Y-%m-%d %H:%M:%S')} đến {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Lấy dữ liệu theo batch để tránh giới hạn
        batch_count = 0
        while current_start < end_date:
            try:
                # Lấy một batch
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=binance_interval,
                    start_str=current_start.strftime("%d %b %Y %H:%M:%S"),
                    end_str=end_date.strftime("%d %b %Y %H:%M:%S"),
                    limit=limit
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                batch_count += 1
                
                # Cập nhật start time cho batch tiếp theo
                last_timestamp = klines[-1][0] / 1000  # Convert từ milliseconds
                current_start = datetime.fromtimestamp(last_timestamp) + timedelta(minutes=1)
                
                logger.info(f"Đã lấy batch {batch_count}: {len(klines)} nến (Tổng: {len(all_klines)} nến)")
                
                # Tránh rate limit
                time.sleep(0.1)
                
                # Nếu số lượng nến trả về < limit, đã lấy hết
                if len(klines) < limit:
                    break
                    
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu batch {batch_count}: {e}")
                break
        
        if not all_klines:
            raise ValueError(f"Không lấy được dữ liệu từ Binance cho {symbol} {interval}")
        
        # Chuyển đổi sang DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Chọn các cột cần thiết và chuyển đổi kiểu dữ liệu
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Sắp xếp theo thời gian và loại bỏ trùng lặp
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        logger.info(f"Hoàn thành! Đã lấy {len(df)} nến dữ liệu")
        logger.info(f"Thời gian: {df['timestamp'].min()} đến {df['timestamp'].max()}")
        
        return df
    
    def _fetch_with_requests(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> pd.DataFrame:
        """Lấy dữ liệu sử dụng requests (fallback khi không có python-binance)"""
        if not REQUESTS_AVAILABLE:
            raise ImportError("Cần cài đặt 'requests' hoặc 'python-binance' để lấy dữ liệu từ Binance")
        
        import requests
        
        all_klines = []
        
        # Xác định thời gian
        if start_date is None:
            if end_date is None:
                end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
        
        if end_date is None:
            end_date = datetime.now()
        
        current_start = start_date
        
        logger.info(f"Lấy dữ liệu từ {current_start.strftime('%Y-%m-%d %H:%M:%S')} đến {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        base_url = "https://api.binance.com/api/v3/klines"
        batch_count = 0
        
        while current_start < end_date:
            try:
                # Chuyển đổi thời gian sang milliseconds
                start_time = int(current_start.timestamp() * 1000)
                end_time = int(min(end_date, current_start + timedelta(days=30)).timestamp() * 1000)
                
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': limit
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                klines = response.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                batch_count += 1
                
                # Cập nhật start time
                last_timestamp = klines[-1][0] / 1000
                current_start = datetime.fromtimestamp(last_timestamp) + timedelta(minutes=1)
                
                logger.info(f"Đã lấy batch {batch_count}: {len(klines)} nến (Tổng: {len(all_klines)} nến)")
                
                time.sleep(0.2)  # Tránh rate limit
                
                if len(klines) < limit:
                    break
                    
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu batch {batch_count}: {e}")
                break
        
        if not all_klines:
            raise ValueError(f"Không lấy được dữ liệu từ Binance cho {symbol} {interval}")
        
        # Chuyển đổi sang DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        logger.info(f"Hoàn thành! Đã lấy {len(df)} nến dữ liệu")
        logger.info(f"Thời gian: {df['timestamp'].min()} đến {df['timestamp'].max()}")
        
        return df
    
    def fetch_recent_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        days: int = 365
    ) -> pd.DataFrame:
        """
        Lấy dữ liệu gần đây
        
        Args:
            symbol: Cặp giao dịch
            interval: Khung thời gian
            days: Số ngày dữ liệu cần lấy
            
        Returns:
            DataFrame chứa dữ liệu OHLCV
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
