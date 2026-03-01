"""
Điểm vào chương trình chính

Khởi động quy trình tối ưu hóa
"""
import pandas as pd
import numpy as np
import logging
import sys
import io
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import config
from backend.services.optimization_service import OptimizationService
from backend.database.models import Base
from backend.database.repository import ModelRunRepository
from backend.database.connection import (
    create_database_engine,
    test_connection,
    init_database,
    get_session,
    get_database_info
)
from backend.data.binance_fetcher import BinanceDataFetcher
from backend.data.feature_engineering import create_all_features


# Cấu hình logging
def setup_logging(log_level: str = "INFO"):
    """
    Thiết lập logging với format đẹp và màu sắc
    
    Args:
        log_level: Mức độ log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Set UTF-8 encoding for stdout trên Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    # Tạo formatter với format đẹp
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler cho console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Cấu hình root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # Tắt logging từ các thư viện bên ngoài (trừ khi DEBUG)
    if log_level.upper() != 'DEBUG':
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)


def load_data_from_binance(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    days: int = 365,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Tải dữ liệu từ Binance API
    
    Args:
        symbol: Cặp giao dịch (mặc định: BTCUSDT)
        interval: Khung thời gian (15m, 1h, 4h, 1d, ...)
        days: Số ngày dữ liệu cần lấy
        use_cache: Có sử dụng cache từ file CSV không
        
    Returns:
        DataFrame chứa dữ liệu giá OHLCV
    """
    logger = logging.getLogger(__name__)
    
    # Kiểm tra cache file
    if use_cache:
        cache_file = config.data_path / f"{symbol.lower()}_{interval}.csv"
        if cache_file.exists():
            try:
                logger.info(f"📁 Đang tải dữ liệu từ cache: {cache_file}")
                data = pd.read_csv(cache_file)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                logger.info(f"✅ Đã tải {len(data)} nến từ cache")
                return data
            except Exception as e:
                logger.warning(f"⚠️  Không thể đọc cache file: {e}, sẽ tải từ Binance")
    
    # Lấy dữ liệu từ Binance
    logger.info(f"🌐 Đang kết nối Binance API để lấy dữ liệu {symbol} {interval}...")
    
    try:
        fetcher = BinanceDataFetcher()
        data = fetcher.fetch_recent_data(
            symbol=symbol,
            interval=interval,
            days=days
        )
        
        # Lưu vào cache
        if use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(cache_file, index=False)
            logger.info(f"💾 Đã lưu dữ liệu vào cache: {cache_file}")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy dữ liệu từ Binance: {e}")
        logger.info("🔄 Đang thử tải từ file CSV hoặc tạo dữ liệu mẫu...")
        raise


def load_data(data_path: str) -> pd.DataFrame:
    """
    Tải dữ liệu từ file CSV (fallback)
    
    Args:
        data_path: Đường dẫn file dữ liệu
        
    Returns:
        DataFrame chứa dữ liệu giá
    """
    logger = logging.getLogger(__name__)
    
    if Path(data_path).exists():
        logger.info(f"📁 Đang tải dữ liệu từ file: {data_path}")
        data = pd.read_csv(data_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        logger.info(f"✅ Đã tải {len(data)} nến từ file")
        return data
    else:
        # Tạo dữ liệu mẫu (để test)
        logger.warning("⚠️  File dữ liệu không tồn tại, đang tạo dữ liệu mẫu...")
        dates = pd.date_range('2023-01-01', periods=10000, freq='15min')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(10000).cumsum() + 50000,
            'high': np.random.randn(10000).cumsum() + 50100,
            'low': np.random.randn(10000).cumsum() + 49900,
            'close': np.random.randn(10000).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 10000)
        })
        logger.info(f"✅ Đã tạo {len(data)} nến dữ liệu mẫu")
        return data


def create_sample_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Tạo các đặc trưng sử dụng module feature engineering mới
    
    Args:
        data: Dữ liệu giá
        
    Returns:
        Tuple (data_with_features, feature_list)
    """
    data_with_features, features = create_all_features(data, timestamp_col='timestamp')
    return data_with_features, features


def main():
    """Hàm chính"""
    # Thiết lập logging
    setup_logging(log_level=config.log_level)
    logger = logging.getLogger(__name__)
    
    # Header
    logger.info("=" * 80)
    logger.info("🚀 Nền tảng Tối ưu hóa Hệ thống Giao dịch Hợp đồng Tương lai BTCUSDT 15 phút")
    logger.info("=" * 80)
    
    # 1. Khởi tạo cơ sở dữ liệu
    logger.info("\n📊 [BƯỚC 1/5] Khởi tạo cơ sở dữ liệu...")
    db_info = get_database_info()
    logger.info(f"   🔗 Host: {db_info['host']}:{db_info['port']}")
    logger.info(f"   📁 Database: {db_info['database']}")
    logger.info(f"   👤 User: {db_info['user']}")
    
    try:
        # Test kết nối
        engine = create_database_engine(echo=False)
        if test_connection(engine):
            # Khởi tạo database (tạo tables)
            if init_database(engine):
                db_session = get_session(engine)
                logger.info("✅ Kết nối PostgreSQL thành công và đã tạo tables")
            else:
                raise Exception("Không thể tạo tables")
        else:
            raise Exception("Không thể kết nối đến PostgreSQL")
    except Exception as e:
        logger.error(f"❌ Kết nối PostgreSQL thất bại: {e}")
        logger.error(f"   Chi tiết lỗi: {type(e).__name__}")
        logger.info("\n💡 Hướng dẫn khắc phục:")
        logger.info("   1. Kiểm tra PostgreSQL đã được cài đặt và đang chạy")
        logger.info("      - Windows: Kiểm tra Services hoặc chạy: pg_ctl status")
        logger.info("      - Linux/Mac: sudo systemctl status postgresql")
        logger.info("   2. Kiểm tra thông tin kết nối:")
        logger.info("      - Tạo file .env trong thư mục gốc với nội dung:")
        logger.info(f"        POSTGRES_HOST={config.database.postgres_host}")
        logger.info(f"        POSTGRES_PORT={config.database.postgres_port}")
        logger.info(f"        POSTGRES_USER={config.database.postgres_user}")
        logger.info(f"        POSTGRES_PASSWORD=your_password")
        logger.info(f"        POSTGRES_DB={config.database.postgres_db}")
        logger.info("   3. Đảm bảo database đã được tạo:")
        logger.info(f"      psql -U {config.database.postgres_user} -c 'CREATE DATABASE {config.database.postgres_db};'")
        logger.info("   4. Kiểm tra quyền truy cập của user")
        logger.info("\n🔄 Sử dụng SQLite trong bộ nhớ (chế độ test)...")
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        db_session = SessionLocal()
        logger.warning("⚠️  Lưu ý: Dữ liệu sẽ không được lưu khi chương trình kết thúc")
    
    # 2. Tải dữ liệu từ Binance
    logger.info("\n📥 [BƯỚC 2/5] Tải dữ liệu từ Binance API...")
    try:
        data = load_data_from_binance(
            symbol="BTCUSDT",
            interval="15m",
            days=365,  # Lấy 1 năm dữ liệu
            use_cache=True
        )
        logger.info(f"✅ Hoàn thành! Đã tải {len(data):,} nến dữ liệu")
        logger.info(f"   📅 Thời gian: {data['timestamp'].min()} → {data['timestamp'].max()}")
        logger.info(f"   💰 Giá: ${data['close'].min():,.2f} → ${data['close'].max():,.2f}")
    except Exception as e:
        logger.error(f"❌ Không thể lấy dữ liệu từ Binance: {e}")
        logger.info("🔄 Đang thử tải từ file CSV...")
        data_path = config.data_path / "btcusdt_15m.csv"
        data = load_data(str(data_path))
        logger.info(f"✅ Đã tải {len(data):,} nến từ file CSV")
    
    # 3. Tạo đặc trưng
    logger.info("\n🔧 [BƯỚC 3/5] Tạo đặc trưng (features)...")
    data, features = create_sample_features(data)
    logger.info(f"✅ Đã tạo {len(features)} đặc trưng:")
    # Hiển thị theo nhóm
    feature_groups = {
        'RSI': [f for f in features if f.startswith('rsi')],
        'SMA': [f for f in features if f.startswith('sma')],
        'EMA': [f for f in features if f.startswith('ema')],
        'ATR': [f for f in features if f.startswith('atr')],
        'Volatility': [f for f in features if f.startswith('volatility')],
        'Volume': [f for f in features if 'volume' in f],
        'Regime': [f for f in features if 'regime' in f or 'position' in f],
        'Other': [f for f in features if not any(f.startswith(prefix) or prefix in f 
                 for prefix in ['rsi', 'sma', 'ema', 'atr', 'volatility', 'volume', 'regime', 'position'])]
    }
    for group_name, group_features in feature_groups.items():
        if group_features:
            logger.info(f"   {group_name} ({len(group_features)}): {', '.join(group_features[:5])}")
            if len(group_features) > 5:
                logger.info(f"      ... và {len(group_features) - 5} features khác")
    
    # 4. Khởi tạo dịch vụ tối ưu hóa
    logger.info("\n⚙️  [BƯỚC 4/5] Khởi tạo dịch vụ tối ưu hóa...")
    optimization_service = OptimizationService(config, db_session)
    logger.info("✅ Đã khởi tạo các thành phần:")
    logger.info("   • Walk-Forward Engine")
    logger.info("   • Feature Optimizer")
    logger.info("   • Label Optimizer")
    logger.info("   • Hyperparameter Optimizer")
    logger.info("   • Risk Optimizer")
    
    # 5. Chạy tối ưu hóa
    logger.info("\n🎯 [BƯỚC 5/5] Bắt đầu quy trình tối ưu hóa...")
    logger.info("=" * 80)
    
    # Chọn chế độ: single model hoặc multi-model
    import os
    use_multi_model = os.getenv('USE_MULTI_MODEL', 'false').lower() == 'true'
    n_candidates = int(os.getenv('N_CANDIDATES', '20'))
    
    if use_multi_model:
        logger.info(f"🚀 Chế độ Multi-Model: Training {n_candidates} models song song...")
        results = optimization_service.run_multi_model_optimization(
            data=data,
            available_features=features,
            n_candidates=n_candidates,
            price_col='close',
            timestamp_col='timestamp',
            initial_equity=100000.0
        )
        
        # Hiển thị kết quả multi-model
        if 'best_candidate' in results:
            best = results['best_candidate']
            logger.info("\n" + "=" * 80)
            logger.info("🏆 MODEL TỐT NHẤT (Multi-Model Optimization)")
            logger.info("=" * 80)
            logger.info(f"   • Candidate ID: {best.candidate_id}")
            logger.info(f"   • Sharpe Ratio: {best.metrics.get('sharpe_ratio', 0.0):.4f}")
            logger.info(f"   • Max Drawdown: {best.metrics.get('max_drawdown', 1.0)*100:.2f}%")
            logger.info(f"   • Profit Factor: {best.metrics.get('profit_factor', 0.0):.4f}")
            logger.info(f"   • Total Trades: {best.metrics.get('total_trades', 0)}")
            logger.info(f"   • Win Rate: {best.metrics.get('win_rate', 0.0)*100:.2f}%")
            logger.info(f"   • Features: {len(best.features)} features")
            if best.model_path:
                logger.info(f"   • Model Path: {best.model_path}")
            
            # Top 5 models
            if 'top_5' in results:
                logger.info("\n📊 TOP 5 MODELS:")
                for i, candidate in enumerate(results['top_5'], 1):
                    logger.info(f"   {i}. Candidate {candidate.candidate_id}: "
                              f"Sharpe={candidate.metrics.get('sharpe_ratio', 0.0):.4f}, "
                              f"DD={candidate.metrics.get('max_drawdown', 1.0)*100:.2f}%, "
                              f"Trades={candidate.metrics.get('total_trades', 0)}")
    else:
        logger.info("🔧 Chế độ Single Model: Tối ưu hóa tuần tự...")
        results = optimization_service.run_full_optimization(
            data=data,
            available_features=features,
            price_col='close',
            timestamp_col='timestamp',
            initial_equity=100000.0
        )
    
    # 6. Hiển thị kết quả
    logger.info("\n" + "=" * 80)
    logger.info("📊 TÓM TẮT KẾT QUẢ TỐI ƯU HÓA")
    logger.info("=" * 80)
    logger.info(f"🆔 Run ID: {results.get('run_id', 'N/A')}")
    
    feature_set = results.get('feature_set', {})
    logger.info(f"\n📈 Tập đặc trưng tối ưu: {len(feature_set.features)} đặc trưng")
    if feature_set.features:
        logger.info(f"   Features: {', '.join(feature_set.features)}")
    
    label_config = results.get('label_config', {})
    logger.info(f"\n🏷️  Cấu hình nhãn tối ưu:")
    logger.info(f"   • Horizon: {label_config.horizon} nến")
    logger.info(f"   • Threshold: {label_config.threshold*100:.2f}%")
    
    logger.info(f"\n📊 Chỉ số hiệu suất cuối cùng:")
    final_metrics = results.get('final_metrics', {})
    important_metrics = ['sharpe_ratio', 'max_drawdown', 'profit_factor', 
                        'total_return', 'win_rate', 'total_trades']
    for key in important_metrics:
        if key in final_metrics:
            value = final_metrics[key]
            if key == 'sharpe_ratio':
                logger.info(f"   • Sharpe Ratio: {value:.4f}")
            elif key == 'max_drawdown':
                logger.info(f"   • Max Drawdown: {value*100:.2f}%")
            elif key == 'profit_factor':
                logger.info(f"   • Profit Factor: {value:.4f}")
            elif key == 'total_return':
                logger.info(f"   • Total Return: {value*100:.2f}%")
            elif key == 'win_rate':
                logger.info(f"   • Win Rate: {value*100:.2f}%")
            elif key == 'total_trades':
                logger.info(f"   • Total Trades: {int(value)}")
    
    # Hiển thị thông tin về prediction analysis nếu có
    if 'prediction_analysis' in results:
        pred_analysis = results['prediction_analysis']
        logger.info(f"\n🔍 Phân tích phân phối dự đoán:")
        logger.info(f"   • Mean probability: {pred_analysis.mean_probability:.4f}")
        logger.info(f"   • Std probability: {pred_analysis.std_probability:.4f}")
        logger.info(f"   • Collapse về 0.5: {'❌ CÓ' if pred_analysis.collapse_to_05 else '✅ KHÔNG'}")
        logger.info(f"   • Class imbalance: {'❌ CÓ' if pred_analysis.class_imbalance else '✅ KHÔNG'}")
        logger.info(f"   • Always predict 0: {'❌ CÓ' if pred_analysis.always_predict_zero else '✅ KHÔNG'}")
        logger.info(f"   • Always predict 1: {'❌ CÓ' if pred_analysis.always_predict_one else '✅ KHÔNG'}")
    
    # 7. Lấy Top 5 cấu hình
    logger.info("\n" + "=" * 80)
    logger.info("🏆 TOP 5 CẤU HÌNH (theo Tỷ lệ Sharpe)")
    logger.info("=" * 80)
    model_run_repo = ModelRunRepository(db_session)
    top_runs = model_run_repo.get_top_by_sharpe(top_n=5)
    
    if top_runs:
        for i, run in enumerate(top_runs, 1):
            logger.info(f"\n{i}. Run ID: {run.id}")
            logger.info(f"   📈 Sharpe Ratio: {run.sharpe_ratio:.4f}")
            logger.info(f"   📉 Max Drawdown: {run.max_drawdown*100:.2f}%")
            logger.info(f"   💰 Profit Factor: {run.profit_factor:.4f}")
            logger.info(f"   🔢 Total Trades: {run.total_trades}")
    else:
        logger.info("   (Chưa có dữ liệu)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TỐI ƯU HÓA HOÀN TẤT!")
    logger.info("=" * 80)
    
    db_session.close()


if __name__ == "__main__":
    main()
