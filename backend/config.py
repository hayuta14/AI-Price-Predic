"""
Mô-đun Cấu hình - Định nghĩa cấu hình và tham số hệ thống toàn cục

Mô-đun này định nghĩa các tham số cấu hình cốt lõi của toàn bộ khung nghiên cứu định lượng, bao gồm:
- Tham số xác thực Walk-forward
- Tham số rủi ro
- Không gian tìm kiếm tối ưu hóa
- Kết nối cơ sở dữ liệu
- Tham số chi phí giao dịch
"""
import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()


@dataclass
class WalkForwardConfig:
    """Cấu hình xác thực Walk-forward"""
    train_window_days: int = 60  # Cửa sổ huấn luyện (ngày)
    test_window_days: int = 7    # Cửa sổ kiểm tra (ngày)
    step_days: int = 7            # Bước cuộn (ngày)
    min_train_samples: int = 1000  # Số mẫu huấn luyện tối thiểu
    expanding_window: bool = True  # Có sử dụng cửa sổ mở rộng không (True=mở rộng, False=cuộn)
    min_n_folds: int = 5          # Số folds tối thiểu (để kiểm tra robustness)
    max_n_folds: int = 10         # Số folds tối đa


@dataclass
class RiskConfig:
    """Cấu hình quản lý rủi ro"""
    risk_per_trade: float = 0.01      # Rủi ro mỗi giao dịch (phần trăm vốn tài khoản) - Giảm từ 2% xuống 1%
    max_leverage: float = 10.0        # Hệ số đòn bẩy tối đa
    max_daily_loss: float = 0.05      # Thua lỗ một ngày tối đa (phần trăm vốn tài khoản)
    max_drawdown_stop: float = 0.20   # Dừng lỗ vẽ lùi tối đa (phần trăm vốn tài khoản)
    volatility_scaling: bool = True   # Có bật tỷ lệ theo biến động không
    volatility_lookback: int = 20     # Kỳ nhìn lại tính biến động (số nến)


@dataclass
class TradingConfig:
    """Cấu hình chi phí giao dịch"""
    fee_rate: float = 0.0005          # Tỷ lệ phí (0.05%)
    slippage_rate: float = 0.0002     # Tỷ lệ trượt giá (0.02%)
    funding_rate: float = 0.0001     # Tỷ lệ tài trợ (giữ chỗ, 0.01%)


@dataclass
class LabelConfig:
    """Cấu hình nhãn"""
    horizons: List[int] = None        # Phạm vi thời gian dự đoán (số nến)
    thresholds: List[float] = None    # Ngưỡng lợi nhuận (phần trăm)
    use_dynamic_threshold: bool = True  # Sử dụng dynamic threshold theo ATR
    atr_multiplier_range: List[float] = None  # Hệ số nhân ATR cho dynamic threshold
    default_atr_multiplier: float = 0.5  # Mặc định 0.5 * ATR như yêu cầu
    use_asymmetric_labels: bool = False  # Sử dụng asymmetric labels (long/short khác nhau)
    long_thresholds: List[float] = None  # Long thresholds (nếu asymmetric)
    short_thresholds: List[float] = None  # Short thresholds (nếu asymmetric)
    
    def __post_init__(self):
        if self.horizons is None:
            # Thêm horizon 8 để tăng trade frequency
            self.horizons = [6, 8, 10, 12]
        if self.thresholds is None:
            # Thêm threshold 0.3% để tăng trade frequency
            self.thresholds = [0.003, 0.004, 0.005, 0.01]
        if self.atr_multiplier_range is None:
            # Hệ số nhân ATR: ưu tiên 0.5 như yêu cầu, thêm các giá trị khác để test
            self.atr_multiplier_range = [0.5, 0.75, 1.0, 1.5]
        if self.long_thresholds is None:
            # Long thresholds cho asymmetric labels (ví dụ: 0.4%)
            self.long_thresholds = [0.003, 0.004, 0.005]
        if self.short_thresholds is None:
            # Short thresholds cho asymmetric labels (ví dụ: 0.6%)
            self.short_thresholds = [0.005, 0.006, 0.007]


@dataclass
class XGBoostConfig:
    """Không gian tìm kiếm siêu tham số XGBoost"""
    # Giảm model capacity để tránh memorize noise
    max_depth_range: tuple = (3, 4)  # Giảm từ (3, 8) xuống (3, 4)
    learning_rate_range: tuple = (0.03, 0.08)  # Giảm từ (0.01, 0.2) xuống (0.03, 0.08)
    n_estimators_range: tuple = (150, 300)  # Điều chỉnh từ (100, 500)
    subsample_range: tuple = (0.7, 0.9)  # Giữ trong range hợp lý
    colsample_bytree_range: tuple = (0.7, 0.9)  # Giữ trong range hợp lý
    n_trials: int = 50                # Số lần thử nghiệm tối ưu hóa Optuna


@dataclass
class MonteCarloConfig:
    """Cấu hình mô phỏng Monte Carlo"""
    n_simulations: int = 1000          # Số lần mô phỏng
    confidence_level: float = 0.05    # Mức tin cậy (để tính toán trường hợp xấu nhất)


@dataclass
class DatabaseConfig:
    """Cấu hình cơ sở dữ liệu"""
    postgres_host: str = None
    postgres_port: int = None
    postgres_user: str = None
    postgres_password: str = None
    postgres_db: str = None
    mongo_host: str = None
    mongo_port: int = None
    mongo_db: str = None
    
    def __post_init__(self):
        # Load từ biến môi trường nếu có, nếu không thì dùng giá trị mặc định
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.postgres_user = os.getenv("POSTGRES_USER", "postgres")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.postgres_db = os.getenv("POSTGRES_DB", "trading_research")
        self.mongo_host = os.getenv("MONGO_HOST", "localhost")
        self.mongo_port = int(os.getenv("MONGO_PORT", "27017"))
        self.mongo_db = os.getenv("MONGO_DB", "trading_data")
    
    def get_postgres_url(self) -> str:
        """Tạo connection string cho PostgreSQL"""
        return (
            f"postgresql://{self.postgres_user}:"
            f"{self.postgres_password}@"
            f"{self.postgres_host}:"
            f"{self.postgres_port}/"
            f"{self.postgres_db}"
        )


@dataclass
class SystemConfig:
    """Cấu hình hệ thống toàn cục"""
    data_path: Path = Path("data")
    results_path: Path = Path("results")
    log_level: str = "INFO"
    random_seed: int = 42
    
    # Cấu hình con
    walkforward: WalkForwardConfig = None
    risk: RiskConfig = None
    trading: TradingConfig = None
    label: LabelConfig = None
    xgboost: XGBoostConfig = None
    montecarlo: MonteCarloConfig = None
    database: DatabaseConfig = None
    
    def __post_init__(self):
        if self.walkforward is None:
            self.walkforward = WalkForwardConfig()
        if self.risk is None:
            self.risk = RiskConfig()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.label is None:
            self.label = LabelConfig()
        if self.xgboost is None:
            self.xgboost = XGBoostConfig()
        if self.montecarlo is None:
            self.montecarlo = MonteCarloConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        
        # Đảm bảo thư mục tồn tại
        self.data_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)


# Thể hiện cấu hình toàn cục
config = SystemConfig()
