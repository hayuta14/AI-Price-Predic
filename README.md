# Nền tảng tối ưu hoá hệ thống giao dịch Futures BTCUSDT 15m

Nền tảng nghiên cứu định lượng **modular, production-ready** để thiết kế, tối ưu hoá và backtest hệ thống giao dịch futures BTCUSDT trên khung thời gian **15 phút**.

### Tính năng chính

- **Walk‑forward validation chuẩn time‑series**: không data leakage, không lookahead bias.
- **Target dựa trên return**: multi-class `-1/0/1` và log‑return cho regression head, tránh dự đoán giá tuyệt đối.
- **Feature engineering chuyên cho futures**: funding rate, open interest, long/short ratio, liquidation, volatility regime, volume & microstructure.
- **Quản trị rủi ro đa tầng**: RiskEngine, FuturesRiskManager, adaptive sizing theo volatility + regime + confidence + ATR.
- **Backtest thực tế hơn**: phí giao dịch, slippage, funding, path‑dependency, Monte Carlo.
- **Tối ưu hoá toàn diện**: feature set, label config, risk params, XGBoost hyperparams, multi‑model search.

---

## Cấu trúc dự án

```text
backend/
├── api/              # FastAPI routes (REST API cho tối ưu & truy vấn kết quả)
├── core/             # Core engines (walk-forward, backtest, risk, metrics, regime, validation)
├── optimization/     # Tối ưu hoá feature/label/risk/hyperparams
├── analysis/         # Phân tích Monte Carlo, regime, performance report
├── services/         # Orchestrator: training, optimization, multi-model
├── database/         # SQLAlchemy models + repositories (PostgreSQL)
├── data/             # Data fetcher (Binance), feature engineering
├── config.py         # Khai báo & khởi tạo SystemConfig
└── main.py           # Entry point CLI: chạy quy trình tối ưu hoá end‑to‑end
```

---

## Quy trình pipeline (backend/main.py)

Quy trình chính trong `main()` gồm 5 bước:

- **Bước 1 – Khởi tạo database**
  - Kết nối PostgreSQL theo `backend/config.py` (`SystemConfig.database`).
  - Nếu lỗi kết nối, tự động fallback sang SQLite in‑memory (chế độ test).

- **Bước 2 – Tải dữ liệu BTCUSDT 15m**
  - Ưu tiên đọc từ cache CSV: `data/btcusdt_15m.csv` hoặc `{symbol}_{interval}.csv`.
  - Nếu không có, gọi `BinanceDataFetcher` để lấy dữ liệu OHLCV từ Binance, sau đó lưu lại CSV.

- **Bước 3 – Tạo features + targets + regimes**
  - Gọi `create_all_features(data, timestamp_col='timestamp')` để tạo full feature set:
    - RSI multi‑TF, SMA/EMA, ATR/ATR%, rolling volatility, momentum/returns, VWAP, liquidity, market structure, multi‑TF alignment, time‑of‑day…
    - Futures features: `funding_rate_zscore`, `funding_rate_cumsum_8h`, `oi_change_pct`, `oi_zscore`, `price_oi_divergence`, `ls_ratio_zscore`, `liq_imbalance`, …
    - Volatility regime: `rv_1h`, `rv_4h`, `rv_24h`, `vol_ratio`, `atr_normalized`, `bb_width`, `bb_position`.
    - Volume & microstructure: `volume_ratio`, `vwap_long`, `vwap_deviation`, `taker_ratio`, `taker_ratio_ma` (nếu có dữ liệu).
  - Tạo target return‑based:
    - `create_targets(data, forward_periods=3, threshold=0.002)` sinh:
      - `log_return_fwd`: log‑return làm regression target.
      - `target_class`: -1/0/1 làm classification label.
      - `target_ev`: EV thô (có thể dùng cho sizing).
  - Detect market regime:
    - `RegimeDetector.detect_regime()` thêm:
      - `adx`, `vol_regime`, `trend_direction`, `regime`, `regime_size_mult`.

- **Bước 4 – Tối ưu hoá mô hình**
  - Khởi tạo `OptimizationService` với `SystemConfig` + DB session.
  - Gọi `run_full_optimization(...)`:
    - **LabelOptimizer**: tối ưu horizon/threshold cho nhãn.
    - **FeatureOptimizer**: chọn tập feature tối ưu dựa trên WF Sharpe + SHAP.
    - **HyperparameterOptimizer**: dùng Optuna để tối ưu XGBoost hyperparams với mục tiêu Sharpe (walk‑forward), không dùng accuracy/F1.
    - **RiskOptimizer**: tối ưu risk per trade, ATR multiplier, RR.
    - Phân tích thêm: prediction distribution, AUC, calibration, feature dominance.
  - Chạy **final backtest** với threshold đã tối ưu chỉ trên **test set out‑of‑sample** (tránh threshold optimization bias).

- **Bước 5 – Lưu kết quả & báo cáo**
  - Lưu run vào PostgreSQL (`ModelRunRepository`) với toàn bộ config + metrics.
  - Lưu model cuối cùng và metadata vào `results/models/` (`ModelPersistence`).
  - Có thể dùng API (`backend/api/routes.py`) để truy vấn các run, top Sharpe, top stability.

---

## Cài đặt & chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Khởi động database (khuyến nghị PostgreSQL qua Docker)

```bash
docker-compose up -d
```

Hoặc tự cài PostgreSQL rồi chỉnh thông số trong `.env`:

- `POSTGRES_HOST`, `POSTGRES_PORT`
- `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `POSTGRES_DB`

### 3. Chạy tối ưu hoá full pipeline

```bash
python backend/main.py
```

Log sẽ hiển thị rõ:
- Bước 1–5, số nến, khoảng thời gian data, thống kê basic.
- Số feature được tạo, phân loại theo nhóm.
- Kết quả walk‑forward, final backtest, metrics quan trọng (Sharpe, max DD, PF, win rate, số trades).

---

## Các module cốt lõi

### 1. Walk-Forward Engine – `backend/core/walkforward_engine.py`

- Sinh các fold train/test đúng thứ tự thời gian (expanding/rolling).
- Bảo đảm **train_end < test_start**, không lookahead.
- Tính toán & gom metric per‑fold (mean Sharpe, max DD, Sharpe stability…).

### 2. Backtest Engine + Realistic Execution – `backend/core/backtest_engine.py`

- Mô phỏng đầy đủ:
  - Mở/đóng vị thế, SL/TP theo ATR, phí, slippage, funding.
  - Lưu trade log chi tiết (`Trade` dataclass).
  - Sinh `BacktestResults`: equity curve, returns, daily_returns, metrics, trade_log.
- **RealisticExecutionSimulator**:
  - `calculate_fill_price`: slippage phụ thuộc volatility + volume ratio.
  - `calculate_funding_cost`: funding mỗi 8 giờ (32 nến 15m).
  - `simulate_partial_fill`: mô phỏng partial fills khi thanh khoản thấp.

### 3. Risk Engine – `backend/core/risk_engine.py`

- Risk‑per‑trade cố định theo % equity (`RiskConfig.risk_per_trade`).
- Sizing dựa trên **khoảng cách SL** (entry – stop) & volatility scaling.
- Giới hạn:
  - Max leverage.
  - Max daily loss.
  - Max drawdown stop.

### 4. Futures Risk Manager – `backend/core/risk_manager.py`

- `FuturesRiskManager` thêm một lớp risk chuyên cho futures:
  - Base risk per trade.
  - Volatility scalar (high vol → giảm size).
  - Regime scalar (sideways/high vol → giảm mạnh).
  - Confidence scalar (dựa trên xác suất model).
  - Kill‑switch ở **15% drawdown** với log cảnh báo.

### 5. Feature Engineering – `backend/data/feature_engineering.py`

- `create_all_features`:
  - OHLCV classic (RSI/SMA/EMA/ATR/vol).
  - Momentum, return, trend slope, price position.
  - Liquidity & spread proxies, VWAP & vwap distance.
  - Multi‑TF trend alignment, market structure (HH/HL/LH/LL).
  - Time‑of‑day (hour/day‑of‑week, sinusoidal).
- Futures‑specific:
  - Funding: `funding_rate_zscore`, `funding_rate_cumsum_8h`, `funding_extreme`.
  - Open interest: `oi_change_pct`, `oi_zscore`, `price_oi_divergence`.
  - Long/short ratio: `ls_ratio_zscore`, `ls_extreme_long`, `ls_extreme_short`.
  - Liquidations: `liq_imbalance`, `liq_spike`.
- Volatility & microstructure:
  - `rv_1h`, `rv_4h`, `rv_24h`, `vol_ratio`, `atr_normalized`, `bb_width`, `bb_position`.
  - `volume_ratio`, `vwap_long`, `vwap_deviation`, `taker_ratio`, `taker_ratio_ma`.

### 6. Targets – `create_targets`

- Return‑based target chuẩn hoá:
  - `log_return_fwd` (regression).
  - `target_class` (-1/0/1).
  - `target_ev` (hiện là log‑return, có thể dùng để xây EV‑based sizing sau này).

### 7. Hyperparameter Optimization – `backend/optimization/hyperparameter_optimizer.py`

- Dùng Optuna để tối ưu XGBoost:
  - Space lấy từ `XGBoostConfig`.
  - Mục tiêu: **Sharpe Ratio** trên walk‑forward, trừ penalty DD:
    - Nếu tổng số trades < 10: reject (`-999`).
    - Score = Sharpe – 0.5 * max(0, maxDD – 15%).
- Có `quick_backtest` vectorized dành riêng cho Optuna, tính Sharpe/Sortino/DD/PF/win rate/total trades.

### 8. Validation & Regime Testing – `backend/core/validation.py`

- `RobustValidator.walk_forward_validate`:
  - Purged WF (train → gap → test) với `gap_periods`.
  - Trả về Sharpe trung bình, std, min Sharpe, consistency.
- `regime_split_validate`:
  - Đánh giá riêng trên từng regime (`trending_up`, `trending_down`, `sideways`, `high_volatility`) nếu đủ mẫu.

---

## Database & API

- **Database**:
  - PostgreSQL là backend chính; nếu không khả dụng, fallback SQLite in‑memory để dễ test.
  - Bảng chính:
    - `model_runs`: lưu mỗi run (features, label config, risk config, hyperparams, metrics).
    - `optimization_trials`: lưu log các trial Optuna.

- **API** (`backend/api/routes.py`):
  - `POST /optimize/start`: khởi chạy job tối ưu mới.
  - `GET /optimize/status/{id}`: trạng thái job.
  - `GET /optimize/results/{id}`: kết quả chi tiết.
  - `GET /runs`, `/runs/{id}`, `/runs/top/sharpe`, `/runs/top/stability`.

---

## Lưu ý khi sử dụng

- Đảm bảo dữ liệu:
  - Được sort theo `timestamp`.
  - Có đủ các cột: `timestamp, open, high, low, close, volume` (và thêm funding/OI nếu muốn futures features).
- Lần chạy đầu:
  - Kiểm tra Docker PostgreSQL đã chạy.
  - Hoặc cấu hình `.env` đúng thông tin database.
- Tối ưu hoá có thể tốn thời gian (đặc biệt với nhiều trials & nhiều folds WF):
  - Khuyến nghị chạy trên server riêng hoặc background job.

---

## Giấy phép

MIT License
