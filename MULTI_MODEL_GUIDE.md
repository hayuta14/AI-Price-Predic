# Hướng dẫn Multi-Model Training

## Tổng quan

Hệ thống hỗ trợ 2 chế độ training:

1. **Single Model Optimization**: Tối ưu hóa tuần tự (mặc định)
2. **Multi-Model Optimization**: Train nhiều models song song để tìm model tốt nhất

## Cách sử dụng

### 1. Single Model Optimization (Mặc định)

Chạy như bình thường:

```bash
python backend/main.py
```

Hoặc:

```bash
# Windows
set USE_MULTI_MODEL=false
python backend/main.py
```

### 2. Multi-Model Optimization

Để train nhiều models song song:

```bash
# Windows PowerShell
$env:USE_MULTI_MODEL="true"
$env:N_CANDIDATES="20"
python backend/main.py

# Windows CMD
set USE_MULTI_MODEL=true
set N_CANDIDATES=20
python backend/main.py

# Linux/Mac
export USE_MULTI_MODEL=true
export N_CANDIDATES=20
python backend/main.py
```

**Tham số:**
- `USE_MULTI_MODEL`: `true` để bật multi-model mode
- `N_CANDIDATES`: Số lượng models để train (mặc định: 20)

## Các tính năng

### Model Persistence

Tất cả models được tự động lưu vào thư mục `results/models/` với:
- Model file (`.pkl`)
- Metadata file (`.json`) chứa:
  - Features sử dụng
  - Hyperparameters
  - Metrics (Sharpe, Max Drawdown, etc.)
  - Label config
  - Risk config

### Multi-Model Training

- **Parallel Training**: Sử dụng ThreadPoolExecutor để train nhiều models song song
- **Auto Worker Detection**: Tự động detect số CPU cores và sử dụng tối ưu
- **Model Comparison**: Tự động so sánh và rank các models theo Sharpe ratio
- **Best Model Selection**: Tự động chọn model tốt nhất dựa trên:
  - Primary metric: Sharpe ratio
  - Minimum trades requirement
  - Validation metrics

### Candidate Generation

Hệ thống tự động generate candidates với:
- **Feature Combinations**: Nhiều combinations của features
- **Label Configs**: Fixed và dynamic thresholds
- **Hyperparameters**: Grid search trong search space

## Ví dụ Output

### Multi-Model Mode

```
🚀 Chế độ Multi-Model: Training 20 models song song...

[1/3] Đang tạo candidates...
✅ Đã tạo 20 candidates

[2/3] Đang train 20 models song song...
Training candidate 0...
Training candidate 1...
...
Candidate 0 completed: Sharpe=1.2345
Candidate 1 completed: Sharpe=0.9876
...

[3/3] Đang chọn model tốt nhất...

🏆 Model tốt nhất:
   • Candidate ID: 5
   • Sharpe Ratio: 1.4567
   • Max Drawdown: 12.34%
   • Profit Factor: 1.89
   • Total Trades: 156
   • Features: 12 features
   • Model Path: results/models/model_run_5_20240101_120000.pkl

📊 Top 5 Models:
   1. Candidate 5: Sharpe=1.4567, DD=12.34%, Trades=156
   2. Candidate 12: Sharpe=1.3456, DD=13.45%, Trades=142
   3. Candidate 8: Sharpe=1.2345, DD=14.56%, Trades=138
   ...
```

## Load Model đã lưu

```python
from backend.services.model_persistence import ModelPersistence
from pathlib import Path

# Khởi tạo
model_persistence = ModelPersistence()

# Load model mới nhất
latest_model = model_persistence.get_latest_model()
if latest_model:
    model_data = model_persistence.load_model(latest_model)
    model = model_data['model']
    features = model_data['features']
    metrics = model_data['metrics']
    
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
    print(f"Features: {features}")

# Load model tốt nhất theo Sharpe
best_model_data = model_persistence.get_best_model(metric='sharpe_ratio')
if best_model_data:
    model = best_model_data['model']
    # Sử dụng model để predict...

# So sánh nhiều models
model_paths = [
    Path("results/models/model_run_1_20240101_120000.pkl"),
    Path("results/models/model_run_2_20240101_130000.pkl"),
]
comparison_df = model_persistence.compare_models(model_paths)
print(comparison_df)
```

## Lưu ý

1. **Memory Usage**: Multi-model training sử dụng nhiều RAM hơn. Đảm bảo có đủ RAM.

2. **CPU Usage**: Sử dụng tất cả CPU cores (trừ 1 core cho hệ thống). Có thể điều chỉnh trong code.

3. **Time**: Multi-model training mất nhiều thời gian hơn nhưng explore được nhiều combinations hơn.

4. **Storage**: Mỗi model chiếm ~1-10MB. Với 20 models = ~20-200MB.

5. **Best Practice**: 
   - Chạy multi-model khi muốn explore nhiều combinations
   - Chạy single-model khi đã biết config tốt và muốn optimize chi tiết

## Troubleshooting

### Lỗi "No valid model found"
- Kiểm tra xem có đủ data không
- Giảm `min_trades` requirement trong `select_best_model()`

### Lỗi Memory
- Giảm `N_CANDIDATES`
- Giảm số features trong candidates

### Models không được lưu
- Kiểm tra quyền ghi vào thư mục `results/models/`
- Kiểm tra disk space
