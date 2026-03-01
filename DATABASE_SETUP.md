# Hướng dẫn cấu hình PostgreSQL

## Tổng quan

Backend đã được cấu hình để kết nối với PostgreSQL. Hệ thống sẽ tự động:
- Kết nối đến PostgreSQL nếu có sẵn
- Fallback sang SQLite trong bộ nhớ nếu không kết nối được PostgreSQL

## Cấu hình PostgreSQL

### 1. Tạo file `.env` (khuyến nghị)

Tạo file `.env` trong thư mục gốc của project với nội dung:

```env
# PostgreSQL Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=trading_research
```

### 2. Hoặc chỉnh sửa trong `backend/config.py`

Nếu không dùng file `.env`, bạn có thể chỉnh sửa trực tiếp trong `backend/config.py`:

```python
@dataclass
class DatabaseConfig:
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "your_password"
    postgres_db: str = "trading_research"
```

## Thiết lập Database

### 1. Tạo database

```bash
# Kết nối PostgreSQL
psql -U postgres

# Tạo database
CREATE DATABASE trading_research;

# Thoát
\q
```

### 2. Kiểm tra kết nối

Chạy script test:

```bash
python test_db_connection.py
```

Script này sẽ:
- Kiểm tra kết nối đến PostgreSQL
- Hiển thị thông tin database
- Tạo các tables cần thiết (nếu chưa có)

## Cấu trúc Database

Hệ thống tự động tạo 2 tables:

1. **model_runs**: Lưu trữ kết quả của mỗi lần chạy optimization
   - Cấu hình features, labels, risk, hyperparameters
   - Các chỉ số hiệu suất (Sharpe ratio, drawdown, profit factor, ...)
   - Thời gian tạo và cập nhật

2. **optimization_trials**: Lưu trữ chi tiết các thử nghiệm optimization
   - Liên kết với model_runs
   - Loại thử nghiệm (feature, label, risk, hyperparameter)
   - Tham số và kết quả của từng thử nghiệm

## Sử dụng trong code

### Kết nối database

```python
from backend.database.connection import (
    create_database_engine,
    test_connection,
    init_database,
    get_session
)

# Tạo engine
engine = create_database_engine()

# Test kết nối
if test_connection(engine):
    # Khởi tạo database
    init_database(engine)
    
    # Lấy session
    db_session = get_session(engine)
    
    # Sử dụng session...
    db_session.close()
```

### Sử dụng Repository

```python
from backend.database.repository import ModelRunRepository

# Tạo repository
repo = ModelRunRepository(db_session)

# Lấy top 5 runs theo Sharpe ratio
top_runs = repo.get_top_by_sharpe(top_n=5)

# Lấy run theo ID
run = repo.get_by_id(run_id)
```

## Troubleshooting

### Lỗi: "Kết nối PostgreSQL thất bại"

1. **Kiểm tra PostgreSQL đang chạy:**
   ```bash
   # Windows
   Get-Service postgresql*
   
   # Linux/Mac
   sudo systemctl status postgresql
   ```

2. **Kiểm tra thông tin kết nối:**
   - Host và port có đúng không?
   - Username và password có đúng không?
   - Database đã được tạo chưa?

3. **Kiểm tra firewall:**
   - Port 5432 có bị chặn không?

4. **Kiểm tra quyền truy cập:**
   ```sql
   -- Kết nối PostgreSQL
   psql -U postgres
   
   -- Kiểm tra database
   \l
   
   -- Kiểm tra quyền
   \du
   ```

### Lỗi: "relation does not exist"

Database chưa có tables. Chạy lại script test:

```bash
python test_db_connection.py
```

Hoặc trong code:

```python
from backend.database.connection import init_database
init_database()
```

## Ghi chú

- Hệ thống tự động fallback sang SQLite nếu không kết nối được PostgreSQL
- Dữ liệu trong SQLite sẽ mất khi chương trình kết thúc
- Khuyến nghị sử dụng PostgreSQL cho production
