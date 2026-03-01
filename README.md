# BTCUSDT 15分钟期货交易系统优化平台

生产级模块化量化研究框架，用于优化BTCUSDT 15分钟期货交易系统。

## 系统特性

- **严格的时间序列walk-forward验证** - 无数据泄漏，无lookahead bias
- **动态风险管理** - 基于波动率的仓位管理，多重风险控制
- **全面优化** - 特征、标签、风险参数、超参数自动优化
- **稳健性分析** - 蒙特卡洛模拟、市场状态分析
- **模块化架构** - 易于扩展和维护

## 技术栈

- Python 3.11
- pandas, numpy
- xgboost, shap, optuna
- fastapi, pydantic, SQLAlchemy
- PostgreSQL, MongoDB

## 项目结构

```
backend/
├── api/              # FastAPI路由
├── core/             # 核心引擎（walk-forward, backtest, risk, metrics）
├── optimization/     # 优化器模块
├── analysis/         # 分析模块（蒙特卡洛、市场状态、报告）
├── services/         # 服务层
├── database/         # 数据库模型和仓库
├── config.py        # 配置管理
└── main.py          # 主程序入口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动数据库

```bash
docker-compose up -d
```

### 3. 运行优化

```bash
python backend/main.py
```

## 核心模块说明

### Walk-Forward引擎
严格的时间序列验证，支持扩展窗口和滚动窗口，确保无数据泄漏。

### 回测引擎
完整的交易模拟，包括：
- ATR-based止损
- 风险/收益比止盈
- 手续费（0.05%）
- 滑点（0.02%）
- 资金费率

### 风险管理引擎
- 动态波动率缩放
- 最大杠杆限制
- 单日亏损限制
- 最大回撤保护

### 优化器
- **特征优化器**: 基于SHAP重要性的特征选择
- **标签优化器**: 优化预测时间范围和阈值
- **风险优化器**: 优化风险参数组合
- **超参数优化器**: Optuna Bayesian优化

### 分析模块
- **蒙特卡洛模拟**: 1000次模拟，评估策略稳健性
- **市场状态分析**: 按波动率和趋势分段分析
- **性能报告**: 多维度排名和综合评估

## API接口

- `POST /optimize/start` - 启动优化任务
- `GET /optimize/status/{id}` - 获取优化状态
- `GET /optimize/results/{id}` - 获取优化结果
- `GET /runs` - 获取所有运行记录
- `GET /runs/{id}` - 获取单个运行记录
- `GET /runs/top/sharpe` - 获取Top Sharpe配置
- `GET /runs/top/stability` - 获取最稳定配置

## 配置说明

主要配置在 `backend/config.py` 中：

- `WalkForwardConfig`: Walk-forward验证参数
- `RiskConfig`: 风险管理参数
- `TradingConfig`: 交易成本参数
- `LabelConfig`: 标签配置
- `XGBoostConfig`: XGBoost超参数搜索空间
- `MonteCarloConfig`: 蒙特卡洛模拟配置

## 数据库

使用PostgreSQL存储优化结果：
- `model_runs`: 模型运行记录
- `optimization_trials`: 优化试验记录

## 注意事项

1. 确保数据按时间排序
2. 数据必须包含必要的价格列（open, high, low, close）
3. 首次运行会创建数据库表结构
4. 优化过程可能需要较长时间，建议使用后台任务

## 许可证

MIT License
