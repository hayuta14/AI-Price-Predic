"""
超参数优化器

使用Optuna进行Bayesian超参数优化：
- XGBoost超参数搜索
- 使用walk-forward Sharpe作为目标函数
- 限制搜索空间以避免过拟合
"""
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb

from backend.config import XGBoostConfig
from backend.core.walkforward_engine import WalkForwardEngine
from backend.core.metrics import MetricsCalculator


@dataclass
class HyperparameterSet:
    """超参数集"""
    max_depth: int
    learning_rate: float
    n_estimators: int
    subsample: float
    colsample_bytree: float
    sharpe_ratio: float
    max_drawdown: float


class HyperparameterOptimizer:
    """
    超参数优化器
    
    使用Optuna进行Bayesian优化
    """
    
    def __init__(
        self,
        walkforward_engine: WalkForwardEngine,
        config: XGBoostConfig
    ):
        """
        初始化超参数优化器
        
        Args:
            walkforward_engine: Walk-forward验证引擎
            config: XGBoost配置
        """
        self.walkforward_engine = walkforward_engine
        self.config = config
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.Study] = None
        self.metrics_calculator = MetricsCalculator()
    
    def create_objective_function(
        self,
        data: pd.DataFrame,
        features: List[str],
        target_col: str
    ) -> Callable:
        """
        创建Optuna目标函数
        
        Args:
            data: 完整数据集
            features: 特征列表
            target_col: 目标列名
            
        Returns:
            目标函数
        """
        def objective(trial: optuna.Trial) -> float:
            # 建议超参数
            max_depth = trial.suggest_int(
                'max_depth',
                self.config.max_depth_range[0],
                self.config.max_depth_range[1]
            )
            
            learning_rate = trial.suggest_float(
                'learning_rate',
                self.config.learning_rate_range[0],
                self.config.learning_rate_range[1],
                log=True
            )
            
            n_estimators = trial.suggest_int(
                'n_estimators',
                self.config.n_estimators_range[0],
                self.config.n_estimators_range[1]
            )
            
            subsample = trial.suggest_float(
                'subsample',
                self.config.subsample_range[0],
                self.config.subsample_range[1]
            )
            
            colsample_bytree = trial.suggest_float(
                'colsample_bytree',
                self.config.colsample_bytree_range[0],
                self.config.colsample_bytree_range[1]
            )
            
            # 定义模型训练函数
            def train_model_internal(X_train, y_train, **kwargs):
                model = xgb.XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
                model.fit(X_train, y_train)
                return model
            
            # Wrapper để extract features và target từ train_data
            def train_wrapper(train_data, **kwargs):
                X_train = train_data[features]
                y_train = train_data[target_col]
                return train_model_internal(X_train, y_train, **kwargs)
            
            # 定义预测函数
            def predict(model, test_data):
                X_test = test_data[features]
                return model.predict_proba(X_test)[:, 1]  # 返回正类概率
            
            # 定义指标计算函数
            def calculate_metrics(predictions, test_data):
                # 将预测转换为收益率（简化实现）
                if 'close' not in test_data.columns:
                    return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
                
                price_returns = test_data['close'].pct_change().fillna(0.0)
                signals = (predictions > 0.5).astype(int) * 2 - 1
                strategy_returns = price_returns * signals
                
                if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                    return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
                
                equity = (1 + strategy_returns).cumprod()
                metrics = self.metrics_calculator.calculate_all_metrics(
                    strategy_returns,
                    equity
                )
                
                return {
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown
                }
            
            # 运行walk-forward验证
            try:
                results = self.walkforward_engine.run_validation(
                    data=data,
                    model_train_fn=train_wrapper,
                    model_predict_fn=predict,
                    metrics_fn=calculate_metrics
                )
                
                # 返回平均Sharpe（作为优化目标）
                sharpe = results.aggregated_metrics.get(
                    'mean_sharpe_ratio',
                    results.aggregated_metrics.get('sharpe_ratio', 0.0)
                )
                
                return sharpe
            except Exception as e:
                print(f"Trial失败: {e}")
                return -np.inf
        
        return objective
    
    def optimize(
        self,
        data: pd.DataFrame,
        features: List[str],
        target_col: str,
        n_trials: Optional[int] = None,
        study_name: Optional[str] = None
    ) -> HyperparameterSet:
        """
        优化超参数
        
        Args:
            data: 完整数据集
            features: 特征列表
            target_col: 目标列名
            n_trials: 优化试验次数（如果为None，使用配置值）
            study_name: Optuna study名称
            
        Returns:
            最优超参数集
        """
        # Kiểm tra features có rỗng không
        if not features or len(features) == 0:
            print("Cảnh báo: Không có features để tối ưu hóa, trả về giá trị mặc định")
            return HyperparameterSet(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                sharpe_ratio=0.0,
                max_drawdown=1.0
            )
        
        if n_trials is None:
            n_trials = self.config.n_trials
        
        # 创建study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name or 'xgboost_optimization'
        )
        
        # 创建目标函数
        objective = self.create_objective_function(data, features, target_col)
        
        # 运行优化
        print(f"开始超参数优化，共 {n_trials} 次试验...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.study = study
        self.best_params = study.best_params
        
        # 评估最优参数
        best_sharpe = study.best_value
        
        # 重新运行一次以获取完整指标
        def train_model_internal(X_train, y_train, **kwargs):
            model = xgb.XGBClassifier(
                max_depth=self.best_params['max_depth'],
                learning_rate=self.best_params['learning_rate'],
                n_estimators=self.best_params['n_estimators'],
                subsample=self.best_params['subsample'],
                colsample_bytree=self.best_params['colsample_bytree'],
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            model.fit(X_train, y_train)
            return model
        
        # Wrapper để extract features và target từ train_data
        def train_wrapper(train_data, **kwargs):
            X_train = train_data[features]
            y_train = train_data[target_col]
            return train_model_internal(X_train, y_train, **kwargs)
        
        def predict(model, test_data):
            X_test = test_data[features]
            return model.predict_proba(X_test)[:, 1]
        
        def calculate_metrics(predictions, test_data):
            if 'close' not in test_data.columns:
                return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
            
            price_returns = test_data['close'].pct_change().fillna(0.0)
            signals = (predictions > 0.5).astype(int) * 2 - 1
            strategy_returns = price_returns * signals
            
            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
            
            equity = (1 + strategy_returns).cumprod()
            metrics = self.metrics_calculator.calculate_all_metrics(
                strategy_returns,
                equity
            )
            
            return {
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown
            }
        
        results = self.walkforward_engine.run_validation(
            data=data,
            model_train_fn=train_wrapper,
            model_predict_fn=predict,
            metrics_fn=calculate_metrics
        )
        
        max_dd = results.aggregated_metrics.get('max_max_drawdown', 1.0)
        
        return HyperparameterSet(
            max_depth=self.best_params['max_depth'],
            learning_rate=self.best_params['learning_rate'],
            n_estimators=self.best_params['n_estimators'],
            subsample=self.best_params['subsample'],
            colsample_bytree=self.best_params['colsample_bytree'],
            sharpe_ratio=best_sharpe,
            max_drawdown=max_dd
        )
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        获取优化历史
        
        Returns:
            DataFrame包含所有试验的结果
        """
        if self.study is None:
            return pd.DataFrame()
        
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                'trial_number': trial.number,
                'max_depth': trial.params.get('max_depth'),
                'learning_rate': trial.params.get('learning_rate'),
                'n_estimators': trial.params.get('n_estimators'),
                'subsample': trial.params.get('subsample'),
                'colsample_bytree': trial.params.get('colsample_bytree'),
                'sharpe_ratio': trial.value,
                'state': trial.state.name
            })
        
        return pd.DataFrame(trials_data)
