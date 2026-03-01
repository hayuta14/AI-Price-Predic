"""
Walk-Forward验证引擎

实现严格的时间序列walk-forward验证，确保：
1. 无数据泄漏（严格的时间顺序）
2. 扩展窗口或滚动窗口支持
3. 返回每个fold的详细指标
4. 聚合所有fold的总体性能

核心原则：
- 训练集时间 < 测试集时间（严格）
- 不允许任何未来信息泄露
- 支持扩展窗口（训练集累积增长）和滚动窗口（固定大小）
"""
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.config import WalkForwardConfig


@dataclass
class WalkForwardFold:
    """单个walk-forward fold的数据结构"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int
    train_data: pd.DataFrame
    test_data: pd.DataFrame


@dataclass
class WalkForwardResults:
    """Walk-forward验证结果"""
    folds: List[WalkForwardFold]
    fold_metrics: List[Dict[str, float]]  # 每个fold的指标
    aggregated_metrics: Dict[str, float]  # 聚合指标
    total_train_samples: int
    total_test_samples: int
    n_folds: int
    sharpe_stability: float = 0.0  # Sharpe stability across folds
    sharpe_by_period: Optional[pd.DataFrame] = None  # Sharpe ratio by time period


class WalkForwardEngine:
    """
    Walk-Forward验证引擎
    
    实现hedge-fund级别的严格时间序列验证：
    - 扩展窗口：训练集从初始点累积增长
    - 滚动窗口：训练集保持固定大小
    - 严格时间顺序：确保无lookahead bias
    """
    
    def __init__(self, config: WalkForwardConfig):
        """
        初始化walk-forward引擎
        
        Args:
            config: Walk-forward配置参数
        """
        self.config = config
        self.folds: List[WalkForwardFold] = []
    
    def generate_folds(
        self, 
        data: pd.DataFrame,
        date_column: str = 'timestamp'
    ) -> List[WalkForwardFold]:
        """
        生成walk-forward folds
        
        Args:
            data: 时间序列数据（必须按时间排序）
            date_column: 日期列名
            
        Returns:
            Walk-forward folds列表
            
        Raises:
            ValueError: 如果数据不足或配置无效
        """
        # 确保数据按时间排序
        if date_column not in data.columns:
            raise ValueError(f"日期列 '{date_column}' 不存在于数据中")
        
        data = data.sort_values(by=date_column).reset_index(drop=True)
        data[date_column] = pd.to_datetime(data[date_column])
        
        # 验证数据量
        if len(data) < self.config.min_train_samples:
            raise ValueError(
                f"数据量不足：需要至少 {self.config.min_train_samples} 个样本，"
                f"实际只有 {len(data)} 个"
            )
        
        # 计算时间窗口（转换为时间戳）
        train_window = timedelta(days=self.config.train_window_days)
        test_window = timedelta(days=self.config.test_window_days)
        step = timedelta(days=self.config.step_days)
        
        folds = []
        fold_id = 0
        
        # 确定初始训练集结束时间
        first_date = data[date_column].min()
        train_end = first_date + train_window
        
        # 生成folds直到数据用完
        while True:
            # 测试集时间范围
            test_start = train_end
            test_end = test_start + test_window
            
            # 检查是否有足够的测试数据
            test_data = data[
                (data[date_column] >= test_start) & 
                (data[date_column] < test_end)
            ]
            
            if len(test_data) == 0:
                break  # 没有更多测试数据
            
            # 训练集时间范围
            if self.config.expanding_window:
                # 扩展窗口：从初始点到train_end
                train_start = first_date
            else:
                # 滚动窗口：保持固定大小
                train_start = train_end - train_window
            
            train_data = data[
                (data[date_column] >= train_start) & 
                (data[date_column] < train_end)
            ]
            
            # 验证训练集大小
            if len(train_data) < self.config.min_train_samples:
                # 如果使用滚动窗口且训练集不足，尝试扩展窗口
                if not self.config.expanding_window:
                    train_start = first_date
                    train_data = data[
                        (data[date_column] >= train_start) & 
                        (data[date_column] < train_end)
                    ]
                
                if len(train_data) < self.config.min_train_samples:
                    # 跳过这个fold，继续下一个
                    train_end = test_end
                    continue
            
            # 严格验证：确保训练集时间 < 测试集时间
            if train_data[date_column].max() >= test_data[date_column].min():
                raise ValueError(
                    f"Fold {fold_id}: 时间顺序错误！"
                    f"训练集结束时间 {train_data[date_column].max()} "
                    f"必须 < 测试集开始时间 {test_data[date_column].min()}"
                )
            
            # 创建fold
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=len(train_data),
                test_size=len(test_data),
                train_data=train_data.copy(),
                test_data=test_data.copy()
            )
            
            folds.append(fold)
            fold_id += 1
            
            # 移动到下一个fold
            train_end = test_end + step
            
            # 检查是否还有足够的数据
            if train_end > data[date_column].max():
                break
        
        if len(folds) == 0:
            raise ValueError("无法生成任何walk-forward fold，请检查配置和数据")
        
        self.folds = folds
        return folds
    
    def validate_fold(self, fold: WalkForwardFold) -> bool:
        """
        验证fold的时间顺序正确性
        
        Args:
            fold: 要验证的fold
            
        Returns:
            True if valid, False otherwise
        """
        date_col = 'timestamp'
        
        # 检查训练集和测试集的时间顺序
        train_max = fold.train_data[date_col].max()
        test_min = fold.test_data[date_col].min()
        
        if train_max >= test_min:
            return False
        
        # 检查数据完整性
        if len(fold.train_data) == 0 or len(fold.test_data) == 0:
            return False
        
        return True
    
    def get_fold_summary(self) -> pd.DataFrame:
        """
        获取所有folds的摘要信息
        
        Returns:
            DataFrame包含每个fold的详细信息
        """
        if not self.folds:
            return pd.DataFrame()
        
        summary_data = []
        for fold in self.folds:
            summary_data.append({
                'fold_id': fold.fold_id,
                'train_start': fold.train_start,
                'train_end': fold.train_end,
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'train_size': fold.train_size,
                'test_size': fold.test_size,
                'train_days': (fold.train_end - fold.train_start).days,
                'test_days': (fold.test_end - fold.test_start).days
            })
        
        return pd.DataFrame(summary_data)
    
    def run_validation(
        self,
        data: pd.DataFrame,
        model_train_fn,
        model_predict_fn,
        metrics_fn,
        date_column: str = 'timestamp',
        **kwargs
    ) -> WalkForwardResults:
        """
        执行完整的walk-forward验证
        
        Args:
            data: 时间序列数据
            model_train_fn: 模型训练函数 (train_data, **kwargs) -> model
            model_predict_fn: 模型预测函数 (model, test_data) -> predictions
            metrics_fn: 指标计算函数 (predictions, test_data) -> dict
            date_column: 日期列名
            **kwargs: 传递给训练函数的额外参数
            
        Returns:
            WalkForwardResults对象
        """
        # 生成folds
        folds = self.generate_folds(data, date_column)
        
        # Kiểm tra số lượng folds
        if len(folds) < self.config.min_n_folds:
            print(f"\n⚠️  Warning: Chỉ có {len(folds)} folds, cần ít nhất {self.config.min_n_folds} folds để kiểm tra robustness")
            print(f"   → Có thể cần điều chỉnh train_window_days, test_window_days, hoặc step_days")
        elif len(folds) > self.config.max_n_folds:
            print(f"\n⚠️  Warning: Có {len(folds)} folds, nhiều hơn max_n_folds={self.config.max_n_folds}")
            print(f"   → Có thể giảm step_days để có nhiều folds hơn hoặc tăng max_n_folds")
        
        print(f"\n📊 Walk-Forward Configuration:")
        print(f"   • Total Folds: {len(folds)}")
        print(f"   • Train Window: {self.config.train_window_days} days")
        print(f"   • Test Window: {self.config.test_window_days} days")
        print(f"   • Step: {self.config.step_days} days")
        print(f"   • Expanding Window: {self.config.expanding_window}")
        
        # 验证所有folds
        for fold in folds:
            if not self.validate_fold(fold):
                raise ValueError(f"Fold {fold.fold_id} 验证失败")
        
        # 对每个fold进行训练和评估
        fold_metrics = []
        
        for fold in folds:
            # 训练模型（严格使用训练集）
            model = model_train_fn(fold.train_data, **kwargs)
            
            # 预测（严格使用测试集）
            predictions = model_predict_fn(model, fold.test_data)
            
            # 计算指标
            metrics = metrics_fn(predictions, fold.test_data)
            metrics['fold_id'] = fold.fold_id
            fold_metrics.append(metrics)
        
        # 聚合指标（跨所有folds）
        aggregated_metrics = self._aggregate_metrics(fold_metrics)
        
        # 计算总样本数
        total_train_samples = sum(fold.train_size for fold in folds)
        total_test_samples = sum(fold.test_size for fold in folds)
        
        # Tính Sharpe stability và Sharpe theo từng period
        sharpe_stability = aggregated_metrics.get('sharpe_stability', 0.0)
        
        # Tạo DataFrame với Sharpe theo từng period
        sharpe_by_period_data = []
        for i, fold in enumerate(folds):
            fold_metric = fold_metrics[i] if i < len(fold_metrics) else {}
            sharpe_by_period_data.append({
                'fold_id': fold.fold_id,
                'period_start': fold.test_start,
                'period_end': fold.test_end,
                'sharpe_ratio': fold_metric.get('sharpe_ratio', fold_metric.get('sharpe', 0.0)),
                'max_drawdown': fold_metric.get('max_drawdown', 1.0),
                'profit_factor': fold_metric.get('profit_factor', 0.0),
                'total_trades': fold_metric.get('total_trades', 0)
            })
        sharpe_by_period = pd.DataFrame(sharpe_by_period_data) if sharpe_by_period_data else None
        
        return WalkForwardResults(
            folds=folds,
            fold_metrics=fold_metrics,
            aggregated_metrics=aggregated_metrics,
            total_train_samples=total_train_samples,
            total_test_samples=total_test_samples,
            n_folds=len(folds),
            sharpe_stability=sharpe_stability,
            sharpe_by_period=sharpe_by_period
        )
    
    def _aggregate_metrics(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        聚合所有folds的指标
        
        Args:
            fold_metrics: 每个fold的指标列表
            
        Returns:
            聚合后的指标字典
        """
        if not fold_metrics:
            return {}
        
        # 转换为DataFrame便于聚合
        df = pd.DataFrame(fold_metrics)
        
        aggregated = {}
        
        # 对于不同指标使用不同的聚合方法
        # 收益率相关：平均
        for col in ['return', 'sharpe', 'calmar', 'profit_factor']:
            if col in df.columns:
                aggregated[f'mean_{col}'] = df[col].mean()
                aggregated[f'std_{col}'] = df[col].std()
        
        # 回撤相关：取最大值
        for col in ['max_drawdown', 'max_dd_duration']:
            if col in df.columns:
                aggregated[f'max_{col}'] = df[col].max()
                aggregated[f'mean_{col}'] = df[col].mean()
        
        # 交易统计：求和
        for col in ['total_trades', 'winning_trades', 'losing_trades']:
            if col in df.columns:
                aggregated[f'total_{col}'] = df[col].sum()
        
        # 胜率
        if 'winning_trades' in df.columns and 'total_trades' in df.columns:
            total_wins = df['winning_trades'].sum()
            total_trades = df['total_trades'].sum()
            if total_trades > 0:
                aggregated['overall_win_rate'] = total_wins / total_trades
        
        # 稳定性指标：跨folds的Sharpe标准差
        if 'sharpe_ratio' in df.columns:
            sharpe_values = df['sharpe_ratio']
            aggregated['mean_sharpe_ratio'] = sharpe_values.mean()
            aggregated['std_sharpe_ratio'] = sharpe_values.std()
            aggregated['min_sharpe_ratio'] = sharpe_values.min()
            aggregated['max_sharpe_ratio'] = sharpe_values.max()
            # Sharpe stability: 1 / (1 + coefficient of variation)
            if abs(sharpe_values.mean()) > 1e-6:
                cv = sharpe_values.std() / abs(sharpe_values.mean())
                aggregated['sharpe_stability'] = 1.0 / (1.0 + cv)
            else:
                aggregated['sharpe_stability'] = 0.0
        elif 'sharpe' in df.columns:
            sharpe_values = df['sharpe']
            aggregated['mean_sharpe_ratio'] = sharpe_values.mean()
            aggregated['std_sharpe_ratio'] = sharpe_values.std()
            aggregated['min_sharpe_ratio'] = sharpe_values.min()
            aggregated['max_sharpe_ratio'] = sharpe_values.max()
            if abs(sharpe_values.mean()) > 1e-6:
                cv = sharpe_values.std() / abs(sharpe_values.mean())
                aggregated['sharpe_stability'] = 1.0 / (1.0 + cv)
            else:
                aggregated['sharpe_stability'] = 0.0
        
        return aggregated
