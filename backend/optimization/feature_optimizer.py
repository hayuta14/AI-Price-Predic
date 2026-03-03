"""
特征优化器

通过迭代添加/移除特征来优化特征集：
- 基于SHAP重要性评估特征
- 使用walk-forward验证评估特征集性能
- 存储特征性能历史
- 避免过拟合（严格使用out-of-sample评估）
"""
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

from backend.core.walkforward_engine import WalkForwardEngine, WalkForwardResults
from backend.core.metrics import MetricsCalculator
from backend.optimization.information_gain_filter import InformationGainFilter


@dataclass
class FeatureSet:
    """特征集"""
    features: List[str]
    sharpe_ratio: float
    max_drawdown: float
    n_features: int
    shap_importance: Dict[str, float]


class FeatureOptimizer:
    """
    特征优化器
    
    使用迭代方法优化特征集：
    1. 从空集或基础特征集开始
    2. 逐个添加特征，评估性能
    3. 移除低重要性特征
    4. 使用SHAP分析特征重要性
    """
    
    def __init__(
        self,
        walkforward_engine: WalkForwardEngine,
        initial_features: Optional[List[str]] = None,
        use_mi_filter: bool = True,
        min_mi: float = 0.001
    ):
        """
        初始化特征优化器
        
        Args:
            walkforward_engine: Walk-forward验证引擎
            initial_features: 初始特征列表（如果为None，从数据中推断）
            use_mi_filter: Có sử dụng Information Gain filter không
            min_mi: Minimum Mutual Information threshold
        """
        self.walkforward_engine = walkforward_engine
        self.initial_features = initial_features or []
        self.feature_history: List[FeatureSet] = []
        self.metrics_calculator = MetricsCalculator()
        self.use_mi_filter = use_mi_filter
        self.mi_filter = InformationGainFilter(min_mi=min_mi) if use_mi_filter else None
    
    def calculate_shap_importance(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """
        计算SHAP特征重要性
        
        Args:
            model: 训练好的XGBoost模型
            X: 特征数据
            sample_size: 采样大小（用于加速计算）
            
        Returns:
            特征重要性字典（特征名 -> 重要性分数）
        """
        # 采样以加速计算
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # 计算平均绝对SHAP值作为重要性
        if isinstance(shap_values, list):
            # 多分类情况，取第一个类别
            shap_values = shap_values[0]
        
        importance = np.abs(shap_values).mean(axis=0)
        
        # 转换为字典
        feature_importance = {
            feature: float(importance[i])
            for i, feature in enumerate(X.columns)
        }
        
        return feature_importance
    
    def evaluate_feature_set(
        self,
        data: pd.DataFrame,
        features: List[str],
        target_col: str,
        model_train_fn: Callable,
        model_predict_fn: Callable
    ) -> Dict[str, float]:
        """
        评估特征集的性能
        
        Args:
            data: 完整数据集
            features: 特征列表
            target_col: 目标列名
            model_train_fn: 模型训练函数
            model_predict_fn: 模型预测函数
            
        Returns:
            性能指标字典
        """
        # 准备特征数据
        if not features:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
        
        # 运行walk-forward验证
        def train_wrapper(train_data, **kwargs):
            X_train = train_data[features]
            y_train = train_data[target_col]
            return model_train_fn(X_train, y_train, **kwargs)
        
        def predict_wrapper(model, test_data):
            X_test = test_data[features]
            return model_predict_fn(model, X_test)
        
        def metrics_wrapper(predictions, test_data):
            # 这里简化：假设predictions是概率，需要转换为收益率
            # 实际应用中需要根据具体业务逻辑调整
            returns = self._predictions_to_returns(predictions, test_data)
            equity = (1 + returns).cumprod()
            metrics = self.metrics_calculator.calculate_all_metrics(returns, equity)
            return {
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'profit_factor': metrics.profit_factor
            }
        
        results = self.walkforward_engine.run_validation(
            data=data,
            model_train_fn=train_wrapper,
            model_predict_fn=predict_wrapper,
            metrics_fn=metrics_wrapper
        )
        
        return results.aggregated_metrics
    
    def _predictions_to_returns(
        self,
        predictions: np.ndarray,
        test_data: pd.DataFrame
    ) -> pd.Series:
        """
        将模型预测转换为收益率序列
        
        这是一个占位符实现，实际应用中需要根据具体策略调整
        
        Args:
            predictions: 模型预测值
            test_data: 测试数据
            
        Returns:
            收益率序列
        """
        # 简化实现：假设预测是上涨概率
        # 如果概率>0.5，做多；否则做空
        # 收益率 = 价格变化 * 方向
        
        if 'close' not in test_data.columns:
            # 如果没有价格数据，返回零收益
            return pd.Series(0.0, index=test_data.index)
        
        price_returns = test_data['close'].pct_change().fillna(0.0)
        
        # 根据预测调整方向
        signals = (predictions > 0.5).astype(int) * 2 - 1  # -1 or 1
        
        strategy_returns = price_returns * signals
        
        return strategy_returns
    
    def optimize_features(
        self,
        data: pd.DataFrame,
        available_features: List[str],
        target_col: str,
        model_train_fn: Callable,
        model_predict_fn: Callable,
        max_iterations: int = 20,
        min_sharpe_improvement: float = 0.01,
        apply_mi_filter: bool = True,
        max_features_to_try_per_iteration: int = 10
    ) -> FeatureSet:
        """
        优化特征集
        
        使用前向选择和后向消除的组合策略
        
        Args:
            data: 完整数据集
            available_features: 可用特征列表
            target_col: 目标列名
            model_train_fn: 模型训练函数
            model_predict_fn: 模型预测函数
            max_iterations: 最大迭代次数
            min_sharpe_improvement: 最小Sharpe改进阈值
            apply_mi_filter: Có áp dụng Information Gain filter không
            
        Returns:
            最优特征集
        """
        # Pre-filter features bằng Information Gain nếu được bật
        if apply_mi_filter and self.mi_filter is not None:
            print("\n🔍 Đang kiểm tra Information Gain của features...")
            # Chuẩn bị data
            valid_data = data.iloc[:-10].copy() if len(data) > 10 else data.copy()
            valid_labels = valid_data[target_col] if target_col in valid_data.columns else pd.Series()
            
            if len(valid_labels) > 0 and target_col in valid_data.columns:
                # Chỉ lấy features có sẵn trong data
                available_in_data = [f for f in available_features if f in valid_data.columns]
                if available_in_data:
                    X_for_mi = valid_data[available_in_data]
                    
                    # In báo cáo
                    self.mi_filter.print_analysis_report(X_for_mi, valid_labels, sample_size=5000)
                    
                    # Filter features
                    filtered_X, selected_features, mi_scores = self.mi_filter.filter_features(
                        X_for_mi, valid_labels, min_mi=self.mi_filter.min_mi
                    )
                    
                    print(f"\n✅ Đã filter: {len(available_in_data)} → {len(selected_features)} features "
                          f"(loại bỏ {len(available_in_data) - len(selected_features)} features vô nghĩa)")
                    
                    # Cập nhật available_features
                    available_features = selected_features
        
        # 初始化：使用初始特征或空集
        current_features = set(self.initial_features) if self.initial_features else set()
        remaining_features = set(available_features) - current_features
        
        best_sharpe = -np.inf
        best_feature_set = None
        
        for iteration in range(max_iterations):
            improved = False
            
            # 前向选择：尝试添加特征（限制每轮评估数量，降低复杂度）
            if remaining_features:
                best_new_feature = None
                best_new_sharpe = best_sharpe

                candidate_add_features = sorted(list(remaining_features))[:max_features_to_try_per_iteration]

                for feature in candidate_add_features:
                    test_features = list(current_features | {feature})
                    
                    try:
                        metrics = self.evaluate_feature_set(
                            data=data,
                            features=test_features,
                            target_col=target_col,
                            model_train_fn=model_train_fn,
                            model_predict_fn=model_predict_fn
                        )
                        
                        sharpe = metrics.get('mean_sharpe_ratio', metrics.get('sharpe_ratio', 0.0))
                        
                        if sharpe > best_new_sharpe + min_sharpe_improvement:
                            best_new_sharpe = sharpe
                            best_new_feature = feature
                    except Exception as e:
                        # 跳过有问题的特征
                        print(f"评估特征 {feature} 时出错: {e}")
                        continue
                
                if best_new_feature:
                    current_features.add(best_new_feature)
                    remaining_features.remove(best_new_feature)
                    best_sharpe = best_new_sharpe
                    improved = True
            
            # 后向消除：尝试移除特征
            if len(current_features) > 1:
                worst_feature = None
                best_removed_sharpe = best_sharpe
                
                candidate_remove_features = sorted(list(current_features))[:max_features_to_try_per_iteration]

                for feature in candidate_remove_features:
                    test_features = list(current_features - {feature})
                    
                    try:
                        metrics = self.evaluate_feature_set(
                            data=data,
                            features=test_features,
                            target_col=target_col,
                            model_train_fn=model_train_fn,
                            model_predict_fn=model_predict_fn
                        )
                        
                        sharpe = metrics.get('mean_sharpe_ratio', metrics.get('sharpe_ratio', 0.0))
                        
                        if sharpe > best_removed_sharpe + min_sharpe_improvement:
                            best_removed_sharpe = sharpe
                            worst_feature = feature
                    except Exception as e:
                        continue
                
                if worst_feature:
                    current_features.remove(worst_feature)
                    remaining_features.add(worst_feature)
                    best_sharpe = best_removed_sharpe
                    improved = True
            
            # 评估当前特征集
            if current_features:
                metrics = self.evaluate_feature_set(
                    data=data,
                    features=list(current_features),
                    target_col=target_col,
                    model_train_fn=model_train_fn,
                    model_predict_fn=model_predict_fn
                )
                
                sharpe = metrics.get('mean_sharpe_ratio', metrics.get('sharpe_ratio', 0.0))
                max_dd = metrics.get('max_max_drawdown', metrics.get('max_drawdown', 1.0))
                
                # 计算SHAP重要性（使用最后一个fold的模型）
                shap_importance = {}
                if self.walkforward_engine.folds:
                    # 使用最后一个fold训练模型计算SHAP
                    last_fold = self.walkforward_engine.folds[-1]
                    X_train = last_fold.train_data[list(current_features)]
                    y_train = last_fold.train_data[target_col]
                    model = model_train_fn(X_train, y_train)
                    shap_importance = self.calculate_shap_importance(model, X_train)
                
                feature_set = FeatureSet(
                    features=list(current_features),
                    sharpe_ratio=sharpe,
                    max_drawdown=max_dd,
                    n_features=len(current_features),
                    shap_importance=shap_importance
                )
                
                self.feature_history.append(feature_set)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_feature_set = feature_set
            
            if not improved:
                break
        
        return best_feature_set if best_feature_set else FeatureSet(
            features=list(current_features),
            sharpe_ratio=0.0,
            max_drawdown=1.0,
            n_features=len(current_features),
            shap_importance={}
        )
    
    def get_feature_history(self) -> pd.DataFrame:
        """
        获取特征优化历史
        
        Returns:
            DataFrame包含每次迭代的特征集和性能
        """
        if not self.feature_history:
            return pd.DataFrame()
        
        history_data = []
        for i, feature_set in enumerate(self.feature_history):
            history_data.append({
                'iteration': i,
                'n_features': feature_set.n_features,
                'features': ','.join(feature_set.features),
                'sharpe_ratio': feature_set.sharpe_ratio,
                'max_drawdown': feature_set.max_drawdown
            })
        
        return pd.DataFrame(history_data)
