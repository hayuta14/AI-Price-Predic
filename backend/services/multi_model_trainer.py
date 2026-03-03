"""
Multi-Model Trainer Service

Train nhiều models song song để tìm ra model tốt nhất:
- Parallel training với multiprocessing
- So sánh performance giữa các models
- Chọn model tốt nhất dựa trên multiple metrics
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import as_completed
import multiprocessing as mp
from pathlib import Path
import logging

from backend.core.walkforward_engine import WalkForwardEngine, WalkForwardResults
from backend.services.training_service import TrainingService
from backend.services.model_persistence import ModelPersistence
from backend.optimization.feature_optimizer import FeatureSet
from backend.optimization.label_optimizer import LabelConfiguration
from backend.optimization.hyperparameter_optimizer import HyperparameterSet


@dataclass
class ModelCandidate:
    """Ứng viên model"""
    candidate_id: int
    features: List[str]
    label_config: LabelConfiguration
    hyperparams: HyperparameterSet
    metrics: Dict[str, float]
    model_path: Optional[Path] = None


class MultiModelTrainer:
    """Train nhiều models song song"""
    
    def __init__(
        self,
        walkforward_engine: WalkForwardEngine,
        training_service: TrainingService,
        model_persistence: ModelPersistence,
        n_workers: Optional[int] = None
    ):
        """
        Khởi tạo MultiModelTrainer
        
        Args:
            walkforward_engine: Walk-forward engine
            training_service: Training service
            model_persistence: Model persistence service
            n_workers: Số workers cho parallel processing (mặc định: CPU count)
        """
        self.walkforward_engine = walkforward_engine
        self.training_service = training_service
        self.model_persistence = model_persistence
        
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)  # Để lại 1 core cho hệ thống
        self.n_workers = n_workers
        
        self.logger = logging.getLogger(__name__)
    
    def train_single_model(
        self,
        candidate_id: int,
        data: pd.DataFrame,
        features: List[str],
        label_config: LabelConfiguration,
        hyperparams: HyperparameterSet,
        price_col: str = 'close',
        timestamp_col: str = 'timestamp'
    ) -> ModelCandidate:
        """
        Train một model đơn lẻ
        
        Args:
            candidate_id: ID của candidate
            data: Dữ liệu
            features: Danh sách features
            label_config: Cấu hình label
            hyperparams: Hyperparameters
            price_col: Tên cột giá
            timestamp_col: Tên cột timestamp
            
        Returns:
            ModelCandidate với metrics
        """
        try:
            self.logger.info(f"Training candidate {candidate_id}...")
            
            # Tạo labels
            from backend.optimization.label_optimizer import LabelOptimizer
            label_optimizer = LabelOptimizer(self.walkforward_engine, None)
            
            use_dynamic = label_config.threshold < 0
            # Sử dụng default_atr_multiplier từ config (0.5)
            from backend.config import config
            default_atr_mult = config.label.default_atr_multiplier if hasattr(config.label, 'default_atr_multiplier') else 0.5
            atr_multiplier = abs(label_config.threshold) if use_dynamic else default_atr_mult
            threshold = label_config.threshold if not use_dynamic else 0.0
            
            atr_values = None
            if use_dynamic and all(col in data.columns for col in ['high', 'low', 'close']):
                from backend.data.feature_engineering import calculate_atr
                atr_values = calculate_atr(data, 14)
            
            labels = label_optimizer.create_labels(
                data=data,
                price_col=price_col,
                horizon=label_config.horizon,
                threshold=threshold,
                use_dynamic_threshold=use_dynamic,
                atr_values=atr_values,
                atr_multiplier=atr_multiplier
            )
            
            # Chuẩn bị data
            valid_data = data.iloc[:-label_config.horizon].copy() if label_config.horizon > 0 else data.copy()
            valid_labels = labels.iloc[:-label_config.horizon] if label_config.horizon > 0 else labels
            valid_data['label'] = valid_labels
            
            # Train với walk-forward
            def train_fn(train_data, **kwargs):
                X_train = train_data[features]
                y_train = train_data['label']
                return self.training_service.train_xgboost_model(
                    X_train, y_train,
                    hyperparams={
                        'max_depth': hyperparams.max_depth,
                        'learning_rate': hyperparams.learning_rate,
                        'n_estimators': hyperparams.n_estimators,
                        'subsample': hyperparams.subsample,
                        'colsample_bytree': hyperparams.colsample_bytree
                    },
                    handle_imbalance=True  # Xử lý class imbalance
                )
            
            def predict_fn(model, test_data):
                X_test = test_data[features]
                return self.training_service.predict_proba(model, X_test)
            
            def metrics_fn(predictions, test_data):
                from backend.core.metrics import MetricsCalculator
                
                if 'close' not in test_data.columns:
                    return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
                
                price_returns = test_data['close'].pct_change().fillna(0.0)
                signals = (predictions > 0.5).astype(int) * 2 - 1
                strategy_returns = price_returns * signals
                
                if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                    return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
                
                metrics_calc = MetricsCalculator()
                equity = (1 + strategy_returns).cumprod()
                metrics = metrics_calc.calculate_all_metrics(strategy_returns, equity)
                
                return {
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'profit_factor': metrics.profit_factor,
                    'total_trades': metrics.total_trades,
                    'win_rate': metrics.win_rate,
                    'total_return': metrics.total_return
                }
            
            # Run walk-forward validation
            results = self.walkforward_engine.run_validation(
                data=valid_data,
                model_train_fn=train_fn,
                model_predict_fn=predict_fn,
                metrics_fn=metrics_fn,
                date_column=timestamp_col
            )
            
            metrics = results.aggregated_metrics
            
            # Train model cuối cùng với toàn bộ data để save
            final_model = self.training_service.train_xgboost_model(
                valid_data[features],
                valid_data['label'],
                hyperparams={
                    'max_depth': hyperparams.max_depth,
                    'learning_rate': hyperparams.learning_rate,
                    'n_estimators': hyperparams.n_estimators,
                    'subsample': hyperparams.subsample,
                    'colsample_bytree': hyperparams.colsample_bytree
                }
            )
            
            return ModelCandidate(
                candidate_id=candidate_id,
                features=features,
                label_config=label_config,
                hyperparams=hyperparams,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error training candidate {candidate_id}: {e}")
            return ModelCandidate(
                candidate_id=candidate_id,
                features=features,
                label_config=label_config,
                hyperparams=hyperparams,
                metrics={'sharpe_ratio': -999.0, 'max_drawdown': 1.0}
            )
    
    def train_multiple_models(
        self,
        data: pd.DataFrame,
        candidates: List[Dict[str, Any]],
        price_col: str = 'close',
        timestamp_col: str = 'timestamp',
        save_models: bool = True
    ) -> List[ModelCandidate]:
        """
        Train nhiều models song song
        
        Args:
            data: Dữ liệu
            candidates: List các candidates, mỗi candidate là dict chứa:
                - features: List[str]
                - label_config: LabelConfiguration
                - hyperparams: HyperparameterSet
            price_col: Tên cột giá
            timestamp_col: Tên cột timestamp
            save_models: Có lưu models không
            
        Returns:
            List ModelCandidate đã train
        """
        self.logger.info(f"Training {len(candidates)} models in parallel with {self.n_workers} workers...")
        
        trained_candidates = []
        
        # Với workload CPU-bound (XGBoost đã multithread nội bộ),
        # chạy tuần tự ở mức candidate để tránh oversubscription CPU và contention.
        for i, candidate in enumerate(candidates):
            candidate_id = i
            try:
                candidate = self.train_single_model(
                    candidate_id=i,
                    data=data,
                    features=candidate['features'],
                    label_config=candidate['label_config'],
                    hyperparams=candidate['hyperparams'],
                    price_col=price_col,
                    timestamp_col=timestamp_col
                )
                trained_candidates.append(candidate)

                # Save model nếu cần
                if save_models and candidate.metrics.get('sharpe_ratio', -999) > -100:
                    try:
                        # Train lại model cuối cùng để save
                        from backend.optimization.label_optimizer import LabelOptimizer
                        label_optimizer = LabelOptimizer(self.walkforward_engine, None)

                        use_dynamic = candidate.label_config.threshold < 0
                        from backend.config import config
                        default_atr_mult = config.label.default_atr_multiplier if hasattr(config.label, 'default_atr_multiplier') else 0.5
                        atr_multiplier = abs(candidate.label_config.threshold) if use_dynamic else default_atr_mult
                        threshold = candidate.label_config.threshold if not use_dynamic else 0.0

                        atr_values = None
                        if use_dynamic and all(col in data.columns for col in ['high', 'low', 'close']):
                            from backend.data.feature_engineering import calculate_atr
                            atr_values = calculate_atr(data, 14)

                        labels = label_optimizer.create_labels(
                            data=data,
                            price_col=price_col,
                            horizon=candidate.label_config.horizon,
                            threshold=threshold,
                            use_dynamic_threshold=use_dynamic,
                            atr_values=atr_values,
                            atr_multiplier=atr_multiplier
                        )

                        valid_data = data.iloc[:-candidate.label_config.horizon].copy() if candidate.label_config.horizon > 0 else data.copy()
                        valid_labels = labels.iloc[:-candidate.label_config.horizon] if candidate.label_config.horizon > 0 else labels
                        valid_data['label'] = valid_labels

                        final_model = self.training_service.train_xgboost_model(
                            valid_data[candidate.features],
                            valid_data['label'],
                            hyperparams={
                                'max_depth': candidate.hyperparams.max_depth,
                                'learning_rate': candidate.hyperparams.learning_rate,
                                'n_estimators': candidate.hyperparams.n_estimators,
                                'subsample': candidate.hyperparams.subsample,
                                'colsample_bytree': candidate.hyperparams.colsample_bytree
                            },
                            handle_imbalance=True  # Xử lý class imbalance
                        )

                        model_path = self.model_persistence.save_model(
                            model=final_model,
                            run_id=candidate.candidate_id,
                            features=candidate.features,
                            metrics=candidate.metrics,
                            label_config={
                                'horizon': candidate.label_config.horizon,
                                'threshold': candidate.label_config.threshold
                            },
                            hyperparams={
                                'max_depth': candidate.hyperparams.max_depth,
                                'learning_rate': candidate.hyperparams.learning_rate,
                                'n_estimators': candidate.hyperparams.n_estimators,
                                'subsample': candidate.hyperparams.subsample,
                                'colsample_bytree': candidate.hyperparams.colsample_bytree
                            }
                        )
                        candidate.model_path = model_path
                        self.logger.info(f"Saved model for candidate {candidate.candidate_id}: {model_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not save model for candidate {candidate.candidate_id}: {e}")

                self.logger.info(
                    f"Candidate {candidate_id} completed: "
                    f"Sharpe={candidate.metrics.get('sharpe_ratio', 0.0):.4f}"
                )
            except Exception as e:
                self.logger.error(f"Error getting result for candidate {candidate_id}: {e}")
        
        # Sắp xếp theo Sharpe ratio
        trained_candidates.sort(
            key=lambda x: x.metrics.get('sharpe_ratio', -999.0),
            reverse=True
        )
        
        return trained_candidates
    
    def select_best_model(
        self,
        candidates: List[ModelCandidate],
        primary_metric: str = 'sharpe_ratio',
        min_trades: int = 10
    ) -> Optional[ModelCandidate]:
        """
        Chọn model tốt nhất từ các candidates
        
        Args:
            candidates: List các ModelCandidate
            primary_metric: Metric chính để chọn (mặc định: sharpe_ratio)
            min_trades: Số trades tối thiểu để chấp nhận model
            
        Returns:
            ModelCandidate tốt nhất hoặc None
        """
        if not candidates:
            return None
        
        # Lọc các candidates hợp lệ
        valid_candidates = [
            c for c in candidates
            if c.metrics.get('total_trades', 0) >= min_trades
            and c.metrics.get('sharpe_ratio', -999) > -100
        ]
        
        if not valid_candidates:
            self.logger.warning("No valid candidates found")
            return None
        
        # Chọn theo primary metric
        best_candidate = max(
            valid_candidates,
            key=lambda x: x.metrics.get(primary_metric, -999.0)
        )
        
        return best_candidate
