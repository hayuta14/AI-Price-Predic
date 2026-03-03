"""
Dịch vụ Tối ưu hóa

Điều phối tất cả các bộ tối ưu hóa, thực thi quy trình tối ưu hóa đầy đủ
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from backend.config import SystemConfig
from backend.core.walkforward_engine import WalkForwardEngine
from backend.core.backtest_engine import BacktestEngine
from backend.core.risk_engine import RiskEngine
from backend.optimization.feature_optimizer import FeatureOptimizer, FeatureSet
from backend.optimization.label_optimizer import LabelOptimizer
from backend.optimization.risk_optimizer import RiskOptimizer, RiskConfiguration
from backend.optimization.hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterSet
from backend.services.training_service import TrainingService
from backend.analysis.regime_analysis import RegimeAnalyzer
from backend.analysis.montecarlo import MonteCarloSimulator
from backend.analysis.prediction_analysis import PredictionAnalyzer
from backend.services.model_persistence import ModelPersistence
from backend.services.multi_model_trainer import MultiModelTrainer
from backend.services.candidate_generator import CandidateGenerator
from backend.core.volatility_filter import VolatilityFilter
from backend.core.adaptive_position_sizing import AdaptivePositionSizing
from backend.optimization.threshold_optimizer import ThresholdOptimizer
from backend.analysis.auc_evaluator import AUCEvaluator
from backend.analysis.feature_importance_analyzer import FeatureImportanceAnalyzer
from backend.analysis.calibration_analyzer import CalibrationAnalyzer
from backend.analysis.probability_calibrator import ProbabilityCalibrator
from backend.database.repository import ModelRunRepository, OptimizationTrialRepository


class OptimizationService:
    """
    Dịch vụ Tối ưu hóa
    
    Điều phối tất cả các mô-đun tối ưu hóa, thực thi quy trình tối ưu hóa đầy đủ
    """
    
    def __init__(
        self,
        config: SystemConfig,
        db_session
    ):
        """
        Khởi tạo dịch vụ tối ưu hóa
        
        Args:
            config: Cấu hình hệ thống
            db_session: Phiên cơ sở dữ liệu
        """
        self.config = config
        self.db_session = db_session
        
        # Khởi tạo các thành phần cốt lõi
        self.walkforward_engine = WalkForwardEngine(config.walkforward)
        self.training_service = TrainingService(self.walkforward_engine)
        
        # Khởi tạo các bộ tối ưu hóa
        self.feature_optimizer = FeatureOptimizer(self.walkforward_engine)
        self.label_optimizer = LabelOptimizer(self.walkforward_engine, config.label)
        self.risk_optimizer = RiskOptimizer(self.walkforward_engine)
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            self.walkforward_engine,
            config.xgboost
        )
        
        # Khởi tạo các bộ phân tích
        self.regime_analyzer = RegimeAnalyzer()
        self.montecarlo_simulator = MonteCarloSimulator(config.montecarlo)
        self.prediction_analyzer = PredictionAnalyzer()
        
        # Khởi tạo model persistence và multi-model trainer
        self.model_persistence = ModelPersistence(models_dir=config.results_path / "models")
        self.multi_model_trainer = MultiModelTrainer(
            walkforward_engine=self.walkforward_engine,
            training_service=self.training_service,
            model_persistence=self.model_persistence,
            n_workers=None  # Auto-detect
        )
        self.candidate_generator = CandidateGenerator(random_seed=config.random_seed)
        
        # Khởi tạo volatility filter và adaptive position sizing
        self.volatility_filter = VolatilityFilter(use_volatility_filter=True)
        self.adaptive_position_sizing = AdaptivePositionSizing(
            base_risk=config.risk.risk_per_trade,
            use_adaptive=True
        )
        self.threshold_optimizer = ThresholdOptimizer(self.walkforward_engine)
        self.auc_evaluator = AUCEvaluator(min_auc=0.52)
        self.feature_importance_analyzer = FeatureImportanceAnalyzer(dominance_threshold=0.9)
        self.calibration_analyzer = CalibrationAnalyzer(n_bins=10)
        self.probability_calibrator = ProbabilityCalibrator(method='isotonic')
        
        # Khởi tạo kho lưu trữ
        self.model_run_repo = ModelRunRepository(db_session)
        self.trial_repo = OptimizationTrialRepository(db_session)
    
    def run_full_optimization(
        self,
        data: pd.DataFrame,
        available_features: List[str],
        price_col: str = 'close',
        timestamp_col: str = 'timestamp',
        initial_equity: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Chạy quy trình tối ưu hóa đầy đủ
        
        Args:
            data: Dữ liệu giá
            available_features: Danh sách đặc trưng có sẵn
            price_col: Tên cột giá
            timestamp_col: Tên cột thời gian
            initial_equity: Vốn ban đầu
            
        Returns:
            Từ điển kết quả tối ưu hóa
        """
        print("=" * 80)
        print("Bắt đầu quy trình tối ưu hóa đầy đủ")
        print("=" * 80)
        
        results = {}
        
        # 1. Tối ưu hóa nhãn
        print("\n[1/4] Đang tối ưu hóa cấu hình nhãn...")
        label_config = self.label_optimizer.optimize_labels(
            data=data,
            features=available_features[:10] if len(available_features) >= 10 else available_features,  # Dùng một phần đặc trưng trước
            price_col=price_col,
            model_train_fn=self.training_service.train_xgboost_model,
            model_predict_fn=self.training_service.predict_proba
        )
        results['label_config'] = label_config
        print(f"Cấu hình nhãn tối ưu: horizon={label_config.horizon}, threshold={label_config.threshold}")
        
        # Tạo label từ label_config đã tối ưu để sử dụng cho feature optimization
        labels = self.label_optimizer.create_triple_barrier_labels(
            data=data,
            price_col=price_col,
            horizon=label_config.horizon,
            sl_multiplier=label_config.sl_multiplier,
            tp_multiplier=label_config.tp_multiplier,
            atr_period=14,
            min_ret=label_config.min_ret
        )
        data_with_label = data.copy()
        valid_data_for_features = data_with_label.iloc[:-label_config.horizon].copy() if label_config.horizon > 0 else data_with_label.copy()
        valid_labels_for_features = labels.iloc[:-label_config.horizon] if label_config.horizon > 0 else labels
        valid_data_for_features['label'] = valid_labels_for_features
        
        # 2. Tối ưu hóa đặc trưng
        print("\n[2/4] Đang tối ưu hóa tập đặc trưng...")
        feature_set = self.feature_optimizer.optimize_features(
            data=valid_data_for_features,
            available_features=available_features,
            target_col='label',
            model_train_fn=self.training_service.train_xgboost_model,
            model_predict_fn=self.training_service.predict_proba
        )
        results['feature_set'] = feature_set
        print(f"Tập đặc trưng tối ưu: {len(feature_set.features)} đặc trưng")
        
        # Kiểm tra số lượng features
        if len(feature_set.features) < 3:
            print(f"\n⚠️  CẢNH BÁO: Chỉ có {len(feature_set.features)} features được chọn")
            print("   → Quá ít features, không đủ để có alpha mới")
            print("   → Dừng optimization")
            results['error'] = f"Insufficient features: only {len(feature_set.features)} features selected"
            results['should_continue'] = False
            results['final_metrics'] = {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
            return results
        
        # Nếu không có feature nào được chọn
        if len(feature_set.features) == 0:
            print("\n❌ LỖI: Không có feature nào được chọn")
            print("   → Dừng optimization")
            results['error'] = "No features selected"
            results['should_continue'] = False
            results['final_metrics'] = {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
            return results
        
        # 3. Tối ưu hóa siêu tham số
        print("\n[3/4] Đang tối ưu hóa siêu tham số XGBoost...")
        # Sử dụng data_with_label đã tạo ở bước 2
        valid_data = valid_data_for_features.copy()
        
        # Kiểm tra xem có features không
        if len(feature_set.features) == 0:
            print("Cảnh báo: Không có features để tối ưu hóa hyperparameters, sử dụng giá trị mặc định...")
            from backend.optimization.hyperparameter_optimizer import HyperparameterSet
            hyperparams = HyperparameterSet(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                sharpe_ratio=0.0,
                max_drawdown=1.0
            )
        else:
            hyperparams = self.hyperparameter_optimizer.optimize(
                data=valid_data,
                features=feature_set.features,
                target_col='label'
            )
        results['hyperparams'] = hyperparams
        print(f"Siêu tham số tối ưu: max_depth={hyperparams.max_depth}, lr={hyperparams.learning_rate}")
        
        # 4. Tối ưu hóa tham số rủi ro (cần huấn luyện mô hình trước để tạo dự đoán)
        print("\n[4/4] Đang tối ưu hóa tham số rủi ro...")
        # Tạo dự đoán walk-forward để tránh lookahead bias
        predictions = self._generate_walkforward_predictions(
            data=valid_data,
            features=feature_set.features,
            target_col='label',
            hyperparams={
                'max_depth': hyperparams.max_depth,
                'learning_rate': hyperparams.learning_rate,
                'n_estimators': hyperparams.n_estimators,
                'subsample': hyperparams.subsample,
                'colsample_bytree': hyperparams.colsample_bytree
            }
        )
        
        # Tính toán ATR (đơn giản hóa)
        if 'high' in data.columns and 'low' in data.columns:
            atr_values = (data['high'] - data['low']).rolling(window=14).mean()
        else:
            atr_values = data[price_col] * 0.01  # Mặc định 1%

        # Tạo tín hiệu từ dự đoán (sẽ được optimize sau trong _run_final_backtest)
        # Tạm thời dùng default thresholds
        valid_signals = pd.Series(0, index=valid_data.index, dtype=int)
        valid_signals[predictions > 0.55] = 1
        valid_signals[predictions < 0.45] = -1
        signals = pd.Series(0, index=data.index, dtype=int)
        signals.loc[valid_signals.index] = valid_signals.values
        
        risk_config = self.risk_optimizer.optimize_risk_parameters(
            data=data,
            predictions=predictions.reindex(data.index, fill_value=0.5),
            atr_values=atr_values,
            price_col=price_col,
            timestamp_col=timestamp_col,
            signals=signals
        )
        results['risk_config'] = risk_config
        
        # 5. Phân tích phân phối dự đoán
        print("\n[5/6] Đang phân tích phân phối dự đoán...")
        predictions_for_analysis = self._generate_walkforward_predictions(
            data=valid_data,
            features=feature_set.features,
            target_col='label',
            hyperparams={
                'max_depth': hyperparams.max_depth,
                'learning_rate': hyperparams.learning_rate,
                'n_estimators': hyperparams.n_estimators,
                'subsample': hyperparams.subsample,
                'colsample_bytree': hyperparams.colsample_bytree
            }
        )
        
        # Phân tích phân phối dự đoán
        pred_analysis = self.prediction_analyzer.analyze(
            predictions=predictions_for_analysis.values,
            labels=valid_data['label'].values if 'label' in valid_data.columns else None
        )
        self.prediction_analyzer.print_analysis(pred_analysis)
        results['prediction_analysis'] = pred_analysis
        
        # 5.1. Đánh giá AUC (quan trọng - kiểm tra predictive power trước khi backtest)
        print("\n[5.1/7] Đang đánh giá AUC để kiểm tra predictive power...")
        if 'label' in valid_data.columns:
            auc_results = self.auc_evaluator.evaluate_walkforward(
                predictions=predictions_for_analysis,
                labels=valid_data['label']
            )
            self.auc_evaluator.print_evaluation(auc_results)
            results['auc_evaluation'] = auc_results
            
            # Kiểm tra xem có nên tiếp tục không
            if not self.auc_evaluator.should_continue_backtest(auc_results):
                print("\n⚠️  CẢNH BÁO: Model không có predictive power (AUC < 0.52)")
                print("   → Dừng optimization, không tiếp tục backtest")
                print("   → Cần cải thiện features hoặc model configuration")
                results['should_continue'] = False
                results['final_metrics'] = {'sharpe_ratio': 0.0, 'max_drawdown': 1.0, 'auc': auc_results.get('auc', 0.5)}
                return results
        else:
            print("⚠️  Không có labels để đánh giá AUC")
            results['auc_evaluation'] = {'auc': 0.5, 'is_random': True}
        
        results['should_continue'] = True
        
        # 5.2. Phân tích Feature Importance
        print("\n[5.2/7] Đang phân tích Feature Importance...")
        try:
            # Train model cuối cùng để phân tích importance
            final_model = self.training_service.train_xgboost_model(
                valid_data[feature_set.features],
                valid_data['label'],
                hyperparams={
                    'max_depth': hyperparams.max_depth,
                    'learning_rate': hyperparams.learning_rate,
                    'n_estimators': hyperparams.n_estimators,
                    'subsample': hyperparams.subsample,
                    'colsample_bytree': hyperparams.colsample_bytree
                },
                handle_imbalance=True
            )
            
            importance_analysis = self.feature_importance_analyzer.analyze(
                model=final_model,
                X=valid_data[feature_set.features],
                feature_names=feature_set.features
            )
            self.feature_importance_analyzer.print_analysis(importance_analysis)
            results['feature_importance'] = importance_analysis
            
            # Kiểm tra feature dominance
            if importance_analysis.get('has_issues', False):
                print("\n⚠️  CẢNH BÁO: Có feature dominance issues")
                print("   → Cần xem xét lại feature selection")
        except Exception as e:
            print(f"⚠️  Không thể phân tích feature importance: {e}")
            results['feature_importance'] = {}
        
        # 5.3. Phân tích Calibration
        print("\n[5.3/7] Đang phân tích Probability Calibration...")
        if 'label' in valid_data.columns:
            calibration_results = self.calibration_analyzer.analyze(
                predictions=predictions_for_analysis.values,
                labels=valid_data['label'].values
            )
            self.calibration_analyzer.print_analysis(calibration_results)
            results['calibration'] = calibration_results
            
            # Kiểm tra calibration
            if not self.calibration_analyzer.should_use_thresholds(calibration_results):
                print("\n⚠️  CẢNH BÁO: Model không calibrated")
                print("   → Thresholds có thể meaningless")
                print("   → Cần calibration trước khi sử dụng thresholds")
        else:
            print("⚠️  Không có labels để đánh giá calibration")
            results['calibration'] = {'is_calibrated': False}
        
        # 6. Kiểm tra ngược và đánh giá cuối cùng
        print("\n[6/7] Đang thực thi kiểm tra ngược cuối cùng...")
        final_metrics = self._run_final_backtest(
            data=data,
            features=feature_set.features,
            label_config=label_config,
            hyperparams=hyperparams,
            risk_config=risk_config,
            price_col=price_col,
            timestamp_col=timestamp_col,
            initial_equity=initial_equity
        )
        results['final_metrics'] = final_metrics
        results['threshold_config'] = {
            'long_threshold': final_metrics.get('long_threshold', 0.55),
            'short_threshold': final_metrics.get('short_threshold', 0.45)
        }
        
        # 7. Lưu vào cơ sở dữ liệu
        print("\nĐang lưu kết quả vào cơ sở dữ liệu...")
        
        # Helper function để convert numpy types sang Python types
        def convert_to_python_type(value):
            """Convert numpy types to Python native types"""
            if value is None:
                return None
            if isinstance(value, (np.integer, np.floating)):
                return float(value) if isinstance(value, np.floating) else int(value)
            if isinstance(value, (int, float)):
                return value
            return value
        
        model_run = self.model_run_repo.create(
            feature_config={'features': feature_set.features},
            label_config={
                'horizon': label_config.horizon,
                'threshold': label_config.threshold
            },
            risk_config={
                'risk_per_trade': risk_config.risk_per_trade,
                'atr_multiplier': risk_config.atr_multiplier,
                'reward_risk_ratio': risk_config.reward_risk_ratio
            },
            hyperparams={
                'max_depth': hyperparams.max_depth,
                'learning_rate': hyperparams.learning_rate,
                'n_estimators': hyperparams.n_estimators,
                'subsample': hyperparams.subsample,
                'colsample_bytree': hyperparams.colsample_bytree
            },
            sharpe_ratio=convert_to_python_type(final_metrics.get('sharpe_ratio', 0.0)),
            max_drawdown=convert_to_python_type(final_metrics.get('max_drawdown', 1.0)),
            profit_factor=convert_to_python_type(final_metrics.get('profit_factor', 0.0)),
            calmar_ratio=convert_to_python_type(final_metrics.get('calmar_ratio', 0.0)),
            total_return=convert_to_python_type(final_metrics.get('total_return', 0.0)),
            annualized_return=convert_to_python_type(final_metrics.get('annualized_return', 0.0)),
            volatility=convert_to_python_type(final_metrics.get('volatility', 0.0)),
            win_rate=convert_to_python_type(final_metrics.get('win_rate', 0.0)),
            total_trades=int(final_metrics.get('total_trades', 0))
        )
        results['run_id'] = model_run.id
        
        # 8. Lưu model cuối cùng
        print("\nĐang lưu model cuối cùng...")
        try:
            # Train model cuối cùng với toàn bộ data
            final_model = self.training_service.train_xgboost_model(
                valid_data[feature_set.features],
                valid_data['label'],
                hyperparams={
                    'max_depth': hyperparams.max_depth,
                    'learning_rate': hyperparams.learning_rate,
                    'n_estimators': hyperparams.n_estimators,
                    'subsample': hyperparams.subsample,
                    'colsample_bytree': hyperparams.colsample_bytree
                },
                handle_imbalance=True  # Xử lý class imbalance
            )
            
            model_path = self.model_persistence.save_model(
                model=final_model,
                run_id=results['run_id'],
                features=feature_set.features,
                metrics=final_metrics,
                label_config={
                    'horizon': label_config.horizon,
                    'threshold': label_config.threshold
                },
                hyperparams={
                    'max_depth': hyperparams.max_depth,
                    'learning_rate': hyperparams.learning_rate,
                    'n_estimators': hyperparams.n_estimators,
                    'subsample': hyperparams.subsample,
                    'colsample_bytree': hyperparams.colsample_bytree
                },
                risk_config={
                    'risk_per_trade': risk_config.risk_per_trade,
                    'atr_multiplier': risk_config.atr_multiplier,
                    'reward_risk_ratio': risk_config.reward_risk_ratio
                }
            )
            results['model_path'] = str(model_path)
            print(f"✅ Đã lưu model: {model_path}")
        except Exception as e:
            print(f"⚠️  Không thể lưu model: {e}")
        
        print("\n" + "=" * 80)
        print("Tối ưu hóa hoàn tất!")
        print("=" * 80)
        
        return results
    
    def run_multi_model_optimization(
        self,
        data: pd.DataFrame,
        available_features: List[str],
        n_candidates: int = 20,
        price_col: str = 'close',
        timestamp_col: str = 'timestamp',
        initial_equity: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Chạy tối ưu hóa với nhiều models song song
        
        Args:
            data: Dữ liệu giá
            available_features: Danh sách đặc trưng có sẵn
            n_candidates: Số lượng candidates để train
            price_col: Tên cột giá
            timestamp_col: Tên cột thời gian
            initial_equity: Vốn ban đầu
            
        Returns:
            Từ điển kết quả với best model
        """
        print("=" * 80)
        print(f"Bắt đầu Multi-Model Optimization với {n_candidates} candidates")
        print("=" * 80)
        
        # Generate candidates
        print("\n[1/3] Đang tạo candidates...")
        candidates = self.candidate_generator.generate_candidates_from_config(
            available_features=available_features,
            label_config_options={
                'horizons': self.config.label.horizons,
                'thresholds': self.config.label.thresholds,
                'use_dynamic': self.config.label.use_dynamic_threshold,
                'atr_multipliers': self.config.label.atr_multiplier_range
            },
            hyperparam_options={
                'max_depths': list(range(
                    self.config.xgboost.max_depth_range[0],
                    self.config.xgboost.max_depth_range[1] + 1
                )),
                'learning_rates': [0.01, 0.05, 0.1, 0.15],
                'n_estimators_list': [100, 200, 300],
                'subsamples': [0.7, 0.8, 0.9],
                'colsample_bytrees': [0.7, 0.8, 0.9]
            },
            n_candidates=n_candidates,
            max_features_per_candidate=min(20, len(available_features))
        )
        print(f"✅ Đã tạo {len(candidates)} candidates")
        
        # Train tất cả candidates song song
        print(f"\n[2/3] Đang train {len(candidates)} models song song...")
        trained_candidates = self.multi_model_trainer.train_multiple_models(
            data=data,
            candidates=candidates,
            price_col=price_col,
            timestamp_col=timestamp_col,
            save_models=True
        )
        
        # Chọn model tốt nhất
        print("\n[3/3] Đang chọn model tốt nhất...")
        best_candidate = self.multi_model_trainer.select_best_model(
            trained_candidates,
            primary_metric='sharpe_ratio',
            min_trades=10
        )
        
        if best_candidate is None:
            print("⚠️  Không tìm thấy model hợp lệ")
            return {'error': 'No valid model found'}
        
        print(f"\n🏆 Model tốt nhất:")
        print(f"   • Candidate ID: {best_candidate.candidate_id}")
        print(f"   • Sharpe Ratio: {best_candidate.metrics.get('sharpe_ratio', 0.0):.4f}")
        print(f"   • Max Drawdown: {best_candidate.metrics.get('max_drawdown', 1.0)*100:.2f}%")
        print(f"   • Profit Factor: {best_candidate.metrics.get('profit_factor', 0.0):.4f}")
        print(f"   • Total Trades: {best_candidate.metrics.get('total_trades', 0)}")
        print(f"   • Features: {len(best_candidate.features)} features")
        
        # So sánh top 5 models
        print("\n📊 Top 5 Models:")
        top_5 = trained_candidates[:5]
        for i, candidate in enumerate(top_5, 1):
            print(f"   {i}. Candidate {candidate.candidate_id}: "
                  f"Sharpe={candidate.metrics.get('sharpe_ratio', 0.0):.4f}, "
                  f"DD={candidate.metrics.get('max_drawdown', 1.0)*100:.2f}%, "
                  f"Trades={candidate.metrics.get('total_trades', 0)}")
        
        return {
            'best_candidate': best_candidate,
            'all_candidates': trained_candidates,
            'top_5': top_5,
            'model_path': str(best_candidate.model_path) if best_candidate.model_path else None
        }
    
    def _generate_walkforward_predictions(
        self,
        data: pd.DataFrame,
        features: List[str],
        target_col: str,
        hyperparams: Dict[str, Any],
        date_column: str = 'timestamp'
    ) -> pd.Series:
        """Tạo dự đoán out-of-sample theo từng fold walk-forward."""
        predictions = pd.Series(0.5, index=data.index, dtype=float)

        folds = self.walkforward_engine.generate_folds(data, date_column)
        for fold in folds:
            if len(fold.test_data) == 0:
                continue

            model = self.training_service.train_xgboost_model(
                fold.train_data[features],
                fold.train_data[target_col],
                hyperparams
            )
            fold_pred = self.training_service.predict_proba(
                model,
                fold.test_data[features]
            )
            predictions.loc[fold.test_data.index] = fold_pred

        return predictions
    
    def _run_final_backtest(
        self,
        data: pd.DataFrame,
        features: List[str],
        label_config,
        hyperparams,
        risk_config,
        price_col: str,
        timestamp_col: str,
        initial_equity: float
    ) -> Dict[str, float]:
        """
        Chạy kiểm tra ngược cuối cùng
        
        Returns:
            Từ điển chỉ số hiệu suất
        """
        # Tạo nhãn
        from backend.optimization.label_optimizer import LabelOptimizer
        label_opt = LabelOptimizer(self.walkforward_engine, self.config.label)
        
        labels = label_opt.create_triple_barrier_labels(
            data=data,
            price_col=price_col,
            horizon=label_config.horizon,
            sl_multiplier=label_config.sl_multiplier,
            tp_multiplier=label_config.tp_multiplier,
            atr_period=14,
            min_ret=label_config.min_ret
        )
        
        # Chuẩn bị dữ liệu
        valid_data = data.iloc[:-label_config.horizon].copy()
        valid_data['label'] = labels.iloc[:-label_config.horizon]
        
        # Tạo dự đoán walk-forward trên tập dữ liệu hợp lệ
        predictions = self._generate_walkforward_predictions(
            data=valid_data,
            features=features,
            target_col='label',
            hyperparams={
                'max_depth': hyperparams.max_depth,
                'learning_rate': hyperparams.learning_rate,
                'n_estimators': hyperparams.n_estimators,
                'subsample': hyperparams.subsample,
                'colsample_bytree': hyperparams.colsample_bytree
            }
        )
        
        # 6.1. Optimize probability thresholds trên VALIDATION SET (không dùng test set)
        print("\n[6.1/7] Đang optimize probability thresholds trên VALIDATION SET...")
        print("   ⚠️  Lưu ý: Thresholds chỉ được optimize trên validation, không dùng test set")
        threshold_config = self.threshold_optimizer.optimize(
            predictions=predictions,
            data=valid_data,
            price_col=price_col,
            date_column=timestamp_col,
            validation_split=0.7  # 70% validation, 30% test
        )
        print(f"✅ Threshold tối ưu (validation): long={threshold_config.long_threshold:.3f}, "
              f"short={threshold_config.short_threshold:.3f}, "
              f"Validation Sharpe={threshold_config.sharpe_ratio:.4f}")
        if threshold_config.test_sharpe_ratio > 0:
            print(f"   • Test Sharpe (out-of-sample): {threshold_config.test_sharpe_ratio:.4f}")
        
        # Tạo tín hiệu giao dịch từ dự đoán với optimized thresholds
        valid_signals = pd.Series(0, index=valid_data.index, dtype=int)
        valid_signals[predictions > threshold_config.long_threshold] = 1
        valid_signals[predictions < threshold_config.short_threshold] = -1
        
        # 6.2. Apply volatility filter
        print("\n[6.2/7] Đang áp dụng volatility filter...")
        if all(col in valid_data.columns for col in ['volatility_20', 'atr_14_pct', 'hl_range_ma']):
            valid_signals = self.volatility_filter.filter_signals(
                valid_signals,
                valid_data,
                volatility_col='volatility_20',
                atr_col='atr_14_pct',
                hl_range_col='hl_range_ma'
            )
            filtered_count = (valid_signals != 0).sum()
            print(f"✅ Đã filter signals: {filtered_count} signals sau filter")
        
        # Chỉ chạy final backtest trên TEST SET để tránh optimization bias
        n_total = len(valid_data)
        n_validation = int(n_total * 0.7)
        test_data = valid_data.iloc[n_validation:].copy()

        # Fallback nếu test quá nhỏ
        if len(test_data) == 0:
            test_data = valid_data.copy()

        test_predictions = predictions.reindex(test_data.index, fill_value=0.5)
        test_signals = valid_signals.reindex(test_data.index, fill_value=0)

        # Chạy kiểm tra ngược
        from backend.core.backtest_engine import BacktestEngine
        from backend.core.risk_engine import RiskEngine
        from backend.config import RiskConfig, TradingConfig
        
        risk_cfg = RiskConfig(risk_per_trade=risk_config.risk_per_trade)
        trading_cfg = TradingConfig()
        
        risk_engine = RiskEngine(risk_cfg)
        risk_engine.reset(initial_equity)
        
        backtest_engine = BacktestEngine(risk_engine, trading_cfg, initial_equity)
        
        # Tính toán ATR
        if 'high' in test_data.columns and 'low' in test_data.columns:
            atr_values = (test_data['high'] - test_data['low']).rolling(window=14).mean()
        else:
            atr_values = test_data[price_col] * 0.01
        
        backtest_results = backtest_engine.run_backtest(
            data=test_data,
            predictions=test_predictions,
            signals=test_signals,
            atr_values=atr_values,
            atr_multiplier=risk_config.atr_multiplier,
            reward_risk_ratio=risk_config.reward_risk_ratio,
            timestamp_col=timestamp_col,
            price_col=price_col
        )
        
        # Thêm threshold config vào metrics
        metrics = backtest_results.metrics.copy()
        metrics['long_threshold'] = threshold_config.long_threshold
        metrics['short_threshold'] = threshold_config.short_threshold
        
        return metrics
