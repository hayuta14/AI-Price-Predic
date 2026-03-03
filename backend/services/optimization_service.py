"""
Optimization service.

Coordinates all optimization modules and executes the full optimization pipeline.
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
from backend.data.feature_engineering import create_all_features


class OptimizationService:
    """Optimization service."""

    def __init__(
        self,
        config: SystemConfig,
        db_session
    ):
        self.config = config
        self.db_session = db_session

        self.walkforward_engine = WalkForwardEngine(config.walkforward)
        self.training_service = TrainingService(self.walkforward_engine)

        self.feature_optimizer = FeatureOptimizer(self.walkforward_engine)
        self.label_optimizer = LabelOptimizer(self.walkforward_engine, config.label)
        self.risk_optimizer = RiskOptimizer(self.walkforward_engine)
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            self.walkforward_engine,
            config.xgboost
        )

        self.regime_analyzer = RegimeAnalyzer()
        self.montecarlo_simulator = MonteCarloSimulator(config.montecarlo)
        self.prediction_analyzer = PredictionAnalyzer()

        self.model_persistence = ModelPersistence(models_dir=config.results_path / "models")
        self.multi_model_trainer = MultiModelTrainer(
            walkforward_engine=self.walkforward_engine,
            training_service=self.training_service,
            model_persistence=self.model_persistence,
            n_workers=None
        )
        self.candidate_generator = CandidateGenerator(random_seed=config.random_seed)

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
        print("=" * 80)
        print("Starting full optimization pipeline")
        print("=" * 80)

        results = {}

        print("\n[1/4] Optimizing label configuration...")
        label_config = self.label_optimizer.optimize_labels(
            data=data,
            features=available_features[:10] if len(available_features) >= 10 else available_features,
            price_col=price_col,
            model_train_fn=self.training_service.train_xgboost_model,
            model_predict_fn=self.training_service.predict_proba
        )
        results['label_config'] = label_config

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

        print("\n[2/4] Optimizing feature set...")
        feature_set = self.feature_optimizer.optimize_features(
            data=valid_data_for_features,
            available_features=available_features,
            target_col='label',
            model_train_fn=self.training_service.train_xgboost_model,
            model_predict_fn=self.training_service.predict_proba
        )
        results['feature_set'] = feature_set

        if len(feature_set.features) < 3:
            results['error'] = f"Insufficient features: only {len(feature_set.features)} features selected"
            results['should_continue'] = False
            results['final_metrics'] = {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
            return results

        if len(feature_set.features) == 0:
            results['error'] = "No features selected"
            results['should_continue'] = False
            results['final_metrics'] = {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}
            return results

        print("\n[3/4] Optimizing XGBoost hyperparameters...")
        valid_data = valid_data_for_features.copy()

        if len(feature_set.features) == 0:
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

        print("\n[4/4] Optimizing risk parameters...")
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
            },
            feature_engineering_fn=create_all_features
        )

        if 'high' in data.columns and 'low' in data.columns:
            atr_values = (data['high'] - data['low']).rolling(window=14).mean()
        else:
            atr_values = data[price_col] * 0.01

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

        print("\n[5/6] Analyzing prediction distribution...")
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
            },
            feature_engineering_fn=create_all_features
        )

        pred_analysis = self.prediction_analyzer.analyze(
            predictions=predictions_for_analysis.values,
            labels=valid_data['label'].values if 'label' in valid_data.columns else None
        )
        results['prediction_analysis'] = pred_analysis

        print("\n[5.1/7] Evaluating walk-forward AUC...")
        if 'label' in valid_data.columns:
            fold_predictions = []
            wf_folds = self.walkforward_engine.generate_folds(valid_data, timestamp_col)
            for fold in wf_folds:
                y_true = fold.test_data['label']
                y_pred = predictions_for_analysis.reindex(fold.test_data.index)
                fold_predictions.append((y_true.values, y_pred.values))

            auc_results = self.auc_evaluator.evaluate_walkforward(fold_predictions)
            self.auc_evaluator.print_evaluation(auc_results)
            results['auc_evaluation'] = auc_results

            if not self.auc_evaluator.should_continue_backtest(auc_results):
                results['should_continue'] = False
                results['final_metrics'] = {'sharpe_ratio': 0.0, 'max_drawdown': 1.0, 'auc': auc_results.get('auc', 0.5)}
                return results
        else:
            results['auc_evaluation'] = {'auc': 0.5, 'is_random': True}

        results['should_continue'] = True

        print("\n[6/7] Running final backtest...")
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

        def convert_to_python_type(value):
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
        trained_candidates = self.multi_model_trainer.train_multiple_models(
            data=data,
            candidates=self.candidate_generator.generate_candidates_from_config(
                available_features=available_features,
                label_config_options={
                    'horizons': self.config.label.horizons,
                    'thresholds': self.config.label.thresholds,
                    'use_dynamic': self.config.label.use_dynamic_threshold,
                    'atr_multipliers': self.config.label.atr_multiplier_range
                },
                hyperparam_options={
                    'max_depths': list(range(self.config.xgboost.max_depth_range[0], self.config.xgboost.max_depth_range[1] + 1)),
                    'learning_rates': [0.01, 0.05, 0.1, 0.15],
                    'n_estimators_list': [100, 200, 300],
                    'subsamples': [0.7, 0.8, 0.9],
                    'colsample_bytrees': [0.7, 0.8, 0.9]
                },
                n_candidates=n_candidates,
                max_features_per_candidate=min(20, len(available_features))
            ),
            price_col=price_col,
            timestamp_col=timestamp_col,
            save_models=True
        )

        best_candidate = self.multi_model_trainer.select_best_model(
            trained_candidates,
            primary_metric='sharpe_ratio',
            min_trades=10
        )

        if best_candidate is None:
            return {'error': 'No valid model found'}

        return {
            'best_candidate': best_candidate,
            'all_candidates': trained_candidates,
            'top_5': trained_candidates[:5],
            'model_path': str(best_candidate.model_path) if best_candidate.model_path else None
        }

    def _generate_walkforward_predictions(
        self,
        data: pd.DataFrame,
        features: List[str],
        target_col: str,
        hyperparams: Dict[str, Any],
        date_column: str = 'timestamp',
        feature_engineering_fn=None
    ) -> pd.Series:
        predictions = pd.Series(0.5, index=data.index, dtype=float)

        folds = self.walkforward_engine.generate_folds(data, date_column)
        for fold in folds:
            if len(fold.test_data) == 0:
                continue

            train_data = fold.train_data.copy()
            test_data = fold.test_data.copy()

            if feature_engineering_fn is not None:
                train_out = feature_engineering_fn(train_data)
                train_data = train_out[0] if isinstance(train_out, tuple) else train_out

                warmup_data = pd.concat([train_data.tail(200), test_data], axis=0)
                test_out = feature_engineering_fn(warmup_data)
                test_with_warmup = test_out[0] if isinstance(test_out, tuple) else test_out
                test_data = test_with_warmup.tail(len(test_data)).copy()

            model = self.training_service.train_xgboost_model(
                train_data[features],
                train_data[target_col],
                hyperparams
            )
            fold_pred = self.training_service.predict_proba(
                model,
                test_data[features]
            )
            predictions.loc[test_data.index] = fold_pred

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

        valid_data = data.iloc[:-label_config.horizon].copy()
        valid_data['label'] = labels.iloc[:-label_config.horizon]

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
            },
            feature_engineering_fn=create_all_features
        )

        print("\n[6.1/7] Optimizing thresholds on held-out middle split...")
        threshold_config = self.threshold_optimizer.optimize(
            predictions=predictions,
            data=valid_data,
            price_col=price_col,
            date_column=timestamp_col,
            validation_split=0.75
        )

        valid_signals = pd.Series(0, index=valid_data.index, dtype=int)
        valid_signals[predictions > threshold_config.long_threshold] = 1
        valid_signals[predictions < threshold_config.short_threshold] = -1

        if all(col in valid_data.columns for col in ['volatility_20', 'atr_14_pct', 'hl_range_ma']):
            valid_signals = self.volatility_filter.filter_signals(
                valid_signals,
                valid_data,
                volatility_col='volatility_20',
                atr_col='atr_14_pct',
                hl_range_col='hl_range_ma'
            )

        n_total = len(valid_data)
        train_split = int(n_total * 0.6)
        final_test_split = int(n_total * 0.8)

        threshold_data = valid_data.iloc[train_split:final_test_split].copy()
        final_test_data = valid_data.iloc[final_test_split:].copy()

        if len(final_test_data) == 0:
            final_test_data = threshold_data.copy()

        final_test_predictions = predictions.reindex(final_test_data.index, fill_value=0.5)
        final_test_signals = valid_signals.reindex(final_test_data.index, fill_value=0)

        from backend.config import RiskConfig, TradingConfig
        risk_cfg = RiskConfig(risk_per_trade=risk_config.risk_per_trade)
        trading_cfg = TradingConfig()

        risk_engine = RiskEngine(risk_cfg)
        risk_engine.reset(initial_equity)

        backtest_engine = BacktestEngine(risk_engine, trading_cfg, initial_equity)

        if 'high' in final_test_data.columns and 'low' in final_test_data.columns:
            atr_values = (final_test_data['high'] - final_test_data['low']).rolling(window=14).mean()
        else:
            atr_values = final_test_data[price_col] * 0.01

        backtest_results = backtest_engine.run_backtest(
            data=final_test_data,
            predictions=final_test_predictions,
            signals=final_test_signals,
            atr_values=atr_values,
            atr_multiplier=risk_config.atr_multiplier,
            reward_risk_ratio=risk_config.reward_risk_ratio,
            timestamp_col=timestamp_col,
            price_col=price_col
        )

        metrics = backtest_results.metrics.copy()
        metrics['long_threshold'] = threshold_config.long_threshold
        metrics['short_threshold'] = threshold_config.short_threshold

        return metrics
