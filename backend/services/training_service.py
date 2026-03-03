"""
Training service.

Wraps model training logic and provides a unified training interface.
"""
from typing import List, Optional, Callable
import pandas as pd
import numpy as np
import xgboost as xgb

from backend.core.walkforward_engine import WalkForwardEngine, WalkForwardResults


class TrainingService:
    """
    Training service.

    Provides a unified interface for model training and prediction.
    """

    def __init__(self, walkforward_engine: WalkForwardEngine):
        """Initialize the training service."""
        self.walkforward_engine = walkforward_engine

    def train_xgboost_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparams: Optional[dict] = None,
        handle_imbalance: bool = True
    ) -> xgb.XGBClassifier:
        """Train an XGBoost model."""
        if hyperparams is None:
            hyperparams = {
                'max_depth': 3,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }

        # Binary handling for triple-barrier labels: long (1) vs not-long (-1/0)
        scale_pos_weight = 1.0
        y_train_binary = (y_train == 1).astype(int)
        if handle_imbalance:
            positive_count = y_train_binary.sum()
            negative_count = (y_train_binary == 0).sum()
            if positive_count > 0 and negative_count > 0:
                scale_pos_weight = negative_count / positive_count
                scale_pos_weight = max(0.1, min(10.0, scale_pos_weight))

        model = xgb.XGBClassifier(
            max_depth=hyperparams.get('max_depth', 5),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            n_estimators=hyperparams.get('n_estimators', 100),
            subsample=hyperparams.get('subsample', 0.8),
            colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

        model.fit(X_train, y_train)
        return model

    def predict_proba(
        self,
        model: xgb.XGBClassifier,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """Predict positive-class probabilities."""
        return model.predict_proba(X_test)[:, 1]

    def train_with_walkforward(
        self,
        data: pd.DataFrame,
        features: List[str],
        target_col: str,
        hyperparams: Optional[dict] = None,
        date_column: str = 'timestamp',
        feature_engineering_fn: Optional[Callable] = None
    ) -> WalkForwardResults:
        """Train with walk-forward validation."""
        def train_fn(train_data, **kwargs):
            X_train = train_data[features]
            y_train = train_data[target_col]
            return self.train_xgboost_model(X_train, y_train, hyperparams, handle_imbalance=True)

        def predict_fn(model, test_data):
            X_test = test_data[features]
            return self.predict_proba(model, X_test)

        def metrics_fn(predictions, test_data):
            if 'close' not in test_data.columns:
                return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}

            price_returns = test_data['close'].pct_change().fillna(0.0)
            signals = (predictions > 0.5).astype(int) * 2 - 1
            strategy_returns = price_returns * signals

            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}

            from backend.core.metrics import MetricsCalculator
            metrics_calc = MetricsCalculator()
            equity = (1 + strategy_returns).cumprod()
            metrics = metrics_calc.calculate_all_metrics(strategy_returns, equity)

            return {
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate
            }

        results = self.walkforward_engine.run_validation(
            data=data,
            model_train_fn=train_fn,
            model_predict_fn=predict_fn,
            metrics_fn=metrics_fn,
            date_column=date_column,
            feature_engineering_fn=feature_engineering_fn
        )

        return results
