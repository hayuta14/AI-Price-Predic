"""
AUC evaluator.

Evaluates AUC (Area Under ROC Curve) to check predictive power.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')


class AUCEvaluator:
    """Evaluate AUC to assess predictive power."""

    def __init__(self, min_auc: float = 0.52):
        """Initialize the AUC evaluator."""
        self.min_auc = min_auc

    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate AUC on one prediction/label set."""
        predictions = np.array(predictions)
        labels = np.array(labels)

        if sample_size and len(predictions) > sample_size:
            indices = np.random.choice(len(predictions), size=sample_size, replace=False)
            predictions = predictions[indices]
            labels = labels[indices]

        valid_mask = np.isfinite(predictions) & np.isfinite(labels)
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]

        if len(predictions) == 0 or len(np.unique(labels)) < 2:
            return {
                'auc': 0.5,
                'is_random': True,
                'has_predictive_power': False,
                'n_samples': 0
            }

        try:
            auc = roc_auc_score(labels, predictions)
        except Exception as e:
            print(f"Warning: Error calculating AUC: {e}")
            auc = 0.5

        try:
            fpr, tpr, thresholds = roc_curve(labels, predictions)
        except Exception:
            fpr, tpr, thresholds = None, None, None

        is_random = auc < self.min_auc
        has_predictive_power = auc >= self.min_auc

        return {
            'auc': float(auc),
            'is_random': is_random,
            'has_predictive_power': has_predictive_power,
            'n_samples': len(predictions),
            'fpr': fpr.tolist() if fpr is not None else None,
            'tpr': tpr.tolist() if tpr is not None else None,
            'thresholds': thresholds.tolist() if thresholds is not None else None
        }

    def evaluate_walkforward(self, fold_predictions: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """
        Evaluate AUC per walk-forward fold.

        Args:
            fold_predictions: list of (y_true, y_pred_proba) tuples, one per fold.

        Returns:
            {'per_fold_auc': [...], 'mean_auc': float, 'std_auc': float, ...}
        """
        fold_aucs: List[float] = []

        for y_true, y_pred in fold_predictions:
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
            y_true_arr = y_true_arr[valid_mask]
            y_pred_arr = y_pred_arr[valid_mask]
            if len(y_true_arr) == 0 or len(np.unique(y_true_arr)) < 2:
                continue
            try:
                fold_aucs.append(float(roc_auc_score(y_true_arr, y_pred_arr)))
            except Exception:
                continue

        if not fold_aucs:
            return {
                'per_fold_auc': [],
                'mean_auc': 0.5,
                'std_auc': 0.0,
                'auc': 0.5,
                'is_random': True,
                'has_predictive_power': False,
                'n_folds': 0,
                'n_samples': 0
            }

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))

        return {
            'per_fold_auc': fold_aucs,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'auc': mean_auc,
            'is_random': mean_auc < self.min_auc,
            'has_predictive_power': mean_auc >= self.min_auc,
            'n_folds': len(fold_aucs),
            'n_samples': len(fold_aucs)
        }

    def print_evaluation(self, results: Dict[str, float]):
        """Print AUC evaluation results."""
        auc = results.get('mean_auc', results.get('auc', 0.5))
        is_random = results.get('is_random', True)
        n_folds = results.get('n_folds', 0)

        print("\n" + "=" * 80)
        print("📊 PREDICTIVE POWER EVALUATION (AUC)")
        print("=" * 80)

        print(f"\n📈 Mean AUC Score: {auc:.4f}")
        if n_folds:
            print(f"   • Folds: {n_folds}")
            print(f"   • Std AUC: {results.get('std_auc', 0.0):.4f}")
            print(f"   • Per-fold AUC: {results.get('per_fold_auc', [])}")

        if auc < 0.52:
            print("   • Status: ❌ RANDOM (AUC < 0.52)")
        elif auc < 0.55:
            print("   • Status: ⚠️  WEAK (0.52 ≤ AUC < 0.55)")
        elif auc < 0.60:
            print("   • Status: ✅ FAIR (0.55 ≤ AUC < 0.60)")
        elif auc < 0.70:
            print("   • Status: ✅ GOOD (0.60 ≤ AUC < 0.70)")
        else:
            print("   • Status: ✅ EXCELLENT (AUC ≥ 0.70)")

        if is_random:
            print(f"\n⚠️  WARNING: Model AUC < {self.min_auc}")
        else:
            print(f"\n✅ Model has predictive power (AUC ≥ {self.min_auc})")

        print("=" * 80)

    def should_continue_backtest(self, results: Dict[str, float]) -> bool:
        """Return whether backtest should continue based on AUC quality."""
        return results.get('has_predictive_power', False)
