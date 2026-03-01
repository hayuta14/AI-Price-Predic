"""
AUC Evaluator

Đánh giá AUC (Area Under ROC Curve) để kiểm tra predictive power:
- AUC ≈ 0.50-0.52 → model gần random
- AUC > 0.55 → có predictive power
- Kiểm tra trước khi backtest để tránh waste time
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')


class AUCEvaluator:
    """Đánh giá AUC để kiểm tra predictive power"""
    
    def __init__(self, min_auc: float = 0.52):
        """
        Khởi tạo AUC Evaluator
        
        Args:
            min_auc: AUC tối thiểu để chấp nhận model (mặc định: 0.52)
        """
        self.min_auc = min_auc
    
    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Đánh giá AUC
        
        Args:
            predictions: Model predictions (probabilities)
            labels: True labels (0/1)
            sample_size: Sample size để tính (None = use all)
            
        Returns:
            Dictionary với AUC và các metrics khác
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Sample nếu cần
        if sample_size and len(predictions) > sample_size:
            indices = np.random.choice(len(predictions), size=sample_size, replace=False)
            predictions = predictions[indices]
            labels = labels[indices]
        
        # Loại bỏ NaN và Inf
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
        
        # Tính AUC
        try:
            auc = roc_auc_score(labels, predictions)
        except Exception as e:
            print(f"Warning: Error calculating AUC: {e}")
            auc = 0.5
        
        # Tính ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(labels, predictions)
        except Exception:
            fpr, tpr, thresholds = None, None, None
        
        # Đánh giá
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
    
    def evaluate_walkforward(
        self,
        predictions: pd.Series,
        labels: pd.Series
    ) -> Dict[str, float]:
        """
        Đánh giá AUC trên walk-forward predictions
        
        Args:
            predictions: Predictions từ walk-forward
            labels: True labels
            
        Returns:
            Dictionary với AUC metrics
        """
        # Align indices
        common_idx = predictions.index.intersection(labels.index)
        if len(common_idx) == 0:
            return {
                'auc': 0.5,
                'is_random': True,
                'has_predictive_power': False,
                'n_samples': 0
            }
        
        pred_aligned = predictions.loc[common_idx]
        labels_aligned = labels.loc[common_idx]
        
        return self.evaluate(pred_aligned.values, labels_aligned.values)
    
    def print_evaluation(self, results: Dict[str, float]):
        """
        In kết quả đánh giá AUC
        
        Args:
            results: Kết quả từ evaluate()
        """
        auc = results.get('auc', 0.5)
        is_random = results.get('is_random', True)
        has_power = results.get('has_predictive_power', False)
        n_samples = results.get('n_samples', 0)
        
        print("\n" + "=" * 80)
        print("📊 ĐÁNH GIÁ PREDICTIVE POWER (AUC)")
        print("=" * 80)
        
        print(f"\n📈 AUC Score: {auc:.4f}")
        print(f"   • Samples: {n_samples}")
        
        # Interpretation
        if auc < 0.52:
            print(f"   • Status: ❌ RANDOM (AUC < 0.52)")
            print(f"   • Interpretation: Model không có predictive power, gần như random")
        elif auc < 0.55:
            print(f"   • Status: ⚠️  WEAK (0.52 ≤ AUC < 0.55)")
            print(f"   • Interpretation: Predictive power yếu, cần cải thiện")
        elif auc < 0.60:
            print(f"   • Status: ✅ FAIR (0.55 ≤ AUC < 0.60)")
            print(f"   • Interpretation: Có predictive power, có thể tiếp tục")
        elif auc < 0.70:
            print(f"   • Status: ✅ GOOD (0.60 ≤ AUC < 0.70)")
            print(f"   • Interpretation: Predictive power tốt")
        else:
            print(f"   • Status: ✅ EXCELLENT (AUC ≥ 0.70)")
            print(f"   • Interpretation: Predictive power rất tốt")
        
        if is_random:
            print(f"\n⚠️  CẢNH BÁO: Model có AUC < {self.min_auc}")
            print(f"   → Không nên tiếp tục backtest với model này")
            print(f"   → Cần cải thiện features hoặc model")
        else:
            print(f"\n✅ Model có predictive power (AUC ≥ {self.min_auc})")
            print(f"   → Có thể tiếp tục với backtest")
        
        print("=" * 80)
    
    def should_continue_backtest(self, results: Dict[str, float]) -> bool:
        """
        Kiểm tra xem có nên tiếp tục backtest không
        
        Args:
            results: Kết quả AUC evaluation
            
        Returns:
            True nếu nên tiếp tục, False nếu không
        """
        return results.get('has_predictive_power', False)
