"""
Calibration Analyzer

Kiểm tra probability calibration:
- Plot calibration curve
- Tính Brier score
- Kiểm tra xem probability có calibrated không
- Nếu không calibrated → threshold meaningless
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')


class CalibrationAnalyzer:
    """Phân tích probability calibration"""
    
    def __init__(self, n_bins: int = 10):
        """
        Khởi tạo Calibration Analyzer
        
        Args:
            n_bins: Số bins cho calibration curve
        """
        self.n_bins = n_bins
    
    def analyze(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Phân tích calibration
        
        Args:
            predictions: Model predictions (probabilities)
            labels: True labels (0/1)
            n_bins: Số bins (None = use self.n_bins)
            
        Returns:
            Dictionary với calibration metrics
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Loại bỏ NaN và Inf
        valid_mask = np.isfinite(predictions) & np.isfinite(labels)
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
        
        if len(predictions) == 0:
            return {
                'brier_score': 1.0,
                'is_calibrated': False,
                'fraction_of_positives': None,
                'mean_predicted_value': None,
                'calibration_error': 1.0
            }
        
        if n_bins is None:
            n_bins = self.n_bins
        
        # Tính calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                labels, predictions, n_bins=n_bins, strategy='uniform'
            )
        except Exception as e:
            print(f"Warning: Error calculating calibration curve: {e}")
            fraction_of_positives = np.array([0.5] * n_bins)
            mean_predicted_value = np.linspace(0, 1, n_bins)
        
        # Tính Brier score
        brier = brier_score_loss(labels, predictions)
        
        # Tính calibration error (mean absolute difference)
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Đánh giá calibration
        # Calibrated nếu calibration_error < 0.1 và Brier score hợp lý
        is_calibrated = calibration_error < 0.1 and brier < 0.25
        
        return {
            'brier_score': float(brier),
            'is_calibrated': is_calibrated,
            'calibration_error': float(calibration_error),
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'n_bins': n_bins,
            'n_samples': len(predictions)
        }
    
    def print_analysis(self, results: Dict[str, any]):
        """
        In kết quả phân tích calibration
        
        Args:
            results: Kết quả từ analyze()
        """
        brier = results.get('brier_score', 1.0)
        is_calibrated = results.get('is_calibrated', False)
        cal_error = results.get('calibration_error', 1.0)
        n_samples = results.get('n_samples', 0)
        
        print("\n" + "=" * 80)
        print("📊 PHÂN TÍCH PROBABILITY CALIBRATION")
        print("=" * 80)
        
        print(f"\n📈 Calibration Metrics:")
        print(f"   • Brier Score: {brier:.4f}")
        print(f"   • Calibration Error: {cal_error:.4f}")
        print(f"   • Samples: {n_samples}")
        
        # Interpretation
        if brier < 0.1:
            brier_interpretation = "Excellent"
        elif brier < 0.2:
            brier_interpretation = "Good"
        elif brier < 0.25:
            brier_interpretation = "Fair"
        else:
            brier_interpretation = "Poor"
        
        print(f"   • Brier Score Interpretation: {brier_interpretation}")
        
        if cal_error < 0.05:
            cal_interpretation = "Excellent"
        elif cal_error < 0.1:
            cal_interpretation = "Good"
        elif cal_error < 0.15:
            cal_interpretation = "Fair"
        else:
            cal_interpretation = "Poor"
        
        print(f"   • Calibration Error Interpretation: {cal_interpretation}")
        
        # Status
        if is_calibrated:
            print(f"\n✅ Model CALIBRATED")
            print(f"   → Probabilities có ý nghĩa")
            print(f"   → Thresholds có thể được sử dụng")
        else:
            print(f"\n⚠️  Model KHÔNG CALIBRATED")
            print(f"   → Probabilities không đáng tin cậy")
            print(f"   → Thresholds có thể meaningless")
            print(f"   → Cần calibration (Platt scaling hoặc isotonic regression)")
        
        # Calibration curve data
        frac_pos = results.get('fraction_of_positives')
        mean_pred = results.get('mean_predicted_value')
        
        if frac_pos and mean_pred:
            print(f"\n📊 Calibration Curve (sample):")
            print(f"   Predicted | Actual")
            for i in range(min(5, len(mean_pred))):
                print(f"   {mean_pred[i]:.2f}      | {frac_pos[i]:.2f}")
        
        print("=" * 80)
    
    def should_use_thresholds(self, results: Dict[str, any]) -> bool:
        """
        Kiểm tra xem có nên sử dụng thresholds không
        
        Args:
            results: Kết quả calibration analysis
            
        Returns:
            True nếu nên sử dụng thresholds
        """
        return results.get('is_calibrated', False)
