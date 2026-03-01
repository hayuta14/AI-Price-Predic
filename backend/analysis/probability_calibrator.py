"""
Probability Calibrator

Calibrate model probabilities trước khi optimize thresholds:
- Platt Scaling (Logistic Regression)
- Isotonic Regression
- Kiểm tra AUC tăng và calibration error giảm sau calibration
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class ProbabilityCalibrator:
    """Calibrate model probabilities"""
    
    def __init__(self, method: str = 'isotonic'):
        """
        Khởi tạo Probability Calibrator
        
        Args:
            method: 'platt' (Platt scaling) hoặc 'isotonic' (Isotonic regression)
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def calibrate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        method: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calibrate probabilities
        
        Args:
            predictions: Raw model predictions (probabilities)
            labels: True labels (0/1)
            method: Calibration method ('platt' hoặc 'isotonic')
            
        Returns:
            Tuple of (calibrated_predictions, metrics_dict)
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Loại bỏ NaN và Inf
        valid_mask = np.isfinite(predictions) & np.isfinite(labels)
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
        
        if len(predictions) == 0 or len(np.unique(labels)) < 2:
            return predictions, {
                'auc_before': 0.5,
                'auc_after': 0.5,
                'auc_improvement': 0.0,
                'cal_error_before': 1.0,
                'cal_error_after': 1.0,
                'cal_error_reduction': 0.0,
                'brier_before': 1.0,
                'brier_after': 1.0,
                'brier_improvement': 0.0
            }
        
        method = method or self.method
        
        # Tính metrics trước calibration
        try:
            auc_before = roc_auc_score(labels, predictions)
        except:
            auc_before = 0.5
        
        try:
            brier_before = brier_score_loss(labels, predictions)
        except:
            brier_before = 1.0
        
        try:
            frac_pos, mean_pred = calibration_curve(labels, predictions, n_bins=10, strategy='uniform')
            cal_error_before = np.mean(np.abs(frac_pos - mean_pred))
        except:
            cal_error_before = 1.0
        
        # Calibrate
        calibrated_predictions = predictions.copy()
        
        try:
            if method == 'platt':
                # Platt Scaling (Logistic Regression)
                lr = LogisticRegression()
                lr.fit(predictions.reshape(-1, 1), labels)
                calibrated_predictions = lr.predict_proba(predictions.reshape(-1, 1))[:, 1]
                self.calibrator = lr
                
            elif method == 'isotonic':
                # Isotonic Regression
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                calibrated_predictions = iso_reg.fit_transform(predictions, labels)
                self.calibrator = iso_reg
                
            else:
                raise ValueError(f"Unknown calibration method: {method}")
            
            self.is_fitted = True
            
            # Giới hạn trong [0, 1]
            calibrated_predictions = np.clip(calibrated_predictions, 0.0, 1.0)
            
        except Exception as e:
            print(f"Warning: Error during calibration: {e}")
            calibrated_predictions = predictions.copy()
        
        # Tính metrics sau calibration
        try:
            auc_after = roc_auc_score(labels, calibrated_predictions)
        except:
            auc_after = auc_before
        
        try:
            brier_after = brier_score_loss(labels, calibrated_predictions)
        except:
            brier_after = brier_before
        
        try:
            frac_pos, mean_pred = calibration_curve(labels, calibrated_predictions, n_bins=10, strategy='uniform')
            cal_error_after = np.mean(np.abs(frac_pos - mean_pred))
        except:
            cal_error_after = cal_error_before
        
        # Tính improvements
        auc_improvement = auc_after - auc_before
        cal_error_reduction = cal_error_before - cal_error_after
        brier_improvement = brier_before - brier_after
        
        metrics = {
            'auc_before': float(auc_before),
            'auc_after': float(auc_after),
            'auc_improvement': float(auc_improvement),
            'cal_error_before': float(cal_error_before),
            'cal_error_after': float(cal_error_after),
            'cal_error_reduction': float(cal_error_reduction),
            'brier_before': float(brier_before),
            'brier_after': float(brier_after),
            'brier_improvement': float(brier_improvement),
            'method': method
        }
        
        return calibrated_predictions, metrics
    
    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Transform new predictions using fitted calibrator
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Calibrated predictions
        """
        if not self.is_fitted or self.calibrator is None:
            return predictions
        
        predictions = np.array(predictions)
        valid_mask = np.isfinite(predictions)
        predictions_valid = predictions[valid_mask]
        
        try:
            if isinstance(self.calibrator, LogisticRegression):
                calibrated = self.calibrator.predict_proba(predictions_valid.reshape(-1, 1))[:, 1]
            elif isinstance(self.calibrator, IsotonicRegression):
                calibrated = self.calibrator.transform(predictions_valid)
            else:
                calibrated = predictions_valid
            
            # Giới hạn trong [0, 1]
            calibrated = np.clip(calibrated, 0.0, 1.0)
            
            # Map lại về original shape
            result = predictions.copy()
            result[valid_mask] = calibrated
            result[~valid_mask] = 0.5  # Default cho NaN
            
            return result
            
        except Exception as e:
            print(f"Warning: Error transforming predictions: {e}")
            return predictions
    
    def print_calibration_results(self, metrics: Dict[str, float]):
        """
        In kết quả calibration
        
        Args:
            metrics: Metrics từ calibrate()
        """
        print("\n" + "=" * 80)
        print("📊 KẾT QUẢ CALIBRATION")
        print("=" * 80)
        
        method = metrics.get('method', 'unknown')
        print(f"\n🔧 Method: {method.upper()}")
        
        print(f"\n📈 Metrics Trước Calibration:")
        print(f"   • AUC: {metrics['auc_before']:.4f}")
        print(f"   • Calibration Error: {metrics['cal_error_before']:.4f}")
        print(f"   • Brier Score: {metrics['brier_before']:.4f}")
        
        print(f"\n📈 Metrics Sau Calibration:")
        print(f"   • AUC: {metrics['auc_after']:.4f}")
        print(f"   • Calibration Error: {metrics['cal_error_after']:.4f}")
        print(f"   • Brier Score: {metrics['brier_after']:.4f}")
        
        print(f"\n📊 Improvements:")
        auc_imp = metrics['auc_improvement']
        cal_red = metrics['cal_error_reduction']
        brier_imp = metrics['brier_improvement']
        
        print(f"   • AUC Improvement: {auc_imp:+.4f} {'✅' if auc_imp > 0 else '❌'}")
        print(f"   • Calibration Error Reduction: {cal_red:+.4f} {'✅' if cal_red > 0 else '❌'}")
        print(f"   • Brier Score Improvement: {brier_imp:+.4f} {'✅' if brier_imp > 0 else '❌'}")
        
        # Đánh giá
        if auc_imp > 0.01 and cal_red > 0.05:
            print(f"\n✅ Model có hy vọng sau calibration!")
            print(f"   → AUC tăng và Calibration Error giảm mạnh")
            print(f"   → Có thể tiếp tục với threshold optimization")
        elif auc_imp > 0 or cal_red > 0:
            print(f"\n⚠️  Calibration có cải thiện nhưng không đáng kể")
        else:
            print(f"\n❌ Calibration không cải thiện model")
            print(f"   → Cần xem xét lại features hoặc model")
        
        print("=" * 80)
    
    def should_continue(self, metrics: Dict[str, float]) -> bool:
        """
        Kiểm tra xem có nên tiếp tục sau calibration không
        
        Args:
            metrics: Metrics từ calibrate()
            
        Returns:
            True nếu nên tiếp tục
        """
        auc_imp = metrics.get('auc_improvement', 0.0)
        cal_red = metrics.get('cal_error_reduction', 0.0)
        
        # Tiếp tục nếu AUC tăng hoặc calibration error giảm đáng kể
        return auc_imp > 0.005 or cal_red > 0.03
