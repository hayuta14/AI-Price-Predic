"""
Prediction Distribution Analysis

Kiểm tra phân phối dự đoán của mô hình:
- Probability output có bị collapse về 0.5 không?
- Có bị class imbalance không?
- Có bị always predict 0 không?
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PredictionAnalysis:
    """Kết quả phân tích dự đoán"""
    mean_probability: float
    std_probability: float
    median_probability: float
    collapse_to_05: bool  # Có bị collapse về 0.5 không
    class_imbalance: bool  # Có bị class imbalance không
    positive_ratio: float  # Tỷ lệ dự đoán positive
    negative_ratio: float  # Tỷ lệ dự đoán negative
    always_predict_zero: bool  # Có luôn dự đoán 0 không
    always_predict_one: bool  # Có luôn dự đoán 1 không
    prediction_distribution: Dict[str, float]  # Phân phối dự đoán
    warnings: List[str]  # Cảnh báo
    collapse_threshold: float = 0.1  # Ngưỡng để xác định collapse


class PredictionAnalyzer:
    """Phân tích phân phối dự đoán"""
    
    def __init__(self, collapse_threshold: float = 0.1):
        """
        Khởi tạo analyzer
        
        Args:
            collapse_threshold: Ngưỡng để xác định collapse (std < threshold)
        """
        self.collapse_threshold = collapse_threshold
    
    def analyze(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        probability_threshold: float = 0.5
    ) -> PredictionAnalysis:
        """
        Phân tích phân phối dự đoán
        
        Args:
            predictions: Mảng dự đoán (probability)
            labels: Nhãn thực tế (optional, để kiểm tra class imbalance)
            probability_threshold: Ngưỡng để chuyển probability thành class
            
        Returns:
            PredictionAnalysis object
        """
        predictions = np.array(predictions)
        warnings = []
        
        # 1. Kiểm tra collapse về 0.5
        mean_prob = float(np.mean(predictions))
        std_prob = float(np.std(predictions))
        median_prob = float(np.median(predictions))
        
        # Collapse nếu std quá nhỏ hoặc mean quá gần 0.5
        collapse_to_05 = (
            std_prob < self.collapse_threshold or
            abs(mean_prob - 0.5) < 0.05
        )
        
        if collapse_to_05:
            warnings.append(
                f"⚠️  Dự đoán có thể bị collapse về 0.5: "
                f"mean={mean_prob:.3f}, std={std_prob:.3f}"
            )
        
        # 2. Kiểm tra class imbalance trong dự đoán
        binary_predictions = (predictions > probability_threshold).astype(int)
        positive_count = np.sum(binary_predictions == 1)
        negative_count = np.sum(binary_predictions == 0)
        total = len(binary_predictions)
        
        positive_ratio = positive_count / total if total > 0 else 0.0
        negative_ratio = negative_count / total if total > 0 else 0.0
        
        # Class imbalance nếu một class chiếm > 90% hoặc < 10%
        class_imbalance = (
            positive_ratio > 0.9 or positive_ratio < 0.1 or
            negative_ratio > 0.9 or negative_ratio < 0.1
        )
        
        if class_imbalance:
            warnings.append(
                f"⚠️  Class imbalance trong dự đoán: "
                f"positive={positive_ratio:.1%}, negative={negative_ratio:.1%}"
            )
        
        # 3. Kiểm tra always predict 0 hoặc 1
        always_predict_zero = positive_count == 0
        always_predict_one = negative_count == 0
        
        if always_predict_zero:
            warnings.append("❌ Mô hình luôn dự đoán 0 (negative)")
        if always_predict_one:
            warnings.append("❌ Mô hình luôn dự đoán 1 (positive)")
        
        # 4. Phân phối dự đoán
        prediction_distribution = {
            '0.0-0.1': np.sum((predictions >= 0.0) & (predictions < 0.1)),
            '0.1-0.2': np.sum((predictions >= 0.1) & (predictions < 0.2)),
            '0.2-0.3': np.sum((predictions >= 0.2) & (predictions < 0.3)),
            '0.3-0.4': np.sum((predictions >= 0.3) & (predictions < 0.4)),
            '0.4-0.5': np.sum((predictions >= 0.4) & (predictions < 0.5)),
            '0.5-0.6': np.sum((predictions >= 0.5) & (predictions < 0.6)),
            '0.6-0.7': np.sum((predictions >= 0.6) & (predictions < 0.7)),
            '0.7-0.8': np.sum((predictions >= 0.7) & (predictions < 0.8)),
            '0.8-0.9': np.sum((predictions >= 0.8) & (predictions < 0.9)),
            '0.9-1.0': np.sum((predictions >= 0.9) & (predictions <= 1.0)),
        }
        
        # Chuyển sang tỷ lệ phần trăm
        total_pred = len(predictions)
        if total_pred > 0:
            prediction_distribution = {
                k: v / total_pred * 100
                for k, v in prediction_distribution.items()
            }
        
        # 5. Kiểm tra class imbalance trong labels (nếu có)
        if labels is not None:
            labels = np.array(labels)
            label_positive = np.sum(labels == 1)
            label_negative = np.sum(labels == 0)
            label_total = len(labels)
            
            if label_total > 0:
                label_positive_ratio = label_positive / label_total
                if label_positive_ratio > 0.9 or label_positive_ratio < 0.1:
                    warnings.append(
                        f"⚠️  Class imbalance trong labels: "
                        f"positive={label_positive_ratio:.1%}"
                    )
        
        return PredictionAnalysis(
            mean_probability=mean_prob,
            std_probability=std_prob,
            median_probability=median_prob,
            collapse_to_05=collapse_to_05,
            collapse_threshold=self.collapse_threshold,
            class_imbalance=class_imbalance,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            always_predict_zero=always_predict_zero,
            always_predict_one=always_predict_one,
            prediction_distribution=prediction_distribution,
            warnings=warnings
        )
    
    def print_analysis(self, analysis: PredictionAnalysis):
        """
        In kết quả phân tích
        
        Args:
            analysis: Kết quả phân tích
        """
        print("\n" + "=" * 80)
        print("📊 PHÂN TÍCH PHÂN PHỐI DỰ ĐOÁN")
        print("=" * 80)
        
        print(f"\n📈 Thống kê cơ bản:")
        print(f"   • Mean probability: {analysis.mean_probability:.4f}")
        print(f"   • Std probability: {analysis.std_probability:.4f}")
        print(f"   • Median probability: {analysis.median_probability:.4f}")
        
        print(f"\n🎯 Phân phối dự đoán:")
        for range_name, percentage in analysis.prediction_distribution.items():
            bar = "█" * int(percentage / 2)
            print(f"   {range_name}: {percentage:5.1f}% {bar}")
        
        print(f"\n⚖️  Class balance:")
        print(f"   • Positive predictions: {analysis.positive_ratio:.1%}")
        print(f"   • Negative predictions: {analysis.negative_ratio:.1%}")
        
        print(f"\n🔍 Kiểm tra:")
        print(f"   • Collapse về 0.5: {'❌ CÓ' if analysis.collapse_to_05 else '✅ KHÔNG'}")
        print(f"   • Class imbalance: {'❌ CÓ' if analysis.class_imbalance else '✅ KHÔNG'}")
        print(f"   • Always predict 0: {'❌ CÓ' if analysis.always_predict_zero else '✅ KHÔNG'}")
        print(f"   • Always predict 1: {'❌ CÓ' if analysis.always_predict_one else '✅ KHÔNG'}")
        
        if analysis.warnings:
            print(f"\n⚠️  Cảnh báo:")
            for warning in analysis.warnings:
                print(f"   {warning}")
        else:
            print(f"\n✅ Không có cảnh báo - Phân phối dự đoán hợp lý")
        
        print("=" * 80)
