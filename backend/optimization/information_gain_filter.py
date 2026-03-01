"""
Information Gain Filter

Kiểm tra Information Gain của features để loại bỏ features vô nghĩa:
- Sử dụng mutual_info_classif để tính Information Gain
- Filter features có MI ≈ 0
- Giữ lại features có MI cao
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


class InformationGainFilter:
    """Filter features dựa trên Information Gain"""
    
    def __init__(self, min_mi: float = 0.001, random_state: int = 42):
        """
        Khởi tạo Information Gain Filter
        
        Args:
            min_mi: Minimum Mutual Information để giữ lại feature (mặc định: 0.001)
            random_state: Random state cho reproducibility
        """
        self.min_mi = min_mi
        self.random_state = random_state
        self.feature_mi_scores: Dict[str, float] = {}
    
    def calculate_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_size: Optional[int] = None
    ) -> pd.Series:
        """
        Tính Mutual Information cho tất cả features
        
        Args:
            X: Feature matrix
            y: Target labels
            sample_size: Sample size để tính nhanh (None = use all data)
            
        Returns:
            Series với MI scores cho mỗi feature
        """
        # Sample data nếu cần (để tăng tốc)
        if sample_size and len(X) > sample_size:
            sample_idx = np.random.RandomState(self.random_state).choice(
                len(X), size=sample_size, replace=False
            )
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y
        
        # Loại bỏ NaN và Inf
        X_clean = X_sample.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Tính Mutual Information
        try:
            mi_scores = mutual_info_classif(
                X_clean,
                y_sample,
                random_state=self.random_state,
                discrete_features=False
            )
        except Exception as e:
            print(f"Warning: Error calculating MI: {e}")
            # Return zero scores nếu có lỗi
            mi_scores = np.zeros(len(X_clean.columns))
        
        # Tạo Series với feature names
        mi_series = pd.Series(mi_scores, index=X_clean.columns)
        
        # Lưu scores
        self.feature_mi_scores = mi_series.to_dict()
        
        return mi_series
    
    def filter_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_mi: Optional[float] = None,
        top_k: Optional[int] = None,
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str], pd.Series]:
        """
        Filter features dựa trên Information Gain
        
        Args:
            X: Feature matrix
            y: Target labels
            min_mi: Minimum MI threshold (None = use self.min_mi)
            top_k: Chỉ giữ lại top K features (None = use min_mi)
            sample_size: Sample size để tính MI (None = use all)
            
        Returns:
            Tuple (filtered_X, selected_features, mi_scores)
        """
        if min_mi is None:
            min_mi = self.min_mi
        
        # Tính MI scores
        mi_scores = self.calculate_mutual_information(X, y, sample_size)
        
        # Select features
        if top_k:
            # Chọn top K features
            selected_features = mi_scores.nlargest(top_k).index.tolist()
        else:
            # Chọn features có MI >= min_mi
            selected_features = mi_scores[mi_scores >= min_mi].index.tolist()
        
        # Filter X
        filtered_X = X[selected_features]
        
        return filtered_X, selected_features, mi_scores
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Lấy feature importance dựa trên MI scores
        
        Returns:
            DataFrame với features và MI scores, sorted by MI
        """
        if not self.feature_mi_scores:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': list(self.feature_mi_scores.keys()),
            'mutual_information': list(self.feature_mi_scores.values())
        })
        
        df = df.sort_values('mutual_information', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def analyze_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_size: Optional[int] = 5000
    ) -> Dict[str, any]:
        """
        Phân tích features và trả về report
        
        Args:
            X: Feature matrix
            y: Target labels
            sample_size: Sample size để tính MI
            
        Returns:
            Dictionary chứa analysis results
        """
        # Tính MI scores
        mi_scores = self.calculate_mutual_information(X, y, sample_size)
        
        # Phân tích
        total_features = len(mi_scores)
        meaningful_features = (mi_scores >= self.min_mi).sum()
        meaningless_features = (mi_scores < self.min_mi).sum()
        
        # Top features
        top_10 = mi_scores.nlargest(10)
        bottom_10 = mi_scores.nsmallest(10)
        
        return {
            'total_features': total_features,
            'meaningful_features': meaningful_features,
            'meaningless_features': meaningless_features,
            'meaningful_ratio': meaningful_features / total_features if total_features > 0 else 0.0,
            'mean_mi': mi_scores.mean(),
            'median_mi': mi_scores.median(),
            'std_mi': mi_scores.std(),
            'top_10_features': top_10.to_dict(),
            'bottom_10_features': bottom_10.to_dict(),
            'mi_scores': mi_scores.to_dict()
        }
    
    def print_analysis_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_size: Optional[int] = 5000
    ):
        """
        In báo cáo phân tích features
        
        Args:
            X: Feature matrix
            y: Target labels
            sample_size: Sample size để tính MI
        """
        analysis = self.analyze_features(X, y, sample_size)
        
        print("\n" + "=" * 80)
        print("📊 PHÂN TÍCH INFORMATION GAIN (Mutual Information)")
        print("=" * 80)
        
        print(f"\n📈 Tổng quan:")
        print(f"   • Tổng số features: {analysis['total_features']}")
        print(f"   • Features có nghĩa (MI >= {self.min_mi}): {analysis['meaningful_features']}")
        print(f"   • Features vô nghĩa (MI < {self.min_mi}): {analysis['meaningless_features']}")
        print(f"   • Tỷ lệ features có nghĩa: {analysis['meaningful_ratio']*100:.1f}%")
        
        print(f"\n📊 Thống kê MI:")
        print(f"   • Mean MI: {analysis['mean_mi']:.6f}")
        print(f"   • Median MI: {analysis['median_mi']:.6f}")
        print(f"   • Std MI: {analysis['std_mi']:.6f}")
        
        print(f"\n🏆 Top 10 Features (MI cao nhất):")
        for i, (feature, mi) in enumerate(analysis['top_10_features'].items(), 1):
            print(f"   {i:2d}. {feature:30s} : {mi:.6f}")
        
        print(f"\n⚠️  Bottom 10 Features (MI thấp nhất - có thể loại bỏ):")
        for i, (feature, mi) in enumerate(analysis['bottom_10_features'].items(), 1):
            print(f"   {i:2d}. {feature:30s} : {mi:.6f}")
        
        print("=" * 80)
