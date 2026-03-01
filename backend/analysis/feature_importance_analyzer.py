"""
Feature Importance Analyzer

Kiểm tra feature importance để đảm bảo không có feature dominance:
- XGBoost gain importance
- SHAP values
- Kiểm tra feature có thực sự đóng góp không
- Cảnh báo nếu 1 feature chiếm >90% gain
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available, will skip SHAP analysis")


class FeatureImportanceAnalyzer:
    """Phân tích feature importance"""
    
    def __init__(self, dominance_threshold: float = 0.9):
        """
        Khởi tạo Feature Importance Analyzer
        
        Args:
            dominance_threshold: Ngưỡng để cảnh báo feature dominance (mặc định: 0.9 = 90%)
        """
        self.dominance_threshold = dominance_threshold
    
    def analyze_gain_importance(
        self,
        model: xgb.XGBClassifier,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Phân tích Gain Importance từ XGBoost
        
        Args:
            model: Trained XGBoost model
            feature_names: Tên các features
            
        Returns:
            DataFrame với feature importance
        """
        # Lấy feature importance từ model
        importance_dict = model.get_booster().get_score(importance_type='gain')
        
        # Tạo DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'gain': [importance_dict.get(f'f{i}', 0.0) for i in range(len(feature_names))]
        })
        
        # Tính tỷ lệ phần trăm
        total_gain = importance_df['gain'].sum()
        if total_gain > 0:
            importance_df['gain_pct'] = importance_df['gain'] / total_gain * 100
        else:
            importance_df['gain_pct'] = 0.0
        
        # Sắp xếp theo gain giảm dần
        importance_df = importance_df.sort_values('gain', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def analyze_shap_importance(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        sample_size: int = 1000
    ) -> pd.DataFrame:
        """
        Phân tích SHAP Importance
        
        Args:
            model: Trained XGBoost model
            X: Feature matrix
            sample_size: Sample size để tính SHAP (để tăng tốc)
            
        Returns:
            DataFrame với SHAP importance
        """
        if not SHAP_AVAILABLE:
            return pd.DataFrame()
        
        # Sample để tăng tốc
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        try:
            # Tính SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Xử lý SHAP values (có thể là list cho multi-class)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Lấy class 1
            
            # Tính mean absolute SHAP value
            shap_importance = np.abs(shap_values).mean(axis=0)
            
            # Tạo DataFrame
            importance_df = pd.DataFrame({
                'feature': X.columns.tolist(),
                'shap_importance': shap_importance
            })
            
            # Tính tỷ lệ phần trăm
            total_shap = importance_df['shap_importance'].sum()
            if total_shap > 0:
                importance_df['shap_pct'] = importance_df['shap_importance'] / total_shap * 100
            else:
                importance_df['shap_pct'] = 0.0
            
            # Sắp xếp
            importance_df = importance_df.sort_values('shap_importance', ascending=False)
            importance_df['rank'] = range(1, len(importance_df) + 1)
            
            return importance_df
            
        except Exception as e:
            print(f"Warning: Error calculating SHAP: {e}")
            return pd.DataFrame()
    
    def check_feature_dominance(
        self,
        importance_df: pd.DataFrame,
        importance_col: str = 'gain_pct'
    ) -> Dict[str, any]:
        """
        Kiểm tra feature dominance
        
        Args:
            importance_df: DataFrame với feature importance
            importance_col: Tên cột importance để kiểm tra
            
        Returns:
            Dictionary với kết quả kiểm tra
        """
        if len(importance_df) == 0:
            return {
                'has_dominance': False,
                'dominant_feature': None,
                'dominance_pct': 0.0,
                'top_3_pct': 0.0,
                'warnings': []
            }
        
        top_feature = importance_df.iloc[0]
        top_pct = top_feature[importance_col]
        
        top_3_pct = importance_df.head(3)[importance_col].sum()
        
        warnings = []
        has_dominance = False
        
        if top_pct > self.dominance_threshold * 100:
            has_dominance = True
            warnings.append(
                f"⚠️  Feature dominance: {top_feature['feature']} chiếm {top_pct:.1f}% "
                f"(>{self.dominance_threshold*100:.0f}%) - Nguy hiểm!"
            )
        
        if top_3_pct > 0.95 * 100:
            warnings.append(
                f"⚠️  Top 3 features chiếm {top_3_pct:.1f}% - Quá tập trung"
            )
        
        return {
            'has_dominance': has_dominance,
            'dominant_feature': top_feature['feature'] if has_dominance else None,
            'dominance_pct': top_pct,
            'top_3_pct': top_3_pct,
            'warnings': warnings
        }
    
    def analyze(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Phân tích đầy đủ feature importance
        
        Args:
            model: Trained XGBoost model
            X: Feature matrix
            feature_names: Tên features (None = dùng từ X.columns)
            
        Returns:
            Dictionary với kết quả phân tích
        """
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        # Gain importance
        gain_df = self.analyze_gain_importance(model, feature_names)
        
        # SHAP importance
        shap_df = self.analyze_shap_importance(model, X)
        
        # Kiểm tra dominance
        gain_dominance = self.check_feature_dominance(gain_df, 'gain_pct')
        shap_dominance = self.check_feature_dominance(shap_df, 'shap_pct') if not shap_df.empty else {}
        
        return {
            'gain_importance': gain_df,
            'shap_importance': shap_df,
            'gain_dominance': gain_dominance,
            'shap_dominance': shap_dominance,
            'has_issues': gain_dominance.get('has_dominance', False) or shap_dominance.get('has_dominance', False)
        }
    
    def print_analysis(self, analysis: Dict[str, any]):
        """
        In kết quả phân tích
        
        Args:
            analysis: Kết quả từ analyze()
        """
        print("\n" + "=" * 80)
        print("📊 PHÂN TÍCH FEATURE IMPORTANCE")
        print("=" * 80)
        
        gain_df = analysis['gain_importance']
        shap_df = analysis['shap_importance']
        
        # Gain Importance
        print(f"\n📈 Gain Importance (XGBoost):")
        print(f"   Top 10 Features:")
        for i, row in gain_df.head(10).iterrows():
            print(f"   {row['rank']:2d}. {row['feature']:30s} : {row['gain_pct']:6.2f}%")
        
        # SHAP Importance
        if not shap_df.empty:
            print(f"\n📊 SHAP Importance:")
            print(f"   Top 10 Features:")
            for i, row in shap_df.head(10).iterrows():
                print(f"   {row['rank']:2d}. {row['feature']:30s} : {row['shap_pct']:6.2f}%")
        
        # Dominance Check
        gain_dom = analysis['gain_dominance']
        if gain_dom.get('has_dominance', False):
            print(f"\n⚠️  CẢNH BÁO - Feature Dominance:")
            for warning in gain_dom['warnings']:
                print(f"   {warning}")
            print(f"   → Feature '{gain_dom['dominant_feature']}' chiếm quá nhiều importance")
            print(f"   → Có thể dẫn đến overfitting hoặc model không robust")
        else:
            print(f"\n✅ Không có feature dominance - Feature distribution hợp lý")
        
        if not shap_df.empty:
            shap_dom = analysis['shap_dominance']
            if shap_dom.get('has_dominance', False):
                print(f"\n⚠️  CẢNH BÁO - SHAP Dominance:")
                for warning in shap_dom['warnings']:
                    print(f"   {warning}")
        
        print("=" * 80)
