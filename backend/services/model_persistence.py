"""
Model Persistence Service

Lưu và tải model để tái sử dụng:
- Save model với metadata đầy đủ
- Load model để sử dụng trong production
- Quản lý version models
- So sánh performance giữa các models
"""
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import xgboost as xgb
import json


class ModelPersistence:
    """Lưu và tải model"""
    
    def __init__(self, models_dir: Path = None):
        """
        Khởi tạo ModelPersistence
        
        Args:
            models_dir: Thư mục lưu models (mặc định: models/)
        """
        if models_dir is None:
            models_dir = Path("models")
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: xgb.XGBClassifier,
        run_id: int,
        features: List[str],
        metrics: Dict[str, float],
        label_config: Dict = None,
        hyperparams: Dict = None,
        risk_config: Dict = None,
        metadata: Dict = None
    ) -> Path:
        """
        Lưu model với metadata đầy đủ
        
        Args:
            model: XGBoost model đã train
            run_id: ID của run
            features: Danh sách features được sử dụng
            metrics: Metrics của model (sharpe, max_drawdown, etc.)
            label_config: Cấu hình label
            hyperparams: Hyperparameters
            risk_config: Cấu hình risk
            metadata: Metadata bổ sung
            
        Returns:
            Đường dẫn đến file model đã lưu
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_run_{run_id}_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        model_data = {
            'model': model,
            'features': features,
            'metrics': metrics,
            'label_config': label_config or {},
            'hyperparams': hyperparams or {},
            'risk_config': risk_config or {},
            'metadata': metadata or {},
            'timestamp': timestamp,
            'run_id': run_id,
            'model_type': 'xgboost'
        }
        
        # Lưu model bằng joblib (tốt hơn pickle cho sklearn/xgboost)
        joblib.dump(model_data, model_path)
        
        # Lưu metadata riêng dưới dạng JSON để dễ đọc
        metadata_path = self.models_dir / f"metadata_run_{run_id}_{timestamp}.json"
        metadata_dict = {
            'run_id': run_id,
            'timestamp': timestamp,
            'features': features,
            'metrics': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                       for k, v in metrics.items()},
            'label_config': label_config or {},
            'hyperparams': hyperparams or {},
            'risk_config': risk_config or {},
            'model_path': str(model_path)
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        return model_path
    
    def load_model(self, model_path: Path) -> Dict[str, Any]:
        """
        Tải model từ file
        
        Args:
            model_path: Đường dẫn đến file model
            
        Returns:
            Dictionary chứa model và metadata
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)
    
    def get_latest_model(self, run_id: Optional[int] = None) -> Optional[Path]:
        """
        Lấy model mới nhất
        
        Args:
            run_id: Nếu chỉ định, chỉ lấy model của run_id đó
            
        Returns:
            Đường dẫn đến model mới nhất hoặc None
        """
        if run_id:
            pattern = f"model_run_{run_id}_*.pkl"
        else:
            pattern = "model_run_*.pkl"
        
        models = list(self.models_dir.glob(pattern))
        if not models:
            return None
        
        return max(models, key=lambda p: p.stat().st_mtime)
    
    def list_models(self, run_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Liệt kê tất cả models
        
        Args:
            run_id: Nếu chỉ định, chỉ liệt kê models của run_id đó
            
        Returns:
            List các dictionary chứa thông tin models
        """
        if run_id:
            pattern = f"model_run_{run_id}_*.pkl"
        else:
            pattern = "model_run_*.pkl"
        
        models = list(self.models_dir.glob(pattern))
        model_info_list = []
        
        for model_path in models:
            try:
                model_data = self.load_model(model_path)
                model_info = {
                    'path': str(model_path),
                    'run_id': model_data.get('run_id'),
                    'timestamp': model_data.get('timestamp'),
                    'metrics': model_data.get('metrics', {}),
                    'n_features': len(model_data.get('features', [])),
                    'sharpe_ratio': model_data.get('metrics', {}).get('sharpe_ratio', 0.0)
                }
                model_info_list.append(model_info)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue
        
        # Sắp xếp theo Sharpe ratio giảm dần
        model_info_list.sort(key=lambda x: x.get('sharpe_ratio', 0.0), reverse=True)
        
        return model_info_list
    
    def get_best_model(self, metric: str = 'sharpe_ratio') -> Optional[Dict[str, Any]]:
        """
        Lấy model tốt nhất theo metric
        
        Args:
            metric: Metric để so sánh (mặc định: sharpe_ratio)
            
        Returns:
            Dictionary chứa model và metadata của model tốt nhất
        """
        models = self.list_models()
        if not models:
            return None
        
        # Tìm model có metric cao nhất
        best_model_info = max(
            models,
            key=lambda x: x.get('metrics', {}).get(metric, 0.0)
        )
        
        return self.load_model(Path(best_model_info['path']))
    
    def compare_models(self, model_paths: List[Path]) -> pd.DataFrame:
        """
        So sánh nhiều models
        
        Args:
            model_paths: List đường dẫn đến các models cần so sánh
            
        Returns:
            DataFrame chứa thông tin so sánh
        """
        comparison_data = []
        
        for model_path in model_paths:
            try:
                model_data = self.load_model(model_path)
                comparison_data.append({
                    'run_id': model_data.get('run_id'),
                    'timestamp': model_data.get('timestamp'),
                    'sharpe_ratio': model_data.get('metrics', {}).get('sharpe_ratio', 0.0),
                    'max_drawdown': model_data.get('metrics', {}).get('max_drawdown', 1.0),
                    'profit_factor': model_data.get('metrics', {}).get('profit_factor', 0.0),
                    'total_return': model_data.get('metrics', {}).get('total_return', 0.0),
                    'win_rate': model_data.get('metrics', {}).get('win_rate', 0.0),
                    'total_trades': model_data.get('metrics', {}).get('total_trades', 0),
                    'n_features': len(model_data.get('features', [])),
                    'model_path': str(model_path)
                })
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue
        
        return pd.DataFrame(comparison_data).sort_values('sharpe_ratio', ascending=False)
