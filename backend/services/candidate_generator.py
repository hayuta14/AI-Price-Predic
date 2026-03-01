"""
Candidate Generator

Tạo các candidates để train nhiều models:
- Generate combinations của features, labels, hyperparams
- Tạo diverse set of candidates để explore search space
"""
from typing import List, Dict, Any
import itertools
import random
from backend.optimization.label_optimizer import LabelConfiguration
from backend.optimization.hyperparameter_optimizer import HyperparameterSet


class CandidateGenerator:
    """Generate candidates cho multi-model training"""
    
    def __init__(self, random_seed: int = 42):
        """
        Khởi tạo CandidateGenerator
        
        Args:
            random_seed: Random seed để reproduce
        """
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def generate_candidates_from_optimization_results(
        self,
        feature_sets: List[List[str]],
        label_configs: List[LabelConfiguration],
        hyperparam_sets: List[HyperparameterSet],
        n_candidates: int = None,
        strategy: str = 'top_combinations'
    ) -> List[Dict[str, Any]]:
        """
        Generate candidates từ kết quả optimization
        
        Args:
            feature_sets: List các feature sets
            label_configs: List các label configurations
            hyperparam_sets: List các hyperparameter sets
            n_candidates: Số candidates muốn tạo (None = tất cả combinations)
            strategy: Strategy để chọn candidates:
                - 'top_combinations': Top combinations theo performance
                - 'random': Random sampling
                - 'diverse': Diverse sampling để explore space
                - 'all': Tất cả combinations
                
        Returns:
            List các candidates
        """
        # Tạo tất cả combinations
        all_combinations = list(itertools.product(
            feature_sets,
            label_configs,
            hyperparam_sets
        ))
        
        if strategy == 'all' or n_candidates is None or n_candidates >= len(all_combinations):
            candidates = [
                {
                    'features': features,
                    'label_config': label_config,
                    'hyperparams': hyperparams
                }
                for features, label_config, hyperparams in all_combinations
            ]
        elif strategy == 'random':
            selected = random.sample(all_combinations, min(n_candidates, len(all_combinations)))
            candidates = [
                {
                    'features': features,
                    'label_config': label_config,
                    'hyperparams': hyperparams
                }
                for features, label_config, hyperparams in selected
            ]
        elif strategy == 'top_combinations':
            # Sắp xếp theo performance (nếu có)
            # Ở đây chúng ta sẽ chọn top N combinations
            # Trong thực tế, có thể rank theo expected performance
            selected = all_combinations[:n_candidates]
            candidates = [
                {
                    'features': features,
                    'label_config': label_config,
                    'hyperparams': hyperparams
                }
                for features, label_config, hyperparams in selected
            ]
        else:  # diverse
            # Chọn diverse set để explore search space
            step = len(all_combinations) // n_candidates if n_candidates > 0 else 1
            selected = all_combinations[::step][:n_candidates]
            candidates = [
                {
                    'features': features,
                    'label_config': label_config,
                    'hyperparams': hyperparams
                }
                for features, label_config, hyperparams in selected
            ]
        
        return candidates
    
    def generate_candidates_from_config(
        self,
        available_features: List[str],
        label_config_options: Dict[str, Any],
        hyperparam_options: Dict[str, Any],
        n_candidates: int = 20,
        max_features_per_candidate: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Generate candidates từ config options
        
        Args:
            available_features: Tất cả features có sẵn
            label_config_options: Dict chứa options cho label config:
                - horizons: List[int]
                - thresholds: List[float]
                - use_dynamic: bool
                - atr_multipliers: List[float]
            hyperparam_options: Dict chứa options cho hyperparams:
                - max_depths: List[int]
                - learning_rates: List[float]
                - n_estimators_list: List[int]
                - subsamples: List[float]
                - colsample_bytrees: List[float]
            n_candidates: Số candidates muốn tạo
            max_features_per_candidate: Số features tối đa mỗi candidate
            
        Returns:
            List các candidates
        """
        candidates = []
        
        # Generate feature combinations
        feature_combinations = []
        for n_features in range(5, min(max_features_per_candidate + 1, len(available_features) + 1)):
            # Random sample n_features từ available_features
            n_combinations = min(3, n_candidates // 5)  # Mỗi số features có vài combinations
            for _ in range(n_combinations):
                features = random.sample(available_features, n_features)
                feature_combinations.append(features)
        
        # Generate label configs
        label_configs = []
        horizons = label_config_options.get('horizons', [6, 8, 10, 12])
        thresholds = label_config_options.get('thresholds', [0.003, 0.005, 0.01])
        use_dynamic = label_config_options.get('use_dynamic', True)
        atr_multipliers = label_config_options.get('atr_multipliers', [0.5, 1.0, 1.5, 2.0])
        
        for horizon in horizons:
            # Fixed thresholds
            for threshold in thresholds:
                label_configs.append(LabelConfiguration(
                    horizon=horizon,
                    threshold=threshold,
                    sharpe_ratio=0.0,
                    max_drawdown=1.0,
                    profit_factor=0.0,
                    total_trades=0,
                    win_rate=0.0
                ))
            
            # Dynamic thresholds
            if use_dynamic:
                for atr_mult in atr_multipliers:
                    label_configs.append(LabelConfiguration(
                        horizon=horizon,
                        threshold=-atr_mult,  # Negative để đánh dấu dynamic
                        sharpe_ratio=0.0,
                        max_drawdown=1.0,
                        profit_factor=0.0,
                        total_trades=0,
                        win_rate=0.0
                    ))
        
        # Generate hyperparam sets
        hyperparam_sets = []
        max_depths = hyperparam_options.get('max_depths', [3, 5, 7])
        learning_rates = hyperparam_options.get('learning_rates', [0.01, 0.05, 0.1])
        n_estimators_list = hyperparam_options.get('n_estimators_list', [100, 200, 300])
        subsamples = hyperparam_options.get('subsamples', [0.8, 0.9])
        colsample_bytrees = hyperparam_options.get('colsample_bytrees', [0.8, 0.9])
        
        # Tạo combinations của hyperparams
        hyperparam_combinations = list(itertools.product(
            max_depths,
            learning_rates,
            n_estimators_list,
            subsamples,
            colsample_bytrees
        ))
        
        # Sample một số combinations
        n_hyperparams = min(n_candidates // 3, len(hyperparam_combinations))
        selected_hyperparams = random.sample(hyperparam_combinations, n_hyperparams)
        
        for max_depth, lr, n_est, subsample, colsample in selected_hyperparams:
            hyperparam_sets.append(HyperparameterSet(
                max_depth=max_depth,
                learning_rate=lr,
                n_estimators=n_est,
                subsample=subsample,
                colsample_bytree=colsample,
                sharpe_ratio=0.0,
                max_drawdown=1.0
            ))
        
        # Combine tất cả
        all_combinations = list(itertools.product(
            feature_combinations[:n_candidates // 2],
            label_configs[:n_candidates // 2],
            hyperparam_sets
        ))
        
        # Sample n_candidates
        if len(all_combinations) > n_candidates:
            selected = random.sample(all_combinations, n_candidates)
        else:
            selected = all_combinations
        
        candidates = [
            {
                'features': features,
                'label_config': label_config,
                'hyperparams': hyperparams
            }
            for features, label_config, hyperparams in selected
        ]
        
        return candidates
