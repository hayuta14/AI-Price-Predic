"""
Robust validation utilities.
"""
from __future__ import annotations

from typing import Callable, Dict, Any, List
import numpy as np
import pandas as pd


class RobustValidator:
    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        model_fn: Callable[[], Any],
        n_splits: int = 5,
        train_pct: float = 0.7,
        gap_periods: int = 96,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        fold_size = len(df) // max(n_splits, 1)

        for i in range(n_splits):
            start = i * fold_size
            end = len(df) if i == n_splits - 1 else start + fold_size
            split = start + int((end - start) * train_pct)

            train_end = max(start, split - gap_periods)
            test_start = split

            train_df = df.iloc[start:train_end]
            test_df = df.iloc[test_start:end]

            if len(train_df) < 100 or len(test_df) < 50:
                continue

            results.append(
                {
                    'fold': i,
                    'train_period': f"{train_df.index.min()} → {train_df.index.max()}",
                    'test_period': f"{test_df.index.min()} → {test_df.index.max()}",
                    'metrics': self._evaluate_fold(model_fn, train_df, test_df),
                }
            )

        return self._aggregate_results(results)

    def regime_split_validate(self, df: pd.DataFrame, model_fn: Callable[[], Any]) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for regime in ['trending_up', 'trending_down', 'sideways', 'high_volatility']:
            regime_df = df[df['regime'] == regime] if 'regime' in df.columns else pd.DataFrame()
            if len(regime_df) < 200:
                continue
            split = int(len(regime_df) * 0.7)
            train_df = regime_df.iloc[:split]
            test_df = regime_df.iloc[split:]
            results[regime] = self._evaluate_fold(model_fn, train_df, test_df)
        return results

    def _evaluate_fold(self, model_fn: Callable[[], Any], train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
        model = model_fn()
        if hasattr(model, 'fit'):
            model.fit(train_df)
        if hasattr(model, 'backtest'):
            return model.backtest(test_df)
        return {'sharpe_ratio': 0.0}

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {'mean_sharpe': 0.0, 'std_sharpe': 0.0, 'min_sharpe': 0.0, 'consistency': 0.0, 'folds': []}
        all_sharpes = [r['metrics'].get('sharpe_ratio', 0.0) for r in results]
        return {
            'mean_sharpe': float(np.mean(all_sharpes)),
            'std_sharpe': float(np.std(all_sharpes)),
            'min_sharpe': float(np.min(all_sharpes)),
            'consistency': float(np.mean([s > 0 for s in all_sharpes])),
            'folds': results,
        }


if __name__ == '__main__':
    print('RobustValidator module loaded.')
