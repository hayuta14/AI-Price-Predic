"""
性能报告生成模块

生成综合性能报告，包括：
- 按Sharpe排序的配置排名
- 按最大回撤排序
- 按跨状态稳定性排序
- 综合排名表
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from backend.core.metrics import PerformanceMetrics


class PerformanceReporter:
    """
    性能报告生成器
    
    生成各种格式的性能报告和排名
    """
    
    def __init__(self):
        """初始化性能报告生成器"""
        pass
    
    def create_ranking_table(
        self,
        results: List[Dict],
        sort_by: str = 'sharpe_ratio',
        ascending: bool = False,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        创建排名表
        
        Args:
            results: 结果字典列表，每个字典包含配置和性能指标
            sort_by: 排序字段
            ascending: 是否升序
            top_n: 返回前N名（如果为None，返回全部）
            
        Returns:
            排名DataFrame
        """
        if not results:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 排序
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # 添加排名
        df.insert(0, '排名', range(1, len(df) + 1))
        
        # 返回前N名
        if top_n is not None:
            df = df.head(top_n)
        
        return df
    
    def create_comprehensive_ranking(
        self,
        results: List[Dict],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        创建综合排名
        
        使用多个指标加权排序
        
        Args:
            results: 结果字典列表
            weights: 指标权重字典（如果为None，使用默认权重）
            
        Returns:
            综合排名DataFrame
        """
        if not results:
            return pd.DataFrame()
        
        # 默认权重
        if weights is None:
            weights = {
                'sharpe_ratio': 0.4,
                'max_drawdown': -0.3,  # 负权重（越小越好）
                'profit_factor': 0.2,
                'calmar_ratio': 0.1
            }
        
        df = pd.DataFrame(results)
        
        # 归一化指标（0-1范围）
        normalized_scores = {}
        
        for metric, weight in weights.items():
            if metric in df.columns:
                values = df[metric].values
                
                # 处理负权重（对于越小越好的指标）
                if weight < 0:
                    # 反转：最大值变为最小值
                    max_val = values.max()
                    min_val = values.min()
                    if max_val != min_val:
                        normalized = 1 - (values - min_val) / (max_val - min_val)
                    else:
                        normalized = np.ones_like(values)
                    weight = abs(weight)
                else:
                    # 正常归一化
                    max_val = values.max()
                    min_val = values.min()
                    if max_val != min_val:
                        normalized = (values - min_val) / (max_val - min_val)
                    else:
                        normalized = np.ones_like(values)
                
                normalized_scores[metric] = normalized * weight
        
        # 计算综合得分
        composite_score = sum(normalized_scores.values())
        df['综合得分'] = composite_score
        
        # 排序
        df = df.sort_values(by='综合得分', ascending=False)
        df.insert(0, '综合排名', range(1, len(df) + 1))
        
        return df
    
    def create_stability_ranking(
        self,
        results: List[Dict],
        regime_sharpe_std_col: str = 'sharpe_stability'
    ) -> pd.DataFrame:
        """
        创建稳定性排名（按跨状态Sharpe标准差）
        
        Args:
            results: 结果字典列表
            regime_sharpe_std_col: 跨状态Sharpe标准差列名
            
        Returns:
            稳定性排名DataFrame（Sharpe标准差越小越好）
        """
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        if regime_sharpe_std_col not in df.columns:
            # 如果没有该列，尝试计算
            if 'regime_sharpe_std' in df.columns:
                regime_sharpe_std_col = 'regime_sharpe_std'
            else:
                # 无法计算稳定性，返回空
                return pd.DataFrame()
        
        # 按Sharpe标准差排序（越小越好）
        df = df.sort_values(by=regime_sharpe_std_col, ascending=True)
        df.insert(0, '稳定性排名', range(1, len(df) + 1))
        
        return df
    
    def generate_summary_report(
        self,
        results: List[Dict],
        top_n: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        生成综合摘要报告
        
        Args:
            results: 结果字典列表
            top_n: 每个排名表返回前N名
            
        Returns:
            包含各种排名表的字典
        """
        reports = {}
        
        # 按Sharpe排序
        reports['sharpe_ranking'] = self.create_ranking_table(
            results,
            sort_by='sharpe_ratio',
            ascending=False,
            top_n=top_n
        )
        
        # 按最大回撤排序（越小越好）
        reports['drawdown_ranking'] = self.create_ranking_table(
            results,
            sort_by='max_drawdown',
            ascending=True,
            top_n=top_n
        )
        
        # 按盈亏比排序
        reports['profit_factor_ranking'] = self.create_ranking_table(
            results,
            sort_by='profit_factor',
            ascending=False,
            top_n=top_n
        )
        
        # 综合排名
        reports['comprehensive_ranking'] = self.create_comprehensive_ranking(
            results
        ).head(top_n)
        
        # 稳定性排名
        reports['stability_ranking'] = self.create_stability_ranking(results).head(top_n)
        
        return reports
    
    def format_report_for_display(
        self,
        df: pd.DataFrame,
        precision: int = 4
    ) -> str:
        """
        格式化报告用于显示
        
        Args:
            df: 报告DataFrame
            precision: 数值精度
            
        Returns:
            格式化的字符串
        """
        if df.empty:
            return "无数据"
        
        # 设置显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # 格式化数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_formatted = df.copy()
        
        for col in numeric_cols:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{x:.{precision}f}" if pd.notna(x) else "N/A"
            )
        
        return df_formatted.to_string(index=False)
    
    def export_report_to_csv(
        self,
        df: pd.DataFrame,
        filepath: str
    ):
        """
        导出报告到CSV
        
        Args:
            df: 报告DataFrame
            filepath: 文件路径
        """
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
