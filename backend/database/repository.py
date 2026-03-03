"""
数据库仓库层

提供数据访问接口，封装数据库操作
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.models import ModelRun, OptimizationTrial


class ModelRunRepository:
    """模型运行记录仓库"""
    
    def __init__(self, db: Session):
        """
        初始化仓库
        
        Args:
            db: SQLAlchemy数据库会话
        """
        self.db = db
    
    def create(
        self,
        feature_config: Optional[Dict] = None,
        label_config: Optional[Dict] = None,
        risk_config: Optional[Dict] = None,
        hyperparams: Optional[Dict] = None,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        profit_factor: Optional[float] = None,
        calmar_ratio: Optional[float] = None,
        total_return: Optional[float] = None,
        annualized_return: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        total_trades: Optional[int] = None,
        sharpe_stability: Optional[float] = None,
        regime_metrics: Optional[Dict] = None
    ) -> ModelRun:
        """
        创建新的模型运行记录
        
        Returns:
            ModelRun对象
        """
        model_run = ModelRun(
            feature_config=feature_config,
            label_config=label_config,
            risk_config=risk_config,
            hyperparams=hyperparams,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            win_rate=win_rate,
            total_trades=total_trades,
            sharpe_stability=sharpe_stability,
            regime_metrics=regime_metrics
        )
        
        self.db.add(model_run)
        self.db.commit()
        self.db.refresh(model_run)
        
        return model_run
    
    def get_by_id(self, run_id: int) -> Optional[ModelRun]:
        """
        根据ID获取模型运行记录
        
        Args:
            run_id: 运行ID
            
        Returns:
            ModelRun对象或None
        """
        return self.db.query(ModelRun).filter(ModelRun.id == run_id).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: str = 'created_at',
        descending: bool = True
    ) -> List[ModelRun]:
        """
        获取所有模型运行记录
        
        Args:
            skip: 跳过记录数
            limit: 返回记录数
            order_by: 排序字段
            descending: 是否降序
            
        Returns:
            ModelRun列表
        """
        query = self.db.query(ModelRun)
        
        # 排序
        if hasattr(ModelRun, order_by):
            order_column = getattr(ModelRun, order_by)
            if descending:
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(order_column)
        
        return query.offset(skip).limit(limit).all()
    
    def get_top_by_sharpe(self, top_n: int = 5) -> List[Any]:
        """
        获取Sharpe比率最高的N个模型运行

        仅查询展示所需字段，避免因数据库表尚未包含新增列导致查询失败。
        
        Args:
            top_n: 返回数量
            
        Returns:
            包含展示字段的记录列表
        """
        return self.db.query(
            ModelRun.id,
            ModelRun.sharpe_ratio,
            ModelRun.max_drawdown,
            ModelRun.profit_factor,
            ModelRun.total_trades,
        ).filter(
            ModelRun.sharpe_ratio.isnot(None)
        ).order_by(desc(ModelRun.sharpe_ratio)).limit(top_n).all()
    
    def get_top_by_stability(self, top_n: int = 5) -> List[ModelRun]:
        """
        获取最稳定的N个模型运行（Sharpe标准差最小）
        
        Args:
            top_n: 返回数量
            
        Returns:
            ModelRun列表
        """
        return self.db.query(ModelRun).filter(
            ModelRun.sharpe_stability.isnot(None)
        ).order_by(ModelRun.sharpe_stability).limit(top_n).all()
    
    def update(
        self,
        run_id: int,
        **kwargs
    ) -> Optional[ModelRun]:
        """
        更新模型运行记录
        
        Args:
            run_id: 运行ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的ModelRun对象或None
        """
        model_run = self.get_by_id(run_id)
        if not model_run:
            return None
        
        for key, value in kwargs.items():
            if hasattr(model_run, key):
                setattr(model_run, key, value)
        
        model_run.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(model_run)
        
        return model_run
    
    def delete(self, run_id: int) -> bool:
        """
        删除模型运行记录
        
        Args:
            run_id: 运行ID
            
        Returns:
            是否删除成功
        """
        model_run = self.get_by_id(run_id)
        if not model_run:
            return False
        
        self.db.delete(model_run)
        self.db.commit()
        
        return True


class OptimizationTrialRepository:
    """优化试验记录仓库"""
    
    def __init__(self, db: Session):
        """
        初始化仓库
        
        Args:
            db: SQLAlchemy数据库会话
        """
        self.db = db
    
    def create(
        self,
        run_id: int,
        trial_type: str,
        parameters: Optional[Dict] = None,
        sharpe_ratio: Optional[float] = None,
        drawdown: Optional[float] = None,
        profit_factor: Optional[float] = None
    ) -> OptimizationTrial:
        """
        创建新的优化试验记录
        
        Returns:
            OptimizationTrial对象
        """
        trial = OptimizationTrial(
            run_id=run_id,
            trial_type=trial_type,
            parameters=parameters,
            sharpe_ratio=sharpe_ratio,
            drawdown=drawdown,
            profit_factor=profit_factor
        )
        
        self.db.add(trial)
        self.db.commit()
        self.db.refresh(trial)
        
        return trial
    
    def get_by_run_id(self, run_id: int) -> List[OptimizationTrial]:
        """
        获取指定运行的所有试验
        
        Args:
            run_id: 运行ID
            
        Returns:
            OptimizationTrial列表
        """
        return self.db.query(OptimizationTrial).filter(
            OptimizationTrial.run_id == run_id
        ).all()
    
    def get_by_type(
        self,
        run_id: int,
        trial_type: str
    ) -> List[OptimizationTrial]:
        """
        获取指定运行和类型的试验
        
        Args:
            run_id: 运行ID
            trial_type: 试验类型
            
        Returns:
            OptimizationTrial列表
        """
        return self.db.query(OptimizationTrial).filter(
            OptimizationTrial.run_id == run_id,
            OptimizationTrial.trial_type == trial_type
        ).all()
