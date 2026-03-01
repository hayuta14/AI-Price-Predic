"""
数据库模型定义

使用SQLAlchemy ORM定义数据表结构：
- ModelRun: 模型运行记录
- OptimizationTrial: 优化试验记录
"""
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import json

Base = declarative_base()


class ModelRun(Base):
    """
    模型运行记录
    
    存储每次完整优化的配置和结果
    """
    __tablename__ = 'model_runs'
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 配置信息（JSON格式）
    feature_config = Column(JSON, nullable=True)  # 特征配置
    label_config = Column(JSON, nullable=True)    # 标签配置（horizon, threshold）
    risk_config = Column(JSON, nullable=True)     # 风险配置
    hyperparams = Column(JSON, nullable=True)     # XGBoost超参数
    
    # 性能指标
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    total_return = Column(Float, nullable=True)
    annualized_return = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    
    # 稳定性指标
    sharpe_stability = Column(Float, nullable=True)  # 跨状态Sharpe标准差
    regime_metrics = Column(JSON, nullable=True)    # 各状态性能指标
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联关系
    optimization_trials = relationship("OptimizationTrial", back_populates="model_run", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'feature_config': self.feature_config,
            'label_config': self.label_config,
            'risk_config': self.risk_config,
            'hyperparams': self.hyperparams,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
            'calmar_ratio': self.calmar_ratio,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'sharpe_stability': self.sharpe_stability,
            'regime_metrics': self.regime_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class OptimizationTrial(Base):
    """
    优化试验记录
    
    存储每次优化试验的详细参数和结果
    """
    __tablename__ = 'optimization_trials'
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey('model_runs.id'), nullable=False, index=True)
    
    # 试验类型
    trial_type = Column(String(50), nullable=False)  # 'feature', 'label', 'risk', 'hyperparameter'
    
    # 试验参数（JSON格式）
    parameters = Column(JSON, nullable=True)
    
    # 性能指标
    sharpe_ratio = Column(Float, nullable=True)
    drawdown = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # 关联关系
    model_run = relationship("ModelRun", back_populates="optimization_trials")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'run_id': self.run_id,
            'trial_type': self.trial_type,
            'parameters': self.parameters,
            'sharpe_ratio': self.sharpe_ratio,
            'drawdown': self.drawdown,
            'profit_factor': self.profit_factor,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
