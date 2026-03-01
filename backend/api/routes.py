"""
FastAPI路由定义

提供RESTful API接口
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.repository import ModelRunRepository, OptimizationTrialRepository
from backend.database.models import Base
from backend.config import SystemConfig, config


# Pydantic模型
class OptimizationRequest(BaseModel):
    """优化请求"""
    data_path: Optional[str] = None
    available_features: Optional[List[str]] = None
    initial_equity: float = 100000.0


class OptimizationResponse(BaseModel):
    """优化响应"""
    run_id: int
    status: str
    message: str


# 创建路由器
router = APIRouter()


# 依赖注入：获取数据库会话
def get_db():
    """获取数据库会话（需要在实际应用中实现）"""
    # 这里需要实际的数据库连接
    # 暂时返回None，实际应用中应该从依赖注入获取
    return None


@router.post("/optimize/start", response_model=OptimizationResponse)
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    启动优化任务
    
    Returns:
        优化任务ID和状态
    """
    # TODO: 实现实际的优化逻辑
    # 这里应该：
    # 1. 加载数据
    # 2. 在后台任务中运行优化
    # 3. 返回任务ID
    
    return OptimizationResponse(
        run_id=1,
        status="started",
        message="优化任务已启动"
    )


@router.get("/optimize/status/{run_id}")
async def get_optimization_status(
    run_id: int,
    db: Session = Depends(get_db)
):
    """
    获取优化任务状态
    
    Args:
        run_id: 运行ID
        
    Returns:
        任务状态
    """
    model_run_repo = ModelRunRepository(db)
    model_run = model_run_repo.get_by_id(run_id)
    
    if not model_run:
        raise HTTPException(status_code=404, detail="运行记录不存在")
    
    return {
        "run_id": run_id,
        "status": "completed" if model_run.sharpe_ratio is not None else "running",
        "sharpe_ratio": model_run.sharpe_ratio,
        "max_drawdown": model_run.max_drawdown
    }


@router.get("/optimize/results/{run_id}")
async def get_optimization_results(
    run_id: int,
    db: Session = Depends(get_db)
):
    """
    获取优化结果
    
    Args:
        run_id: 运行ID
        
    Returns:
        优化结果详情
    """
    model_run_repo = ModelRunRepository(db)
    model_run = model_run_repo.get_by_id(run_id)
    
    if not model_run:
        raise HTTPException(status_code=404, detail="运行记录不存在")
    
    return model_run.to_dict()


@router.get("/runs")
async def get_all_runs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    获取所有运行记录
    
    Args:
        skip: 跳过记录数
        limit: 返回记录数
        
    Returns:
        运行记录列表
    """
    model_run_repo = ModelRunRepository(db)
    runs = model_run_repo.get_all(skip=skip, limit=limit)
    
    return [run.to_dict() for run in runs]


@router.get("/runs/{run_id}")
async def get_run(
    run_id: int,
    db: Session = Depends(get_db)
):
    """
    获取单个运行记录
    
    Args:
        run_id: 运行ID
        
    Returns:
        运行记录详情
    """
    model_run_repo = ModelRunRepository(db)
    model_run = model_run_repo.get_by_id(run_id)
    
    if not model_run:
        raise HTTPException(status_code=404, detail="运行记录不存在")
    
    return model_run.to_dict()


@router.get("/runs/top/sharpe")
async def get_top_runs_by_sharpe(
    top_n: int = 5,
    db: Session = Depends(get_db)
):
    """
    获取Sharpe比率最高的运行记录
    
    Args:
        top_n: 返回数量
        
    Returns:
        运行记录列表
    """
    model_run_repo = ModelRunRepository(db)
    runs = model_run_repo.get_top_by_sharpe(top_n=top_n)
    
    return [run.to_dict() for run in runs]


@router.get("/runs/top/stability")
async def get_top_runs_by_stability(
    top_n: int = 5,
    db: Session = Depends(get_db)
):
    """
    获取最稳定的运行记录
    
    Args:
        top_n: 返回数量
        
    Returns:
        运行记录列表
    """
    model_run_repo = ModelRunRepository(db)
    runs = model_run_repo.get_top_by_stability(top_n=top_n)
    
    return [run.to_dict() for run in runs]
