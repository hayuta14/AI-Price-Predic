"""
Database Connection Utility

Cung cấp các hàm tiện ích để kết nối và quản lý database
"""
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from backend.config import config
from backend.database.models import Base

logger = logging.getLogger(__name__)


def create_database_engine(echo: bool = False):
    """
    Tạo database engine với cấu hình từ config
    
    Args:
        echo: Có in SQL queries ra console không (để debug)
        
    Returns:
        SQLAlchemy Engine
    """
    database_url = config.database.get_postgres_url()
    
    engine = create_engine(
        database_url,
        pool_pre_ping=True,  # Kiểm tra kết nối trước khi sử dụng
        pool_size=5,  # Số kết nối trong pool
        max_overflow=10,  # Số kết nối tối đa có thể vượt quá pool_size
        echo=echo
    )
    
    return engine


def test_connection(engine=None) -> bool:
    """
    Test kết nối database
    
    Args:
        engine: SQLAlchemy Engine (nếu None sẽ tạo mới)
        
    Returns:
        True nếu kết nối thành công, False nếu thất bại
    """
    if engine is None:
        engine = create_database_engine()
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"✅ Kết nối PostgreSQL thành công")
            logger.info(f"   Version: {version.split(',')[0]}")
            return True
    except Exception as e:
        logger.error(f"❌ Kết nối PostgreSQL thất bại: {e}")
        return False


def init_database(engine=None) -> bool:
    """
    Khởi tạo database (tạo tables nếu chưa có)
    
    Args:
        engine: SQLAlchemy Engine (nếu None sẽ tạo mới)
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    if engine is None:
        engine = create_database_engine()
    
    try:
        Base.metadata.create_all(engine)
        logger.info("✅ Đã tạo/kiểm tra tables trong database")
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi khi tạo tables: {e}")
        return False


def get_session(engine=None) -> Session:
    """
    Tạo database session
    
    Args:
        engine: SQLAlchemy Engine (nếu None sẽ tạo mới)
        
    Returns:
        SQLAlchemy Session
    """
    if engine is None:
        engine = create_database_engine()
    
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def get_database_info(engine=None) -> dict:
    """
    Lấy thông tin về database
    
    Args:
        engine: SQLAlchemy Engine (nếu None sẽ tạo mới)
        
    Returns:
        Dictionary chứa thông tin database
    """
    if engine is None:
        engine = create_database_engine()
    
    info = {
        'host': config.database.postgres_host,
        'port': config.database.postgres_port,
        'database': config.database.postgres_db,
        'user': config.database.postgres_user,
        'connected': False,
        'version': None,
        'tables': []
    }
    
    try:
        with engine.connect() as conn:
            # Lấy version
            result = conn.execute(text("SELECT version()"))
            info['version'] = result.scalar()
            info['connected'] = True
            
            # Lấy danh sách tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            info['tables'] = [row[0] for row in result]
            
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin database: {e}")
    
    return info
