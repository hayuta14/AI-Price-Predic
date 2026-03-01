"""
Script test kết nối PostgreSQL

Chạy script này để kiểm tra kết nối database:
    python test_db_connection.py
"""
import sys
import logging
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config import config
from backend.database.connection import (
    test_connection,
    init_database,
    get_database_info,
    create_database_engine
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Test kết nối database"""
    logger.info("=" * 80)
    logger.info("🔍 TEST KẾT NỐI POSTGRESQL")
    logger.info("=" * 80)
    
    # Hiển thị thông tin cấu hình
    logger.info("\n📋 Thông tin cấu hình:")
    logger.info(f"   Host: {config.database.postgres_host}")
    logger.info(f"   Port: {config.database.postgres_port}")
    logger.info(f"   Database: {config.database.postgres_db}")
    logger.info(f"   User: {config.database.postgres_user}")
    
    # Test kết nối
    logger.info("\n🔗 Đang kiểm tra kết nối...")
    engine = create_database_engine(echo=False)
    
    if test_connection(engine):
        logger.info("\n✅ Kết nối thành công!")
        
        # Lấy thông tin database
        logger.info("\n📊 Thông tin database:")
        db_info = get_database_info(engine)
        if db_info['connected']:
            logger.info(f"   Version: {db_info['version']}")
            logger.info(f"   Tables: {len(db_info['tables'])} tables")
            if db_info['tables']:
                logger.info(f"   Danh sách tables: {', '.join(db_info['tables'])}")
            else:
                logger.info("   (Chưa có tables)")
        
        # Khởi tạo database
        logger.info("\n🔧 Đang khởi tạo database (tạo tables)...")
        if init_database(engine):
            logger.info("✅ Đã tạo/kiểm tra tables thành công")
            
            # Kiểm tra lại tables
            db_info = get_database_info(engine)
            logger.info(f"\n📊 Sau khi khởi tạo: {len(db_info['tables'])} tables")
            if db_info['tables']:
                logger.info(f"   Tables: {', '.join(db_info['tables'])}")
        else:
            logger.error("❌ Không thể tạo tables")
            return 1
    else:
        logger.error("\n❌ Kết nối thất bại!")
        logger.info("\n💡 Hãy kiểm tra:")
        logger.info("   1. PostgreSQL đã được cài đặt và đang chạy")
        logger.info("   2. Thông tin kết nối trong file .env hoặc config.py")
        logger.info("   3. Database đã được tạo")
        logger.info("   4. User có quyền truy cập")
        return 1
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TEST HOÀN TẤT!")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
