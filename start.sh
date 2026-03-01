#!/bin/bash

# BTCUSDT 15分钟期货交易系统优化平台 - Startup Script
# Script để khởi động dự án

set -e  # Dừng nếu có lỗi

# Màu sắc cho output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}BTCUSDT 15分钟期货交易系统优化平台${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Hàm kiểm tra command có tồn tại không
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Kiểm tra Python
echo -e "${YELLOW}[1/6] Kiểm tra Python...${NC}"
if command_exists python3; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python đã cài đặt: $PYTHON_VERSION${NC}"
elif command_exists python; then
    PYTHON_CMD="python"
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python đã cài đặt: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python chưa được cài đặt!${NC}"
    echo -e "${YELLOW}Vui lòng cài đặt Python 3.11+ từ https://www.python.org/downloads/${NC}"
    exit 1
fi

# Kiểm tra phiên bản Python (cần >= 3.11)
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${RED}✗ Python version quá cũ! Cần Python 3.11 hoặc mới hơn.${NC}"
    exit 1
fi

# 2. Kiểm tra pip
echo -e "${YELLOW}[2/6] Kiểm tra pip...${NC}"
if command_exists pip3; then
    PIP_CMD="pip3"
elif command_exists pip; then
    PIP_CMD="pip"
else
    echo -e "${RED}✗ pip chưa được cài đặt!${NC}"
    echo -e "${YELLOW}Đang cài đặt pip...${NC}"
    $PYTHON_CMD -m ensurepip --upgrade
    PIP_CMD="$PYTHON_CMD -m pip"
fi
echo -e "${GREEN}✓ pip đã sẵn sàng${NC}"

# 3. Cài đặt dependencies
echo -e "${YELLOW}[3/6] Cài đặt dependencies từ requirements.txt...${NC}"
if [ -f "requirements.txt" ]; then
    $PIP_CMD install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies đã được cài đặt${NC}"
else
    echo -e "${RED}✗ Không tìm thấy requirements.txt!${NC}"
    exit 1
fi

# 4. Kiểm tra Docker
echo -e "${YELLOW}[4/6] Kiểm tra Docker...${NC}"
if command_exists docker; then
    echo -e "${GREEN}✓ Docker đã cài đặt${NC}"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ Docker chưa được cài đặt hoặc không chạy${NC}"
    echo -e "${YELLOW}Chương trình sẽ sử dụng SQLite in-memory (chỉ để test)${NC}"
    DOCKER_AVAILABLE=false
fi

# 5. Khởi động Docker containers
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo -e "${YELLOW}[5/6] Khởi động Docker containers (PostgreSQL & MongoDB)...${NC}"
    
    if command_exists docker-compose; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        echo -e "${RED}✗ docker-compose không khả dụng!${NC}"
        DOCKER_AVAILABLE=false
    fi
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        # Kiểm tra xem containers đã chạy chưa
        if docker ps | grep -q "trading_postgres\|trading_mongodb"; then
            echo -e "${GREEN}✓ Docker containers đã đang chạy${NC}"
        else
            $COMPOSE_CMD up -d
            echo -e "${GREEN}✓ Docker containers đã được khởi động${NC}"
            
            # Đợi PostgreSQL sẵn sàng
            echo -e "${YELLOW}Đang đợi PostgreSQL sẵn sàng...${NC}"
            for i in {1..30}; do
                if docker exec trading_postgres pg_isready -U postgres >/dev/null 2>&1; then
                    echo -e "${GREEN}✓ PostgreSQL đã sẵn sàng${NC}"
                    break
                fi
                if [ $i -eq 30 ]; then
                    echo -e "${YELLOW}⚠ PostgreSQL chưa sẵn sàng sau 30 giây, tiếp tục...${NC}"
                else
                    sleep 1
                fi
            done
            
            # Đợi MongoDB sẵn sàng
            echo -e "${YELLOW}Đang đợi MongoDB sẵn sàng...${NC}"
            for i in {1..30}; do
                if docker exec trading_mongodb mongosh --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
                    echo -e "${GREEN}✓ MongoDB đã sẵn sàng${NC}"
                    break
                fi
                if [ $i -eq 30 ]; then
                    echo -e "${YELLOW}⚠ MongoDB chưa sẵn sàng sau 30 giây, tiếp tục...${NC}"
                else
                    sleep 1
                fi
            done
        fi
    fi
else
    echo -e "${YELLOW}[5/6] Bỏ qua Docker (không có Docker)${NC}"
fi

# 6. Tạo thư mục cần thiết
echo -e "${YELLOW}[6/6] Tạo thư mục cần thiết...${NC}"
mkdir -p data
mkdir -p results
echo -e "${GREEN}✓ Thư mục đã được tạo${NC}"

# 7. Chạy chương trình chính
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Bắt đầu chạy chương trình...${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Ensure we're in the project root directory
cd "$(dirname "$0")"

# Set PYTHONPATH to include current directory
export PYTHONPATH="${PWD}:${PYTHONPATH}"

$PYTHON_CMD backend/main.py

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Hoàn thành!${NC}"
echo -e "${GREEN}========================================${NC}"
